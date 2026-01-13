//! Phase 2.2: Checkerboard Async Training
//!
//! Trains the DiffLogicCA with asynchronous updates (fire rate masking) to develop
//! self-healing capabilities. The model learns to generate a checkerboard pattern
//! that can recover from damage.
//!
//! Key differences from sync training:
//! - Fire rate = 0.6: Only 60% of cells update per step (stochastic)
//! - 50 steps per epoch (vs 20 for sync): More steps needed for convergence
//! - Batch size = 1 (reference uses 1 for async)
//! - Self-healing: Model recovers from random damage
//!
//! Options:
//!   --small           Use smaller model for fast testing
//!   --epochs=N        Number of epochs to train (default: 1000)
//!   --log-interval=N  How often to log accuracy/loss (default: 50)
//!   --log=FILE        Write training log to file (append mode)
//!   --save=PATH       Save trained model as HardCircuit JSON at end of training

use logicars::{
    compute_checkerboard_accuracy, create_checkerboard, create_checkerboard_async_model,
    create_random_seed, create_small_checkerboard_model, HardCircuit, SimpleRng, TrainingConfig,
    TrainingLoop, CHECKERBOARD_ASYNC_GRID_SIZE, CHECKERBOARD_ASYNC_STEPS, CHECKERBOARD_CHANNELS,
    CHECKERBOARD_SQUARE_SIZE,
};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() {
    println!("=== Phase 2.2: Checkerboard Async Training ===\n");
    println!("Training with fire rate masking for self-healing capability.\n");

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let use_small_model = args.iter().any(|a| a == "--small");

    // Parse --epochs=N for custom epoch count
    let epochs: usize = args
        .iter()
        .find(|a| a.starts_with("--epochs="))
        .and_then(|a| a.strip_prefix("--epochs="))
        .and_then(|s| s.parse().ok())
        .unwrap_or(if use_small_model { 200 } else { 1000 });

    // Parse --log-interval=N for custom logging frequency
    let eval_interval: usize = args
        .iter()
        .find(|a| a.starts_with("--log-interval="))
        .and_then(|a| a.strip_prefix("--log-interval="))
        .and_then(|s| s.parse().ok())
        .unwrap_or(if use_small_model { 20 } else { 50 });

    // Parse --log=FILE for logging to file
    let log_file: Option<String> = args
        .iter()
        .find(|a| a.starts_with("--log="))
        .and_then(|a| a.strip_prefix("--log="))
        .map(|s| s.to_string());

    // Parse --save=PATH for model saving
    let save_path: Option<String> = args
        .iter()
        .find(|a| a.starts_with("--save="))
        .and_then(|a| a.strip_prefix("--save="))
        .map(|s| s.to_string());

    // Create model - async uses deeper network (14×256 vs 10×256 for sync)
    let model = if use_small_model {
        println!("Using SMALL model for fast testing...\n");
        create_small_checkerboard_model()
    } else {
        println!("Using ASYNC checkerboard model (14×256 hidden layers)...\n");
        create_checkerboard_async_model()
    };

    println!("Model architecture:");
    println!(
        "  Perception: {} gates ({} kernels)",
        model.perception.total_gates(),
        model.perception.num_kernels
    );
    println!(
        "  Update: {} gates ({} layers)",
        model.update.total_gates(),
        model.update.layers.len()
    );
    println!("  Total: {} gates\n", model.total_gates());

    // Create async config
    let config = TrainingConfig::checkerboard_async();
    println!("Training configuration (ASYNC mode):");
    println!(
        "  Grid size: {}×{}",
        CHECKERBOARD_ASYNC_GRID_SIZE, CHECKERBOARD_ASYNC_GRID_SIZE
    );
    println!("  Channels: {}", CHECKERBOARD_CHANNELS);
    println!("  Steps per epoch: {} (async)", config.num_steps);
    println!("  Fire rate: {:.1}%", config.fire_rate * 100.0);
    println!("  Epochs: {}", epochs);
    println!("  Log interval: {}", eval_interval);
    println!("  Non-periodic boundaries: {}", !config.periodic);
    println!(
        "  Batch size: {} (async uses single samples)",
        config.batch_size
    );
    if let Some(ref log_path) = log_file {
        println!("  Log file: {}", log_path);
    }
    if let Some(ref save) = save_path {
        println!("  Model save path: {}", save);
    }
    println!();

    // Create training loop
    let mut training_loop = TrainingLoop::new(model, config);
    training_loop.set_seed(42); // For reproducibility

    // Create target pattern
    let target = create_checkerboard(
        CHECKERBOARD_ASYNC_GRID_SIZE,
        CHECKERBOARD_SQUARE_SIZE,
        CHECKERBOARD_CHANNELS,
    );

    println!(
        "Target: {}×{} checkerboard with {}×{} squares\n",
        CHECKERBOARD_ASYNC_GRID_SIZE,
        CHECKERBOARD_ASYNC_GRID_SIZE,
        CHECKERBOARD_SQUARE_SIZE,
        CHECKERBOARD_SQUARE_SIZE
    );

    // Training loop
    let mut best_accuracy = 0.0;
    let start = Instant::now();
    let mut rng = SimpleRng::new(23); // Match reference seed

    // Open log file if specified (append mode for resuming)
    let mut log_writer: Option<BufWriter<File>> = log_file.as_ref().map(|path| {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect(&format!("Failed to open log file: {}", path));
        BufWriter::new(file)
    });

    // Write header to log file if new
    if let Some(ref mut writer) = log_writer {
        writeln!(writer, "# Checkerboard Async Training Log").unwrap();
        writeln!(writer, "# Started: {:?}", std::time::SystemTime::now()).unwrap();
        writeln!(
            writer,
            "# Model: {} gates",
            training_loop.model.total_gates()
        )
        .unwrap();
        writeln!(writer, "# Fire rate: {}", training_loop.config.fire_rate).unwrap();
        writeln!(
            writer,
            "# epoch,soft_loss,hard_loss,accuracy,best_accuracy,elapsed_s"
        )
        .unwrap();
        writer.flush().unwrap();
    }

    println!(
        "Training with async updates (fire rate = {:.1}%)...\n",
        training_loop.config.fire_rate * 100.0
    );

    let mut prev_loss: Option<f64> = None;
    let mut prev_acc: Option<f64> = None;
    let mut current_lr = training_loop.config.learning_rate;
    let mut cooldown_triggered = false;
    let mut cooldown_epoch: Option<usize> = None;
    let mut early_stop_counter = 0;
    let early_stop_patience = 5; // Number of evals with perfect acc/loss before stopping
    let mut finalized = false;

    for epoch in 0..epochs {
        // Create random seed for this epoch
        let input = create_random_seed(
            CHECKERBOARD_ASYNC_GRID_SIZE,
            CHECKERBOARD_CHANNELS,
            &mut rng,
        );

        // Train step (single sample for async)
        let (soft_loss, hard_loss) = training_loop.train_step(&input, &target);

        // Evaluate periodically
        let is_last = epoch == epochs - 1;
        if epoch % eval_interval == 0 || is_last {
            // Run hard evaluation with async steps
            let test_input = create_random_seed(
                CHECKERBOARD_ASYNC_GRID_SIZE,
                CHECKERBOARD_CHANNELS,
                &mut rng,
            );
            let output = training_loop.run_steps(&test_input, CHECKERBOARD_ASYNC_STEPS);
            let accuracy = compute_checkerboard_accuracy(&output, &target);

            // Detect first time reaching 100% accuracy
            if !cooldown_triggered && accuracy >= 1.0 {
                cooldown_triggered = true;
                cooldown_epoch = Some(epoch);
                current_lr = 0.05; // Cool off LR sharply for fine-tuning
                training_loop.set_learning_rate(current_lr);
                println!(
                    "[LR SCHEDULE] 100% accuracy reached at epoch {}. LR cooled to {:.5}",
                    epoch, current_lr
                );
            }

            // LR schedule: adjust based on loss/accuracy trend (only before cooldown)
            if !cooldown_triggered {
                if let (Some(prev_l), Some(prev_a)) = (prev_loss, prev_acc) {
                    if soft_loss > prev_l && accuracy < prev_a {
                        // Bad jump: decrease LR
                        current_lr *= 0.95;
                        training_loop.set_learning_rate(current_lr);
                    } else if soft_loss < prev_l && accuracy > prev_a {
                        // Good direction: increase LR
                        current_lr *= 1.05;
                        training_loop.set_learning_rate(current_lr);
                    }
                }
            }
            prev_loss = Some(soft_loss);
            prev_acc = Some(accuracy);

            if accuracy > best_accuracy {
                best_accuracy = accuracy;
            }

            // Early stopping: if hard_loss == 0 and acc == 1.0 for N evals, finalize
            if accuracy >= 1.0 && hard_loss == 0.0 {
                early_stop_counter += 1;
                if early_stop_counter >= early_stop_patience && !finalized {
                    println!(
                        "[EARLY STOP] Finalized: hard_loss=0 and acc=100% for {} evals (epoch {})",
                        early_stop_patience, epoch
                    );
                    finalized = true;
                    break;
                }
            } else {
                early_stop_counter = 0;
            }

            let elapsed = start.elapsed().as_secs_f32();

            // Print to stdout
            println!(
                "Epoch {:4}: soft_loss={:.4}, hard_loss={:.4}, acc={:.2}% (best: {:.2}%) [{:.1}s] LR={:.5}",
                epoch, soft_loss, hard_loss, accuracy * 100.0, best_accuracy * 100.0, elapsed, current_lr
            );

            // Write to log file
            if let Some(ref mut writer) = log_writer {
                writeln!(
                    writer,
                    "{},{:.6},{:.4},{:.6},{:.6},{:.1},{:.5}",
                    epoch, soft_loss, hard_loss, accuracy, best_accuracy, elapsed, current_lr
                )
                .unwrap();
                writer.flush().unwrap();
            }
        }
    }

    // Final evaluation
    println!("\n=== Final Evaluation ===\n");

    // Evaluate on training size
    let mut total_acc = 0.0;
    let num_eval = 10;
    for _ in 0..num_eval {
        let test_input = create_random_seed(
            CHECKERBOARD_ASYNC_GRID_SIZE,
            CHECKERBOARD_CHANNELS,
            &mut rng,
        );
        let output = training_loop.run_steps(&test_input, CHECKERBOARD_ASYNC_STEPS);
        total_acc += compute_checkerboard_accuracy(&output, &target);
    }
    let train_size_acc = total_acc / num_eval as f64;

    println!(
        "Training size ({}×{}): {:.2}% accuracy",
        CHECKERBOARD_ASYNC_GRID_SIZE,
        CHECKERBOARD_ASYNC_GRID_SIZE,
        train_size_acc * 100.0
    );

    // Test self-healing: damage the grid and see if it recovers
    println!("\n=== Self-Healing Test ===\n");

    // Start from pattern (run until converged)
    let mut grid = create_random_seed(
        CHECKERBOARD_ASYNC_GRID_SIZE,
        CHECKERBOARD_CHANNELS,
        &mut rng,
    );
    for _ in 0..100 {
        grid = training_loop.run_steps(&grid, 1);
    }
    let initial_acc = compute_checkerboard_accuracy(&grid, &target);
    println!(
        "Initial accuracy (before damage): {:.2}%",
        initial_acc * 100.0
    );

    // Damage: zero out center 4x4 region
    let damage_size = 4;
    let damage_start = (CHECKERBOARD_ASYNC_GRID_SIZE - damage_size) / 2;
    for y in damage_start..(damage_start + damage_size) {
        for x in damage_start..(damage_start + damage_size) {
            for c in 0..CHECKERBOARD_CHANNELS {
                grid.set(x, y, c, 0.0);
            }
        }
    }
    let damaged_acc = compute_checkerboard_accuracy(&grid, &target);
    println!(
        "After damage ({}×{} center zeroed): {:.2}%",
        damage_size,
        damage_size,
        damaged_acc * 100.0
    );

    // Run recovery steps
    for step in [10, 25, 50, 100] {
        let mut recovery_grid = grid.clone();
        for _ in 0..step {
            recovery_grid = training_loop.run_steps(&recovery_grid, 1);
        }
        let recovery_acc = compute_checkerboard_accuracy(&recovery_grid, &target);
        println!(
            "After {} recovery steps: {:.2}%",
            step,
            recovery_acc * 100.0
        );
    }

    // Summary
    let total_time = start.elapsed().as_secs_f32();
    println!("\n=== Summary ===");
    println!("  Total time: {:.1}s", total_time);
    println!("  Best accuracy: {:.2}%", best_accuracy * 100.0);
    println!("  Gates: {}", training_loop.model.total_gates());
    println!(
        "  Fire rate: {:.1}%",
        training_loop.config.fire_rate * 100.0
    );

    // Save model if requested
    if let Some(ref path) = save_path {
        println!("\n=== Saving Model ===");
        let circuit = HardCircuit::from_soft(&training_loop.model);
        match circuit.save(path) {
            Ok(()) => {
                println!("  Saved HardCircuit to: {}", path);
                println!("  Total gates: {}", circuit.total_gate_count());
                println!("  Active gates: {}", circuit.active_gate_count());
            }
            Err(e) => {
                eprintln!("  Error saving model: {}", e);
            }
        }
    }
}
