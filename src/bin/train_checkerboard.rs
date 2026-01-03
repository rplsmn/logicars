//! Phase 2.1: Checkerboard Sync Training
//!
//! Trains the DiffLogicCA to generate a checkerboard pattern from random seeds.
//! Uses 8-bit state (8 channels), non-periodic boundaries, 20-step rollout.
//!
//! This is the first multi-channel experiment, validating the N-bit architecture.

use logicars::{
    create_checkerboard, create_random_seed, create_small_checkerboard_model,
    create_checkerboard_model, compute_checkerboard_accuracy,
    TrainingLoop, TrainingConfig, SimpleRng,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE,
    CHECKERBOARD_SYNC_STEPS,
};
use std::time::Instant;

fn main() {
    println!("=== Phase 2.1: Checkerboard Sync Training ===\n");

    // Parse command line args
    let use_small_model = std::env::args().any(|a| a == "--small");
    let epochs: usize = std::env::args()
        .position(|a| a == "--epochs")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(if use_small_model { 100 } else { 500 });

    // Create model
    let model = if use_small_model {
        println!("Using SMALL model for fast testing...\n");
        create_small_checkerboard_model()
    } else {
        println!("Using FULL checkerboard model...\n");
        create_checkerboard_model()
    };

    println!("Model architecture:");
    println!("  Perception: {} gates ({} kernels)", 
             model.perception.total_gates(), 
             model.perception.num_kernels);
    println!("  Update: {} gates ({} layers)", 
             model.update.total_gates(),
             model.update.layers.len());
    println!("  Total: {} gates\n", model.total_gates());

    // Create config
    let config = TrainingConfig::checkerboard_sync();
    println!("Training configuration:");
    println!("  Grid size: {}×{}", CHECKERBOARD_GRID_SIZE, CHECKERBOARD_GRID_SIZE);
    println!("  Channels: {}", CHECKERBOARD_CHANNELS);
    println!("  Steps per epoch: {}", config.num_steps);
    println!("  Epochs: {}", epochs);
    println!("  Non-periodic boundaries: {}", !config.periodic);
    println!();

    // Create training loop
    let mut training_loop = TrainingLoop::new(model, config);

    // Create target pattern
    let target = create_checkerboard(
        CHECKERBOARD_GRID_SIZE,
        CHECKERBOARD_SQUARE_SIZE,
        CHECKERBOARD_CHANNELS,
    );

    println!("Target: {}×{} checkerboard with {}×{} squares\n",
             CHECKERBOARD_GRID_SIZE, CHECKERBOARD_GRID_SIZE,
             CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_SQUARE_SIZE);

    // Training loop
    let eval_interval = if use_small_model { 10 } else { 50 };
    let mut best_accuracy = 0.0;
    let start = Instant::now();
    let mut rng = SimpleRng::new(23); // Match reference seed

    println!("Training...\n");

    for epoch in 0..epochs {
        // Create random seed for this epoch
        let input = create_random_seed(
            CHECKERBOARD_GRID_SIZE,
            CHECKERBOARD_CHANNELS,
            &mut rng,
        );

        // Train step (multi-step rollout with loss at final step)
        let (soft_loss, hard_loss) = training_loop.train_step(&input, &target);

        // Evaluate periodically
        if epoch % eval_interval == 0 || epoch == epochs - 1 {
            // Run hard evaluation
            let test_input = create_random_seed(
                CHECKERBOARD_GRID_SIZE,
                CHECKERBOARD_CHANNELS,
                &mut rng,
            );
            let output = training_loop.run_steps(&test_input, CHECKERBOARD_SYNC_STEPS);
            let accuracy = compute_checkerboard_accuracy(&output, &target);
            
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
            }

            let elapsed = start.elapsed().as_secs_f32();
            println!(
                "Epoch {:4}: soft_loss={:.4}, hard_loss={:.4}, acc={:.2}% (best: {:.2}%) [{:.1}s]",
                epoch, soft_loss, hard_loss, accuracy * 100.0, best_accuracy * 100.0, elapsed
            );
        }
    }

    // Final evaluation
    println!("\n=== Final Evaluation ===\n");

    // Evaluate on training size
    let mut total_acc = 0.0;
    let num_eval = 10;
    for _ in 0..num_eval {
        let test_input = create_random_seed(
            CHECKERBOARD_GRID_SIZE,
            CHECKERBOARD_CHANNELS,
            &mut rng,
        );
        let output = training_loop.run_steps(&test_input, CHECKERBOARD_SYNC_STEPS);
        total_acc += compute_checkerboard_accuracy(&output, &target);
    }
    let train_size_acc = total_acc / num_eval as f64;

    println!("Training size ({}×{}): {:.2}% accuracy",
             CHECKERBOARD_GRID_SIZE, CHECKERBOARD_GRID_SIZE, train_size_acc * 100.0);

    // Evaluate on larger grid (generalization test)
    let large_size = 64;
    let large_target = create_checkerboard(large_size, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
    let large_input = create_random_seed(large_size, CHECKERBOARD_CHANNELS, &mut rng);
    let large_output = training_loop.run_steps(&large_input, CHECKERBOARD_SYNC_STEPS);
    let large_acc = compute_checkerboard_accuracy(&large_output, &large_target);

    println!("Large size ({}×{}): {:.2}% accuracy (generalization test)",
             large_size, large_size, large_acc * 100.0);

    // Print sample output
    println!("\n=== Sample Output (channel 0, 8×8 top-left) ===\n");
    let sample_input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
    let sample_output = training_loop.run_steps(&sample_input, CHECKERBOARD_SYNC_STEPS);

    print!("Target:    ");
    for x in 0..8 {
        let v = if target.get(x, 0, 0) > 0.5 { "█" } else { "░" };
        print!("{}", v);
    }
    println!();

    print!("Predicted: ");
    for x in 0..8 {
        let v = if sample_output.get(x, 0, 0) > 0.5 { "█" } else { "░" };
        print!("{}", v);
    }
    println!("\n");

    // Summary
    let total_time = start.elapsed().as_secs_f32();
    println!("=== Summary ===");
    println!("  Total time: {:.1}s", total_time);
    println!("  Best accuracy: {:.2}%", best_accuracy * 100.0);
    println!("  Gates: {}", training_loop.model.total_gates());
}
