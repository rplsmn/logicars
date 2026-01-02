//! Phase 1.5: Game of Life Validation
//!
//! Trains the full DiffLogicCA architecture on Game of Life rules.
//! Uses all 512 neighborhood configurations.
//!
//! Exit criteria: >95% hard accuracy
//! Also validates with gliders and blinkers simulation.

use logicars::{
    DiffLogicCA, DiffLogicCATrainer, GolTruthTable, NGrid, NNeighborhood,
    PerceptionModule, UpdateModule, ConnectionType,
};
use std::time::Instant;

fn main() {
    println!("=== Phase 1.5: Game of Life Validation ===\n");

    // Use smaller model for faster iteration (can scale up later)
    let use_small_model = std::env::args().any(|a| a == "--small");
    
    let model = if use_small_model {
        println!("Using SMALL model for fast testing...\n");
        create_small_model()
    } else {
        println!("Using FULL GoL model...\n");
        DiffLogicCA::gol()
    };
    
    println!("Model architecture:");
    println!("  Perception: {} gates", model.perception.total_gates());
    println!("  Update: {} gates", model.update.total_gates());
    println!("  Total: {} gates\n", model.total_gates());

    // Create trainer
    let mut trainer = DiffLogicCATrainer::new(model, 0.05);

    // Get all 512 GoL configurations
    let truth_table = GolTruthTable::new();
    let alive_count = truth_table.targets.iter().filter(|&&t| t).count();
    println!("Training data:");
    println!("  Total configurations: 512");
    println!("  Alive outcomes: {}", alive_count);
    println!("  Dead outcomes: {}\n", 512 - alive_count);

    // Training loop
    let max_epochs = if use_small_model { 1000 } else { 5000 };
    let target_accuracy = 0.95;
    let eval_interval = if use_small_model { 50 } else { 100 };

    println!("Training...");
    println!("Target: >{:.0}% hard accuracy\n", target_accuracy * 100.0);

    let mut best_accuracy = 0.0;
    let start = Instant::now();

    for epoch in 0..max_epochs {
        let mut epoch_loss = 0.0;

        // Train on all 512 configurations
        for idx in 0..512 {
            let neighborhood = NNeighborhood::from_gol_index(idx);
            let target = if truth_table.target(idx) { 1.0 } else { 0.0 };
            let loss = trainer.train_step(&neighborhood, &[target]);
            epoch_loss += loss;
        }
        epoch_loss /= 512.0;

        // Evaluate periodically
        if epoch % eval_interval == 0 || epoch == max_epochs - 1 {
            let accuracy = evaluate_accuracy(&trainer.model, &truth_table);
            let elapsed = start.elapsed().as_secs_f32();
            
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
            }

            println!(
                "Epoch {:5}: Loss = {:.6}, Acc = {:.2}% (best: {:.2}%) [{:.1}s]",
                epoch, epoch_loss, accuracy * 100.0, best_accuracy * 100.0, elapsed
            );

            if accuracy >= target_accuracy {
                println!("\n🎉 TARGET ACCURACY ACHIEVED!\n");
                break;
            }
        }
    }

    // Final evaluation
    let final_accuracy = evaluate_accuracy(&trainer.model, &truth_table);
    println!("\nFinal Results:");
    println!("  Hard accuracy: {:.2}%", final_accuracy * 100.0);
    println!("  Target: >{:.0}%", target_accuracy * 100.0);
    println!("  Exit criteria met: {}", final_accuracy >= target_accuracy);

    // Simulation tests
    println!("\n=== Simulation Tests ===\n");

    // Test 1: Blinker (period 2 oscillator)
    println!("--- Blinker Test ---");
    test_blinker(&trainer.model);

    // Test 2: Glider (moves diagonally)
    println!("\n--- Glider Test ---");
    test_glider(&trainer.model);

    // Summary
    println!("\n=== Summary ===");
    if final_accuracy >= target_accuracy {
        println!("✅ Phase 1.5 COMPLETE: All exit criteria met");
    } else {
        println!("❌ Phase 1.5 INCOMPLETE: Accuracy {:.2}% < {:.0}%", 
            final_accuracy * 100.0, target_accuracy * 100.0);
    }
}

/// Evaluate hard accuracy on all 512 configurations
fn evaluate_accuracy(model: &DiffLogicCA, truth_table: &GolTruthTable) -> f64 {
    let mut correct = 0;

    for idx in 0..512 {
        let neighborhood = NNeighborhood::from_gol_index(idx);
        let output = model.forward_hard(&neighborhood);
        let predicted = output[0] > 0.5;
        let target = truth_table.target(idx);

        if predicted == target {
            correct += 1;
        }
    }

    correct as f64 / 512.0
}

/// Test blinker oscillator (period 2)
fn test_blinker(model: &DiffLogicCA) {
    // Blinker: vertical line of 3 cells → horizontal → vertical
    // Initial state (5x5 grid, blinker in center):
    // . . . . .
    // . . X . .
    // . . X . .
    // . . X . .
    // . . . . .

    let mut grid = NGrid::periodic(5, 5, 1);
    grid.set(2, 1, 0, 1.0);
    grid.set(2, 2, 0, 1.0);
    grid.set(2, 3, 0, 1.0);

    println!("Initial state (vertical):");
    print_grid(&grid);

    // Step 1: should become horizontal
    let grid1 = step_grid(model, &grid);
    println!("After step 1 (should be horizontal):");
    print_grid(&grid1);

    // Step 2: should return to vertical
    let grid2 = step_grid(model, &grid1);
    println!("After step 2 (should be vertical):");
    print_grid(&grid2);

    // Check if period 2
    let matches = grids_match(&grid, &grid2);
    if matches {
        println!("✅ Blinker works correctly (period 2)");
    } else {
        println!("❌ Blinker does not oscillate correctly");
    }
}

/// Test glider (moves diagonally)
fn test_glider(model: &DiffLogicCA) {
    // Glider pattern:
    // . X .
    // . . X
    // X X X

    let mut grid = NGrid::periodic(10, 10, 1);
    grid.set(1, 0, 0, 1.0);
    grid.set(2, 1, 0, 1.0);
    grid.set(0, 2, 0, 1.0);
    grid.set(1, 2, 0, 1.0);
    grid.set(2, 2, 0, 1.0);

    println!("Initial glider:");
    print_grid(&grid);

    // Run for 4 steps (one full cycle, should move diagonally by 1)
    let mut current = grid.clone();
    for step in 1..=4 {
        current = step_grid(model, &current);
        if step == 4 {
            println!("After 4 steps:");
            print_grid(&current);
        }
    }

    // Check if glider moved (should have same pattern shifted)
    let live_cells: Vec<(usize, usize)> = (0..10)
        .flat_map(|y| (0..10).map(move |x| (x, y)))
        .filter(|&(x, y)| current.get(x as isize, y as isize, 0) > 0.5)
        .collect();

    println!("Live cells after 4 steps: {:?}", live_cells);
    
    if live_cells.len() == 5 {
        println!("✅ Glider maintains 5 cells");
    } else {
        println!("❌ Glider cell count changed: {} (expected 5)", live_cells.len());
    }
}

/// Step the entire grid through the model
fn step_grid(model: &DiffLogicCA, grid: &NGrid) -> NGrid {
    let mut output = NGrid::periodic(grid.width, grid.height, 1);

    for y in 0..grid.height {
        for x in 0..grid.width {
            let neighborhood = grid.neighborhood(x, y);
            let result = model.forward_hard(&neighborhood);
            output.set(x, y, 0, result[0]);
        }
    }

    output
}

/// Check if two grids match
fn grids_match(a: &NGrid, b: &NGrid) -> bool {
    for y in 0..a.height {
        for x in 0..a.width {
            let va = a.get(x as isize, y as isize, 0) > 0.5;
            let vb = b.get(x as isize, y as isize, 0) > 0.5;
            if va != vb {
                return false;
            }
        }
    }
    true
}

/// Print grid to console
fn print_grid(grid: &NGrid) {
    for y in 0..grid.height {
        for x in 0..grid.width {
            let v = grid.get(x as isize, y as isize, 0);
            print!("{}", if v > 0.5 { "█" } else { "·" });
        }
        println!();
    }
    println!();
}

/// Create a smaller model for fast testing
fn create_small_model() -> DiffLogicCA {
    // Medium perception: 8 kernels, [9→8→4→2→1] (60 gates)
    let perception = PerceptionModule::new(
        1,  // channels
        8,  // kernels (vs 16 for full)
        &[9, 8, 4, 2, 1],
        &[
            ConnectionType::FirstKernel,
            ConnectionType::Unique,
            ConnectionType::Unique,
            ConnectionType::Unique,
        ],
    );
    
    // Medium update: Input = 1 (center) + 8 (kernels) = 9
    // Need more capacity to learn GoL. Reference uses [17→128×10→...→1]
    // We'll use [9→16→16→16→8→4→2→1] = ~70 gates
    let update = UpdateModule::new(&[9, 16, 16, 16, 8, 4, 2, 1]);
    
    DiffLogicCA::new(perception, update)
}
