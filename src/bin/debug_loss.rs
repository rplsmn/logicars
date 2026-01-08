//! Debug actual loss computation
use logicars::{
    create_small_checkerboard_model, create_checkerboard, create_random_seed,
    SimpleRng, NGrid,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE,
    update::DiffLogicCA,
};

fn forward_grid_soft(model: &DiffLogicCA, input: &NGrid) -> NGrid {
    let mut output = NGrid::new(input.width, input.height, input.channels, input.boundary);
    for y in 0..input.height {
        for x in 0..input.width {
            let neighborhood = input.neighborhood(x, y);
            let (cell_output, _, _) = model.forward_soft(&neighborhood);
            for (c, &val) in cell_output.iter().enumerate() {
                output.set(x, y, c, val);
            }
        }
    }
    output
}

fn main() {
    let model = create_small_checkerboard_model();
    let mut rng = SimpleRng::new(42);
    let input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
    let target = create_checkerboard(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
    
    // Run 20 steps
    let mut grid = input.clone();
    for _ in 0..20 {
        grid = forward_grid_soft(&model, &grid);
    }
    
    // Compute loss on channel 0 only (matching training)
    let mut loss = 0.0;
    let mut correct = 0;
    for y in 0..CHECKERBOARD_GRID_SIZE {
        for x in 0..CHECKERBOARD_GRID_SIZE {
            let pred = grid.get(x as isize, y as isize, 0);
            let tgt = target.get(x as isize, y as isize, 0);
            loss += (pred - tgt).powi(2);
            if (pred > 0.5) == (tgt > 0.5) {
                correct += 1;
            }
        }
    }
    
    println!("=== Loss Computation Debug ===");
    println!("Soft loss (channel 0): {:.4}", loss);
    println!("Hard accuracy: {} / 256 = {:.2}%", correct, 100.0 * correct as f64 / 256.0);
    
    // Show a few example cells
    println!("\nSample cells (x, pred, target, error^2):");
    for x in 0..8 {
        let pred = grid.get(x, 0, 0);
        let tgt = target.get(x, 0, 0);
        println!("  x={}: pred={:.4}, tgt={:.1}, err^2={:.4}", x, pred, tgt, (pred - tgt).powi(2));
    }
}
