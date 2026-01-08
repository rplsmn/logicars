//! Debug multi-step rollout
use logicars::{
    create_small_checkerboard_model, create_checkerboard, create_random_seed,
    TrainingConfig, SimpleRng, NGrid, BoundaryCondition, NNeighborhood,
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
    
    println!("=== Soft Multi-Step Rollout ===");
    
    let mut grid = input.clone();
    for step in 0..=20 {
        if step == 0 || step == 1 || step == 5 || step == 10 || step == 20 {
            // Compute stats for channel 0
            let mut vals: Vec<f64> = Vec::new();
            for y in 0..CHECKERBOARD_GRID_SIZE {
                for x in 0..CHECKERBOARD_GRID_SIZE {
                    vals.push(grid.get(x as isize, y as isize, 0));
                }
            }
            let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let near_half = vals.iter().filter(|&&v| (v - 0.5).abs() < 0.1).count();
            println!("Step {:2}: mean={:.4}, min={:.4}, max={:.4}, near_0.5={}/256", 
                     step, mean, min, max, near_half);
        }
        
        if step < 20 {
            grid = forward_grid_soft(&model, &grid);
        }
    }
}
