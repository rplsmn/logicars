//! Quick test of full model training dynamics
use logicars::{
    create_checkerboard_model, create_checkerboard, create_random_seed,
    TrainingLoop, TrainingConfig, SimpleRng, NGrid,
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
    println!("=== FULL Model Training Dynamics ===\n");
    
    let model = create_checkerboard_model();
    println!("Full model: {} gates, {} update layers", 
             model.total_gates(), model.update.layers.len());
    
    let config = TrainingConfig::checkerboard_sync();
    let mut training = TrainingLoop::new(model, config);
    
    let mut rng = SimpleRng::new(42);
    let target = create_checkerboard(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
    
    for epoch in 0..10 {
        let input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
        let (soft_loss, hard_loss) = training.train_step(&input, &target);
        
        // Check output distribution after 20 steps
        let test_input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
        let mut grid = test_input.clone();
        for _ in 0..20 {
            grid = forward_grid_soft(&training.model, &grid);
        }
        
        let vals: Vec<f64> = (0..256).map(|i| {
            let x = i % 16;
            let y = i / 16;
            grid.get(x as isize, y as isize, 0)
        }).collect();
        
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        println!("Epoch {:2}: soft_loss={:.2}, hard_loss={:.0}, output_range=[{:.3}, {:.3}]", 
                 epoch, soft_loss, hard_loss, min, max);
    }
}
