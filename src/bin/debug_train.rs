//! Debug training dynamics
use logicars::{
    create_small_checkerboard_model, create_checkerboard, create_random_seed,
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
    let model = create_small_checkerboard_model();
    let config = TrainingConfig::checkerboard_sync();
    let mut training = TrainingLoop::new(model, config);
    
    let mut rng = SimpleRng::new(42);
    let target = create_checkerboard(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
    
    println!("=== Training Dynamics Debug ===\n");
    
    for epoch in 0..50 {
        let input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
        let (soft_loss, hard_loss) = training.train_step(&input, &target);
        
        if epoch % 10 == 0 || epoch == 49 {
            // Check output distribution after 20 steps
            let test_input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
            let mut grid = test_input.clone();
            for _ in 0..20 {
                grid = forward_grid_soft(&training.model, &grid);
            }
            
            let mut vals: Vec<f64> = (0..256).map(|i| {
                let x = i % 16;
                let y = i / 16;
                grid.get(x as isize, y as isize, 0)
            }).collect();
            
            let mean: f64 = vals.iter().sum::<f64>() / 256.0;
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            println!("Epoch {:2}: soft_loss={:.2}, hard_loss={:.0}, output_range=[{:.3}, {:.3}], mean={:.3}", 
                     epoch, soft_loss, hard_loss, min, max, mean);
        }
    }
}
