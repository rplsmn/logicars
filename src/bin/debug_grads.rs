//! Debug gradient magnitudes
use logicars::{
    create_small_checkerboard_model, create_checkerboard, create_random_seed,
    TrainingLoop, TrainingConfig, SimpleRng,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE,
};

fn main() {
    let model = create_small_checkerboard_model();
    let config = TrainingConfig::checkerboard_sync();
    let mut training = TrainingLoop::new(model, config);
    
    let mut rng = SimpleRng::new(42);
    let input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
    let target = create_checkerboard(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
    
    // Run forward pass only and check output values
    println!("=== Forward Pass Analysis (Hard Mode) ===");
    let output = training.run_steps(&input, 20);
    
    // Check channel 0 output distribution
    let mut vals: Vec<f64> = Vec::new();
    for y in 0..CHECKERBOARD_GRID_SIZE {
        for x in 0..CHECKERBOARD_GRID_SIZE {
            vals.push(output.get(x as isize, y as isize, 0));
        }
    }
    
    let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    println!("Channel 0 output stats:");
    println!("  Mean: {:.4}", mean);
    println!("  Min:  {:.4}", min);
    println!("  Max:  {:.4}", max);
    println!("  Range: {:.4}", max - min);
    
    // Count how many are > 0.5
    let above_half = vals.iter().filter(|&&v| v > 0.5).count();
    println!("  Values > 0.5: {} / {}", above_half, vals.len());
    
    // Sample first 8 values (should be 0,0,1,1,0,0,1,1 for checkerboard)
    println!("  First 8 values: {:?}", &vals[..8].iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>());
    println!("  Target first 8: {:?}", (0..8).map(|x| target.get(x, 0, 0)).map(|v| format!("{:.1}", v)).collect::<Vec<_>>());
}
