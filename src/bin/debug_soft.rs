//! Debug soft vs hard outputs
use logicars::{
    create_small_checkerboard_model, create_checkerboard, create_random_seed,
    TrainingLoop, TrainingConfig, SimpleRng, NNeighborhood,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE,
};

fn main() {
    let model = create_small_checkerboard_model();
    let config = TrainingConfig::checkerboard_sync();
    let mut training = TrainingLoop::new(model, config);
    
    let mut rng = SimpleRng::new(42);
    let input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
    
    // Run soft forward for one cell to see intermediate values
    println!("=== Single Cell Soft Forward ===");
    
    // Get neighborhood for cell (0,0)
    let mut cells = vec![0.0; 9 * CHECKERBOARD_CHANNELS];
    for pos in 0..9 {
        let (dx, dy) = match pos {
            0 => (-1, -1), 1 => (0, -1), 2 => (1, -1),
            3 => (-1, 0),  4 => (0, 0),  5 => (1, 0),
            6 => (-1, 1),  7 => (0, 1),  8 => (1, 1),
            _ => unreachable!(),
        };
        for c in 0..CHECKERBOARD_CHANNELS {
            cells[pos * CHECKERBOARD_CHANNELS + c] = input.get(dx, dy, c);
        }
    }
    let neighborhood = NNeighborhood::new(CHECKERBOARD_CHANNELS, cells);
    
    // Run forward
    let (output, perc_act, upd_act) = training.model.forward_soft(&neighborhood);
    
    println!("Input neighborhood center (ch0-3): {:?}", 
             (0..4).map(|c| format!("{:.2}", neighborhood.get(4, c))).collect::<Vec<_>>());
    println!("Perception output size: {}", perc_act.len());
    println!("Update activations layers: {}", upd_act.len());
    println!("Final output (ch0-3): {:?}",
             output.iter().take(4).map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    
    // Check if outputs are near 0.5 (uncertain)
    let output_mean: f64 = output.iter().sum::<f64>() / output.len() as f64;
    let output_min = output.iter().cloned().fold(f64::INFINITY, f64::min);
    let output_max = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("Output stats: mean={:.4}, min={:.4}, max={:.4}", output_mean, output_min, output_max);
}
