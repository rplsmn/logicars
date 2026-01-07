//! Debug gate logits after training
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
    let target = create_checkerboard(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
    
    println!("=== Before Training ===");
    print_gate_stats(&training);
    
    // Train for a few epochs
    for epoch in 0..10 {
        let input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng);
        let (soft_loss, hard_loss) = training.train_step(&input, &target);
        if epoch == 0 || epoch == 9 {
            println!("\nEpoch {}: soft_loss={:.4}, hard_loss={:.4}", epoch, soft_loss, hard_loss);
        }
    }
    
    println!("\n=== After 10 Epochs ===");
    print_gate_stats(&training);
}

fn print_gate_stats(training: &TrainingLoop) {
    // Check first update layer gate logits
    let gate = &training.model.update.layers[0].gates[0];
    println!("Update Layer 0, Gate 0 logits:");
    for (i, &logit) in gate.logits.iter().enumerate() {
        if logit.abs() > 0.1 {
            println!("  [{:2}]: {:.4}", i, logit);
        }
    }
    
    // Find max logit index
    let max_idx = gate.logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    println!("  Max logit at index {} (should be 12=A for pass-through)", max_idx);
    
    // Check if logits changed
    let sum: f64 = gate.logits.iter().sum();
    println!("  Sum of logits: {:.4}", sum);
}
