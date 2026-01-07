//! Verify forward pass correctness
use logicars::phase_0_1::ProbabilisticGate;

fn main() {
    // Create a pass-through gate
    let gate = ProbabilisticGate::new();
    
    println!("=== Pass-through Gate Forward Test ===\n");
    
    // Test with various inputs
    for &a in &[0.0, 0.5, 1.0] {
        for &b in &[0.0, 0.5, 1.0] {
            let soft_output = gate.execute_soft(a, b);
            let hard_output = gate.execute_hard(a > 0.5, b > 0.5);
            println!("a={:.1}, b={:.1}: soft={:.4}, hard={}", a, b, soft_output, hard_output);
        }
    }
    
    println!("\n=== Expected for pass-through (A) ===");
    println!("soft output should be ≈ a");
    println!("hard output should be = (a > 0.5)");
}
