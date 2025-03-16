mod logic_gates;
mod circuits;
mod difflogicca;
mod python_bindings;

use pyo3::prelude::*;

// Diagnostics helper function remains in the lib file
fn analyze_gate_distributions(circuits: &[circuits::PerceptionCircuit], epoch: usize) {
    if epoch % 10 != 0 {
        return;
    }
    
    // Count dominant gate types across all circuits
    let mut gate_type_counts = [0; 16];
    let mut total_gates = 0;
    let mut max_probability_sum = 0.0;
    let mut max_probability_count = 0;
    
    // Sample one circuit to show detailed distribution
    if let Some(first_circuit) = circuits.get(0) {
        if let Some(first_layer) = first_circuit.circuit.layers.get(0) {
            if let Some(first_gate) = first_layer.gates.get(0) {
                println!("\nEpoch {}: Sample gate distribution:", epoch);
                println!("{:?}", first_gate.probability);
                
                // Find max probability
                let max_prob: f32 = first_gate.probability.iter().fold(0.0, |a, &b| a.max(b));
                println!("Max probability in sample: {:.4}", max_prob);
            }
        }
    }
    
    // Calculate statistics across all gates
    for circuit in circuits {
        for layer in &circuit.circuit.layers {
            for gate in &layer.gates {
                let max_idx = gate.probability.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                
                gate_type_counts[max_idx] += 1;
                total_gates += 1;
                
                let max_prob: f32 = gate.probability.iter().fold(0.0, |a, &b| a.max(b));
                max_probability_sum += max_prob;
                if max_prob > 0.5 {
                    max_probability_count += 1;
                }
            }
        }
    }
    
    // Print summary statistics
    println!("Gate distribution across all circuits:");
    println!("FALSE: {:.1}%, AND: {:.1}%, A_AND_NOT_B: {:.1}%, A: {:.1}%, NOT_A_AND_B: {:.1}%, B: {:.1}%, XOR: {:.1}%, OR: {:.1}%", 
        gate_type_counts[0] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[1] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[2] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[3] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[4] as f32 / total_gates as f32 * 100.0, 
        gate_type_counts[5] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[6] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[7] as f32 / total_gates as f32 * 100.0);
    
    println!("NOR: {:.1}%, XNOR: {:.1}%, NOT_B: {:.1}%, A_OR_NOT_B: {:.1}%, NOT_A: {:.1}%, NOT_A_OR_B: {:.1}%, NAND: {:.1}%, TRUE: {:.1}%",
        gate_type_counts[8] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[9] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[10] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[11] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[12] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[13] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[14] as f32 / total_gates as f32 * 100.0,
        gate_type_counts[15] as f32 / total_gates as f32 * 100.0);
    
    println!("Avg max probability: {:.3}, Gates with p>0.5: {:.1}%", 
        max_probability_sum / total_gates as f32,
        max_probability_count as f32 / total_gates as f32 * 100.0);
}

// Make public the elements that need to be accessed from other modules
pub use difflogicca::{DiffLogicCA, create_game_of_life, create_glider};
pub use circuits::{PerceptionCircuit, UpdateCircuit, Circuit, GateLayer};
pub use logic_gates::{LogicGate, LogicOp};

// The PyO3 module definition
#[pymodule]
fn logicars(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_bindings::register(py, m)?;
    Ok(())
}