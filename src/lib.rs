use ndarray::{Array2, Array3, Array4, s};
use numpy::IntoPyArray;
use numpy::PyArray3;
use numpy::PyReadonlyArray4;
use rand::{prelude::*, rng};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::PyReadonlyArray3;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// Diagnostics

fn analyze_gate_distributions(circuits: &[PerceptionCircuit], epoch: usize) {
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

// Logic gate operations as per the paper
#[allow(non_camel_case_types)]
#[derive(Clone)]
pub enum LogicOp {
    FALSE,      // 0
    AND,        // a * b
    A_AND_NOT_B, // a * (1-b)
    A,          // a
    NOT_A_AND_B, // (1-a) * b
    B,          // b
    XOR,        // a + b - 2*a*b
    OR,         // a + b - a*b
    NOR,        // 1 - (a + b - a*b)
    XNOR,       // 1 - (a + b - 2*a*b)
    NOT_B,      // 1 - b
    A_OR_NOT_B, // 1 - b + a*b
    NOT_A,      // 1 - a
    NOT_A_OR_B, // 1 - a + a*b
    NAND,       // 1 - a*b
    TRUE,       // 1
}

// Represent a logic gate with its operation and inputs
#[derive(Clone)]
pub struct LogicGate {
    op: LogicOp,
    inputs: (usize, usize), // Indices of input gates or values
    probability: Vec<f32>,  // Probability distribution over operations (for training)
}

impl LogicGate {
    pub fn new(inputs: (usize, usize)) -> Self {

        // Start with a base probability for all gates
        let base_prob = 0.055;
        let mut probability = vec![base_prob; 16];

        // Add stronger bias toward pass-through gates (A and B)
        probability[3] = 0.11; // Bias toward A (LogicOp::A)
        probability[5] = 0.11; // Bias toward B (LogicOp::B)
        
        // Ensure probabilities sum to 1.0
        let sum: f32 = probability.iter().sum();
        for p in &mut probability {
            *p /= sum;
        }

        LogicGate {
            op: LogicOp::A, // Default operation
            inputs,
            probability,
        }
    }
    
    // Compute the hard (binary) output of the gate
    pub fn compute_hard(&self, a: bool, b: bool) -> bool {
        match self.op {
            LogicOp::FALSE => false,
            LogicOp::AND => a && b,
            LogicOp::A_AND_NOT_B => a && !b,
            LogicOp::A => a,
            LogicOp::NOT_A_AND_B => !a && b,
            LogicOp::B => b,
            LogicOp::XOR => a != b,
            LogicOp::OR => a || b,
            LogicOp::NOR => !(a || b),
            LogicOp::XNOR => a == b,
            LogicOp::NOT_B => !b,
            LogicOp::A_OR_NOT_B => a || !b,
            LogicOp::NOT_A => !a,
            LogicOp::NOT_A_OR_B => !a || b,
            LogicOp::NAND => !(a && b),
            LogicOp::TRUE => true,
        }
    }
    
    // Compute the soft (continuous) output for differentiable training
    pub fn compute_soft(&self, a: f32, b: f32) -> f32 {
        // Weighted sum of all possible operations
        let ops = [
            0.0,            // FALSE
            a * b,          // AND
            a * (1.0 - b),  // A_AND_NOT_B
            a,              // A
            (1.0 - a) * b,  // NOT_A_AND_B
            b,              // B
            a + b - 2.0*a*b, // XOR
            a + b - a*b,    // OR
            1.0 - (a + b - a*b), // NOR
            1.0 - (a + b - 2.0*a*b), // XNOR
            1.0 - b,        // NOT_B
            1.0 - b + a*b,  // A_OR_NOT_B
            1.0 - a,        // NOT_A
            1.0 - a + a*b,  // NOT_A_OR_B
            1.0 - a*b,      // NAND
            1.0,            // TRUE
        ];
        
        let mut result = 0.0;
        for (i, &op_value) in ops.iter().enumerate() {
            result += self.probability[i] * op_value;
        }
        result
    }
    
    // Update the gate probabilities during training
    pub fn update_probabilities(&mut self, gradient: f32, learning_rate: f32, a: f32, b: f32, l2_strength: f32, temperature: f32) {
        
        // detect if we're stuck
        let is_dominated = self.probability.iter().any(|&p| p > 0.9);
        let needs_reset = is_dominated && rand::random::<f32>() < 0.01;

        if needs_reset {
            // Reset this gate's probabilities
            self.probability = vec![0.055; 16];
            self.probability[3] = 0.11;
            self.probability[5] = 0.11;

            // Ensure probabilities sum to 1.0
            let sum: f32 = self.probability.iter().sum();
            for p in &mut self.probability {
                *p /= sum;
            }
            // Skip normal update
            return;
        }
        
        // Calculate the current output using weighted probabilities
        let current_output = self.compute_soft(a, b);
        
        // Calculate individual operation outputs
        let ops = [
            0.0, a * b, a * (1.0 - b), a, (1.0 - a) * b, b, 
            a + b - 2.0*a*b, a + b - a*b, 1.0 - (a + b - a*b), 
            1.0 - (a + b - 2.0*a*b), 1.0 - b, 1.0 - b + a*b, 
            1.0 - a, 1.0 - a + a*b, 1.0 - a*b, 1.0
        ];
        
        // Calculate operation-specific gradients based on how each would improve the output
        let mut op_gradients = vec![0.0; 16];
         // Add exploration noise
        let noise_factor = 0.1;

        for i in 0..16 {
            
            let noise = (rand::random::<f32>() * 2.0 - 1.0) * noise_factor;

            // Direct measure of how this operation would improve the output
            op_gradients[i] = -gradient * (ops[i] - current_output) + noise;
        }
        
        // Apply softmax with higher temperature for exploration
        let mut probs = vec![0.0; 16];
        let mut max_prob = f32::MIN;
        
        // First, find max for numerical stability
        for i in 0..16 {
            max_prob = max_prob.max(self.probability[i].ln() / temperature + op_gradients[i]);
        }
        
        // Calculate exp values with subtracted max
        let mut sum_exp = 0.0;
        for i in 0..16 {
            // Add gradient directly to log probabilities (before exp)
            let log_prob = self.probability[i].ln() / temperature + learning_rate * op_gradients[i];
            probs[i] = (log_prob - max_prob).exp();
            sum_exp += probs[i];
        }
        
        // Normalize and apply minimal L2 regularization
        for i in 0..16 {
            probs[i] /= sum_exp;
            
            // Apply very mild L2 only to non-pass-through gates
            if (i != 3 && i != 5) && probs[i] > 0.05 {
                probs[i] -= learning_rate * l2_strength * probs[i];
            }
            
            // Ensure minimum probability
            probs[i] = probs[i].max(0.001);
        }
        
        // Renormalize
        sum_exp = probs.iter().sum();
        for i in 0..16 {
            self.probability[i] = probs[i] / sum_exp;
        }
        
        // Update the current operation
        self.update_current_op();
    }
    
    fn update_current_op(&mut self) {
        let mut max_idx = 0;
        let mut max_prob = self.probability[0];
        
        for i in 1..16 {
            if self.probability[i] > max_prob {
                max_prob = self.probability[i];
                max_idx = i;
            }
        }
        
        self.op = match max_idx {
            0 => LogicOp::FALSE,
            1 => LogicOp::AND,
            2 => LogicOp::A_AND_NOT_B,
            3 => LogicOp::A,
            4 => LogicOp::NOT_A_AND_B,
            5 => LogicOp::B,
            6 => LogicOp::XOR,
            7 => LogicOp::OR,
            8 => LogicOp::NOR,
            9 => LogicOp::XNOR,
            10 => LogicOp::NOT_B,
            11 => LogicOp::A_OR_NOT_B,
            12 => LogicOp::NOT_A,
            13 => LogicOp::NOT_A_OR_B,
            14 => LogicOp::NAND,
            15 => LogicOp::TRUE,
            _ => unreachable!(),
        };
    }

    pub fn get_gate_distribution_stats(&self) -> [f32; 16] {
        let mut stats = [0.0; 16];
        for i in 0..16 {
            stats[i] = self.probability[i];
        }
        stats
    }

    // Calculate gradients for input values based on output gradient
    pub fn backward(&mut self, a: f32, b: f32, output_grad: f32, learning_rate: f32, l2_strength: f32, temperature: f32) -> (f32, f32) {
        // Output gradients for inputs a and b
        let mut grad_a = 0.0;
        let mut grad_b = 0.0;
        
        // Calculate partial derivatives for each operation
        let probs = &self.probability;
        
        // FALSE: constant 0, no gradient
        
        // AND: a * b
        grad_a += probs[1] * b * output_grad;
        grad_b += probs[1] * a * output_grad;
        
        // A_AND_NOT_B: a * (1-b)
        grad_a += probs[2] * (1.0 - b) * output_grad;
        grad_b += probs[2] * (-a) * output_grad;
        
        // A: a
        grad_a += probs[3] * output_grad;
        
        // NOT_A_AND_B: (1-a) * b
        grad_a += probs[4] * (-b) * output_grad;
        grad_b += probs[4] * (1.0 - a) * output_grad;
        
        // B: b
        grad_b += probs[5] * output_grad;
        
        // XOR: a + b - 2*a*b
        grad_a += probs[6] * (1.0 - 2.0 * b) * output_grad;
        grad_b += probs[6] * (1.0 - 2.0 * a) * output_grad;
        
        // OR: a + b - a*b
        grad_a += probs[7] * (1.0 - b) * output_grad;
        grad_b += probs[7] * (1.0 - a) * output_grad;
        
        // NOR: 1 - (a + b - a*b)
        grad_a += probs[8] * (-(1.0 - b)) * output_grad;
        grad_b += probs[8] * (-(1.0 - a)) * output_grad;
        
        // XNOR: 1 - (a + b - 2*a*b)
        grad_a += probs[9] * (-(1.0 - 2.0 * b)) * output_grad;
        grad_b += probs[9] * (-(1.0 - 2.0 * a)) * output_grad;
        
        // NOT_B: 1 - b
        grad_b += probs[10] * (-1.0) * output_grad;
        
        // A_OR_NOT_B: 1 - b + a*b
        grad_a += probs[11] * b * output_grad;
        grad_b += probs[11] * (-1.0 + a) * output_grad;
        
        // NOT_A: 1 - a
        grad_a += probs[12] * (-1.0) * output_grad;
        
        // NOT_A_OR_B: 1 - a + a*b
        grad_a += probs[13] * (-1.0 + b) * output_grad;
        grad_b += probs[13] * a * output_grad;
        
        // NAND: 1 - a*b
        grad_a += probs[14] * (-b) * output_grad;
        grad_b += probs[14] * (-a) * output_grad;
        
        // TRUE: constant 1, no gradient
        
        // Add gradient clipping to prevent exploding gradients
    let clip_value = 5.0;
    grad_a = grad_a.max(-clip_value).min(clip_value);
    grad_b = grad_b.max(-clip_value).min(clip_value);
    
    // Update probabilities with smaller step size for stability
    self.update_probabilities(output_grad, learning_rate, a, b, l2_strength, temperature);
    
    (grad_a, grad_b)

    }

}

// A layer of logic gates in the circuit
#[derive(Clone)]
pub struct GateLayer {
    gates: Vec<LogicGate>,
}

impl GateLayer {
    pub fn new(n_gates: usize, inputs_per_gate: usize, rng: &mut ThreadRng) -> Self {
        let mut gates = Vec::with_capacity(n_gates);
        
        for _ in 0..n_gates {
            let in1 = rng.random_range(0..inputs_per_gate);
            let in2 = rng.random_range(0..inputs_per_gate);
            gates.push(LogicGate::new((in1, in2)));
        }
        
        GateLayer { gates }
    }
    
    pub fn forward_hard(&self, inputs: &[bool]) -> Vec<bool> {
        let mut outputs = vec![false; self.gates.len()];
        
        for (i, gate) in self.gates.iter().enumerate() {
            let a = inputs[gate.inputs.0];
            let b = inputs[gate.inputs.1];
            outputs[i] = gate.compute_hard(a, b);
        }
        
        outputs
    }
    
    pub fn forward_soft(&self, inputs: &[f32]) -> Vec<f32> {
        let mut outputs = vec![0.0; self.gates.len()];
        
        for (i, gate) in self.gates.iter().enumerate() {
            let a = inputs[gate.inputs.0];
            let b = inputs[gate.inputs.1];
            outputs[i] = gate.compute_soft(a, b);
        }
        
        outputs
    }
    
    pub fn backward(&mut self, inputs: &[f32], output_grads: &[f32], learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        let mut input_grads = vec![0.0; inputs.len()];
        
        // Process each gate
        for (i, gate) in self.gates.iter_mut().enumerate() {
            let a_idx = gate.inputs.0;
            let b_idx = gate.inputs.1;
            let a = inputs[a_idx];
            let b = inputs[b_idx];
            
            // Compute gradients for this gate
            let (grad_a, grad_b) = gate.backward(a, b, output_grads[i], learning_rate, l2_strength, temperature);
            
            // Accumulate gradients for inputs
            input_grads[a_idx] += grad_a;
            input_grads[b_idx] += grad_b;
        }
        
        input_grads
    }

}

// Circuit composed of multiple layers of gates
#[derive(Clone)]
pub struct Circuit {
    layers: Vec<GateLayer>,
    layer_sizes: Vec<usize>,
}

impl Circuit {
    pub fn new(layer_sizes: &[usize], rng: &mut ThreadRng) -> Self {
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(GateLayer::new(layer_sizes[i+1], layer_sizes[i], rng));
        }
        
        Circuit {
            layers,
            layer_sizes: layer_sizes.to_vec(),
        }
    }
    
    pub fn forward_hard(&self, inputs: &[bool]) -> Vec<bool> {
        let mut current = inputs.to_vec();
        
        for layer in &self.layers {
            current = layer.forward_hard(&current);
        }
        
        current
    }
    
    pub fn forward_soft(&self, inputs: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut current = inputs.to_vec();
        let mut all_outputs = Vec::with_capacity(self.layer_sizes.len());
        all_outputs.push(current.clone());
        
        for layer in &self.layers {
            current = layer.forward_soft(&current);
            all_outputs.push(current.clone());
        }
        
        (all_outputs, current.clone())
    }
    
    pub fn backward(&mut self, all_inputs: &[Vec<f32>], output_grads: &[f32], learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        let mut gradients = output_grads.to_vec();
        
        // Backpropagate through layers in reverse order
        for i in (0..self.layers.len()).rev() {
            let layer_inputs = &all_inputs[i];
            gradients = self.layers[i].backward(layer_inputs, &gradients, learning_rate, l2_strength, temperature);
        }
        
        gradients
    }
}

// Perception circuit that processes neighborhood states
#[derive(Clone)]
pub struct PerceptionCircuit {
    circuit: Circuit,
}

impl PerceptionCircuit {
    pub fn new(state_size: usize, rng: &mut ThreadRng) -> Self {
        // Architecture for the perception circuit:
        // 9 neighbors * state_size inputs -> 8 -> 4 -> 2 -> 1 outputs
        let layer_sizes = vec![9 * state_size, 8, 4, 2, 1];
        
        PerceptionCircuit {
            circuit: Circuit::new(&layer_sizes, rng),
        }
    }
    
    pub fn process_neighborhood(&self, neighborhood: &[bool]) -> bool {
        let outputs = self.circuit.forward_hard(neighborhood);
        outputs[0]
    }

    pub fn process_neighborhood_soft(&self, neighborhood: &[f32]) -> f32 {
        let outputs = self.circuit.forward_soft(neighborhood);
        outputs.1[0] // Return the final output value
    }

    pub fn backward(&mut self, inputs: &[f32], output_grad: f32, learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        // Perform forward pass to collect intermediate outputs
        let (all_inputs, _) = self.circuit.forward_soft(inputs);
        
        // Backward pass with single output gradient
        self.circuit.backward(&all_inputs, &[output_grad], learning_rate, l2_strength, temperature)
    }

}

// The update circuit that computes the next state
pub struct UpdateCircuit {
    circuit: Circuit,
}

impl UpdateCircuit {
    pub fn new(input_size: usize, _state_size: usize, rng: &mut ThreadRng) -> Self {
        // Architecture for the update circuit:
        // input_size inputs -> 16 layers of 128 nodes -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 outputs
        let mut layer_sizes = vec![input_size];
        
        // Add 16 layers of 128 nodes
        for _ in 0..16 {
            layer_sizes.push(128);
        }
        
        // Add the remaining layers
        layer_sizes.extend_from_slice(&[64, 32, 16, 8, 4, 2, 1]);
        
        UpdateCircuit {
            circuit: Circuit::new(&layer_sizes, rng),
        }
    }
    
    pub fn compute_next_state(&self, perceptions: &[bool], current_state: &[bool]) -> Vec<bool> {
        let mut inputs = Vec::with_capacity(perceptions.len() + current_state.len());
        inputs.extend_from_slice(perceptions);
        inputs.extend_from_slice(current_state);
        
        self.circuit.forward_hard(&inputs)
    }

    pub fn compute_next_state_soft(&self, perceptions: &[f32], current_state: &[f32]) -> Vec<f32> {
        let mut inputs = Vec::with_capacity(perceptions.len() + current_state.len());
        inputs.extend_from_slice(perceptions);
        inputs.extend_from_slice(current_state);
        
        let (_, outputs) = self.circuit.forward_soft(&inputs);
        outputs
    }

    pub fn backward(&mut self, inputs: &[f32], output_grads: &[f32], learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        // Perform forward pass to collect intermediate outputs
        let (all_inputs, _) = self.circuit.forward_soft(inputs);
        
        // Backward pass
        self.circuit.backward(&all_inputs, output_grads, learning_rate, l2_strength, temperature)
    }

}

// The main DiffLogic CA model
pub struct DiffLogicCA {
    width: usize,
    height: usize,
    state_size: usize,
    grid: Array3<bool>,
    perception_circuits: Vec<PerceptionCircuit>,
    update_circuit: UpdateCircuit,
    batch_size: usize,
    l2_strength: f32,
    temperature: f32,
}

impl DiffLogicCA {
    pub fn new(width: usize, height: usize, state_size: usize, n_perception_circuits: usize) -> Self {
        let mut rng = rng();
        
        // Initialize perception circuits
        let mut perception_circuits = Vec::with_capacity(n_perception_circuits);
        for _ in 0..n_perception_circuits {
            perception_circuits.push(PerceptionCircuit::new(state_size, &mut rng));
        }
        
        // Initialize update circuit
        let update_circuit = UpdateCircuit::new(n_perception_circuits + state_size, state_size, &mut rng);
        
        // Initialize grid with random values
        let grid = Array3::<bool>::from_shape_fn((height, width, state_size), |_| rng.random_bool(0.5));
        
        DiffLogicCA {
            width,
            height,
            state_size,
            grid,
            perception_circuits,
            update_circuit,
            batch_size: 64,
            l2_strength: 0.001,
            temperature: 0.1,
        }
    }
    
    // Add setter methods
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }
    
    pub fn set_l2_strength(&mut self, l2_strength: f32) {
        self.l2_strength = l2_strength;
    }
    
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    // Get the Moore neighborhood (8 surrounding cells + center) for a cell
    fn get_neighborhood(&self, row: usize, col: usize) -> Vec<bool> {
        let mut neighborhood = Vec::with_capacity(9 * self.state_size);
        
        // Iterate over 3x3 neighborhood, handling boundary conditions
        for dr in [-1, 0, 1].iter() {
            for dc in [-1, 0, 1].iter() {
                let r = (row as isize + dr).rem_euclid(self.height as isize) as usize;
                let c = (col as isize + dc).rem_euclid(self.width as isize) as usize;
                
                // Add all state bits for this neighbor
                for s in 0..self.state_size {
                    neighborhood.push(self.grid[[r, c, s]]);
                }
            }
        }
        
        neighborhood
    }
    
    // Perform one step of the CA update
    pub fn step(&mut self) {
        // Create a new grid to store the updated states
        let mut new_grid = Array3::<bool>::from_elem((self.height, self.width, self.state_size), false);
        
        // For each cell in the grid
        for row in 0..self.height {
            for col in 0..self.width {
                // Get the neighborhood
                let neighborhood = self.get_neighborhood(row, col);
                
                // Process the neighborhood with each perception circuit
                let mut perceptions = Vec::with_capacity(self.perception_circuits.len());
                for circuit in &self.perception_circuits {
                    perceptions.push(circuit.process_neighborhood(&neighborhood));
                }
                
                // Get the current state of the cell
                let current_state: Vec<bool> = (0..self.state_size)
                    .map(|s| self.grid[[row, col, s]])
                    .collect();
                
                // Compute the next state
                let next_state = self.update_circuit.compute_next_state(&perceptions, &current_state);
                
                // Update the grid
                for s in 0..self.state_size {
                    new_grid[[row, col, s]] = next_state[s];
                }
            }
        }
        
        // Replace the old grid with the new one
        self.grid = new_grid;
    }
    
    // Train the model to learn a specific pattern transition
    pub fn train(&mut self, initial_states: &Array4<bool>, target_states: &Array4<bool>, learning_rate: f32, epochs: usize) {
        let n_samples = initial_states.shape()[0];
        let height = initial_states.shape()[1];
        let width = initial_states.shape()[2];
        
        // Safety check
        assert_eq!(initial_states.shape()[1], target_states.shape()[1], "Height mismatch");
        assert_eq!(initial_states.shape()[2], target_states.shape()[2], "Width mismatch");
        assert_eq!(initial_states.shape()[3], target_states.shape()[3], "Channel mismatch");
        assert_eq!(height, self.height, "Height doesn't match CA grid height");
        assert_eq!(width, self.width, "Width doesn't match CA grid width");
        
        println!("Starting training for {} epochs with {} samples", epochs, n_samples);
        
        for epoch in 0..epochs {
            let (soft_loss, hard_loss) = self.train_epoch_internal(initial_states, target_states, learning_rate, epoch);
            
            // Print progress
            println!("Epoch {}: Soft Loss = {}, Hard Loss = {}", epoch, soft_loss, hard_loss);
        }
        
        println!("Training completed!");
    }
    
    // Train for a single epoch and return soft and hard loss values
fn train_epoch_internal(&mut self, initial_states: &Array4<bool>, target_states: &Array4<bool>, learning_rate: f32, epoch: usize) -> (f32, f32) {
       
    // Clone the immutable data we need from self
    let width = self.width;
    let height = self.height;
    let state_size = self.state_size;
    
    let n_samples = initial_states.shape()[0];
    
    // Process in smaller batches
    let batch_size = self.batch_size;
    let num_batches = (n_samples + batch_size - 1) / batch_size;
    
    // Add adaptive learning rate 
    let base_learning_rate = learning_rate;
    let min_learning_rate = base_learning_rate * 0.01;

    // Create thread-safe shared state for the model
    let perception_circuits = Arc::new(Mutex::new(&mut self.perception_circuits));
    let update_circuit = Arc::new(Mutex::new(&mut self.update_circuit));
    
    // Process batches in parallel
    let batch_results: Vec<(f32, f32)> = (0..num_batches)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&batch_idx| {
            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, n_samples);
            
            // Calculate batch-specific learning rate
            let adjusted_learning_rate = base_learning_rate * (1.0 / (1.0 + 0.01 * batch_idx as f32));
            let adjusted_learning_rate = adjusted_learning_rate.max(min_learning_rate);

            let mut batch_soft_loss = 0.0;
            let mut batch_hard_loss = 0.0;
            
            for i in start_idx..end_idx {
                // Get initial state for this sample
                let init_state = initial_states.slice(s![i, ..height, ..width, ..]);
                
                // Convert to float for soft computation
                let mut soft_state = Array3::<f32>::zeros((height, width, state_size));
                for h in 0..height {
                    for w in 0..width {
                        for s in 0..state_size {
                            soft_state[[h, w, s]] = if init_state[[h, w, s]] { 1.0 } else { 0.0 };
                        }
                    }
                }
                
                // Forward pass - compute predictions
                let mut perceptions = Vec::with_capacity(perception_circuits.lock().unwrap().len());
                for _ in 0..perception_circuits.lock().unwrap().len() {
                    perceptions.push(Array2::<f32>::zeros((height, width)));
                }
                
                // Process each cell with perception circuits
                for h in 0..height {
                    for w in 0..width {
                        // Get neighborhood for this cell
                        let neighborhood = Self::get_neighborhood_soft_static(&soft_state, h, w, height, width, state_size);
                        
                        // Process neighborhood with perception circuits
                        let perception_lock = perception_circuits.lock().unwrap();
                        for (p_idx, circuit) in perception_lock.iter().enumerate() {
                            let result = circuit.process_neighborhood_soft(&neighborhood);
                            perceptions[p_idx][[h, w]] = result;
                        }
                    }
                }
                
                // Process results with update circuit (soft computation)
                let mut next_state_soft = Array3::<f32>::zeros((height, width, state_size));
                let mut next_state_hard = Array3::<bool>::from_elem((height, width, state_size), false);
                
                for h in 0..height {
                    for w in 0..width {
                        let mut perception_values = Vec::with_capacity(perceptions.len());
                        for p in &perceptions {
                            perception_values.push(p[[h, w]]);
                        }
                        
                        let mut current_state = Vec::with_capacity(state_size);
                        for s in 0..state_size {
                            current_state.push(soft_state[[h, w, s]]);
                        }
                        
                        // Compute next state (soft)
                        let update_lock = update_circuit.lock().unwrap();
                        let result_soft = update_lock.compute_next_state_soft(&perception_values, &current_state);
                        
                        // Convert current state to boolean for hard computation
                        let current_state_bool: Vec<bool> = current_state.iter().map(|&x| x > 0.5).collect();
                        
                        // Get perception values as boolean
                        let perception_values_bool: Vec<bool> = perception_values.iter().map(|&x| x > 0.5).collect();
                        
                        // Compute next state (hard)
                        let result_hard = update_lock.compute_next_state(&perception_values_bool, &current_state_bool);
                        
                        for s in 0..state_size {
                            next_state_soft[[h, w, s]] = result_soft[s];
                            next_state_hard[[h, w, s]] = result_hard[s];
                        }
                    }
                }
                                
                if batch_idx == 0 && i == start_idx {
                    // Create a temporary clone for analysis
                    let circuits_clone = {
                        let perception_lock = perception_circuits.lock().unwrap();
                        perception_lock.clone()
                    };
                    
                    analyze_gate_distributions(&circuits_clone, epoch);
                }

                // Compute soft loss
                let mut sample_soft_loss = 0.0;
                let mut sample_hard_loss = 0.0;
                
                for h in 0..height {
                    for w in 0..width {
                        for s in 0..state_size {
                            let target = if target_states[[i, h, w, s]] { 1.0 } else { 0.0 };
                            let target_bool = target_states[[i, h, w, s]];
                            
                            // Soft loss
                            let diff_soft = next_state_soft[[h, w, s]] - target;
                            sample_soft_loss += diff_soft * diff_soft;
                            
                            // Hard loss
                            if next_state_hard[[h, w, s]] != target_bool {
                                sample_hard_loss += 1.0;
                            }
                        }
                    }
                }
                
                batch_soft_loss += sample_soft_loss;
                batch_hard_loss += sample_hard_loss;
                
                // Backward pass
                let mut output_grads = Array3::<f32>::zeros((height, width, state_size));
                for h in 0..height {
                    for w in 0..width {
                        for s in 0..state_size {
                            let target = if target_states[[i, h, w, s]] { 1.0 } else { 0.0 };
                            output_grads[[h, w, s]] = 2.0 * (next_state_soft[[h, w, s]] - target);
                        }
                    }
                }
                
                // Update perception and update circuits
                for h in 0..height {
                    for w in 0..width {
                        // Extract gradients for this cell
                        let mut cell_grads = Vec::with_capacity(state_size);
                        for s in 0..state_size {
                            cell_grads.push(output_grads[[h, w, s]]);
                        }
                        
                        // Get inputs to update circuit
                        let mut update_inputs = Vec::new();
                        for p in &perceptions {
                            update_inputs.push(p[[h, w]]);
                        }
                        let mut current_state = Vec::with_capacity(state_size);
                        for s in 0..state_size {
                            current_state.push(soft_state[[h, w, s]]);
                        }
                        update_inputs.extend_from_slice(&current_state);

                        // Add batch normalization here
                        let batch_mean = cell_grads.iter().sum::<f32>() / cell_grads.len() as f32;
                        let batch_var = cell_grads.iter()
                            .map(|&g| (g - batch_mean).powi(2))
                            .sum::<f32>() / cell_grads.len() as f32;
                        let epsilon = 1e-5;
                        let mut normalized_grads = Vec::with_capacity(cell_grads.len());
                        for &g in &cell_grads {
                            normalized_grads.push((g - batch_mean) / (batch_var + epsilon).sqrt());
                        }
                        
                        // Backpropagate through update circuit
                        let mut update_lock = update_circuit.lock().unwrap();
                        let perception_grads = update_lock.backward(&update_inputs, &normalized_grads, adjusted_learning_rate, self.l2_strength, self.temperature);
                        
                        // Prepare perception gradients
                        let perception_circuit_grads = perception_grads[0..perceptions.len()].to_vec();
                        
                        // Pre-compute all neighborhoods
                        let neighborhood = Self::get_neighborhood_soft_static(&soft_state, h, w, height, width, state_size);
                        
                        // Update each circuit
                        let mut perception_lock = perception_circuits.lock().unwrap();
                        for (p_idx, circuit) in perception_lock.iter_mut().enumerate() {
                            circuit.backward(&neighborhood, perception_circuit_grads[p_idx], adjusted_learning_rate, self.l2_strength, self.temperature);
                        }
                    }
                }
            }
            
            (batch_soft_loss, batch_hard_loss)
        })
        .collect();
    
    // Sum up batch results
    let total_soft_loss: f32 = batch_results.iter().map(|(soft, _)| soft).sum();
    let total_hard_loss: f32 = batch_results.iter().map(|(_, hard)| hard).sum();
    
    // Normalize by number of samples
    let normalized_soft_loss = total_soft_loss / (n_samples as f32);
    let normalized_hard_loss = total_hard_loss / (n_samples as f32);
    
    (normalized_soft_loss, normalized_hard_loss)
}
    
    // Get the current state of the grid
    pub fn get_grid(&self) -> &Array3<bool> {
        &self.grid
    }
    
    // Set the state of the grid
    pub fn set_grid(&mut self, grid: Array3<bool>) {
        assert_eq!(grid.shape(), self.grid.shape());
        self.grid = grid;
    }

fn get_neighborhood_soft_static(grid: &Array3<f32>, row: usize, col: usize, height: usize, width: usize, state_size: usize) -> Vec<f32> {
    let mut neighborhood = Vec::with_capacity(9 * state_size);
    
    // Iterate over 3x3 neighborhood, handling boundary conditions
    for dr in [-1, 0, 1].iter() {
        for dc in [-1, 0, 1].iter() {
            let r = (row as isize + dr).rem_euclid(height as isize) as usize;
            let c = (col as isize + dc).rem_euclid(width as isize) as usize;
            
            // Add all state bits for this neighbor
            for s in 0..state_size {
                neighborhood.push(grid[[r, c, s]]);
            }
        }
    }
    
    neighborhood
}

}

// Example implementation for Conway's Game of Life
pub fn create_game_of_life(width: usize, height: usize) -> DiffLogicCA {
    let state_size = 1; // Game of Life has 1 bit of state per cell
    let n_perception_circuits = 16; // As suggested in the paper
    
    DiffLogicCA::new(width, height, state_size, n_perception_circuits)
}

// Helper function to create a Game of Life pattern
pub fn create_glider(grid: &mut Array3<bool>, row: usize, col: usize) {
    // Create a glider pattern at the specified position
    let pattern = [
        (0, 1), (1, 2), (2, 0), (2, 1), (2, 2)
    ];
    
    // Clear the area
    for r in row..(row+3) {
        for c in col..(col+3) {
            if r < grid.shape()[0] && c < grid.shape()[1] {
                grid[[r, c, 0]] = false;
            }
        }
    }
    
    // Set the glider cells
    for &(dr, dc) in &pattern {
        let r = row + dr;
        let c = col + dc;
        if r < grid.shape()[0] && c < grid.shape()[1] {
            grid[[r, c, 0]] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logic_gate() {
        let mut gate = LogicGate::new((0, 1));
        
        // Test hard computation for AND
        gate.op = LogicOp::AND;
        assert_eq!(gate.compute_hard(true, true), true);
        assert_eq!(gate.compute_hard(true, false), false);
        assert_eq!(gate.compute_hard(false, true), false);
        assert_eq!(gate.compute_hard(false, false), false);
        
        // Test soft computation
        assert!((gate.compute_soft(1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((gate.compute_soft(1.0, 0.0) - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_gate_layer() {
        let mut rng = rng();
        let layer = GateLayer::new(4, 2, &mut rng);
        
        let inputs = vec![true, false];
        let outputs = layer.forward_hard(&inputs);
        
        assert_eq!(outputs.len(), 4);
    }
    
    #[test]
    fn test_circuit() {
        let mut rng = rng();
        let circuit = Circuit::new(&[2, 4, 1], &mut rng);
        
        let inputs = vec![true, false];
        let outputs = circuit.forward_hard(&inputs);
        
        assert_eq!(outputs.len(), 1);
    }
    
    #[test]
    fn test_difflogic_ca() {
        let ca = DiffLogicCA::new(10, 10, 1, 16);
        
        assert_eq!(ca.grid.shape(), &[10, 10, 1]);
    }
    
    #[test]
    fn test_glider_creation() {
        let mut ca = DiffLogicCA::new(10, 10, 1, 16);
        create_glider(&mut ca.grid, 1, 1);
        
        // Check that the glider pattern was created correctly
        assert_eq!(ca.grid[[1, 2, 0]], true);
        assert_eq!(ca.grid[[2, 3, 0]], true);
        assert_eq!(ca.grid[[3, 1, 0]], true);
        assert_eq!(ca.grid[[3, 2, 0]], true);
        assert_eq!(ca.grid[[3, 3, 0]], true);
    }
}

#[pyclass]
struct PyDiffLogicCA {
    model: DiffLogicCA,
}

#[pymethods]
impl PyDiffLogicCA {
    #[new]
    fn new(width: usize, height: usize, state_size: usize, n_perception_circuits: usize) -> Self {
        PyDiffLogicCA {
            model: DiffLogicCA::new(width, height, state_size, n_perception_circuits),
        }
    }
    
    fn step(&mut self) {
        self.model.step();
    }
    
    fn get_grid<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray3<bool>>> {
        let grid = self.model.get_grid().to_owned();
        Ok(grid.into_pyarray(py).into())
    }
    
    fn set_grid(&mut self, grid: PyReadonlyArray3<bool>) {
        let array = grid.as_array().to_owned();
        self.model.set_grid(array);
    }
    
    fn create_glider(&mut self, row: usize, col: usize) {
        let mut grid = self.model.get_grid().to_owned();
        crate::create_glider(&mut grid, row, col);
        self.model.set_grid(grid);
    }

    fn train(&mut self, initial_states: PyReadonlyArray4<bool>, target_states: PyReadonlyArray4<bool>, 
        learning_rate: f32, epochs: usize) {
        let initial_array = initial_states.as_array().to_owned();
        let target_array = target_states.as_array().to_owned();
        self.model.train(&initial_array, &target_array, learning_rate, epochs);
    }
    
    fn train_epoch(&mut self, initial_states: PyReadonlyArray4<bool>, target_states: PyReadonlyArray4<bool>,
        learning_rate: f32, epoch: usize) -> PyResult<(f32, f32)> {
        let initial_array = initial_states.as_array().to_owned();
        let target_array = target_states.as_array().to_owned();
        let (soft_loss, hard_loss) = self.model.train_epoch_internal(&initial_array, &target_array, learning_rate, epoch);
        Ok((soft_loss, hard_loss))
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.model.set_batch_size(batch_size);
    }
    
    fn set_l2_strength(&mut self, l2_strength: f32) {
        self.model.set_l2_strength(l2_strength);
    }
    
    fn set_temperature(&mut self, temperature: f32) {
        self.model.set_temperature(temperature);
    }

}

#[pyfunction]
fn create_gol(width: usize, height: usize) -> PyResult<PyDiffLogicCA> {
    Ok(PyDiffLogicCA {
        model: crate::create_game_of_life(width, height),
    })
}

#[pymodule]
fn logicars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiffLogicCA>()?;
    m.add_function(wrap_pyfunction!(create_gol, m)?)?;
    Ok(())
}