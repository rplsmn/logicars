use ndarray::{Array2, Array3, Array4, s};
use numpy::IntoPyArray;
use numpy::PyArray3;
use numpy::PyReadonlyArray4;
use rand::{prelude::*, rng};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::PyReadonlyArray3;


// Logic gate operations as per the paper
#[allow(non_camel_case_types)]
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
pub struct LogicGate {
    op: LogicOp,
    inputs: (usize, usize), // Indices of input gates or values
    probability: Vec<f32>,  // Probability distribution over operations (for training)
}

impl LogicGate {
    pub fn new(inputs: (usize, usize)) -> Self {
        // Initialize with a bias toward pass-through gates (A or B)
        let mut probability = vec![0.01; 16];
        probability[3] = 0.4; // Bias toward A
        probability[5] = 0.4; // Bias toward B
        
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
    pub fn update_probabilities(&mut self, gradient: f32, learning_rate: f32, a: f32, b: f32) {
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
        
        // Compute gradients for each operation
        let mut gradients = vec![0.0; 16];
        for i in 0..16 {
            gradients[i] = gradient * ops[i];
        }
        
        // Update probabilities with softmax gradient
        let mut sum = 0.0;
        for i in 0..16 {
            self.probability[i] += learning_rate * gradients[i];
            sum += self.probability[i];
        }
        
        // Re-normalize
        for i in 0..16 {
            self.probability[i] /= sum;
        }
        
        // Set the most probable operation
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

    // Calculate gradients for input values based on output gradient
    pub fn backward(&mut self, a: f32, b: f32, output_grad: f32, learning_rate: f32) -> (f32, f32) {
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
        
        // Update operation probabilities
        self.update_probabilities(output_grad, learning_rate, a, b);
        
        (grad_a, grad_b)
    }

}

// A layer of logic gates in the circuit
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
    
    pub fn backward(&mut self, inputs: &[f32], output_grads: &[f32], learning_rate: f32) -> Vec<f32> {
        let mut input_grads = vec![0.0; inputs.len()];
        
        // Process each gate
        for (i, gate) in self.gates.iter_mut().enumerate() {
            let a_idx = gate.inputs.0;
            let b_idx = gate.inputs.1;
            let a = inputs[a_idx];
            let b = inputs[b_idx];
            
            // Compute gradients for this gate
            let (grad_a, grad_b) = gate.backward(a, b, output_grads[i], learning_rate);
            
            // Accumulate gradients for inputs
            input_grads[a_idx] += grad_a;
            input_grads[b_idx] += grad_b;
        }
        
        input_grads
    }

}

// Circuit composed of multiple layers of gates
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
    
    pub fn backward(&mut self, all_inputs: &[Vec<f32>], output_grads: &[f32], learning_rate: f32) -> Vec<f32> {
        let mut gradients = output_grads.to_vec();
        
        // Backpropagate through layers in reverse order
        for i in (0..self.layers.len()).rev() {
            let layer_inputs = &all_inputs[i];
            gradients = self.layers[i].backward(layer_inputs, &gradients, learning_rate);
        }
        
        gradients
    }
}

// Perception circuit that processes neighborhood states
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

    pub fn backward(&mut self, inputs: &[f32], output_grad: f32, learning_rate: f32) -> Vec<f32> {
        // Perform forward pass to collect intermediate outputs
        let (all_inputs, _) = self.circuit.forward_soft(inputs);
        
        // Backward pass with single output gradient
        self.circuit.backward(&all_inputs, &[output_grad], learning_rate)
    }

}

// The update circuit that computes the next state
pub struct UpdateCircuit {
    circuit: Circuit,
}

impl UpdateCircuit {
    pub fn new(input_size: usize, state_size: usize, rng: &mut ThreadRng) -> Self {
        // Architecture for the update circuit:
        // input_size inputs -> 128 -> 64 -> 32 -> 16 -> 8 -> state_size outputs
        let layer_sizes = vec![input_size, 128, 64, 32, 16, 8, state_size];
        
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

    pub fn backward(&mut self, inputs: &[f32], output_grads: &[f32], learning_rate: f32) -> Vec<f32> {
        // Perform forward pass to collect intermediate outputs
        let (all_inputs, _) = self.circuit.forward_soft(inputs);
        
        // Backward pass
        self.circuit.backward(&all_inputs, output_grads, learning_rate)
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
        }
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
        let height = initial_states.shape()[0];
        let width = initial_states.shape()[1];
        
        println!("Starting training for {} epochs", epochs);
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            // For each training example
            for i in 0..initial_states.shape()[0] {
                // Get the initial state
                let init_state = initial_states.slice(s![.., .., .., ..]);
                
                // Convert to float for soft computation
                let mut soft_state = Array3::<f32>::zeros((height, width, self.state_size));
                for h in 0..height {
                    for w in 0..width {
                        for s in 0..self.state_size {
                            soft_state[[h, w, s]] = if init_state[[i, h, w, s]] { 1.0 } else { 0.0 };
                        }
                    }
                }
                
                // Forward pass - compute predictions
                let mut perceptions = Vec::with_capacity(self.perception_circuits.len());
                
                // Process each cell with perception circuits
                for h in 0..height {
                    for w in 0..width {
                        // Get neighborhood for this cell
                        let neighborhood = self.get_neighborhood_soft(&soft_state, h, w);
                        
                        // Process neighborhood with perception circuits
                        for (p_idx, circuit) in self.perception_circuits.iter().enumerate() {
                            // Convert circuit to use soft computation
                            let result = circuit.process_neighborhood_soft(&neighborhood);
                            if perceptions.len() <= p_idx {
                                perceptions.push(Array2::<f32>::zeros((height, width)));
                            }
                            perceptions[p_idx][[h, w]] = result;
                        }
                    }
                }
                
                // Process results with update circuit (soft computation)
                let mut next_state = Array3::<f32>::zeros((height, width, self.state_size));
                for h in 0..height {
                    for w in 0..width {
                        let mut perception_values = Vec::with_capacity(self.perception_circuits.len());
                        for p in &perceptions {
                            perception_values.push(p[[h, w]]);
                        }
                        
                        let mut current_state = Vec::with_capacity(self.state_size);
                        for s in 0..self.state_size {
                            current_state.push(soft_state[[h, w, s]]);
                        }
                        
                        // Compute next state
                        let result = self.update_circuit.compute_next_state_soft(&perception_values, &current_state);
                        
                        for s in 0..self.state_size {
                            next_state[[h, w, s]] = result[s];
                        }
                    }
                }
                
                // Compute loss
                let mut loss = 0.0;
                for h in 0..height {
                    for w in 0..width {
                        for s in 0..self.state_size {
                            let target = if target_states[[i, h, w, s]] { 1.0 } else { 0.0 };
                            let diff = next_state[[h, w, s]] - target;
                            loss += diff * diff;
                        }
                    }
                }
                total_loss += loss;
                
                // Backward pass - update circuit parameters
                // Compute output gradients - derivative of squared error
let mut output_grads = Array3::<f32>::zeros((height, width, self.state_size));
for h in 0..height {
    for w in 0..width {
        for s in 0..self.state_size {
            let target = if target_states[[i, h, w, s]] { 1.0 } else { 0.0 };
            // Gradient of (output - target)²: 2 * (output - target)
            output_grads[[h, w, s]] = 2.0 * (next_state[[h, w, s]] - target);
        }
    }
}

// Backward pass - update perception and update circuits
for h in 0..height {
    for w in 0..width {
        // Extract gradients for this cell
        let mut cell_grads = Vec::with_capacity(self.state_size);
        for s in 0..self.state_size {
            cell_grads.push(output_grads[[h, w, s]]);
        }
        
        // Get inputs to update circuit
        let mut update_inputs = Vec::new();
        for p in &perceptions {
            update_inputs.push(p[[h, w]]);
        }
        let mut current_state = Vec::with_capacity(self.state_size);
        for s in 0..self.state_size {
            current_state.push(soft_state[[h, w, s]]);
        }
        update_inputs.extend_from_slice(&current_state);
        
        // Backpropagate through update circuit
        let perception_grads = self.update_circuit.backward(&update_inputs, &cell_grads, learning_rate);
        
        // Backpropagate through perception circuits
        // Prepare perception gradients
let perception_circuit_grads = perception_grads[0..self.perception_circuits.len()].to_vec();

// Pre-compute all neighborhoods
let neighborhood = self.get_neighborhood_soft(&soft_state, h, w);

// Now update each circuit
for (p_idx, circuit) in self.perception_circuits.iter_mut().enumerate() {
    circuit.backward(&neighborhood, perception_circuit_grads[p_idx], learning_rate);
}
    }
}
            }
            
            // Print progress
            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("Epoch {}: Loss = {}", epoch, total_loss);
            }
        }
        
        println!("Training completed!");
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

// Get the Moore neighborhood for a cell, with soft (float) values
fn get_neighborhood_soft(&self, grid: &Array3<f32>, row: usize, col: usize) -> Vec<f32> {
    let mut neighborhood = Vec::with_capacity(9 * self.state_size);
    
    // Iterate over 3x3 neighborhood, handling boundary conditions
    for dr in [-1, 0, 1].iter() {
        for dc in [-1, 0, 1].iter() {
            let r = (row as isize + dr).rem_euclid(self.height as isize) as usize;
            let c = (col as isize + dc).rem_euclid(self.width as isize) as usize;
            
            // Add all state bits for this neighbor
            for s in 0..self.state_size {
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