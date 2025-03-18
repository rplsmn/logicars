use crate::circuits::{PerceptionCircuit, UpdateCircuit};
use ndarray::{s, Array2, Array3, Array4};
use rand::{prelude::*, rng};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// The main DiffLogic CA model
pub struct DiffLogicCA {
    pub width: usize,
    pub height: usize,
    pub state_size: usize,
    pub grid: Array3<bool>,
    pub perception_circuits: Vec<PerceptionCircuit>,
    pub update_circuit: UpdateCircuit,
    pub batch_size: usize,
    pub l2_strength: f32,
    pub temperature: f32,
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
    pub fn train_epoch_internal(&mut self, initial_states: &Array4<bool>, target_states: &Array4<bool>, learning_rate: f32, epoch: usize) -> (f32, f32) {
        // Clone the immutable data we need from self
        let width = self.width;
        let height = self.height;
        let state_size = self.state_size;
        
        let n_samples = initial_states.shape()[0];
        
        // Process in smaller batches
        let batch_size = self.batch_size;
        let num_batches = (n_samples + batch_size - 1) / batch_size;

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

                    // diagnostics     
                    if batch_idx == 0 && i == start_idx {
                        
                        // Create a temporary clone for analysis
                        let circuits_clone = {
                            let perception_lock = perception_circuits.lock().unwrap();
                            perception_lock.clone()
                        };

                        crate::analyze_gate_distributions(&circuits_clone, epoch);

                        if epoch % 10 == 0 {
                            println!("Current temp: {:?}", &self.temperature);
                            println!("Current learning rate: {:?}", &learning_rate);
                        }
                    }

                    // Compute soft loss
                    let mut sample_soft_loss = 0.0;
                    let mut sample_hard_loss = 0.0;
                    
                    // Only compare the center cell (not all cells)
                    // For a 3x3 grid, the center cell is at (1,1)
                    let h = 1;
                    let w = 1;
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
                    
                    batch_soft_loss += sample_soft_loss;
                    batch_hard_loss += sample_hard_loss;
                    
                    // Backward pass
                    let mut output_grads = Array3::<f32>::zeros((height, width, state_size));
                    // Only calculate gradients for center cell (1,1)
                    let h = 1;
                    let w = 1;
                    for s in 0..state_size {
                        let target = if target_states[[i, h, w, s]] { 1.0 } else { 0.0 };
                        output_grads[[h, w, s]] = 2.0 * (next_state_soft[[h, w, s]] - target);
                    }
                    
                    // Update perception and update circuits
                    // Only update for the center cell
                    let h = 1;
                    let w = 1;

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

                    // Skip normalization 
                    let normalized_grads = cell_grads.clone();
                    
                    // Backpropagate through update circuit
                    let mut update_lock = update_circuit.lock().unwrap();
                    let perception_grads = update_lock.backward(&update_inputs, &normalized_grads, learning_rate, self.l2_strength, self.temperature);

                    // Prepare perception gradients
                    let mut perception_circuit_grads = perception_grads[0..perceptions.len()].to_vec();

                    // Add small noise to break zero gradient problem
                    for grad in &mut perception_circuit_grads {
                        if *grad == 0.0 {
                            *grad = (rand::random::<f32>() * 2.0 - 1.0) * 0.01;
                        }
                    }
                    
                    if batch_idx == 0 && i == start_idx && epoch % 10 == 0 {
                        println!("First few perception_grads: {:?}", &perception_circuit_grads);
                        println!("All zeros: {}", perception_circuit_grads.iter().all(|&x| x == 0.0));
                    }

                    // Pre-compute neighborhood for center cell
                    let neighborhood = Self::get_neighborhood_soft_static(&soft_state, h, w, height, width, state_size);

                    // Update each circuit
                    let mut perception_lock = perception_circuits.lock().unwrap();
                    for (p_idx, circuit) in perception_lock.iter_mut().enumerate() {
                        circuit.backward(&neighborhood, perception_circuit_grads[p_idx], learning_rate, self.l2_strength, self.temperature);
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

    pub fn get_neighborhood_soft_static(grid: &Array3<f32>, row: usize, col: usize, height: usize, width: usize, state_size: usize) -> Vec<f32> {
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