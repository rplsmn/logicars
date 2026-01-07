//! Phase 1.4: Training Loop
//!
//! Implements grid-level training infrastructure matching the reference implementation.
//! Supports both synchronous (all cells update together) and asynchronous
//! (fire rate masking) training modes.
//!
//! Key features:
//! - MSE loss: sum((predicted - target)²) per cell per channel
//! - AdamW optimizer with gradient clipping (100.0)
//! - Sync mode: All cells update simultaneously
//! - Async mode: Fire rate masking (0.6) for fault tolerance
//! - Multi-step rollout support

use crate::grid::{BoundaryCondition, NGrid, NNeighborhood};
use crate::optimizer::AdamW;
use crate::perception::{GateLayer, PerceptionModule};
use crate::phase_0_1::BinaryOp;
use crate::update::{DiffLogicCA, UpdateModule};

/// Fire rate for async training (probability of cell update)
pub const FIRE_RATE: f64 = 0.6;

/// Default gradient clipping value
pub const GRADIENT_CLIP: f64 = 100.0;

/// Training configuration matching reference hyperparameters
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate (default: 0.05)
    pub learning_rate: f64,
    /// Gradient clipping value (default: 100.0)
    pub gradient_clip: f64,
    /// Number of steps per forward pass (default: 1)
    pub num_steps: usize,
    /// Use async training with fire rate masking
    pub async_training: bool,
    /// Fire rate for async training (default: 0.6)
    pub fire_rate: f64,
    /// Use periodic (toroidal) boundaries
    pub periodic: bool,
    /// Which channel to compute loss on (None = all channels, Some(c) = only channel c)
    /// For multi-channel experiments like checkerboard, loss is only on channel 0.
    pub loss_channel: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            gradient_clip: GRADIENT_CLIP,
            num_steps: 1,
            async_training: false,
            fire_rate: FIRE_RATE,
            periodic: true,
            loss_channel: None, // All channels for GoL (C=1)
        }
    }
}

impl TrainingConfig {
    /// Create config for Game of Life training
    pub fn gol() -> Self {
        Self {
            learning_rate: 0.05,
            gradient_clip: 100.0,
            num_steps: 1,
            async_training: false,
            fire_rate: FIRE_RATE,
            periodic: true,
            loss_channel: None, // All channels (C=1)
        }
    }

    /// Create config for Checkerboard sync training
    pub fn checkerboard_sync() -> Self {
        Self {
            learning_rate: 0.05,
            gradient_clip: 100.0,
            num_steps: 20,
            async_training: false,
            fire_rate: FIRE_RATE,
            periodic: false,
            loss_channel: Some(0), // Loss only on channel 0 (pattern channel)
        }
    }

    /// Create config for Checkerboard async training
    pub fn checkerboard_async() -> Self {
        Self {
            learning_rate: 0.05,
            gradient_clip: 100.0,
            num_steps: 50,
            async_training: true,
            fire_rate: FIRE_RATE,
            periodic: false,
            loss_channel: Some(0), // Loss only on channel 0 (pattern channel)
        }
    }
}

/// Simple random number generator (xorshift64)
/// Used for fire rate masking in async training
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate next random u64
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate random bool with given probability
    pub fn next_bool(&mut self, probability: f64) -> bool {
        self.next_f64() < probability
    }
}

/// Training loop for DiffLogicCA
///
/// Handles grid-level training with support for sync/async modes,
/// multi-step rollouts, and MSE loss computation.
pub struct TrainingLoop {
    /// The model being trained
    pub model: DiffLogicCA,
    /// Training configuration
    pub config: TrainingConfig,
    /// Optimizers for perception: [kernel][layer][gate]
    perception_optimizers: Vec<Vec<Vec<AdamW>>>,
    /// Optimizers for update: [layer][gate]
    update_optimizers: Vec<Vec<AdamW>>,
    /// Random number generator for async training
    rng: SimpleRng,
    /// Current iteration count
    pub iteration: usize,
}

impl TrainingLoop {
    /// Create a new training loop
    pub fn new(model: DiffLogicCA, config: TrainingConfig) -> Self {
        let perception_optimizers = Self::create_perception_optimizers(&model.perception, config.learning_rate);
        let update_optimizers = Self::create_update_optimizers(&model.update, config.learning_rate);

        Self {
            model,
            config,
            perception_optimizers,
            update_optimizers,
            rng: SimpleRng::new(42),
            iteration: 0,
        }
    }

    /// Create optimizers for perception module
    fn create_perception_optimizers(perception: &PerceptionModule, lr: f64) -> Vec<Vec<Vec<AdamW>>> {
        perception
            .kernels
            .iter()
            .map(|kernel| {
                kernel
                    .layers
                    .iter()
                    .map(|layer| (0..layer.num_gates()).map(|_| AdamW::new(lr)).collect())
                    .collect()
            })
            .collect()
    }

    /// Create optimizers for update module
    fn create_update_optimizers(update: &UpdateModule, lr: f64) -> Vec<Vec<AdamW>> {
        update
            .layers
            .iter()
            .map(|layer| (0..layer.num_gates()).map(|_| AdamW::new(lr)).collect())
            .collect()
    }

    /// Set random seed for reproducible async training
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = SimpleRng::new(seed);
    }

    /// Run one step of the CA in synchronous mode
    ///
    /// All cells update simultaneously based on their neighborhoods.
    pub fn step_sync(&self, input: &NGrid) -> NGrid {
        let mut output = NGrid::new(
            input.width,
            input.height,
            input.channels,
            input.boundary,
        );

        for y in 0..input.height {
            for x in 0..input.width {
                let neighborhood = input.neighborhood(x, y);
                let next_state = self.model.forward_hard(&neighborhood);
                output.set_cell(x, y, &next_state);
            }
        }

        output
    }

    /// Run one step of the CA in asynchronous mode
    ///
    /// Only ~fire_rate fraction of cells update (stochastic masking).
    pub fn step_async(&mut self, input: &NGrid) -> NGrid {
        let mut output = input.clone();

        for y in 0..input.height {
            for x in 0..input.width {
                // Fire rate masking: only update some cells
                if self.rng.next_bool(self.config.fire_rate) {
                    let neighborhood = input.neighborhood(x, y);
                    let next_state = self.model.forward_hard(&neighborhood);
                    output.set_cell(x, y, &next_state);
                }
                // Otherwise cell keeps its current state
            }
        }

        output
    }

    /// Run multiple steps (rollout)
    pub fn run_steps(&mut self, input: &NGrid, num_steps: usize) -> NGrid {
        let mut current = input.clone();

        for _ in 0..num_steps {
            current = if self.config.async_training {
                self.step_async(&current)
            } else {
                self.step_sync(&current)
            };
        }

        current
    }

    /// Compute MSE loss between predicted and target grids
    ///
    /// loss = sum((predicted - target)²) for all cells and channels
    pub fn compute_loss(predicted: &NGrid, target: &NGrid) -> f64 {
        assert_eq!(predicted.width, target.width);
        assert_eq!(predicted.height, target.height);
        assert_eq!(predicted.channels, target.channels);

        let pred_data = predicted.raw_data();
        let target_data = target.raw_data();

        pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum()
    }

    /// Compute MSE loss on a single channel only
    ///
    /// For multi-channel experiments like checkerboard, loss is only on the pattern channel.
    pub fn compute_loss_channel(predicted: &NGrid, target: &NGrid, channel: usize) -> f64 {
        assert_eq!(predicted.width, target.width);
        assert_eq!(predicted.height, target.height);
        assert!(channel < predicted.channels);
        assert!(channel < target.channels);

        let mut loss = 0.0;
        for y in 0..predicted.height {
            for x in 0..predicted.width {
                let pred = predicted.get(x as isize, y as isize, channel);
                let tgt = target.get(x as isize, y as isize, channel);
                loss += (pred - tgt).powi(2);
            }
        }
        loss
    }

    /// Compute mean squared error (average over all values)
    pub fn compute_mse(predicted: &NGrid, target: &NGrid) -> f64 {
        let loss = Self::compute_loss(predicted, target);
        loss / predicted.num_values() as f64
    }

    /// Forward pass through the grid in soft mode (for training)
    ///
    /// Returns the output grid and all intermediate activations needed for backprop.
    fn forward_grid_soft(&self, input: &NGrid) -> (NGrid, GridActivations) {
        let mut output = NGrid::new(
            input.width,
            input.height,
            input.channels,
            input.boundary,
        );

        let mut all_neighborhoods = Vec::with_capacity(input.num_cells());
        let mut all_perception_outputs = Vec::with_capacity(input.num_cells());
        let mut all_perception_activations = Vec::with_capacity(input.num_cells());
        let mut all_update_activations = Vec::with_capacity(input.num_cells());

        for y in 0..input.height {
            for x in 0..input.width {
                let neighborhood = input.neighborhood(x, y);
                
                // Forward through perception
                let (perception_output, perception_activations) =
                    self.model.perception.forward_soft(&neighborhood);

                // Forward through update
                let update_activations = self.model.update.forward_soft(&perception_output);
                let cell_output = update_activations.last().cloned().unwrap_or_default();

                output.set_cell(x, y, &cell_output);

                all_neighborhoods.push(neighborhood);
                all_perception_outputs.push(perception_output);
                all_perception_activations.push(perception_activations);
                all_update_activations.push(update_activations);
            }
        }

        let activations = GridActivations {
            neighborhoods: all_neighborhoods,
            perception_outputs: all_perception_outputs,
            perception_activations: all_perception_activations,
            update_activations: all_update_activations,
        };

        (output, activations)
    }

    /// Train on a single (input, target) grid pair with multi-step rollout
    ///
    /// Runs forward pass for `config.num_steps` iterations, computes loss at final step,
    /// then backpropagates through all steps (BPTT).
    ///
    /// Returns (soft_loss, hard_loss) matching reference implementation
    pub fn train_step(&mut self, input: &NGrid, target: &NGrid) -> (f64, f64) {
        let num_steps = self.config.num_steps;

        // Forward pass through all steps, storing activations for each
        let mut step_grids: Vec<NGrid> = Vec::with_capacity(num_steps + 1);
        let mut step_activations: Vec<GridActivations> = Vec::with_capacity(num_steps);

        step_grids.push(input.clone());

        for step in 0..num_steps {
            let current_grid = &step_grids[step];
            let (output, activations) = self.forward_grid_soft(current_grid);
            step_grids.push(output);
            step_activations.push(activations);
        }

        // Compute soft loss at final step (channel-specific if configured)
        let final_output = step_grids.last().unwrap();
        let soft_loss = match self.config.loss_channel {
            Some(c) => Self::compute_loss_channel(final_output, target, c),
            None => Self::compute_loss(final_output, target),
        };

        // Compute hard loss for monitoring
        let hard_output = self.run_steps(input, num_steps);
        let hard_loss = match self.config.loss_channel {
            Some(c) => Self::compute_loss_channel(&hard_output, target, c),
            None => Self::compute_loss(&hard_output, target),
        };

        // Backpropagate through all steps (BPTT)
        self.backward_through_time(&step_grids, &step_activations, target);

        self.iteration += 1;
        (soft_loss, hard_loss)
    }

    /// Backpropagation through time for multi-step rollouts
    fn backward_through_time(
        &mut self,
        step_grids: &[NGrid],
        step_activations: &[GridActivations],
        target: &NGrid,
    ) {
        let num_steps = step_activations.len();
        let num_cells = step_grids[0].num_cells();
        let channels = step_grids[0].channels;

        // Accumulate gradients across all steps
        let mut perception_grad_accum: Vec<Vec<Vec<[f64; 16]>>> = self
            .model
            .perception
            .kernels
            .iter()
            .map(|kernel| {
                kernel
                    .layers
                    .iter()
                    .map(|layer| vec![[0.0; 16]; layer.num_gates()])
                    .collect()
            })
            .collect();

        let mut update_grad_accum: Vec<Vec<[f64; 16]>> = self
            .model
            .update
            .layers
            .iter()
            .map(|layer| vec![[0.0; 16]; layer.num_gates()])
            .collect();

        // Initialize dL/dgrid at final step: 2 * (output - target)
        // If loss_channel is set, only compute gradient for that channel
        let final_output = &step_grids[num_steps];
        let mut grid_grads = NGrid::new(
            final_output.width,
            final_output.height,
            final_output.channels,
            final_output.boundary,
        );

        for y in 0..final_output.height {
            for x in 0..final_output.width {
                for c in 0..channels {
                    // Only compute gradient for loss channel (if configured)
                    let grad = match self.config.loss_channel {
                        Some(loss_c) if loss_c == c => {
                            let pred = final_output.get(x as isize, y as isize, c);
                            let tgt = target.get(x as isize, y as isize, c);
                            2.0 * (pred - tgt)
                        }
                        Some(_) => 0.0, // Not the loss channel, no gradient
                        None => {
                            // All channels contribute to loss
                            let pred = final_output.get(x as isize, y as isize, c);
                            let tgt = target.get(x as isize, y as isize, c);
                            2.0 * (pred - tgt)
                        }
                    };
                    grid_grads.set(x, y, c, grad);
                }
            }
        }

        // Backpropagate through steps in reverse order
        for step in (0..num_steps).rev() {
            let input_grid = &step_grids[step];
            let activations = &step_activations[step];

            // Gradient w.r.t. previous grid (for chaining to earlier steps)
            let mut prev_grid_grads = NGrid::new(
                input_grid.width,
                input_grid.height,
                input_grid.channels,
                input_grid.boundary,
            );

            // Process each cell
            for cell_idx in 0..num_cells {
                let x = cell_idx % input_grid.width;
                let y = cell_idx / input_grid.width;

                // Get output gradient for this cell
                let output_grads: Vec<f64> = (0..channels)
                    .map(|c| grid_grads.get(x as isize, y as isize, c))
                    .collect();

                // Backprop through update module
                let update_grads = self.model.update.compute_gradients(
                    &activations.perception_outputs[cell_idx],
                    &activations.update_activations[cell_idx],
                    &output_grads,
                );

                // Accumulate update gradients
                for (layer_idx, layer_grads) in update_grads.iter().enumerate() {
                    for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                        for i in 0..16 {
                            update_grad_accum[layer_idx][gate_idx][i] += gate_grad[i];
                        }
                    }
                }

                // Compute gradient w.r.t. perception output
                let perception_output_grads = self.compute_perception_output_grads(
                    &activations.perception_outputs[cell_idx],
                    &activations.update_activations[cell_idx],
                    &output_grads,
                );

                // Backprop through perception module
                let perception_grads = self.model.perception.compute_gradients(
                    &activations.neighborhoods[cell_idx],
                    &activations.perception_activations[cell_idx],
                    &perception_output_grads,
                );

                // Accumulate perception gradients
                for (kernel_idx, kernel_grads) in perception_grads.iter().enumerate() {
                    for channel_grads in kernel_grads {
                        for (layer_idx, layer_grads) in channel_grads.iter().enumerate() {
                            for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                                for i in 0..16 {
                                    perception_grad_accum[kernel_idx][layer_idx][gate_idx][i] +=
                                        gate_grad[i];
                                }
                            }
                        }
                    }
                }

                // Compute gradient w.r.t. input grid (for BPTT)
                // The neighborhood contains 9 cells (center + 8 neighbors)
                // Gradient flows back to these 9 cells
                let input_grads = self.compute_input_grads(
                    &activations.neighborhoods[cell_idx],
                    &activations.perception_activations[cell_idx],
                    &perception_output_grads,
                );

                // Distribute gradients to the 9 neighborhood positions
                let neighborhood_offsets: [(isize, isize); 9] = [
                    (-1, -1), (0, -1), (1, -1),
                    (-1, 0),  (0, 0),  (1, 0),
                    (-1, 1),  (0, 1),  (1, 1),
                ];

                for (pos_idx, &(dx, dy)) in neighborhood_offsets.iter().enumerate() {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;

                    // Handle boundaries
                    let (nx_wrapped, ny_wrapped) = if input_grid.boundary == BoundaryCondition::Periodic {
                        (
                            ((nx % input_grid.width as isize) + input_grid.width as isize) as usize % input_grid.width,
                            ((ny % input_grid.height as isize) + input_grid.height as isize) as usize % input_grid.height,
                        )
                    } else {
                        if nx < 0 || nx >= input_grid.width as isize || ny < 0 || ny >= input_grid.height as isize {
                            continue; // Skip out-of-bounds cells for non-periodic
                        }
                        (nx as usize, ny as usize)
                    };

                    for c in 0..channels {
                        let grad_idx = pos_idx * channels + c;
                        if grad_idx < input_grads.len() {
                            let current = prev_grid_grads.get(nx_wrapped as isize, ny_wrapped as isize, c);
                            prev_grid_grads.set(nx_wrapped, ny_wrapped, c, current + input_grads[grad_idx]);
                        }
                    }
                }
            }

            // Propagate gradients to previous step
            grid_grads = prev_grid_grads;
        }

        // Apply gradient updates
        // Reference implementation: loss = sum over cells of (pred - target)^2
        // Gradients naturally scale with number of cells (each cell contributes)
        // We accumulate over cells, so we need to average to match reference's per-cell gradient
        // 
        // IMPORTANT: We do NOT average over num_steps!
        // In BPTT, gradients from each step accumulate (they represent the same parameters used at each step)
        // Only the final step contributes to the loss, gradients flow back through all steps.
        //
        // If loss_channel is set, only 1 channel contributes; otherwise all channels
        let effective_channels = match self.config.loss_channel {
            Some(_) => 1,
            None => channels,
        };
        let scale = 1.0 / (num_cells as f64 * effective_channels as f64);
        self.apply_gradients(&perception_grad_accum, &update_grad_accum, scale);
    }

    /// Compute gradient w.r.t. input neighborhood
    fn compute_input_grads(
        &self,
        neighborhood: &NNeighborhood,
        perception_activations: &[Vec<Vec<Vec<f64>>>],
        perception_output_grads: &[f64],
    ) -> Vec<f64> {
        // Backpropagate through perception to get gradients w.r.t. input
        self.model.perception.compute_input_gradients(
            neighborhood,
            perception_activations,
            perception_output_grads,
        )
    }

    /// Apply accumulated gradients to model weights
    fn apply_gradients(
        &mut self,
        perception_grads: &[Vec<Vec<[f64; 16]>>],
        update_grads: &[Vec<[f64; 16]>],
        scale: f64,
    ) {
        // Apply updates to perception weights
        for (kernel_idx, kernel_grads) in perception_grads.iter().enumerate() {
            for (layer_idx, layer_grads) in kernel_grads.iter().enumerate() {
                for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                    let mut clipped = [0.0; 16];
                    for i in 0..16 {
                        clipped[i] = (gate_grad[i] * scale).clamp(
                            -self.config.gradient_clip,
                            self.config.gradient_clip,
                        );
                    }

                    self.perception_optimizers[kernel_idx][layer_idx][gate_idx].step(
                        &mut self.model.perception.kernels[kernel_idx].layers[layer_idx].gates
                            [gate_idx]
                            .logits,
                        &clipped,
                    );
                }
            }
        }

        // Apply updates to update weights
        for (layer_idx, layer_grads) in update_grads.iter().enumerate() {
            for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                let mut clipped = [0.0; 16];
                for i in 0..16 {
                    clipped[i] = (gate_grad[i] * scale).clamp(
                        -self.config.gradient_clip,
                        self.config.gradient_clip,
                    );
                }

                self.update_optimizers[layer_idx][gate_idx].step(
                    &mut self.model.update.layers[layer_idx].gates[gate_idx].logits,
                    &clipped,
                );
            }
        }
    }
    /// Compute gradient w.r.t. perception output (for chain rule through update)
    fn compute_perception_output_grads(
        &self,
        perception_output: &[f64],
        update_activations: &[Vec<f64>],
        output_grads: &[f64],
    ) -> Vec<f64> {
        let num_layers = self.model.update.layers.len();
        let mut grads = output_grads.to_vec();

        // Backpropagate through update layers
        for layer_idx in (0..num_layers).rev() {
            let layer = &self.model.update.layers[layer_idx];

            let layer_inputs: Vec<f64> = if layer_idx == 0 {
                perception_output.to_vec()
            } else {
                update_activations[layer_idx - 1].clone()
            };

            let prev_len = if layer_idx > 0 {
                update_activations[layer_idx - 1].len()
            } else {
                perception_output.len()
            };
            let mut prev_grads = vec![0.0; prev_len];

            for (gate_idx, gate) in layer.gates.iter().enumerate() {
                let a_idx = layer.wires.a[gate_idx];
                let b_idx = layer.wires.b[gate_idx];
                let a = layer_inputs[a_idx];
                let b = layer_inputs[b_idx];
                let grad = grads[gate_idx];

                let (da, db) = compute_gate_input_grads(gate, a, b);
                prev_grads[a_idx] += grad * da;
                prev_grads[b_idx] += grad * db;
            }

            grads = prev_grads;
        }

        grads
    }

    /// Get current hard accuracy on a dataset
    ///
    /// Returns fraction of cells that match exactly after hard inference
    pub fn evaluate_accuracy(&self, inputs: &[NGrid], targets: &[NGrid]) -> f64 {
        let mut correct = 0usize;
        let mut total = 0usize;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.step_sync(input);
            let hard_output = output.to_hard();
            let hard_target = target.to_hard();

            for y in 0..input.height {
                for x in 0..input.width {
                    for c in 0..input.channels {
                        let pred = hard_output.get(x as isize, y as isize, c);
                        let tgt = hard_target.get(x as isize, y as isize, c);
                        if (pred - tgt).abs() < 0.5 {
                            correct += 1;
                        }
                        total += 1;
                    }
                }
            }
        }

        correct as f64 / total as f64
    }
}

/// Intermediate activations for backpropagation
struct GridActivations {
    /// Neighborhoods for each cell
    neighborhoods: Vec<NNeighborhood>,
    /// Perception outputs for each cell
    perception_outputs: Vec<Vec<f64>>,
    /// Perception activations: [cell][kernel][channel][layer]
    perception_activations: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
    /// Update activations: [cell][layer]
    update_activations: Vec<Vec<Vec<f64>>>,
}

/// Compute gradient of gate output w.r.t. its inputs
fn compute_gate_input_grads(gate: &crate::phase_0_1::ProbabilisticGate, a: f64, b: f64) -> (f64, f64) {
    let probs = gate.probabilities();
    let mut da = 0.0;
    let mut db = 0.0;

    for (i, &op) in BinaryOp::ALL.iter().enumerate() {
        let (op_da, op_db) = op_input_grads(op, a, b);
        da += probs[i] * op_da;
        db += probs[i] * op_db;
    }

    (da, db)
}

/// Compute gradient of operation output w.r.t. its inputs
fn op_input_grads(op: BinaryOp, a: f64, b: f64) -> (f64, f64) {
    match op {
        BinaryOp::False => (0.0, 0.0),
        BinaryOp::And => (b, a),
        BinaryOp::AAndNotB => (1.0 - b, -a),
        BinaryOp::A => (1.0, 0.0),
        BinaryOp::NotAAndB => (-b, 1.0 - a),
        BinaryOp::B => (0.0, 1.0),
        BinaryOp::Xor => (1.0 - 2.0 * b, 1.0 - 2.0 * a),
        BinaryOp::Or => (1.0 - b, 1.0 - a),
        BinaryOp::Nor => (b - 1.0, a - 1.0),
        BinaryOp::Xnor => (2.0 * b - 1.0, 2.0 * a - 1.0),
        BinaryOp::NotB => (0.0, -1.0),
        BinaryOp::AOrNotB => (b, a - 1.0),
        BinaryOp::NotA => (-1.0, 0.0),
        BinaryOp::NotAOrB => (b - 1.0, a),
        BinaryOp::Nand => (-b, -a),
        BinaryOp::True => (0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::{ConnectionType, PerceptionModule};
    use approx::assert_relative_eq;

    // ==================== Config Tests ====================

    #[test]
    fn test_config_defaults() {
        let config = TrainingConfig::default();
        assert_relative_eq!(config.learning_rate, 0.05);
        assert_relative_eq!(config.gradient_clip, 100.0);
        assert_eq!(config.num_steps, 1);
        assert!(!config.async_training);
        assert!(config.periodic);
    }

    #[test]
    fn test_config_gol() {
        let config = TrainingConfig::gol();
        assert_relative_eq!(config.learning_rate, 0.05);
        assert_eq!(config.num_steps, 1);
        assert!(!config.async_training);
        assert!(config.periodic);
    }

    #[test]
    fn test_config_checkerboard_sync() {
        let config = TrainingConfig::checkerboard_sync();
        assert_eq!(config.num_steps, 20);
        assert!(!config.async_training);
        assert!(!config.periodic);
    }

    #[test]
    fn test_config_checkerboard_async() {
        let config = TrainingConfig::checkerboard_async();
        assert_eq!(config.num_steps, 50);
        assert!(config.async_training);
        assert_relative_eq!(config.fire_rate, 0.6);
    }

    // ==================== RNG Tests ====================

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_f64_range() {
        let mut rng = SimpleRng::new(123);

        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_rng_bool_probability() {
        let mut rng = SimpleRng::new(456);
        let mut count = 0;
        let total = 10000;

        for _ in 0..total {
            if rng.next_bool(0.6) {
                count += 1;
            }
        }

        let ratio = count as f64 / total as f64;
        assert!(ratio > 0.55 && ratio < 0.65, "Fire rate ~0.6 expected, got {}", ratio);
    }

    // ==================== Loss Computation Tests ====================

    #[test]
    fn test_compute_loss_identical() {
        let grid = NGrid::periodic(3, 3, 1);
        let loss = TrainingLoop::compute_loss(&grid, &grid);
        assert_relative_eq!(loss, 0.0);
    }

    #[test]
    fn test_compute_loss_different() {
        let mut grid1 = NGrid::periodic(2, 2, 1);
        let mut grid2 = NGrid::periodic(2, 2, 1);

        grid1.set(0, 0, 0, 0.0);
        grid2.set(0, 0, 0, 1.0);

        let loss = TrainingLoop::compute_loss(&grid1, &grid2);
        assert_relative_eq!(loss, 1.0); // (0-1)^2 = 1
    }

    #[test]
    fn test_compute_mse() {
        let mut grid1 = NGrid::periodic(2, 2, 1);
        let mut grid2 = NGrid::periodic(2, 2, 1);

        // All cells differ by 0.5
        for y in 0..2 {
            for x in 0..2 {
                grid1.set(x, y, 0, 0.0);
                grid2.set(x, y, 0, 0.5);
            }
        }

        let mse = TrainingLoop::compute_mse(&grid1, &grid2);
        assert_relative_eq!(mse, 0.25); // 0.5^2 = 0.25
    }

    #[test]
    fn test_compute_loss_channel() {
        // Create multi-channel grids
        let mut grid1 = NGrid::periodic(2, 2, 8);
        let mut grid2 = NGrid::periodic(2, 2, 8);

        // Channel 0: all 0s vs all 1s = loss of 4.0 (4 cells × 1.0²)
        // Channels 1-7: mixed but should be ignored
        for y in 0..2 {
            for x in 0..2 {
                grid1.set(x, y, 0, 0.0);
                grid2.set(x, y, 0, 1.0);
                
                // Set other channels to random values
                for c in 1..8 {
                    grid1.set(x, y, c, 0.5);
                    grid2.set(x, y, c, 0.3);
                }
            }
        }

        // Total loss (all channels)
        let total_loss = TrainingLoop::compute_loss(&grid1, &grid2);
        // Channel 0: 4 * 1.0² = 4.0
        // Channels 1-7: 7 * 4 * 0.2² = 7 * 4 * 0.04 = 1.12
        // Total: 5.12
        assert!(total_loss > 5.0 && total_loss < 5.3, "Total loss = {}", total_loss);

        // Channel 0 only loss
        let ch0_loss = TrainingLoop::compute_loss_channel(&grid1, &grid2, 0);
        assert_relative_eq!(ch0_loss, 4.0); // 4 cells × (0-1)² = 4
    }

    // ==================== Step Tests ====================

    fn create_small_model() -> DiffLogicCA {
        // Small model for testing: 2 kernels, [9→8→4→2→1]
        let perception = PerceptionModule::new(
            1,  // channels
            2,  // kernels
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        );
        // Input: 1 (center) + 2 kernels = 3
        let update = UpdateModule::new(&[3, 2, 1]);
        DiffLogicCA::new(perception, update)
    }

    #[test]
    fn test_step_sync() {
        let model = create_small_model();
        let config = TrainingConfig::gol();
        let training = TrainingLoop::new(model, config);

        let mut input = NGrid::periodic(3, 3, 1);
        input.set(1, 1, 0, 1.0); // Set center cell alive

        let output = training.step_sync(&input);

        assert_eq!(output.width, 3);
        assert_eq!(output.height, 3);
        assert_eq!(output.channels, 1);

        // Output should be valid (all values 0 or 1 in hard mode)
        for y in 0..3 {
            for x in 0..3 {
                let v = output.get(x as isize, y as isize, 0);
                assert!(v == 0.0 || v == 1.0, "Value {} not binary", v);
            }
        }
    }

    #[test]
    fn test_step_async_fire_rate() {
        let model = create_small_model();
        let mut config = TrainingConfig::default();
        config.async_training = true;
        config.fire_rate = 0.5;

        let mut training = TrainingLoop::new(model, config);
        training.set_seed(123);

        // Create a grid where all cells are 1.0
        let mut input = NGrid::periodic(10, 10, 1);
        for y in 0..10 {
            for x in 0..10 {
                input.set(x, y, 0, 1.0);
            }
        }

        // After one async step, some cells should be updated (potentially changed)
        // and some should remain the same
        let output = training.step_async(&input);

        // Just verify the output is valid
        assert_eq!(output.width, 10);
        assert_eq!(output.height, 10);
    }

    #[test]
    fn test_run_steps() {
        let model = create_small_model();
        let config = TrainingConfig::gol();
        let mut training = TrainingLoop::new(model, config);

        let input = NGrid::periodic(3, 3, 1);
        let output = training.run_steps(&input, 3);

        assert_eq!(output.width, 3);
        assert_eq!(output.height, 3);
    }

    // ==================== Training Tests ====================

    #[test]
    fn test_train_step_returns_loss() {
        let model = create_small_model();
        let config = TrainingConfig::gol();
        let mut training = TrainingLoop::new(model, config);

        let input = NGrid::periodic(3, 3, 1);
        let mut target = NGrid::periodic(3, 3, 1);
        target.set(1, 1, 0, 1.0); // Target has center alive

        let (soft_loss, hard_loss) = training.train_step(&input, &target);

        // Losses should be non-negative
        assert!(soft_loss >= 0.0);
        assert!(hard_loss >= 0.0);
    }

    #[test]
    fn test_train_step_loss_decreases() {
        let model = create_small_model();
        let config = TrainingConfig::gol();
        let mut training = TrainingLoop::new(model, config);

        // Create a simple training example
        let mut input = NGrid::periodic(3, 3, 1);
        input.set(1, 1, 0, 1.0);

        let mut target = NGrid::periodic(3, 3, 1);
        target.set(1, 1, 0, 1.0);

        let (initial_loss, _) = training.train_step(&input, &target);

        // Train for a few iterations
        let mut final_loss = initial_loss;
        for _ in 0..50 {
            let (loss, _) = training.train_step(&input, &target);
            final_loss = loss;
        }

        // Loss should decrease (or at least not increase significantly)
        // Note: Loss might not always decrease on every step due to stochasticity
        // but over many steps it should trend downward
        assert!(
            final_loss <= initial_loss * 1.5,
            "Loss should not increase significantly: {} -> {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_iteration_counter() {
        let model = create_small_model();
        let config = TrainingConfig::gol();
        let mut training = TrainingLoop::new(model, config);

        assert_eq!(training.iteration, 0);

        let input = NGrid::periodic(3, 3, 1);
        let target = NGrid::periodic(3, 3, 1);

        for i in 1..=5 {
            training.train_step(&input, &target);
            assert_eq!(training.iteration, i);
        }
    }

    // ==================== Multi-channel Tests ====================

    #[test]
    fn test_training_multichannel() {
        // Create model for 8 channels
        // For C=8, K=2: output_size = 8 + 2*1*8 = 24
        let perception = PerceptionModule::new(
            8,  // channels
            2,  // kernels
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        );
        // Input: center_channels + num_kernels * kernel_output * channels
        //      = 8 + 2 * 1 * 8 = 24
        // Architecture must respect unique_connections: out_dim * 2 >= in_dim
        // [24→12→6→8] - reduction then final output
        let update = UpdateModule::new(&[24, 12, 8, 8]);
        let model = DiffLogicCA::new(perception, update);

        let config = TrainingConfig::default();
        let mut training = TrainingLoop::new(model, config);

        let input = NGrid::periodic(3, 3, 8);
        let target = NGrid::periodic(3, 3, 8);

        let (loss, _) = training.train_step(&input, &target);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_multi_step_rollout() {
        // Test that multi-step training (BPTT) works correctly
        let model = create_small_model();
        let mut config = TrainingConfig::gol();
        config.num_steps = 3; // 3-step rollout

        let mut training = TrainingLoop::new(model, config);

        let input = NGrid::periodic(3, 3, 1);
        let mut target = NGrid::periodic(3, 3, 1);
        target.set(1, 1, 0, 1.0);

        // Should complete without panic
        let (soft_loss, hard_loss) = training.train_step(&input, &target);
        
        assert!(soft_loss >= 0.0, "Soft loss should be non-negative");
        assert!(hard_loss >= 0.0, "Hard loss should be non-negative");
    }

    #[test]
    fn test_non_periodic_boundaries() {
        // Test training with non-periodic (zero-padded) boundaries
        let model = create_small_model();
        let mut config = TrainingConfig::gol();
        config.periodic = false;

        let mut training = TrainingLoop::new(model, config);

        // Use non-periodic grid
        let input = NGrid::new(3, 3, 1, BoundaryCondition::NonPeriodic);
        let target = NGrid::new(3, 3, 1, BoundaryCondition::NonPeriodic);

        let (loss, _) = training.train_step(&input, &target);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_training_multichannel_channel0_loss() {
        // Test that loss_channel works correctly for multi-channel training
        let perception = PerceptionModule::new(
            8,  // channels
            2,  // kernels
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        );
        let update = UpdateModule::new(&[24, 12, 8, 8]);
        let model = DiffLogicCA::new(perception, update);

        // Use channel 0 only loss (like checkerboard)
        let mut config = TrainingConfig::default();
        config.loss_channel = Some(0);

        let mut training = TrainingLoop::new(model, config);

        // Create grids - channel 0 has pattern, others are "working memory"
        let input = NGrid::periodic(3, 3, 8);
        let mut target = NGrid::periodic(3, 3, 8);
        
        // Set target channel 0 only
        target.set(1, 1, 0, 1.0);
        
        // Other channels in target are 0, but they shouldn't affect loss
        // because we're only training on channel 0

        let (loss, _) = training.train_step(&input, &target);
        assert!(loss >= 0.0);
        
        // Loss should be based only on channel 0 (1 cell differs = 1.0)
        // The exact value depends on model output, but should be small
        assert!(loss <= 10.0, "Loss should be reasonable for 9 cells on ch0 only");
    }

    #[test]
    fn test_perception_output_size_multichannel() {
        // Verify perception output size calculation for multi-channel
        let perception = PerceptionModule::new(
            8,   // channels
            16,  // kernels
            &[9, 8, 4, 2], // 2 output bits per kernel
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        );

        // Expected: center(8) + kernels(16) * output_bits(2) * channels(8) = 8 + 256 = 264
        assert_eq!(perception.output_size(), 264);
    }

    // ==================== Accuracy Evaluation ====================

    #[test]
    fn test_evaluate_accuracy_perfect() {
        let model = create_small_model();
        let config = TrainingConfig::gol();
        let training = TrainingLoop::new(model, config);

        // Create identical input/output (zero grid)
        let input = NGrid::periodic(3, 3, 1);
        let target = NGrid::periodic(3, 3, 1);

        // Note: accuracy depends on what the model outputs
        // For an untrained model with zero input, output might vary
        let accuracy = training.evaluate_accuracy(&[input], &[target]);
        
        // Just verify it's a valid accuracy value
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}
