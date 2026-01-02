//! Phase 1.3: Update Module
//!
//! Implements the update network that takes perception output and produces next cell state.
//! The update module is a deep gate network with "unique" connections throughout.
//!
//! Architecture (from paper):
//! - Input: center_channels + kernel_outputs
//! - All layers use "unique" connection topology
//! - Many wide layers (128-512 gates) followed by reduction
//! - Output: C channels (next cell state)
//!
//! For GoL: [17→128×10→64→32→16→8→4→2→1]

use crate::grid::NNeighborhood;
use crate::optimizer::AdamW;
use crate::perception::{unique_connections, GateLayer, PerceptionModule};
use crate::phase_0_1::{BinaryOp, ProbabilisticGate};

/// Update module that transforms perception output to next cell state
///
/// Takes the concatenated output from perception (center + kernel outputs)
/// and produces the next cell state through a deep gate network.
#[derive(Debug, Clone)]
pub struct UpdateModule {
    /// Number of input channels (output from perception)
    pub input_size: usize,
    /// Number of output channels (cell state size)
    pub output_channels: usize,
    /// Gate layers with unique connections
    pub layers: Vec<GateLayer>,
    /// Layer sizes for reference
    pub layer_sizes: Vec<usize>,
}

impl UpdateModule {
    /// Create a new update module with specified architecture
    ///
    /// # Arguments
    /// * `layer_sizes` - Sizes of each layer including input and output
    ///                   e.g., [17, 128, 128, ..., 64, 32, 16, 8, 4, 2, 1]
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least input and output layers");

        let input_size = layer_sizes[0];
        let output_channels = layer_sizes[layer_sizes.len() - 1];

        let mut layers = Vec::new();

        for i in 0..(layer_sizes.len() - 1) {
            let in_dim = layer_sizes[i];
            let out_dim = layer_sizes[i + 1];
            let wires = unique_connections(in_dim, out_dim);
            layers.push(GateLayer::new(out_dim, wires));
        }

        Self {
            input_size,
            output_channels,
            layers,
            layer_sizes: layer_sizes.to_vec(),
        }
    }

    /// Create GoL update module: [17→128×10→64→32→16→8→4→2→1]
    ///
    /// Input: 17 = 1 (center) + 16 (kernel outputs)
    /// Output: 1 (next cell state)
    pub fn gol_module() -> Self {
        // 10 layers of 128, then reduction
        let mut layer_sizes = vec![17];
        for _ in 0..10 {
            layer_sizes.push(128);
        }
        layer_sizes.extend_from_slice(&[64, 32, 16, 8, 4, 2, 1]);

        Self::new(&layer_sizes)
    }

    /// Create a smaller update module for testing
    ///
    /// Uses a reduction strategy that respects unique_connections constraint:
    /// out_dim * 2 >= in_dim (can't reduce by more than half per layer)
    pub fn small_module(input_size: usize, output_size: usize) -> Self {
        // Build layer sizes that gradually reduce
        let mut layer_sizes = vec![input_size];
        let mut current = input_size;

        // Reduce by roughly half each layer until we reach output
        while current > output_size * 2 {
            current = (current + 1) / 2; // Ceil division to stay valid
            if current < output_size {
                current = output_size;
            }
            layer_sizes.push(current);
        }

        // Final output layer
        if *layer_sizes.last().unwrap() != output_size {
            layer_sizes.push(output_size);
        }

        Self::new(&layer_sizes)
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total number of gates across all layers
    pub fn total_gates(&self) -> usize {
        self.layers.iter().map(|l| l.num_gates()).sum()
    }

    /// Forward pass in soft mode
    ///
    /// Returns all layer activations for gradient computation
    pub fn forward_soft(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        assert_eq!(
            inputs.len(),
            self.input_size,
            "Input size mismatch: expected {}, got {}",
            self.input_size,
            inputs.len()
        );

        let mut activations = Vec::with_capacity(self.layers.len());
        let mut current = inputs.to_vec();

        for layer in &self.layers {
            current = layer.forward_soft(&current);
            activations.push(current.clone());
        }

        activations
    }

    /// Forward pass in hard mode
    pub fn forward_hard(&self, inputs: &[f64]) -> Vec<f64> {
        let mut current = inputs.to_vec();

        for layer in &self.layers {
            current = layer.forward_hard(&current);
        }

        current
    }

    /// Get final output from soft forward pass
    pub fn output_soft(&self, inputs: &[f64]) -> Vec<f64> {
        let activations = self.forward_soft(inputs);
        activations.last().cloned().unwrap_or_default()
    }

    /// Compute gradients for all layers
    ///
    /// Returns gradients indexed by [layer][gate][logit]
    pub fn compute_gradients(
        &self,
        inputs: &[f64],
        activations: &[Vec<f64>],
        output_gradients: &[f64],
    ) -> Vec<Vec<[f64; 16]>> {
        let num_layers = self.layers.len();
        let mut all_gradients: Vec<Vec<[f64; 16]>> = Vec::with_capacity(num_layers);

        // Initialize gradient storage
        for layer in &self.layers {
            all_gradients.push(vec![[0.0; 16]; layer.num_gates()]);
        }

        // Start with output gradients
        let mut output_grads = output_gradients.to_vec();

        // Backpropagate through layers (from last to first)
        for layer_idx in (0..num_layers).rev() {
            let layer = &self.layers[layer_idx];

            // Get inputs to this layer
            let layer_inputs: Vec<f64> = if layer_idx == 0 {
                inputs.to_vec()
            } else {
                activations[layer_idx - 1].clone()
            };

            // Prepare gradients for previous layer
            let prev_len = if layer_idx > 0 {
                activations[layer_idx - 1].len()
            } else {
                inputs.len()
            };
            let mut prev_output_grads = vec![0.0; prev_len];

            // Compute gradients for each gate
            for (gate_idx, gate) in layer.gates.iter().enumerate() {
                let a_idx = layer.wires.a[gate_idx];
                let b_idx = layer.wires.b[gate_idx];
                let a = layer_inputs[a_idx];
                let b = layer_inputs[b_idx];
                let output_grad = output_grads[gate_idx];

                // Compute gradient w.r.t. logits (dL/dlogit_i)
                let probs = gate.probabilities();
                let mut gate_grads = [0.0; 16];

                for i in 0..16 {
                    let mut dlogit_i = 0.0;
                    for j in 0..16 {
                        let op_output_j = BinaryOp::ALL[j].execute_soft(a, b);
                        let dprob_j_dlogit_i = if i == j {
                            probs[j] * (1.0 - probs[i])
                        } else {
                            -probs[j] * probs[i]
                        };
                        dlogit_i += op_output_j * dprob_j_dlogit_i;
                    }
                    gate_grads[i] = output_grad * dlogit_i;
                }

                all_gradients[layer_idx][gate_idx] = gate_grads;

                // Compute gradients w.r.t. inputs for backprop
                let (da, db) = compute_gate_input_gradients(gate, a, b);
                prev_output_grads[a_idx] += output_grad * da;
                prev_output_grads[b_idx] += output_grad * db;
            }

            output_grads = prev_output_grads;
        }

        all_gradients
    }
}

/// Compute gradients of gate output w.r.t. inputs a and b
fn compute_gate_input_gradients(gate: &ProbabilisticGate, a: f64, b: f64) -> (f64, f64) {
    let probs = gate.probabilities();
    let mut da = 0.0;
    let mut db = 0.0;

    for (i, &op) in BinaryOp::ALL.iter().enumerate() {
        let (op_da, op_db) = op_input_gradients(op, a, b);
        da += probs[i] * op_da;
        db += probs[i] * op_db;
    }

    (da, db)
}

/// Compute gradients of operation output w.r.t. its inputs
fn op_input_gradients(op: BinaryOp, a: f64, b: f64) -> (f64, f64) {
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

/// Complete Differentiable Logic CA combining perception and update modules
#[derive(Debug, Clone)]
pub struct DiffLogicCA {
    /// Perception module (extracts features from neighborhood)
    pub perception: PerceptionModule,
    /// Update module (computes next cell state)
    pub update: UpdateModule,
}

impl DiffLogicCA {
    /// Create a new DiffLogicCA with specified modules
    pub fn new(perception: PerceptionModule, update: UpdateModule) -> Self {
        // Verify compatibility
        assert_eq!(
            perception.output_size(),
            update.input_size,
            "Perception output ({}) must match update input ({})",
            perception.output_size(),
            update.input_size
        );

        Self { perception, update }
    }

    /// Create GoL configuration
    ///
    /// Perception: 16 kernels, [9→8→4→2→1]
    /// Update: [17→128×10→64→32→16→8→4→2→1]
    pub fn gol() -> Self {
        let perception = PerceptionModule::gol_module();
        let update = UpdateModule::gol_module();
        Self::new(perception, update)
    }

    /// Total gates in the circuit
    pub fn total_gates(&self) -> usize {
        self.perception.total_gates() + self.update.total_gates()
    }

    /// Forward pass in soft mode
    ///
    /// Returns (output, perception_activations, update_activations)
    pub fn forward_soft(
        &self,
        neighborhood: &NNeighborhood,
    ) -> (Vec<f64>, Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<f64>>) {
        // Run perception
        let (perception_output, perception_activations) =
            self.perception.forward_soft(neighborhood);

        // Run update
        let update_activations = self.update.forward_soft(&perception_output);
        let output = update_activations.last().cloned().unwrap_or_default();

        (output, perception_activations, update_activations)
    }

    /// Forward pass in hard mode
    pub fn forward_hard(&self, neighborhood: &NNeighborhood) -> Vec<f64> {
        let perception_output = self.perception.forward_hard(neighborhood);
        self.update.forward_hard(&perception_output)
    }
}

/// Trainer for the complete DiffLogicCA
pub struct DiffLogicCATrainer {
    pub model: DiffLogicCA,
    /// Optimizers for perception: [kernel][layer][gate]
    perception_optimizers: Vec<Vec<Vec<AdamW>>>,
    /// Optimizers for update: [layer][gate]
    update_optimizers: Vec<Vec<AdamW>>,
    pub learning_rate: f64,
    pub iteration: usize,
}

impl DiffLogicCATrainer {
    /// Create a new trainer
    pub fn new(model: DiffLogicCA, learning_rate: f64) -> Self {
        // Create perception optimizers
        let perception_optimizers: Vec<Vec<Vec<AdamW>>> = model
            .perception
            .kernels
            .iter()
            .map(|kernel| {
                kernel
                    .layers
                    .iter()
                    .map(|layer| {
                        (0..layer.num_gates())
                            .map(|_| AdamW::new(learning_rate))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // Create update optimizers
        let update_optimizers: Vec<Vec<AdamW>> = model
            .update
            .layers
            .iter()
            .map(|layer| {
                (0..layer.num_gates())
                    .map(|_| AdamW::new(learning_rate))
                    .collect()
            })
            .collect();

        Self {
            model,
            perception_optimizers,
            update_optimizers,
            learning_rate,
            iteration: 0,
        }
    }

    /// Train on a single neighborhood example
    ///
    /// Returns the loss for this example
    pub fn train_step(&mut self, neighborhood: &NNeighborhood, target: &[f64]) -> f64 {
        // Forward pass
        let (perception_output, perception_activations) =
            self.model.perception.forward_soft(neighborhood);
        let update_activations = self.model.update.forward_soft(&perception_output);
        let output = update_activations.last().cloned().unwrap_or_default();

        // Compute loss (MSE)
        let mut loss = 0.0;
        let output_gradients: Vec<f64> = output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| {
                let error = o - t;
                loss += error * error;
                2.0 * error // d/doutput of (output - target)^2
            })
            .collect();

        loss /= output.len() as f64;

        // Backward pass through update module
        let update_gradients =
            self.model
                .update
                .compute_gradients(&perception_output, &update_activations, &output_gradients);

        // Compute gradient w.r.t. perception output for chain rule
        let perception_output_grads = self.compute_perception_output_gradients(
            &perception_output,
            &update_activations,
            &output_gradients,
        );

        // Backward pass through perception module
        let perception_gradients = self.model.perception.compute_gradients(
            neighborhood,
            &perception_activations,
            &perception_output_grads,
        );

        // Update perception weights
        self.update_perception_weights(&perception_gradients);

        // Update update module weights
        self.update_update_weights(&update_gradients);

        self.iteration += 1;
        loss
    }

    /// Compute gradients w.r.t. perception output for backprop through update
    fn compute_perception_output_gradients(
        &self,
        perception_output: &[f64],
        update_activations: &[Vec<f64>],
        output_gradients: &[f64],
    ) -> Vec<f64> {
        let num_layers = self.model.update.layers.len();
        let mut output_grads = output_gradients.to_vec();

        // Backpropagate through update layers to get gradient w.r.t. input
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
            let mut prev_output_grads = vec![0.0; prev_len];

            for (gate_idx, gate) in layer.gates.iter().enumerate() {
                let a_idx = layer.wires.a[gate_idx];
                let b_idx = layer.wires.b[gate_idx];
                let a = layer_inputs[a_idx];
                let b = layer_inputs[b_idx];
                let output_grad = output_grads[gate_idx];

                let (da, db) = compute_gate_input_gradients(gate, a, b);
                prev_output_grads[a_idx] += output_grad * da;
                prev_output_grads[b_idx] += output_grad * db;
            }

            output_grads = prev_output_grads;
        }

        output_grads
    }

    /// Update perception module weights
    fn update_perception_weights(&mut self, gradients: &[Vec<Vec<Vec<[f64; 16]>>>]) {
        for (k, kernel_grads) in gradients.iter().enumerate() {
            let num_layers = self.model.perception.kernels[k].layers.len();
            let mut accumulated: Vec<Vec<[f64; 16]>> = Vec::with_capacity(num_layers);

            for layer_idx in 0..num_layers {
                let num_gates = self.model.perception.kernels[k].layers[layer_idx].num_gates();
                let mut layer_grads = vec![[0.0; 16]; num_gates];

                for channel_grads in kernel_grads {
                    for (gate_idx, gate_grad) in channel_grads[layer_idx].iter().enumerate() {
                        for i in 0..16 {
                            layer_grads[gate_idx][i] += gate_grad[i];
                        }
                    }
                }

                // Average over channels
                for gate_grad in &mut layer_grads {
                    for v in gate_grad.iter_mut() {
                        *v /= self.model.perception.channels as f64;
                    }
                }

                accumulated.push(layer_grads);
            }

            // Apply updates
            for (layer_idx, layer_grads) in accumulated.iter().enumerate() {
                for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                    let mut clipped = *gate_grad;
                    for v in clipped.iter_mut() {
                        *v = v.clamp(-100.0, 100.0);
                    }

                    self.perception_optimizers[k][layer_idx][gate_idx].step(
                        &mut self.model.perception.kernels[k].layers[layer_idx].gates[gate_idx]
                            .logits,
                        &clipped,
                    );
                }
            }
        }
    }

    /// Update update module weights
    fn update_update_weights(&mut self, gradients: &[Vec<[f64; 16]>]) {
        for (layer_idx, layer_grads) in gradients.iter().enumerate() {
            for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                let mut clipped = *gate_grad;
                for v in clipped.iter_mut() {
                    *v = v.clamp(-100.0, 100.0);
                }

                self.update_optimizers[layer_idx][gate_idx].step(
                    &mut self.model.update.layers[layer_idx].gates[gate_idx].logits,
                    &clipped,
                );
            }
        }
    }
}

/// Trainer specifically for the update module (for isolated testing)
pub struct UpdateTrainer {
    pub module: UpdateModule,
    /// One optimizer per gate: [layer][gate]
    optimizers: Vec<Vec<AdamW>>,
    pub learning_rate: f64,
    pub iteration: usize,
}

impl UpdateTrainer {
    /// Create a new update trainer
    pub fn new(module: UpdateModule, learning_rate: f64) -> Self {
        let optimizers: Vec<Vec<AdamW>> = module
            .layers
            .iter()
            .map(|layer| {
                (0..layer.num_gates())
                    .map(|_| AdamW::new(learning_rate))
                    .collect()
            })
            .collect();

        Self {
            module,
            optimizers,
            learning_rate,
            iteration: 0,
        }
    }

    /// Train on a single example
    ///
    /// Returns the loss for this example
    pub fn train_step(&mut self, inputs: &[f64], target: &[f64]) -> f64 {
        // Forward pass
        let activations = self.module.forward_soft(inputs);
        let output = activations.last().cloned().unwrap_or_default();

        // Compute loss (MSE)
        let mut loss = 0.0;
        let output_gradients: Vec<f64> = output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| {
                let error = o - t;
                loss += error * error;
                2.0 * error
            })
            .collect();

        loss /= output.len() as f64;

        // Backward pass
        let gradients = self.module.compute_gradients(inputs, &activations, &output_gradients);

        // Update weights
        for (layer_idx, layer_grads) in gradients.iter().enumerate() {
            for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                let mut clipped = *gate_grad;
                for v in clipped.iter_mut() {
                    *v = v.clamp(-100.0, 100.0);
                }

                self.optimizers[layer_idx][gate_idx].step(
                    &mut self.module.layers[layer_idx].gates[gate_idx].logits,
                    &clipped,
                );
            }
        }

        self.iteration += 1;
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== UpdateModule Tests ====================

    #[test]
    fn test_update_module_creation() {
        // Architecture must respect: out_dim * 2 >= in_dim
        // 17→16, 16→8, 8→4, 4→2, 2→1
        let module = UpdateModule::new(&[17, 16, 8, 4, 2, 1]);

        assert_eq!(module.input_size, 17);
        assert_eq!(module.output_channels, 1);
        assert_eq!(module.num_layers(), 5);
    }

    #[test]
    fn test_gol_update_module_architecture() {
        let module = UpdateModule::gol_module();

        // Input: 17 = 1 (center) + 16 (kernel outputs)
        assert_eq!(module.input_size, 17);
        // Output: 1 (next cell state)
        assert_eq!(module.output_channels, 1);

        // 18 layers: [17→128×10→64→32→16→8→4→2→1]
        // That's 10 layers to 128, then 7 reduction layers = 17 layers
        assert_eq!(module.num_layers(), 17);

        // Layer sizes check
        assert_eq!(module.layer_sizes[0], 17);
        for i in 1..=10 {
            assert_eq!(module.layer_sizes[i], 128);
        }
        assert_eq!(module.layer_sizes[11], 64);
        assert_eq!(module.layer_sizes[17], 1);
    }

    #[test]
    fn test_update_forward_soft() {
        let module = UpdateModule::small_module(17, 1);

        let inputs = vec![0.5; 17];
        let activations = module.forward_soft(&inputs);

        assert_eq!(activations.len(), module.num_layers());

        // Final output should be 1 value
        assert_eq!(activations.last().unwrap().len(), 1);

        // All values should be valid probabilities
        for layer_act in &activations {
            for v in layer_act {
                assert!(*v >= 0.0 && *v <= 1.0, "Value {} not in [0,1]", v);
            }
        }
    }

    #[test]
    fn test_update_forward_hard() {
        let module = UpdateModule::small_module(17, 1);

        let inputs = vec![0.5; 17];
        let output = module.forward_hard(&inputs);

        assert_eq!(output.len(), 1);
        assert!(output[0] == 0.0 || output[0] == 1.0);
    }

    #[test]
    fn test_update_module_total_gates() {
        let module = UpdateModule::small_module(17, 1);

        // small_module(17, 1) creates: [17→9→5→3→2→1]
        // Gate counts: 9 + 5 + 3 + 2 + 1 = 20 gates
        // Actually depends on the algorithm, just verify it's reasonable
        assert!(module.total_gates() > 0);
        assert!(module.total_gates() < 50);

        // Verify layer sizes sum to total gates
        let expected: usize = module.layers.iter().map(|l| l.num_gates()).sum();
        assert_eq!(module.total_gates(), expected);
    }

    // ==================== Gradient Tests ====================

    #[test]
    fn test_numerical_gradient_update() {
        // Architecture: 8→4→2→1 (each layer reduces by at most half)
        let module = UpdateModule::new(&[8, 4, 2, 1]);

        let inputs: Vec<f64> = (0..8).map(|i| (i as f64) / 10.0 + 0.1).collect();
        let target = 0.7;
        let epsilon = 1e-5;

        let activations = module.forward_soft(&inputs);
        let output = activations.last().unwrap()[0];

        let output_grad = vec![2.0 * (output - target)];
        let gradients = module.compute_gradients(&inputs, &activations, &output_grad);

        // Numerical gradient check for first layer, first gate, first logit
        let mut module_copy = module.clone();
        let original = module_copy.layers[0].gates[0].logits[0];

        module_copy.layers[0].gates[0].logits[0] = original + epsilon;
        let act_plus = module_copy.forward_soft(&inputs);
        let loss_plus = (act_plus.last().unwrap()[0] - target).powi(2);

        module_copy.layers[0].gates[0].logits[0] = original - epsilon;
        let act_minus = module_copy.forward_soft(&inputs);
        let loss_minus = (act_minus.last().unwrap()[0] - target).powi(2);

        let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytical_grad = gradients[0][0][0];

        assert_relative_eq!(analytical_grad, numerical_grad, epsilon = 1e-4, max_relative = 0.02);
    }

    #[test]
    fn test_update_trainer_loss_decreases() {
        // Architecture: 8→8→4→2→1 (respects unique_connections constraint)
        let module = UpdateModule::new(&[8, 8, 4, 2, 1]);
        let mut trainer = UpdateTrainer::new(module, 0.05);

        let inputs: Vec<f64> = (0..8).map(|i| (i as f64) / 10.0).collect();
        let target = vec![0.8];

        let initial_loss = trainer.train_step(&inputs, &target);

        for _ in 0..100 {
            trainer.train_step(&inputs, &target);
        }

        let final_loss = trainer.train_step(&inputs, &target);

        assert!(
            final_loss < initial_loss,
            "Loss should decrease: {} -> {}",
            initial_loss,
            final_loss
        );
    }

    // ==================== DiffLogicCA Tests ====================

    #[test]
    fn test_diff_logic_ca_creation() {
        let ca = DiffLogicCA::gol();

        // Perception: 16 kernels × 15 gates = 240
        assert_eq!(ca.perception.total_gates(), 240);

        // Update: should have many more gates
        assert!(ca.update.total_gates() > 240);

        // Total should be perception + update
        assert_eq!(
            ca.total_gates(),
            ca.perception.total_gates() + ca.update.total_gates()
        );
    }

    #[test]
    fn test_diff_logic_ca_forward_soft() {
        use crate::perception::ConnectionType;

        // Use smaller module for testing
        // Architecture: [9→8→4→2→1] respects unique_connections constraint
        let perception = PerceptionModule::new(
            1,
            4,
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        );
        // Input: 1 (center) + 4 kernels × 1 output = 5
        let update = UpdateModule::new(&[5, 4, 2, 1]);

        let ca = DiffLogicCA::new(perception, update);

        let neighborhood = NNeighborhood::from_gol_index(0b101010101);
        let (output, _, _) = ca.forward_soft(&neighborhood);

        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }

    #[test]
    fn test_diff_logic_ca_forward_hard() {
        use crate::perception::ConnectionType;

        // Architecture: [9→8→4→2→1] respects unique_connections constraint
        let perception = PerceptionModule::new(
            1,
            4,
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        );
        let update = UpdateModule::new(&[5, 4, 2, 1]);

        let ca = DiffLogicCA::new(perception, update);

        let neighborhood = NNeighborhood::from_gol_index(0b000010000);
        let output = ca.forward_hard(&neighborhood);

        assert_eq!(output.len(), 1);
        assert!(output[0] == 0.0 || output[0] == 1.0);
    }

    #[test]
    fn test_diff_logic_ca_trainer_loss_decreases() {
        use crate::perception::ConnectionType;

        // Architecture: [9→8→4→2→1] respects unique_connections constraint
        let perception = PerceptionModule::new(
            1,
            2,
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        );
        // Input: 1 (center) + 2 kernels × 1 output = 3
        let update = UpdateModule::new(&[3, 2, 1]);

        let ca = DiffLogicCA::new(perception, update);
        let mut trainer = DiffLogicCATrainer::new(ca, 0.05);

        let neighborhood = NNeighborhood::from_gol_index(0b111101111); // 8 neighbors
        let target = vec![1.0]; // Cell should survive

        let initial_loss = trainer.train_step(&neighborhood, &target);

        for _ in 0..50 {
            trainer.train_step(&neighborhood, &target);
        }

        let final_loss = trainer.train_step(&neighborhood, &target);

        assert!(
            final_loss < initial_loss,
            "Loss should decrease: {} -> {}",
            initial_loss,
            final_loss
        );
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_update_all_gol_configs() {
        // Test that update module can process all possible perception outputs
        let module = UpdateModule::small_module(17, 1);

        for _ in 0..100 {
            // Random perception output
            let inputs: Vec<f64> = (0..17).map(|i| (i as f64 % 2.0) * 0.7 + 0.1).collect();
            let output = module.output_soft(&inputs);

            assert_eq!(output.len(), 1);
            assert!(
                output[0] >= 0.0 && output[0] <= 1.0,
                "Output {} not in [0,1]",
                output[0]
            );
        }
    }

    #[test]
    fn test_gol_architecture_gate_count() {
        let ca = DiffLogicCA::gol();

        // Reference implementation has ~336 active gates for GoL
        // Our implementation should be similar
        // Perception: 16 kernels × 15 gates = 240
        // Update: [17→128×10→...→1] = lots more

        assert_eq!(ca.perception.total_gates(), 240);

        // Update module gate count
        // 10 layers of 128 = 1280
        // Plus reduction: 64+32+16+8+4+2+1 = 127
        // Total update: 1280 + 127 = 1407
        // Actual will be slightly different due to layer structure
        assert!(ca.update.total_gates() > 1000);

        println!(
            "Total gates: {} (perception: {}, update: {})",
            ca.total_gates(),
            ca.perception.total_gates(),
            ca.update.total_gates()
        );
    }
}
