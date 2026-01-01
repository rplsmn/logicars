//! Phase 0.2: Gate Layer
//!
//! Multiple independent gates learning different operations simultaneously.
//! Verify no gradient interference between gates.

use crate::optimizer::AdamW;
use crate::phase_0_1::{BinaryOp, ProbabilisticGate};

/// A layer of multiple independent probabilistic gates
pub struct GateLayer {
    gates: Vec<ProbabilisticGate>,
}

impl GateLayer {
    /// Create a new gate layer with the specified number of gates
    pub fn new(num_gates: usize) -> Self {
        let gates = (0..num_gates)
            .map(|_| ProbabilisticGate::new())
            .collect();

        Self { gates }
    }

    /// Get the number of gates in the layer
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get a reference to a specific gate
    pub fn gate(&self, index: usize) -> &ProbabilisticGate {
        &self.gates[index]
    }

    /// Get a mutable reference to a specific gate (for testing)
    pub fn gate_mut(&mut self, index: usize) -> &mut ProbabilisticGate {
        &mut self.gates[index]
    }

    /// Execute all gates in soft mode (for training)
    /// Each gate gets its own input pair from the inputs slice
    pub fn execute_soft(&self, inputs: &[(f64, f64)]) -> Vec<f64> {
        assert_eq!(
            inputs.len(),
            self.gates.len(),
            "Number of inputs must match number of gates"
        );

        self.gates
            .iter()
            .zip(inputs.iter())
            .map(|(gate, &(a, b))| gate.execute_soft(a, b))
            .collect()
    }

    /// Execute all gates in hard mode (for inference)
    /// Each gate gets its own input pair from the inputs slice
    pub fn execute_hard(&self, inputs: &[(bool, bool)]) -> Vec<bool> {
        assert_eq!(
            inputs.len(),
            self.gates.len(),
            "Number of inputs must match number of gates"
        );

        self.gates
            .iter()
            .zip(inputs.iter())
            .map(|(gate, &(a, b))| gate.execute_hard(a, b))
            .collect()
    }

    /// Compute gradients for all gates
    /// Returns a vector of gradient arrays, one per gate
    pub fn compute_gradients(
        &self,
        inputs: &[(f64, f64)],
        targets: &[f64],
        outputs: &[f64],
    ) -> Vec<[f64; 16]> {
        assert_eq!(
            inputs.len(),
            self.gates.len(),
            "Number of inputs must match number of gates"
        );
        assert_eq!(
            targets.len(),
            self.gates.len(),
            "Number of targets must match number of gates"
        );
        assert_eq!(
            outputs.len(),
            self.gates.len(),
            "Number of outputs must match number of gates"
        );

        self.gates
            .iter()
            .zip(inputs.iter())
            .zip(targets.iter())
            .zip(outputs.iter())
            .map(|(((gate, &(a, b)), &target), &output)| {
                gate.compute_gradients(a, b, target, output)
            })
            .collect()
    }

    /// Get the dominant operation for each gate
    pub fn dominant_operations(&self) -> Vec<(BinaryOp, f64)> {
        self.gates
            .iter()
            .map(|gate| gate.dominant_operation())
            .collect()
    }
}

/// Training dataset for a gate layer: multiple truth tables
pub struct LayerTruthTable {
    /// One truth table per gate
    pub truth_tables: Vec<(Vec<(f64, f64)>, Vec<f64>)>,
}

impl LayerTruthTable {
    /// Create a layer truth table for multiple operations
    pub fn for_operations(ops: &[BinaryOp]) -> Self {
        let truth_tables = ops
            .iter()
            .map(|&op| {
                let inputs = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
                let targets: Vec<f64> = inputs
                    .iter()
                    .map(|&(a, b)| op.execute(a != 0.0, b != 0.0) as u8 as f64)
                    .collect();
                (inputs, targets)
            })
            .collect();

        Self { truth_tables }
    }

    /// Number of gates (operations) in this truth table
    pub fn num_gates(&self) -> usize {
        self.truth_tables.len()
    }

    /// Get all inputs for a specific example index (e.g., 0 for (0,0), 1 for (0,1), etc.)
    /// Returns one input per gate
    pub fn get_inputs(&self, example_idx: usize) -> Vec<(f64, f64)> {
        self.truth_tables
            .iter()
            .map(|(inputs, _)| inputs[example_idx])
            .collect()
    }

    /// Get all targets for a specific example index
    /// Returns one target per gate
    pub fn get_targets(&self, example_idx: usize) -> Vec<f64> {
        self.truth_tables
            .iter()
            .map(|(_, targets)| targets[example_idx])
            .collect()
    }

    /// Compute mean squared error loss for the entire layer
    pub fn compute_loss(&self, layer: &GateLayer) -> f64 {
        assert_eq!(layer.num_gates(), self.num_gates());

        let mut total_loss = 0.0;
        let num_examples = 4; // Truth tables always have 4 examples

        for example_idx in 0..num_examples {
            let inputs = self.get_inputs(example_idx);
            let targets = self.get_targets(example_idx);
            let outputs = layer.execute_soft(&inputs);

            for (output, target) in outputs.iter().zip(targets.iter()) {
                let error = output - target;
                total_loss += error * error;
            }
        }

        total_loss / (self.num_gates() * num_examples) as f64
    }

    /// Compute hard accuracy for each gate
    pub fn compute_hard_accuracies(&self, layer: &GateLayer) -> Vec<f64> {
        assert_eq!(layer.num_gates(), self.num_gates());

        (0..self.num_gates())
            .map(|gate_idx| {
                let mut correct = 0;
                for example_idx in 0..4 {
                    let (a, b) = self.truth_tables[gate_idx].0[example_idx];
                    let output = layer.gate(gate_idx).execute_hard(a != 0.0, b != 0.0);
                    let target = self.truth_tables[gate_idx].1[example_idx] != 0.0;
                    if output == target {
                        correct += 1;
                    }
                }
                correct as f64 / 4.0
            })
            .collect()
    }
}

/// Trainer for a gate layer
pub struct LayerTrainer {
    pub layer: GateLayer,
    /// One optimizer per gate (for independent parameter updates)
    optimizers: Vec<AdamW>,
    pub iteration: usize,
}

impl LayerTrainer {
    /// Create a new layer trainer
    pub fn new(num_gates: usize, learning_rate: f64) -> Self {
        let layer = GateLayer::new(num_gates);
        let optimizers = (0..num_gates).map(|_| AdamW::new(learning_rate)).collect();

        Self {
            layer,
            optimizers,
            iteration: 0,
        }
    }

    /// Train for one epoch on the layer truth table
    pub fn train_epoch(&mut self, truth_table: &LayerTruthTable) -> f64 {
        assert_eq!(self.layer.num_gates(), truth_table.num_gates());

        // Accumulate gradients for each gate
        let mut total_gradients: Vec<[f64; 16]> =
            vec![[0.0; 16]; self.layer.num_gates()];

        // Process all 4 examples
        for example_idx in 0..4 {
            let inputs = truth_table.get_inputs(example_idx);
            let targets = truth_table.get_targets(example_idx);
            let outputs = self.layer.execute_soft(&inputs);

            let grads = self.layer.compute_gradients(&inputs, &targets, &outputs);

            // Accumulate gradients for each gate
            for (gate_idx, gate_grads) in grads.iter().enumerate() {
                for (i, &grad) in gate_grads.iter().enumerate() {
                    total_gradients[gate_idx][i] += grad;
                }
            }
        }

        // Average gradients and update each gate independently
        for gate_idx in 0..self.layer.num_gates() {
            // Average over 4 examples
            for i in 0..16 {
                total_gradients[gate_idx][i] /= 4.0;
            }

            // Update this gate's parameters using its optimizer
            self.optimizers[gate_idx].step(
                &mut self.layer.gates[gate_idx].logits,
                &total_gradients[gate_idx],
            );
        }

        self.iteration += 1;

        // Return current loss
        truth_table.compute_loss(&self.layer)
    }

    /// Train until convergence or max iterations
    pub fn train(
        &mut self,
        truth_table: &LayerTruthTable,
        max_iterations: usize,
        target_loss: f64,
        verbose: bool,
    ) -> LayerTrainingResult {
        let mut losses = Vec::new();
        let mut converged = false;

        for i in 0..max_iterations {
            let loss = self.train_epoch(truth_table);
            losses.push(loss);

            if verbose && (i % 100 == 0 || i == max_iterations - 1) {
                let accuracies = truth_table.compute_hard_accuracies(&self.layer);
                let dominant_ops = self.layer.dominant_operations();

                println!("Iter {:5}: Loss = {:.6}", i, loss);
                for (gate_idx, ((op, prob), acc)) in
                    dominant_ops.iter().zip(accuracies.iter()).enumerate()
                {
                    println!(
                        "  Gate {}: {:?} ({:.1}%), Acc = {:.1}%",
                        gate_idx,
                        op,
                        prob * 100.0,
                        acc * 100.0
                    );
                }
            }

            if loss < target_loss {
                converged = true;
                if verbose {
                    println!("Converged after {} iterations!", i);
                }
                break;
            }
        }

        let final_loss = losses.last().copied().unwrap_or(f64::INFINITY);
        let hard_accuracies = truth_table.compute_hard_accuracies(&self.layer);
        let dominant_ops = self.layer.dominant_operations();

        LayerTrainingResult {
            converged,
            iterations: self.iteration,
            final_loss,
            hard_accuracies,
            dominant_ops,
            losses,
        }
    }
}

/// Result of training a gate layer
#[derive(Debug)]
pub struct LayerTrainingResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_loss: f64,
    pub hard_accuracies: Vec<f64>,
    pub dominant_ops: Vec<(BinaryOp, f64)>,
    pub losses: Vec<f64>,
}

impl LayerTrainingResult {
    /// Check if this meets Phase 0.2 exit criteria
    pub fn meets_exit_criteria(&self, target_ops: &[BinaryOp]) -> bool {
        if !self.converged {
            return false;
        }

        // All gates must have >99% accuracy
        if !self.hard_accuracies.iter().all(|&acc| acc > 0.99) {
            return false;
        }

        // Each gate must have learned its target operation
        for (i, &target_op) in target_ops.iter().enumerate() {
            let (learned_op, _) = self.dominant_ops[i];
            if learned_op != target_op {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_layer_creation() {
        let layer = GateLayer::new(3);
        assert_eq!(layer.num_gates(), 3);
    }

    #[test]
    fn test_layer_initialization() {
        let layer = GateLayer::new(3);

        // All gates should start with pass-through as dominant
        let dominant_ops = layer.dominant_operations();
        for (op, prob) in dominant_ops {
            assert_eq!(op, BinaryOp::A);
            assert!(prob > 0.9);
        }
    }

    #[test]
    fn test_layer_execute_soft() {
        let layer = GateLayer::new(3);
        let inputs = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)];
        let outputs = layer.execute_soft(&inputs);

        assert_eq!(outputs.len(), 3);
        // All gates are pass-through A initially
        assert_relative_eq!(outputs[0], 0.0, epsilon = 0.1);
        assert_relative_eq!(outputs[1], 1.0, epsilon = 0.1);
        assert_relative_eq!(outputs[2], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_layer_execute_hard() {
        let layer = GateLayer::new(3);
        let inputs = vec![(false, false), (true, false), (true, true)];
        let outputs = layer.execute_hard(&inputs);

        assert_eq!(outputs.len(), 3);
        // All gates are pass-through A initially
        assert_eq!(outputs, vec![false, true, true]);
    }

    #[test]
    fn test_layer_truth_table() {
        let ops = vec![BinaryOp::And, BinaryOp::Or, BinaryOp::Xor];
        let tt = LayerTruthTable::for_operations(&ops);

        assert_eq!(tt.num_gates(), 3);

        // Check first example (0, 0)
        let inputs = tt.get_inputs(0);
        assert_eq!(inputs, vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);

        let targets = tt.get_targets(0);
        assert_eq!(targets, vec![0.0, 0.0, 0.0]); // AND, OR, XOR all give 0 for (0,0)

        // Check last example (1, 1)
        let targets = tt.get_targets(3);
        assert_eq!(targets, vec![1.0, 1.0, 0.0]); // AND=1, OR=1, XOR=0 for (1,1)
    }

    #[test]
    fn test_gradient_independence() {
        // Test that gradients for different gates don't interfere
        let mut layer = GateLayer::new(2);

        // Manually set different logits for each gate
        layer.gate_mut(0).logits[8] = 5.0; // Bias towards AND (index 8)
        layer.gate_mut(1).logits[14] = 5.0; // Bias towards OR (index 14)

        let inputs = vec![(1.0, 1.0), (0.0, 1.0)];
        let targets = vec![1.0, 1.0]; // Both should output 1
        let outputs = layer.execute_soft(&inputs);

        let grads = layer.compute_gradients(&inputs, &targets, &outputs);

        // Verify we got gradients for both gates
        assert_eq!(grads.len(), 2);

        // Gradients should be different for each gate (they have different states)
        // This verifies they're computed independently
        assert_ne!(grads[0], grads[1]);
    }

    #[test]
    fn test_layer_loss_computation() {
        let layer = GateLayer::new(2);
        let ops = vec![BinaryOp::And, BinaryOp::Or];
        let tt = LayerTruthTable::for_operations(&ops);

        let loss = tt.compute_loss(&layer);

        // Loss should be finite and non-negative
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
