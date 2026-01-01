//! Phase 0.3: Multi-Layer Circuits
//!
//! Stack gate layers to learn functions requiring depth.
//! Implements backpropagation through multiple layers.
//!
//! Example: Learn XOR from AND/OR/NAND primitives.

use crate::optimizer::AdamW;
use crate::phase_0_1::{BinaryOp, ProbabilisticGate};

/// Connection pattern: maps output indices from previous layer to input pairs for next layer
/// For each gate in the next layer, specifies (output_idx_a, output_idx_b)
#[derive(Debug, Clone)]
pub struct ConnectionPattern {
    /// For each gate: (source_idx_a, source_idx_b) from previous layer's outputs
    pub connections: Vec<(usize, usize)>,
}

impl ConnectionPattern {
    /// Create connections from a list of (a_idx, b_idx) pairs
    pub fn new(connections: Vec<(usize, usize)>) -> Self {
        Self { connections }
    }

    /// Create a simple paired connection: gates[0] gets (0,1), gates[1] gets (2,3), etc.
    pub fn paired(num_gates: usize) -> Self {
        let connections = (0..num_gates)
            .map(|i| (i * 2, i * 2 + 1))
            .collect();
        Self { connections }
    }

    /// Create connections where all gates share the same inputs
    pub fn broadcast(num_gates: usize, a_idx: usize, b_idx: usize) -> Self {
        let connections = (0..num_gates)
            .map(|_| (a_idx, b_idx))
            .collect();
        Self { connections }
    }

    pub fn num_gates(&self) -> usize {
        self.connections.len()
    }
}

/// A multi-layer circuit of probabilistic gates
pub struct Circuit {
    /// The layers of gates
    layers: Vec<Vec<ProbabilisticGate>>,
    /// Connection patterns between layers (patterns[i] connects layer i to layer i+1)
    /// First layer uses external inputs, so patterns has len = layers.len() - 1
    connections: Vec<ConnectionPattern>,
    /// Number of external inputs to the circuit
    num_inputs: usize,
}

impl Circuit {
    /// Create a new circuit with specified layer sizes and connection patterns
    ///
    /// # Arguments
    /// * `num_inputs` - Number of external inputs to the circuit
    /// * `layer_sizes` - Number of gates in each layer
    /// * `connection_patterns` - How outputs connect to next layer's inputs
    ///
    /// Note: `connection_patterns` length should be `layer_sizes.len() - 1`
    /// The first layer takes external inputs directly
    pub fn new(
        num_inputs: usize,
        layer_sizes: &[usize],
        connection_patterns: Vec<ConnectionPattern>,
    ) -> Self {
        assert!(
            !layer_sizes.is_empty(),
            "Must have at least one layer"
        );
        assert_eq!(
            connection_patterns.len(),
            layer_sizes.len() - 1,
            "Need {} connection patterns for {} layers",
            layer_sizes.len() - 1,
            layer_sizes.len()
        );

        let layers: Vec<Vec<ProbabilisticGate>> = layer_sizes
            .iter()
            .map(|&size| (0..size).map(|_| ProbabilisticGate::new()).collect())
            .collect();

        Self {
            layers,
            connections: connection_patterns,
            num_inputs,
        }
    }

    /// Create a simple 2-layer circuit for learning composite operations
    ///
    /// Layer 1: 2 gates taking the same 2 inputs (computes two intermediate values)
    /// Layer 2: 1 gate taking the outputs of layer 1
    pub fn two_layer_composite(num_inputs: usize) -> Self {
        // Layer 1: 2 gates, each gets the same (a, b) inputs
        // Layer 2: 1 gate, gets outputs from both layer 1 gates
        let connection = ConnectionPattern::new(vec![(0, 1)]);

        Self::new(num_inputs, &[2, 1], vec![connection])
    }

    /// Create a 3-layer circuit
    pub fn three_layer(num_inputs: usize, hidden_size: usize) -> Self {
        // Layer 1: hidden_size gates, all getting same inputs
        // Layer 2: hidden_size gates, various connections
        // Layer 3: 1 output gate

        // Connection from layer 1 to layer 2: pair up outputs
        let conn1 = ConnectionPattern::paired(hidden_size);
        // Connection from layer 2 to layer 3: take first two outputs
        let conn2 = ConnectionPattern::new(vec![(0, 1)]);

        Self::new(num_inputs, &[hidden_size * 2, hidden_size, 1], vec![conn1, conn2])
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get a reference to a specific layer
    pub fn layer(&self, idx: usize) -> &[ProbabilisticGate] {
        &self.layers[idx]
    }

    /// Get a mutable reference to a gate
    pub fn gate_mut(&mut self, layer_idx: usize, gate_idx: usize) -> &mut ProbabilisticGate {
        &mut self.layers[layer_idx][gate_idx]
    }

    /// Forward pass in soft mode (for training)
    /// Returns all intermediate activations and final output
    ///
    /// # Arguments
    /// * `inputs` - External inputs, one (a, b) pair per gate in first layer
    ///
    /// # Returns
    /// * Vector of layer activations, including final output
    pub fn forward_soft(&self, inputs: &[(f64, f64)]) -> Vec<Vec<f64>> {
        assert_eq!(
            inputs.len(),
            self.layers[0].len(),
            "Number of input pairs must match first layer size"
        );

        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());

        // First layer takes external inputs
        let first_outputs: Vec<f64> = self.layers[0]
            .iter()
            .zip(inputs.iter())
            .map(|(gate, &(a, b))| gate.execute_soft(a, b))
            .collect();
        activations.push(first_outputs);

        // Subsequent layers take outputs from previous layer
        for layer_idx in 1..self.layers.len() {
            let prev_outputs = &activations[layer_idx - 1];
            let pattern = &self.connections[layer_idx - 1];

            let layer_outputs: Vec<f64> = self.layers[layer_idx]
                .iter()
                .zip(pattern.connections.iter())
                .map(|(gate, &(a_idx, b_idx))| {
                    let a = prev_outputs[a_idx];
                    let b = prev_outputs[b_idx];
                    gate.execute_soft(a, b)
                })
                .collect();

            activations.push(layer_outputs);
        }

        activations
    }

    /// Forward pass in hard mode (for inference)
    pub fn forward_hard(&self, inputs: &[(bool, bool)]) -> Vec<Vec<bool>> {
        assert_eq!(
            inputs.len(),
            self.layers[0].len(),
            "Number of input pairs must match first layer size"
        );

        let mut activations: Vec<Vec<bool>> = Vec::with_capacity(self.layers.len());

        // First layer takes external inputs
        let first_outputs: Vec<bool> = self.layers[0]
            .iter()
            .zip(inputs.iter())
            .map(|(gate, &(a, b))| gate.execute_hard(a, b))
            .collect();
        activations.push(first_outputs);

        // Subsequent layers
        for layer_idx in 1..self.layers.len() {
            let prev_outputs = &activations[layer_idx - 1];
            let pattern = &self.connections[layer_idx - 1];

            let layer_outputs: Vec<bool> = self.layers[layer_idx]
                .iter()
                .zip(pattern.connections.iter())
                .map(|(gate, &(a_idx, b_idx))| {
                    let a = prev_outputs[a_idx];
                    let b = prev_outputs[b_idx];
                    gate.execute_hard(a, b)
                })
                .collect();

            activations.push(layer_outputs);
        }

        activations
    }

    /// Get final output from forward pass (soft mode)
    pub fn output_soft(&self, inputs: &[(f64, f64)]) -> Vec<f64> {
        let activations = self.forward_soft(inputs);
        activations.last().unwrap().clone()
    }

    /// Get final output from forward pass (hard mode)
    pub fn output_hard(&self, inputs: &[(bool, bool)]) -> Vec<bool> {
        let activations = self.forward_hard(inputs);
        activations.last().unwrap().clone()
    }

    /// Compute gradients for all gates using backpropagation
    ///
    /// # Arguments
    /// * `inputs` - External inputs to the circuit
    /// * `targets` - Target outputs
    /// * `activations` - Pre-computed activations from forward pass
    ///
    /// # Returns
    /// * Gradients for each layer, each gate, 16 logits
    pub fn compute_gradients(
        &self,
        inputs: &[(f64, f64)],
        targets: &[f64],
        activations: &[Vec<f64>],
    ) -> Vec<Vec<[f64; 16]>> {
        let num_layers = self.layers.len();
        let mut all_gradients: Vec<Vec<[f64; 16]>> = Vec::with_capacity(num_layers);

        // Initialize gradient storage
        for layer_idx in 0..num_layers {
            all_gradients.push(vec![[0.0; 16]; self.layers[layer_idx].len()]);
        }

        // Compute gradient of loss w.r.t. each output (dL/dy for MSE)
        let final_outputs = activations.last().unwrap();
        let mut output_gradients: Vec<f64> = final_outputs
            .iter()
            .zip(targets.iter())
            .map(|(&output, &target)| 2.0 * (output - target))
            .collect();

        // Backpropagate through layers (from last to first)
        for layer_idx in (0..num_layers).rev() {
            // Get inputs to this layer
            let layer_inputs: Vec<(f64, f64)> = if layer_idx == 0 {
                inputs.to_vec()
            } else {
                let prev_outputs = &activations[layer_idx - 1];
                let pattern = &self.connections[layer_idx - 1];
                pattern
                    .connections
                    .iter()
                    .map(|&(a_idx, b_idx)| (prev_outputs[a_idx], prev_outputs[b_idx]))
                    .collect()
            };

            // Compute gradients for this layer's gates
            let layer_outputs = &activations[layer_idx];

            // Also need to accumulate gradients to pass backward
            let mut prev_output_gradients: Vec<f64> = if layer_idx > 0 {
                vec![0.0; activations[layer_idx - 1].len()]
            } else {
                vec![] // No gradients needed for external inputs
            };

            for (gate_idx, gate) in self.layers[layer_idx].iter().enumerate() {
                let (a, b) = layer_inputs[gate_idx];
                let output = layer_outputs[gate_idx];
                let output_grad = output_gradients[gate_idx];

                // Compute gradient w.r.t. logits
                // We need to adapt the gradient computation for chain rule
                // dL/dlogits = dL/doutput * doutput/dlogits
                let probs = gate.probabilities();
                let mut gate_grads = [0.0; 16];

                for i in 0..16 {
                    let op_output_i = BinaryOp::ALL[i].execute_soft(a, b);
                    let mut dlogit_i = 0.0;

                    for j in 0..16 {
                        let op_output_j = BinaryOp::ALL[j].execute_soft(a, b);
                        // doutput/dlogit_i = sum_j (doutput/dprob_j * dprob_j/dlogit_i)
                        // doutput/dprob_j = op_output_j
                        // dprob_j/dlogit_i = probs[j] * (delta_ij - probs[i])
                        let dprob_j_dlogit_i = if i == j {
                            probs[j] * (1.0 - probs[i])
                        } else {
                            -probs[j] * probs[i]
                        };

                        dlogit_i += op_output_j * dprob_j_dlogit_i;
                    }

                    // Chain rule: multiply by upstream gradient
                    gate_grads[i] = output_grad * dlogit_i;
                }

                all_gradients[layer_idx][gate_idx] = gate_grads;

                // Compute gradients w.r.t. inputs for backprop to previous layer
                if layer_idx > 0 {
                    // dL/da and dL/db for this gate's inputs
                    // doutput/da = sum_i (prob_i * d(op_i(a,b))/da)
                    // This requires derivatives of soft operations w.r.t. inputs
                    let (da, db) = self.compute_input_gradients(gate, a, b);

                    // Chain rule with output gradient
                    let grad_a = output_grad * da;
                    let grad_b = output_grad * db;

                    // Route gradients back through connection pattern
                    let pattern = &self.connections[layer_idx - 1];
                    let (a_idx, b_idx) = pattern.connections[gate_idx];
                    prev_output_gradients[a_idx] += grad_a;
                    prev_output_gradients[b_idx] += grad_b;
                }
            }

            // Update output_gradients for next iteration
            if layer_idx > 0 {
                output_gradients = prev_output_gradients;
            }
        }

        all_gradients
    }

    /// Compute gradients of gate output w.r.t. inputs a and b
    fn compute_input_gradients(&self, gate: &ProbabilisticGate, a: f64, b: f64) -> (f64, f64) {
        let probs = gate.probabilities();
        let mut da = 0.0;
        let mut db = 0.0;

        // For each operation, compute d(op(a,b))/da and d(op(a,b))/db
        for (i, &op) in BinaryOp::ALL.iter().enumerate() {
            let (op_da, op_db) = Self::op_input_gradients(op, a, b);
            da += probs[i] * op_da;
            db += probs[i] * op_db;
        }

        (da, db)
    }

    /// Compute gradients of a specific operation w.r.t. its inputs
    fn op_input_gradients(op: BinaryOp, a: f64, b: f64) -> (f64, f64) {
        match op {
            BinaryOp::False => (0.0, 0.0),
            BinaryOp::And => (b, a),                           // ab: da=b, db=a
            BinaryOp::AAndNotB => (1.0 - b, -a),               // a(1-b): da=1-b, db=-a
            BinaryOp::A => (1.0, 0.0),
            BinaryOp::NotAAndB => (-b, 1.0 - a),               // (1-a)b: da=-b, db=1-a
            BinaryOp::B => (0.0, 1.0),
            BinaryOp::Xor => (1.0 - 2.0 * b, 1.0 - 2.0 * a),   // a + b - 2ab: da=1-2b, db=1-2a
            BinaryOp::Or => (1.0 - b, 1.0 - a),                // a + b - ab: da=1-b, db=1-a
            BinaryOp::Nor => (b - 1.0, a - 1.0),               // (1-a)(1-b): da=b-1, db=a-1
            BinaryOp::Xnor => (2.0 * b - 1.0, 2.0 * a - 1.0),  // ab + (1-a)(1-b): da=2b-1, db=2a-1
            BinaryOp::NotB => (0.0, -1.0),
            BinaryOp::AOrNotB => (b, a - 1.0),         // a + (1-b) - a(1-b) = 1 - b + ab: da=b, db=a-1
            BinaryOp::NotA => (-1.0, 0.0),
            BinaryOp::NotAOrB => (b - 1.0, a),         // (1-a) + b - (1-a)b = 1 - a + ab: da=b-1, db=a
            BinaryOp::Nand => (-b, -a),                        // 1 - ab: da=-b, db=-a
            BinaryOp::True => (0.0, 0.0),
        }
    }

    /// Get dominant operations for all gates in all layers
    pub fn dominant_operations(&self) -> Vec<Vec<(BinaryOp, f64)>> {
        self.layers
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|gate| gate.dominant_operation())
                    .collect()
            })
            .collect()
    }
}

/// Trainer for multi-layer circuits
pub struct CircuitTrainer {
    pub circuit: Circuit,
    /// One optimizer per gate per layer
    optimizers: Vec<Vec<AdamW>>,
    pub iteration: usize,
}

impl CircuitTrainer {
    /// Create a new circuit trainer
    pub fn new(circuit: Circuit, learning_rate: f64) -> Self {
        let optimizers: Vec<Vec<AdamW>> = circuit
            .layers
            .iter()
            .map(|layer| (0..layer.len()).map(|_| AdamW::new(learning_rate)).collect())
            .collect();

        Self {
            circuit,
            optimizers,
            iteration: 0,
        }
    }

    /// Create a trainer for a 2-layer composite circuit
    pub fn new_two_layer_composite(learning_rate: f64) -> Self {
        let circuit = Circuit::two_layer_composite(2);
        Self::new(circuit, learning_rate)
    }

    /// Train for one epoch on a truth table
    ///
    /// # Arguments
    /// * `truth_table` - Training examples: (inputs, target_outputs)
    pub fn train_epoch(&mut self, truth_table: &CircuitTruthTable) -> f64 {
        let num_examples = truth_table.examples.len();

        // Accumulate gradients
        let mut all_gradients: Vec<Vec<[f64; 16]>> = self
            .circuit
            .layers
            .iter()
            .map(|layer| vec![[0.0; 16]; layer.len()])
            .collect();

        let mut total_loss = 0.0;

        // Process all examples
        for (inputs, targets) in &truth_table.examples {
            let activations = self.circuit.forward_soft(inputs);
            let outputs = activations.last().unwrap();

            // Compute loss
            for (&output, &target) in outputs.iter().zip(targets.iter()) {
                let error = output - target;
                total_loss += error * error;
            }

            // Compute gradients
            let grads = self.circuit.compute_gradients(inputs, targets, &activations);

            // Accumulate
            for layer_idx in 0..self.circuit.num_layers() {
                for gate_idx in 0..self.circuit.layers[layer_idx].len() {
                    for i in 0..16 {
                        all_gradients[layer_idx][gate_idx][i] += grads[layer_idx][gate_idx][i];
                    }
                }
            }
        }

        // Average gradients and update
        for layer_idx in 0..self.circuit.num_layers() {
            for gate_idx in 0..self.circuit.layers[layer_idx].len() {
                // Average
                for i in 0..16 {
                    all_gradients[layer_idx][gate_idx][i] /= num_examples as f64;
                }

                // Update
                self.optimizers[layer_idx][gate_idx].step(
                    &mut self.circuit.layers[layer_idx][gate_idx].logits,
                    &all_gradients[layer_idx][gate_idx],
                );
            }
        }

        self.iteration += 1;

        total_loss / (num_examples * truth_table.num_outputs) as f64
    }

    /// Train until convergence or max iterations
    pub fn train(
        &mut self,
        truth_table: &CircuitTruthTable,
        max_iterations: usize,
        target_loss: f64,
        verbose: bool,
    ) -> CircuitTrainingResult {
        let mut losses = Vec::new();
        let mut converged = false;

        for i in 0..max_iterations {
            let loss = self.train_epoch(truth_table);
            losses.push(loss);

            if verbose && (i % 100 == 0 || i == max_iterations - 1) {
                let accuracy = truth_table.compute_hard_accuracy(&self.circuit);
                let dominant_ops = self.circuit.dominant_operations();

                println!("Iter {:5}: Loss = {:.6}, Accuracy = {:.1}%", i, loss, accuracy * 100.0);
                for (layer_idx, layer_ops) in dominant_ops.iter().enumerate() {
                    print!("  Layer {}: ", layer_idx);
                    for (gate_idx, (op, prob)) in layer_ops.iter().enumerate() {
                        print!("G{}={:?}({:.1}%) ", gate_idx, op, prob * 100.0);
                    }
                    println!();
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
        let hard_accuracy = truth_table.compute_hard_accuracy(&self.circuit);
        let dominant_ops = self.circuit.dominant_operations();

        CircuitTrainingResult {
            converged,
            iterations: self.iteration,
            final_loss,
            hard_accuracy,
            dominant_ops,
            losses,
        }
    }
}

/// Training data for a circuit
pub struct CircuitTruthTable {
    /// (inputs, targets) pairs
    pub examples: Vec<(Vec<(f64, f64)>, Vec<f64>)>,
    pub num_outputs: usize,
}

impl CircuitTruthTable {
    /// Create truth table for learning a single binary operation
    /// with a circuit that takes 2 inputs and produces 1 output
    pub fn for_single_output(target_op: impl Fn(bool, bool) -> bool) -> Self {
        let examples = vec![
            (vec![(0.0, 0.0)], vec![target_op(false, false) as u8 as f64]),
            (vec![(0.0, 1.0)], vec![target_op(false, true) as u8 as f64]),
            (vec![(1.0, 0.0)], vec![target_op(true, false) as u8 as f64]),
            (vec![(1.0, 1.0)], vec![target_op(true, true) as u8 as f64]),
        ];

        Self {
            examples,
            num_outputs: 1,
        }
    }

    /// Create truth table for XOR (classic test for multi-layer circuits)
    pub fn for_xor() -> Self {
        Self::for_single_output(|a, b| a ^ b)
    }

    /// Create truth table for a 2-layer composite circuit
    /// First layer has 2 gates, both receiving the same (a, b) input
    /// Second layer has 1 gate taking both outputs
    pub fn for_two_layer_composite(target_op: impl Fn(bool, bool) -> bool) -> Self {
        let examples = vec![
            (
                vec![(0.0, 0.0), (0.0, 0.0)],  // Both layer-1 gates get (0,0)
                vec![target_op(false, false) as u8 as f64]
            ),
            (
                vec![(0.0, 1.0), (0.0, 1.0)],  // Both layer-1 gates get (0,1)
                vec![target_op(false, true) as u8 as f64]
            ),
            (
                vec![(1.0, 0.0), (1.0, 0.0)],  // Both layer-1 gates get (1,0)
                vec![target_op(true, false) as u8 as f64]
            ),
            (
                vec![(1.0, 1.0), (1.0, 1.0)],  // Both layer-1 gates get (1,1)
                vec![target_op(true, true) as u8 as f64]
            ),
        ];

        Self {
            examples,
            num_outputs: 1,
        }
    }

    /// Compute hard accuracy on the circuit
    pub fn compute_hard_accuracy(&self, circuit: &Circuit) -> f64 {
        let mut correct = 0;
        let mut total = 0;

        for (inputs, targets) in &self.examples {
            // Convert soft inputs to hard inputs
            let hard_inputs: Vec<(bool, bool)> = inputs
                .iter()
                .map(|&(a, b)| (a > 0.5, b > 0.5))
                .collect();

            let outputs = circuit.output_hard(&hard_inputs);

            for (&output, &target) in outputs.iter().zip(targets.iter()) {
                if output == (target > 0.5) {
                    correct += 1;
                }
                total += 1;
            }
        }

        correct as f64 / total as f64
    }

    /// Compute soft loss
    pub fn compute_loss(&self, circuit: &Circuit) -> f64 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (inputs, targets) in &self.examples {
            let outputs = circuit.output_soft(inputs);

            for (&output, &target) in outputs.iter().zip(targets.iter()) {
                let error = output - target;
                total_loss += error * error;
                count += 1;
            }
        }

        total_loss / count as f64
    }
}

/// Result of training a circuit
#[derive(Debug)]
pub struct CircuitTrainingResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_loss: f64,
    pub hard_accuracy: f64,
    pub dominant_ops: Vec<Vec<(BinaryOp, f64)>>,
    pub losses: Vec<f64>,
}

impl CircuitTrainingResult {
    /// Check if exit criteria are met
    pub fn meets_exit_criteria(&self) -> bool {
        self.converged && self.hard_accuracy > 0.99
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit::two_layer_composite(2);
        assert_eq!(circuit.num_layers(), 2);
        assert_eq!(circuit.layer(0).len(), 2);
        assert_eq!(circuit.layer(1).len(), 1);
    }

    #[test]
    fn test_circuit_forward_soft() {
        let circuit = Circuit::two_layer_composite(2);

        // With pass-through initialization, first layer passes through A values
        let inputs = vec![(0.0, 0.0), (1.0, 0.0)];
        let activations = circuit.forward_soft(&inputs);

        assert_eq!(activations.len(), 2);
        assert_eq!(activations[0].len(), 2);
        assert_eq!(activations[1].len(), 1);

        // Layer 0 should pass through A (approximately)
        assert_relative_eq!(activations[0][0], 0.0, epsilon = 0.1);
        assert_relative_eq!(activations[0][1], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_circuit_forward_hard() {
        let circuit = Circuit::two_layer_composite(2);

        let inputs = vec![(false, false), (true, false)];
        let activations = circuit.forward_hard(&inputs);

        assert_eq!(activations.len(), 2);
        assert_eq!(activations[0], vec![false, true]);
    }

    #[test]
    fn test_xor_truth_table() {
        let tt = CircuitTruthTable::for_xor();
        assert_eq!(tt.examples.len(), 4);

        // Check XOR targets: 0^0=0, 0^1=1, 1^0=1, 1^1=0
        assert_eq!(tt.examples[0].1, vec![0.0]);
        assert_eq!(tt.examples[1].1, vec![1.0]);
        assert_eq!(tt.examples[2].1, vec![1.0]);
        assert_eq!(tt.examples[3].1, vec![0.0]);
    }

    #[test]
    fn test_connection_pattern_paired() {
        let pattern = ConnectionPattern::paired(2);
        assert_eq!(pattern.connections, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn test_connection_pattern_broadcast() {
        let pattern = ConnectionPattern::broadcast(3, 0, 1);
        assert_eq!(pattern.connections, vec![(0, 1), (0, 1), (0, 1)]);
    }

    #[test]
    fn test_numerical_gradients_through_circuit() {
        // Test that analytical gradients match numerical gradients through the circuit
        let mut circuit = Circuit::two_layer_composite(2);

        // Use some non-trivial logits
        circuit.gate_mut(0, 0).logits[8] = 2.0;  // Bias towards AND
        circuit.gate_mut(0, 1).logits[14] = 2.0; // Bias towards OR
        circuit.gate_mut(1, 0).logits[6] = 2.0;  // Bias towards XOR

        let epsilon = 1e-5;
        let inputs = vec![(0.7, 0.3), (0.7, 0.3)];
        let targets = vec![0.5];

        // Compute analytical gradients
        let activations = circuit.forward_soft(&inputs);
        let analytical_grads = circuit.compute_gradients(&inputs, &targets, &activations);

        // Test gradients for first layer, first gate
        for logit_idx in 0..16 {
            let original = circuit.gate_mut(0, 0).logits[logit_idx];

            // +epsilon
            circuit.gate_mut(0, 0).logits[logit_idx] = original + epsilon;
            let outputs_plus = circuit.output_soft(&inputs);
            let loss_plus: f64 = outputs_plus.iter()
                .zip(targets.iter())
                .map(|(&o, &t)| (o - t).powi(2))
                .sum();

            // -epsilon
            circuit.gate_mut(0, 0).logits[logit_idx] = original - epsilon;
            let outputs_minus = circuit.output_soft(&inputs);
            let loss_minus: f64 = outputs_minus.iter()
                .zip(targets.iter())
                .map(|(&o, &t)| (o - t).powi(2))
                .sum();

            // Restore
            circuit.gate_mut(0, 0).logits[logit_idx] = original;

            let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

            assert_relative_eq!(
                analytical_grads[0][0][logit_idx],
                numerical_grad,
                epsilon = 1e-4,
                max_relative = 1e-2
            );
        }

        // Test gradients for second layer, first gate
        for logit_idx in 0..16 {
            let original = circuit.gate_mut(1, 0).logits[logit_idx];

            // +epsilon
            circuit.gate_mut(1, 0).logits[logit_idx] = original + epsilon;
            let outputs_plus = circuit.output_soft(&inputs);
            let loss_plus: f64 = outputs_plus.iter()
                .zip(targets.iter())
                .map(|(&o, &t)| (o - t).powi(2))
                .sum();

            // -epsilon
            circuit.gate_mut(1, 0).logits[logit_idx] = original - epsilon;
            let outputs_minus = circuit.output_soft(&inputs);
            let loss_minus: f64 = outputs_minus.iter()
                .zip(targets.iter())
                .map(|(&o, &t)| (o - t).powi(2))
                .sum();

            // Restore
            circuit.gate_mut(1, 0).logits[logit_idx] = original;

            let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

            assert_relative_eq!(
                analytical_grads[1][0][logit_idx],
                numerical_grad,
                epsilon = 1e-4,
                max_relative = 1e-2
            );
        }
    }

    #[test]
    fn test_op_input_gradients() {
        // Verify gradient formulas numerically
        let epsilon = 1e-6;
        let a = 0.4;
        let b = 0.7;

        for op in BinaryOp::ALL {
            let (da, db) = Circuit::op_input_gradients(op, a, b);

            // Numerical gradient for a
            let f_plus_a = op.execute_soft(a + epsilon, b);
            let f_minus_a = op.execute_soft(a - epsilon, b);
            let numerical_da = (f_plus_a - f_minus_a) / (2.0 * epsilon);

            // Numerical gradient for b
            let f_plus_b = op.execute_soft(a, b + epsilon);
            let f_minus_b = op.execute_soft(a, b - epsilon);
            let numerical_db = (f_plus_b - f_minus_b) / (2.0 * epsilon);

            assert_relative_eq!(da, numerical_da, epsilon = 1e-4, max_relative = 1e-3);
            assert_relative_eq!(db, numerical_db, epsilon = 1e-4, max_relative = 1e-3);
        }
    }
}
