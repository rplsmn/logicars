//! Training infrastructure for Phase 0.1

use crate::optimizer::AdamW;
use crate::phase_0_1::{BinaryOp, ProbabilisticGate};

/// Training dataset: truth table for a binary operation
pub struct TruthTable {
    pub inputs: Vec<(f64, f64)>,
    pub targets: Vec<f64>,
}

impl TruthTable {
    /// Create a truth table for a given operation
    pub fn for_operation(op: BinaryOp) -> Self {
        let inputs = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];

        let targets: Vec<f64> = inputs
            .iter()
            .map(|&(a, b)| op.execute(a != 0.0, b != 0.0) as u8 as f64)
            .collect();

        Self { inputs, targets }
    }

    /// Mean squared error loss
    pub fn compute_loss(&self, gate: &ProbabilisticGate) -> f64 {
        let mut total_loss = 0.0;

        for (i, &(a, b)) in self.inputs.iter().enumerate() {
            let output = gate.execute_soft(a, b);
            let target = self.targets[i];
            let error = output - target;
            total_loss += error * error;
        }

        total_loss / self.inputs.len() as f64
    }

    /// Compute hard accuracy (using argmax gate)
    pub fn compute_hard_accuracy(&self, gate: &ProbabilisticGate) -> f64 {
        let mut correct = 0;

        for (i, &(a, b)) in self.inputs.iter().enumerate() {
            let output = gate.execute_hard(a != 0.0, b != 0.0);
            let target = self.targets[i] != 0.0;
            if output == target {
                correct += 1;
            }
        }

        correct as f64 / self.inputs.len() as f64
    }
}

/// Trainer for a single probabilistic gate
pub struct GateTrainer {
    pub gate: ProbabilisticGate,
    pub optimizer: AdamW,
    pub iteration: usize,
}

impl GateTrainer {
    /// Create a new trainer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            gate: ProbabilisticGate::new(),
            optimizer: AdamW::new(learning_rate),
            iteration: 0,
        }
    }

    /// Train for one epoch on the truth table
    pub fn train_epoch(&mut self, truth_table: &TruthTable) -> f64 {
        // Accumulate gradients over all examples
        let mut total_gradients = [0.0; 16];

        for (i, &(a, b)) in truth_table.inputs.iter().enumerate() {
            let output = self.gate.execute_soft(a, b);
            let target = truth_table.targets[i];

            let grads = self.gate.compute_gradients(a, b, target, output);

            for j in 0..16 {
                total_gradients[j] += grads[j];
            }
        }

        // Average gradients
        for j in 0..16 {
            total_gradients[j] /= truth_table.inputs.len() as f64;
        }

        // Update parameters
        self.optimizer.step(&mut self.gate.logits, &total_gradients);
        // Invalidate cached probabilities after logit update
        self.gate.invalidate_cache();
        self.iteration += 1;

        // Return current loss
        truth_table.compute_loss(&self.gate)
    }

    /// Train until convergence or max iterations
    pub fn train(
        &mut self,
        truth_table: &TruthTable,
        max_iterations: usize,
        target_loss: f64,
        verbose: bool,
    ) -> TrainingResult {
        let mut losses = Vec::new();
        let mut converged = false;

        for i in 0..max_iterations {
            let loss = self.train_epoch(truth_table);
            losses.push(loss);

            if verbose && (i % 100 == 0 || i == max_iterations - 1) {
                let hard_acc = truth_table.compute_hard_accuracy(&self.gate);
                let (dominant_op, prob) = self.gate.dominant_operation();
                println!(
                    "Iter {:5}: Loss = {:.6}, Hard Acc = {:.2}%, Dominant Op = {:?} ({:.2}%)",
                    i,
                    loss,
                    hard_acc * 100.0,
                    dominant_op,
                    prob * 100.0
                );
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
        let hard_accuracy = truth_table.compute_hard_accuracy(&self.gate);
        let (dominant_op, prob) = self.gate.dominant_operation();

        TrainingResult {
            converged,
            iterations: self.iteration,
            final_loss,
            hard_accuracy,
            dominant_op,
            dominant_prob: prob,
            losses,
        }
    }
}

/// Result of a training run
#[derive(Debug)]
pub struct TrainingResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_loss: f64,
    pub hard_accuracy: f64,
    pub dominant_op: BinaryOp,
    pub dominant_prob: f64,
    pub losses: Vec<f64>,
}

impl TrainingResult {
    /// Check if this meets Phase 0.1 exit criteria
    pub fn meets_exit_criteria(&self, target_op: BinaryOp) -> bool {
        self.converged && self.hard_accuracy > 0.99 && self.dominant_op == target_op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_table_and() {
        let tt = TruthTable::for_operation(BinaryOp::And);
        assert_eq!(tt.inputs.len(), 4);
        assert_eq!(tt.targets, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_truth_table_or() {
        let tt = TruthTable::for_operation(BinaryOp::Or);
        assert_eq!(tt.targets, vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_truth_table_xor() {
        let tt = TruthTable::for_operation(BinaryOp::Xor);
        assert_eq!(tt.targets, vec![0.0, 1.0, 1.0, 0.0]);
    }
}
