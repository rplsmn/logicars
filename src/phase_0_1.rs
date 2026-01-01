//! Phase 0.1: Single Gate Training
//!
//! Implements one probabilistic logic gate with 16 binary operations.
//! Train to learn AND, OR, XOR individually with >99% accuracy.

use rand::Rng;

/// The 16 possible binary operations on two boolean inputs
/// The enum value represents the truth table: bit pattern for inputs (00, 01, 10, 11)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BinaryOp {
    False = 0,           // 0000: always 0
    And = 8,             // 1000: only 1 when both inputs are 1
    AAndNotB = 4,        // 0100: 1 when a=1 and b=0
    A = 12,              // 1100: pass-through A (important for stability!)
    NotAAndB = 2,        // 0010: 1 when a=0 and b=1
    B = 10,              // 1010: pass-through B
    Xor = 6,             // 0110: 1 when inputs differ
    Or = 14,             // 1110: 0 only when both inputs are 0
    Nor = 1,             // 0001: 1 only when both inputs are 0
    Xnor = 9,            // 1001: 1 when inputs are equal
    NotB = 5,            // 0101: opposite of B
    AOrNotB = 13,        // 1101: 0 only when a=0 and b=1
    NotA = 3,            // 0011: opposite of A
    NotAOrB = 11,        // 1011: 0 only when a=1 and b=0
    Nand = 7,            // 0111: 0 only when both inputs are 1
    True = 15,           // 1111: always 1
}

impl BinaryOp {
    /// Execute the binary operation
    #[inline]
    pub fn execute(self, a: bool, b: bool) -> bool {
        // Use truth table lookup based on operation index
        let idx = ((a as u8) << 1) | (b as u8);
        ((self as u8) >> idx) & 1 == 1
    }

    /// Execute with floating point inputs/outputs (for soft computation)
    #[inline]
    pub fn execute_soft(self, a: f64, b: f64) -> f64 {
        // For soft execution, we treat booleans as probabilities
        // and compute the expected output based on the truth table
        let p_not_a = 1.0 - a;
        let p_not_b = 1.0 - b;

        match self {
            BinaryOp::False => 0.0,
            BinaryOp::And => a * b,
            BinaryOp::AAndNotB => a * p_not_b,
            BinaryOp::A => a,
            BinaryOp::NotAAndB => p_not_a * b,
            BinaryOp::B => b,
            BinaryOp::Xor => a * p_not_b + p_not_a * b,
            BinaryOp::Or => a + b - a * b,
            BinaryOp::Nor => p_not_a * p_not_b,
            BinaryOp::Xnor => a * b + p_not_a * p_not_b,
            BinaryOp::NotB => p_not_b,
            BinaryOp::AOrNotB => a + p_not_b - a * p_not_b,
            BinaryOp::NotA => p_not_a,
            BinaryOp::NotAOrB => p_not_a + b - p_not_a * b,
            BinaryOp::Nand => 1.0 - a * b,
            BinaryOp::True => 1.0,
        }
    }

    /// All 16 operations, indexed by their truth table value (0-15)
    pub const ALL: [BinaryOp; 16] = [
        BinaryOp::False,    // 0
        BinaryOp::Nor,      // 1
        BinaryOp::NotAAndB, // 2
        BinaryOp::NotA,     // 3
        BinaryOp::AAndNotB, // 4
        BinaryOp::NotB,     // 5
        BinaryOp::Xor,      // 6
        BinaryOp::Nand,     // 7
        BinaryOp::And,      // 8
        BinaryOp::Xnor,     // 9
        BinaryOp::B,        // 10
        BinaryOp::NotAOrB,  // 11
        BinaryOp::A,        // 12
        BinaryOp::AOrNotB,  // 13
        BinaryOp::Or,       // 14
        BinaryOp::True,     // 15
    ];
}

/// A probabilistic gate that maintains a distribution over all 16 binary operations
pub struct ProbabilisticGate {
    /// Logits for each of the 16 operations (unnormalized log probabilities)
    /// Index 3 (pass-through) should be initialized to 10.0 for stability
    pub logits: [f64; 16],
}

impl ProbabilisticGate {
    /// Create a new gate with pass-through initialization
    pub fn new() -> Self {
        let mut logits = [0.0; 16];
        // Initialize pass-through gate A (value 12) to 10.0 for training stability
        logits[12] = 10.0;
        Self { logits }
    }

    /// Create a gate with random initialization
    pub fn new_random<R: Rng>(rng: &mut R) -> Self {
        let mut logits = [0.0; 16];
        for i in 0..16 {
            logits[i] = rng.random_range(-1.0..1.0);
        }
        // Still bias towards pass-through for stability
        logits[12] += 5.0;
        Self { logits }
    }

    /// Compute softmax probabilities from logits
    pub fn probabilities(&self) -> [f64; 16] {
        // Numerically stable softmax
        let max_logit = self.logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_sum = 0.0;
        let mut probs = [0.0; 16];

        for i in 0..16 {
            let exp_val = (self.logits[i] - max_logit).exp();
            probs[i] = exp_val;
            exp_sum += exp_val;
        }

        for i in 0..16 {
            probs[i] /= exp_sum;
        }

        probs
    }

    /// Soft execution: weighted sum of all gate outputs
    pub fn execute_soft(&self, a: f64, b: f64) -> f64 {
        let probs = self.probabilities();
        let mut output = 0.0;

        for (i, &op) in BinaryOp::ALL.iter().enumerate() {
            output += probs[i] * op.execute_soft(a, b);
        }

        output
    }

    /// Hard execution: use only the highest-probability gate
    pub fn execute_hard(&self, a: bool, b: bool) -> bool {
        let probs = self.probabilities();
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        BinaryOp::ALL[max_idx].execute(a, b)
    }

    /// Get the most probable operation
    pub fn dominant_operation(&self) -> (BinaryOp, f64) {
        let probs = self.probabilities();
        let (idx, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        (BinaryOp::ALL[idx], prob)
    }

    /// Compute gradients of the loss with respect to logits
    ///
    /// This uses the chain rule:
    /// dL/dlogits[i] = dL/doutput * doutput/dprobs[i] * dprobs[i]/dlogits[i]
    pub fn compute_gradients(&self, a: f64, b: f64, target: f64, output: f64) -> [f64; 16] {
        let probs = self.probabilities();
        let mut gradients = [0.0; 16];

        // dL/doutput for MSE loss: 2 * (output - target)
        let dloss_doutput = 2.0 * (output - target);

        // For each operation
        for i in 0..16 {
            let op_output = BinaryOp::ALL[i].execute_soft(a, b);

            // doutput/dprobs[i] = op_output
            let doutput_dprob_i = op_output;

            // dprobs[i]/dlogits[j] using softmax derivative
            // = probs[i] * (delta_ij - probs[j])
            // We need dL/dlogits[i] = sum over all j of: dL/dprobs[j] * dprobs[j]/dlogits[i]

            let mut dlogit_i = 0.0;
            for j in 0..16 {
                let op_output_j = BinaryOp::ALL[j].execute_soft(a, b);
                let dloss_dprob_j = dloss_doutput * op_output_j;

                // Softmax Jacobian: dprobs[j]/dlogits[i]
                let dprob_j_dlogit_i = if i == j {
                    probs[j] * (1.0 - probs[i])
                } else {
                    -probs[j] * probs[i]
                };

                dlogit_i += dloss_dprob_j * dprob_j_dlogit_i;
            }

            gradients[i] = dlogit_i;
        }

        gradients
    }
}

impl Default for ProbabilisticGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_binary_ops_truth_tables() {
        // Test AND
        assert!(!BinaryOp::And.execute(false, false));
        assert!(!BinaryOp::And.execute(false, true));
        assert!(!BinaryOp::And.execute(true, false));
        assert!(BinaryOp::And.execute(true, true));

        // Test OR
        assert!(!BinaryOp::Or.execute(false, false));
        assert!(BinaryOp::Or.execute(false, true));
        assert!(BinaryOp::Or.execute(true, false));
        assert!(BinaryOp::Or.execute(true, true));

        // Test XOR
        assert!(!BinaryOp::Xor.execute(false, false));
        assert!(BinaryOp::Xor.execute(false, true));
        assert!(BinaryOp::Xor.execute(true, false));
        assert!(!BinaryOp::Xor.execute(true, true));

        // Test pass-through A
        assert!(!BinaryOp::A.execute(false, false));
        assert!(!BinaryOp::A.execute(false, true));
        assert!(BinaryOp::A.execute(true, false));
        assert!(BinaryOp::A.execute(true, true));
    }

    #[test]
    fn test_soft_execution_extremes() {
        // When inputs are 0.0 or 1.0, soft should match hard
        let ops = [BinaryOp::And, BinaryOp::Or, BinaryOp::Xor];

        for op in &ops {
            for &a in &[0.0, 1.0] {
                for &b in &[0.0, 1.0] {
                    let soft = op.execute_soft(a, b);
                    let hard = op.execute(a != 0.0, b != 0.0) as u8 as f64;
                    assert!((soft - hard).abs() < 1e-10,
                        "Op {:?}: soft({}, {}) = {}, hard = {}", op, a, b, soft, hard);
                }
            }
        }
    }

    #[test]
    fn test_gate_initialization() {
        let gate = ProbabilisticGate::new();
        let (dominant_op, prob) = gate.dominant_operation();

        // Pass-through should be dominant initially
        assert_eq!(dominant_op, BinaryOp::A);
        assert!(prob > 0.9, "Pass-through should have >90% probability, got {}", prob);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let gate = ProbabilisticGate::new();
        let probs = gate.probabilities();
        let sum: f64 = probs.iter().sum();

        assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_numerical_gradients() {
        // Test that analytical gradients match numerical gradients
        let mut gate = ProbabilisticGate::new();

        // Use random logits for more thorough testing
        use rand::thread_rng;
        gate = ProbabilisticGate::new_random(&mut thread_rng());

        let epsilon = 1e-5;
        let test_cases = vec![
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.5, 0.5, 0.3),
        ];

        for (a, b, target) in test_cases {
            let output = gate.execute_soft(a, b);

            // Compute analytical gradients
            let analytical_grads = gate.compute_gradients(a, b, target, output);

            // Compute numerical gradients for each logit
            for i in 0..16 {
                // Save original logit
                let original = gate.logits[i];

                // Compute loss with logit + epsilon
                gate.logits[i] = original + epsilon;
                let output_plus = gate.execute_soft(a, b);
                let loss_plus = (output_plus - target).powi(2);

                // Compute loss with logit - epsilon
                gate.logits[i] = original - epsilon;
                let output_minus = gate.execute_soft(a, b);
                let loss_minus = (output_minus - target).powi(2);

                // Restore original logit
                gate.logits[i] = original;

                // Numerical gradient
                let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

                // Compare analytical and numerical gradients
                assert_relative_eq!(
                    analytical_grads[i],
                    numerical_grad,
                    epsilon = 1e-4,
                    max_relative = 1e-3,
                );
            }
        }
    }
}
