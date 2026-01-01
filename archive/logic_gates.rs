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
    pub op: LogicOp,
    pub inputs: (usize, usize), // Indices of input gates or values
    pub probability: Vec<f32>,  // Probability distribution over operations (for training)
}

impl LogicGate {
    pub fn new(inputs: (usize, usize)) -> Self {
        // Start with a base probability for all gates
        let base_prob = 0.06;
        let mut probability = vec![base_prob; 16];

        // Add stronger bias toward pass-through gates (A and B)
        probability[3] = 0.08; // Bias toward A (LogicOp::A)
        probability[5] = 0.08; // Bias toward B (LogicOp::B)
        
        // Add small random noise to break symmetry
        for p in &mut probability {
            *p *= 0.95 + 0.1 * rand::random::<f32>();
        }

        // Ensure probabilities sum to 1.0
        let sum: f32 = probability.iter().sum();
        for p in &mut probability {
            *p /= sum;
        }

        let mut gate = LogicGate {
            op: LogicOp::A, // Default operation will be overwritten
            inputs,
            probability,
        };
        gate.update_current_op(); // Set the op based on probability distribution
        return gate;
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
        // Calculate individual operation outputs
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
        
        // Calculate operation-specific gradients based on how each would improve the output
        let mut op_gradients = vec![0.0; 16];
        // Add exploration noise
        let noise_factor = 0.01;

        for i in 0..16 {
            let noise = (rand::random::<f32>() * 2.0 - 1.0) * noise_factor;
            let target_direction = -gradient; // Negative of loss gradient is direction toward target
            op_gradients[i] = target_direction * ops[i] + noise;

            let op_grad_clip = 3.0;
            op_gradients[i] = op_gradients[i].max(-op_grad_clip).min(op_grad_clip);
        }
        
        // Apply softmax 
        let mut probs = vec![0.0; 16];
        let mut max_prob = f32::MIN;
        
        // First, find max for numerical stability
        for i in 0..16 {
            max_prob = max_prob.max(self.probability[i].ln() / temperature + learning_rate * op_gradients[i]);
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
            
            // Apply L2 reg
            probs[i] -= learning_rate * l2_strength * (probs[i] - 1.0/16.0).abs();
            
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
    
    pub fn update_current_op(&mut self) {
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
}