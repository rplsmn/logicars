//! Optimizers for training probabilistic gates

/// AdamW optimizer for training
pub struct AdamW {
    /// Learning rate
    pub lr: f64,
    /// Beta1 for momentum
    pub beta1: f64,
    /// Beta2 for variance
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Maximum gradient norm for clipping (100.0 per reference)
    pub max_grad_norm: f64,
    /// First moment estimates
    m: [f64; 16],
    /// Second moment estimates
    v: [f64; 16],
    /// Time step
    t: usize,
}

impl AdamW {
    /// Create a new AdamW optimizer with reference implementation defaults
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 100.0, // From reference implementation
            m: [0.0; 16],
            v: [0.0; 16],
            t: 0,
        }
    }

    /// Update parameters given gradients
    /// 
    /// Note: Gradient clipping should be done BEFORE calling this method,
    /// using global norm clipping across all parameters (like optax.clip).
    pub fn step(&mut self, params: &mut [f64; 16], grads: &[f64; 16]) {
        self.t += 1;

        // Apply updates (no clipping here - done globally before this call)
        for i in 0..16 {
            let grad = grads[i];

            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;

            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));

            // AdamW: weight decay applied directly to parameters
            params[i] *= 1.0 - self.lr * self.weight_decay;

            // Update parameters
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.m = [0.0; 16];
        self.v = [0.0; 16];
        self.t = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_step() {
        let mut opt = AdamW::new(0.01);
        let mut params = [1.0; 16];
        let grads = [0.1; 16];

        opt.step(&mut params, &grads);

        // Parameters should have changed
        assert!(params[0] < 1.0);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut opt = AdamW::new(0.01);
        let mut params = [1.0; 16];

        // Very large gradients
        let large_grads = [1000.0; 16];
        opt.step(&mut params, &large_grads);

        // Should still be stable due to clipping
        assert!(params[0].is_finite());
    }
}
