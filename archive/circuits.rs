use crate::logic_gates::LogicGate;
use rand::prelude::*;

// A layer of logic gates in the circuit
#[derive(Clone)]
pub struct GateLayer {
    pub gates: Vec<LogicGate>,
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
    
    pub fn backward(&mut self, inputs: &[f32], output_grads: &[f32], learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        let mut input_grads = vec![0.0; inputs.len()];
        
        // Process each gate
        for (i, gate) in self.gates.iter_mut().enumerate() {
            let a_idx = gate.inputs.0;
            let b_idx = gate.inputs.1;
            let a = inputs[a_idx];
            let b = inputs[b_idx];
            
            // Compute gradients for this gate
            let (grad_a, grad_b) = gate.backward(a, b, output_grads[i], learning_rate, l2_strength, temperature);
            
            // Accumulate gradients for inputs
            input_grads[a_idx] += grad_a;
            input_grads[b_idx] += grad_b;
        }
        
        input_grads
    }
}

// Circuit composed of multiple layers of gates
#[derive(Clone)]
pub struct Circuit {
    pub layers: Vec<GateLayer>,
    pub layer_sizes: Vec<usize>,
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
    
    pub fn backward(&mut self, all_inputs: &[Vec<f32>], output_grads: &[f32], learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        let mut gradients = output_grads.to_vec();
        
        // Backpropagate through layers in reverse order
        for i in (0..self.layers.len()).rev() {
            let layer_inputs = &all_inputs[i];
            gradients = self.layers[i].backward(layer_inputs, &gradients, learning_rate, l2_strength, temperature);
        }
        
        gradients
    }
}

// Perception circuit that processes neighborhood states
#[derive(Clone)]
pub struct PerceptionCircuit {
    pub circuit: Circuit,
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

    pub fn backward(&mut self, inputs: &[f32], output_grad: f32, learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        // Perform forward pass to collect intermediate outputs
        let (all_inputs, _) = self.circuit.forward_soft(inputs);
        
        // Backward pass with single output gradient
        self.circuit.backward(&all_inputs, &[output_grad], learning_rate, l2_strength, temperature)
    }
}

// The update circuit that computes the next state
pub struct UpdateCircuit {
    pub circuit: Circuit,
}

impl UpdateCircuit {
    pub fn new(input_size: usize, _state_size: usize, rng: &mut ThreadRng) -> Self {
        // Architecture for the update circuit:
        // input_size inputs -> 16 layers of 128 nodes -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 outputs
        let mut layer_sizes = vec![input_size];
        
        // Add 16 layers of 128 nodes
        for _ in 0..16 {
            layer_sizes.push(128);
        }
        
        // Add the remaining layers
        layer_sizes.extend_from_slice(&[64, 32, 16, 8, 4, 2, 1]);
        
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

    pub fn backward(&mut self, inputs: &[f32], output_grads: &[f32], learning_rate: f32, l2_strength: f32, temperature: f32) -> Vec<f32> {
        // Perform forward pass to collect intermediate outputs
        let (all_inputs, _) = self.circuit.forward_soft(inputs);
        
        // Backward pass
        self.circuit.backward(&all_inputs, output_grads, learning_rate, l2_strength, temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

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
}