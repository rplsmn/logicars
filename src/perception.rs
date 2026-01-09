//! Phase 1.2: Perception Module
//!
//! Implements parallel perception kernels that extract features from 3×3 neighborhoods.
//! Each kernel processes 9C inputs (9 cells × C channels) and outputs feature bits.
//!
//! Architecture (from paper):
//! - K parallel kernels (4-16, configurable)
//! - Each kernel: multi-layer gate network
//! - Layer 1: "first_kernel" topology (center vs 8 neighbors)
//! - Layers 2+: "unique" connections (information mixing)
//! - Output: [center_cell (C bits), kernel_outputs]

use crate::grid::NNeighborhood;
use crate::optimizer::AdamW;
use crate::phase_0_1::{BinaryOp, ProbabilisticGate};

/// Connection topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    /// First layer of perception: center vs each of 8 neighbors
    FirstKernel,
    /// Subsequent layers: unique connections ensuring each gate gets different inputs
    Unique,
}

/// Wire connections for a layer: (input_a_indices, input_b_indices)
/// Each gate i takes inputs from (wires.0[i], wires.1[i])
#[derive(Debug, Clone)]
pub struct Wires {
    /// Indices for first input of each gate
    pub a: Vec<usize>,
    /// Indices for second input of each gate
    pub b: Vec<usize>,
}

impl Wires {
    /// Create new wires
    pub fn new(a: Vec<usize>, b: Vec<usize>) -> Self {
        assert_eq!(a.len(), b.len(), "Wire vectors must have same length");
        Self { a, b }
    }

    /// Number of gates (wire pairs)
    pub fn num_gates(&self) -> usize {
        self.a.len()
    }
}

/// Generate Moore neighborhood connections (center vs 8 neighbors)
///
/// Used for first layer of perception kernels.
/// Returns 8 wire pairs, each comparing center (position 4) to a neighbor.
///
/// For C channels: position indices are 0-8 (cell positions), not channel indices.
/// The input to this layer should be 9 values (one per cell position).
pub fn first_kernel_connections(_seed: u64) -> Wires {
    // Neighbor positions in 3×3 grid (excluding center at position 4)
    // Order: NW=0, N=1, NE=2, W=3, E=5, SW=6, S=7, SE=8
    let neighbors: Vec<usize> = vec![0, 1, 2, 3, 5, 6, 7, 8];

    // Each gate compares a neighbor to the center
    let a = neighbors.clone();
    let b = vec![4; 8]; // All connect to center

    // Note: Reference impl uses random permutation, but for determinism we skip that
    // The network can learn any permutation through training

    Wires::new(a, b)
}

/// Generate unique connections ensuring each gate gets different input pairs
///
/// Used for subsequent layers to ensure information mixing.
/// Algorithm from reference impl:
/// 1. Pair consecutive elements: (0,1), (2,3), ...
/// 2. Then pair offset elements: (1,2), (3,4), ...
/// 3. Continue with larger offsets as needed
pub fn unique_connections(in_dim: usize, out_dim: usize) -> Wires {
    assert!(out_dim * 2 >= in_dim, "Output dimension too small for input");

    let x: Vec<usize> = (0..in_dim).collect();
    let mut a_vec: Vec<usize> = Vec::new();
    let mut b_vec: Vec<usize> = Vec::new();

    // First: pair consecutive even/odd elements
    let a1: Vec<usize> = x.iter().step_by(2).copied().collect();
    let b1: Vec<usize> = x.iter().skip(1).step_by(2).copied().collect();
    let m = a1.len().min(b1.len());
    a_vec.extend_from_slice(&a1[..m]);
    b_vec.extend_from_slice(&b1[..m]);

    if a_vec.len() < out_dim {
        // Second: offset by 1
        let a2: Vec<usize> = x.iter().skip(1).step_by(2).copied().collect();
        let b2: Vec<usize> = x.iter().skip(2).step_by(2).copied().collect();
        let m = a2.len().min(b2.len());
        a_vec.extend_from_slice(&a2[..m]);
        b_vec.extend_from_slice(&b2[..m]);
    }

    // Continue with larger offsets if needed
    let mut offset = 2;
    while a_vec.len() < out_dim && offset < in_dim {
        let a_off: Vec<usize> = (0..in_dim.saturating_sub(offset)).collect();
        let b_off: Vec<usize> = (offset..in_dim).collect();
        a_vec.extend_from_slice(&a_off);
        b_vec.extend_from_slice(&b_off);
        offset += 1;
    }

    // Truncate to desired output dimension
    assert!(
        a_vec.len() >= out_dim,
        "Could not generate enough unique connections: {} < {}",
        a_vec.len(),
        out_dim
    );

    a_vec.truncate(out_dim);
    b_vec.truncate(out_dim);

    Wires::new(a_vec, b_vec)
}

/// Generate connections based on connection type
pub fn generate_connections(conn_type: ConnectionType, in_dim: usize, out_dim: usize) -> Wires {
    match conn_type {
        ConnectionType::FirstKernel => {
            assert_eq!(out_dim, 8, "FirstKernel always produces 8 outputs");
            first_kernel_connections(0)
        }
        ConnectionType::Unique => unique_connections(in_dim, out_dim),
    }
}

/// A single gate layer with fixed wiring
#[derive(Debug, Clone)]
pub struct GateLayer {
    /// Gates in this layer
    pub gates: Vec<ProbabilisticGate>,
    /// Wire connections (a, b indices into previous layer's output)
    pub wires: Wires,
}

impl GateLayer {
    /// Create a new gate layer with specified connections
    pub fn new(out_dim: usize, wires: Wires) -> Self {
        assert_eq!(wires.num_gates(), out_dim);
        let gates = (0..out_dim).map(|_| ProbabilisticGate::new()).collect();
        Self { gates, wires }
    }

    /// Number of gates in this layer
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Forward pass in soft mode
    pub fn forward_soft(&self, inputs: &[f64]) -> Vec<f64> {
        self.gates
            .iter()
            .enumerate()
            .map(|(i, gate)| {
                let a = inputs[self.wires.a[i]];
                let b = inputs[self.wires.b[i]];
                gate.execute_soft(a, b)
            })
            .collect()
    }

    /// Forward pass in hard mode
    pub fn forward_hard(&self, inputs: &[f64]) -> Vec<f64> {
        self.gates
            .iter()
            .enumerate()
            .map(|(i, gate)| {
                let a = inputs[self.wires.a[i]];
                let b = inputs[self.wires.b[i]];
                if gate.execute_hard(a > 0.5, b > 0.5) { 1.0 } else { 0.0 }
            })
            .collect()
    }

    /// Get all logits as flat f32 array [num_gates × 16]
    /// Used for GPU buffer transfer
    #[cfg(feature = "gpu")]
    pub fn get_logits_flat_f32(&self) -> Vec<f32> {
        self.gates
            .iter()
            .flat_map(|g| g.logits.iter().map(|&x| x as f32))
            .collect()
    }

    /// Get all wire indices as flat u32 array [num_gates × 2]
    /// Layout: [gate0_a, gate0_b, gate1_a, gate1_b, ...]
    #[cfg(feature = "gpu")]
    pub fn get_wires_flat_u32(&self) -> Vec<u32> {
        self.gates
            .iter()
            .enumerate()
            .flat_map(|(i, _)| [self.wires.a[i] as u32, self.wires.b[i] as u32])
            .collect()
    }

    /// Output size of this layer (number of gates)
    pub fn output_size(&self) -> usize {
        self.gates.len()
    }
}

/// A single perception kernel (multi-layer gate network)
///
/// Takes 9C inputs (3×3 neighborhood × C channels) and produces output bits.
/// For C=1 GoL: [9→8→4→2→1]
#[derive(Debug, Clone)]
pub struct PerceptionKernel {
    /// Layers of gates
    pub layers: Vec<GateLayer>,
    /// Number of input channels
    pub input_size: usize,
}

impl PerceptionKernel {
    /// Create a new perception kernel with specified architecture
    ///
    /// # Arguments
    /// * `layer_sizes` - Sizes of each layer including input (e.g., [9, 8, 4, 2, 1])
    /// * `connection_types` - Connection type for each layer transition
    pub fn new(layer_sizes: &[usize], connection_types: &[ConnectionType]) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least input and output layers");
        assert_eq!(
            layer_sizes.len() - 1,
            connection_types.len(),
            "Need {} connection types for {} layers",
            layer_sizes.len() - 1,
            layer_sizes.len()
        );

        let input_size = layer_sizes[0];
        let mut layers = Vec::new();

        for i in 0..(layer_sizes.len() - 1) {
            let in_dim = layer_sizes[i];
            let out_dim = layer_sizes[i + 1];
            let wires = generate_connections(connection_types[i], in_dim, out_dim);
            layers.push(GateLayer::new(out_dim, wires));
        }

        Self { layers, input_size }
    }

    /// Create GoL perception kernel: [9→8→4→2→1]
    pub fn gol_kernel() -> Self {
        Self::new(
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        )
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Output size (number of outputs from last layer)
    pub fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.num_gates()).unwrap_or(0)
    }

    /// Total number of gates across all layers
    pub fn total_gates(&self) -> usize {
        self.layers.iter().map(|l| l.num_gates()).sum()
    }

    /// Forward pass in soft mode
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
}

/// Perception module with K parallel kernels
///
/// Takes a 3×3 neighborhood (9 cells × C channels) and produces:
/// Output: [center_cell (C bits), kernel_1_out, ..., kernel_K_out]
#[derive(Debug, Clone)]
pub struct PerceptionModule {
    /// Number of channels per cell
    pub channels: usize,
    /// Number of kernels
    pub num_kernels: usize,
    /// Parallel perception kernels
    pub kernels: Vec<PerceptionKernel>,
    /// Layer sizes for each kernel
    pub layer_sizes: Vec<usize>,
    /// Connection types for each layer
    pub connection_types: Vec<ConnectionType>,
}

impl PerceptionModule {
    /// Create a new perception module
    ///
    /// # Arguments
    /// * `channels` - Number of channels per cell (C)
    /// * `num_kernels` - Number of parallel kernels (K)
    /// * `layer_sizes` - Architecture for each kernel (e.g., [9, 8, 4, 2, 1])
    /// * `connection_types` - Connection types for each layer transition
    pub fn new(
        channels: usize,
        num_kernels: usize,
        layer_sizes: &[usize],
        connection_types: &[ConnectionType],
    ) -> Self {
        assert!(channels >= 1 && channels <= 128);
        assert!(num_kernels >= 1);

        // For multi-channel: first layer size should be 9 (cell positions)
        // Each kernel processes one channel at a time, replicated across all channels

        let kernels: Vec<PerceptionKernel> = (0..num_kernels)
            .map(|_| PerceptionKernel::new(layer_sizes, connection_types))
            .collect();

        Self {
            channels,
            num_kernels,
            kernels,
            layer_sizes: layer_sizes.to_vec(),
            connection_types: connection_types.to_vec(),
        }
    }

    /// Create GoL perception module (16 kernels, [9→8→4→2→1])
    pub fn gol_module() -> Self {
        Self::new(
            1,  // C=1 for GoL
            16, // 16 parallel kernels
            &[9, 8, 4, 2, 1],
            &[
                ConnectionType::FirstKernel,
                ConnectionType::Unique,
                ConnectionType::Unique,
                ConnectionType::Unique,
            ],
        )
    }

    /// Total number of gates in all kernels
    pub fn total_gates(&self) -> usize {
        self.kernels.iter().map(|k| k.total_gates()).sum()
    }

    /// Output size: center_channels + (num_kernels × kernel_output_size × channels)
    pub fn output_size(&self) -> usize {
        let kernel_output = self.kernels[0].output_size();
        // For C=1: center(1) + kernels(16) * 1 = 17
        // For C>1: center(C) + kernels(K) * output_bits * C
        self.channels + self.num_kernels * kernel_output * self.channels
    }

    /// Forward pass in soft mode on a neighborhood
    ///
    /// For each channel, runs all kernels on the 9-cell positions.
    /// Output ordering matches reference: rearrange(x, 'k c s -> (c s k)')
    /// = [center, then for each channel c, for each output_bit s, for each kernel k]
    ///
    /// Returns (output, all_kernel_activations) for gradient computation
    pub fn forward_soft(&self, neighborhood: &NNeighborhood) -> (Vec<f64>, Vec<Vec<Vec<Vec<f64>>>>) {
        assert_eq!(neighborhood.channels, self.channels);

        let kernel_output_size = self.kernels[0].output_size();
        
        // First, compute all kernel outputs and store activations
        // kernel_outputs[k][c] = Vec<f64> of size kernel_output_size
        let mut kernel_outputs: Vec<Vec<Vec<f64>>> = 
            vec![vec![Vec::new(); self.channels]; self.num_kernels];
        
        // Store activations: [kernel_idx][channel_idx][layer_idx][values]
        let mut all_activations: Vec<Vec<Vec<Vec<f64>>>> =
            vec![Vec::with_capacity(self.channels); self.num_kernels];

        // Run all kernels on all channels
        for c in 0..self.channels {
            let channel_inputs: Vec<f64> = (0..9)
                .map(|pos| neighborhood.get(pos, c))
                .collect();

            for (k, kernel) in self.kernels.iter().enumerate() {
                let activations = kernel.forward_soft(&channel_inputs);
                let output = activations.last().cloned().unwrap_or_default();
                
                kernel_outputs[k][c] = output;
                all_activations[k].push(activations);
            }
        }

        // Build output with correct ordering: (c s k)
        // [center, c0_s0_k0, c0_s0_k1, ..., c0_s0_kK, c0_s1_k0, ..., cC_sS_kK]
        let mut output = Vec::with_capacity(self.output_size());
        
        // First: append center cell values (C channels)
        output.extend_from_slice(neighborhood.center());

        // Then: for each channel, for each output bit, for each kernel
        for c in 0..self.channels {
            for s in 0..kernel_output_size {
                for k in 0..self.num_kernels {
                    output.push(kernel_outputs[k][c][s]);
                }
            }
        }

        (output, all_activations)
    }

    /// Forward pass in hard mode
    /// Output ordering matches reference: rearrange(x, 'k c s -> (c s k)')
    pub fn forward_hard(&self, neighborhood: &NNeighborhood) -> Vec<f64> {
        assert_eq!(neighborhood.channels, self.channels);

        let kernel_output_size = self.kernels[0].output_size();
        
        // First, compute all kernel outputs
        // kernel_outputs[k][c] = Vec<f64> of size kernel_output_size
        let mut kernel_outputs: Vec<Vec<Vec<f64>>> = 
            vec![vec![Vec::new(); self.channels]; self.num_kernels];

        for c in 0..self.channels {
            let channel_inputs: Vec<f64> = (0..9)
                .map(|pos| neighborhood.get(pos, c))
                .collect();

            for (k, kernel) in self.kernels.iter().enumerate() {
                kernel_outputs[k][c] = kernel.forward_hard(&channel_inputs);
            }
        }

        // Build output with correct ordering: (c s k)
        let mut output = Vec::with_capacity(self.output_size());

        // First: append center cell values (thresholded)
        for &v in neighborhood.center() {
            output.push(if v > 0.5 { 1.0 } else { 0.0 });
        }

        // Then: for each channel, for each output bit, for each kernel
        for c in 0..self.channels {
            for s in 0..kernel_output_size {
                for k in 0..self.num_kernels {
                    output.push(kernel_outputs[k][c][s]);
                }
            }
        }

        output
    }

    /// Compute gradients for all kernels
    ///
    /// Returns gradients indexed by [kernel][channel][layer][gate][logit]
    /// 
    /// Input output_gradients are in (c s k) ordering to match forward pass.
    pub fn compute_gradients(
        &self,
        neighborhood: &NNeighborhood,
        all_activations: &[Vec<Vec<Vec<f64>>>],
        output_gradients: &[f64],
    ) -> Vec<Vec<Vec<Vec<[f64; 16]>>>> {
        // Gradient structure: [kernel][channel][layer][gate][16 logits]
        let mut all_gradients: Vec<Vec<Vec<Vec<[f64; 16]>>>> = 
            vec![Vec::new(); self.num_kernels];

        let kernel_output_size = self.kernels[0].output_size();
        
        // Reconstruct per-kernel per-channel gradients from (c s k) ordering
        // output_gradients layout after center: [c0_s0_k0, c0_s0_k1, ..., c0_s1_k0, ...]
        // We need to extract gradients for kernel k, channel c as [s0, s1, ...]
        
        // Build kernel_output_grads[k][c] = Vec<f64> of size kernel_output_size
        let mut kernel_output_grads: Vec<Vec<Vec<f64>>> = 
            vec![vec![vec![0.0; kernel_output_size]; self.channels]; self.num_kernels];
        
        let mut grad_idx = self.channels; // Skip center cell gradients
        for c in 0..self.channels {
            for s in 0..kernel_output_size {
                for k in 0..self.num_kernels {
                    kernel_output_grads[k][c][s] = output_gradients[grad_idx];
                    grad_idx += 1;
                }
            }
        }

        // Now compute gradients for each kernel and channel
        for (k, kernel) in self.kernels.iter().enumerate() {
            let mut kernel_grads: Vec<Vec<Vec<[f64; 16]>>> = Vec::new();

            for c in 0..self.channels {
                let channel_inputs: Vec<f64> = (0..9)
                    .map(|pos| neighborhood.get(pos, c))
                    .collect();

                let output_grad = &kernel_output_grads[k][c];
                let activations = &all_activations[k][c];

                let channel_grads = self.compute_kernel_gradients(
                    kernel,
                    &channel_inputs,
                    activations,
                    output_grad,
                );

                kernel_grads.push(channel_grads);
            }

            all_gradients[k] = kernel_grads;
        }

        all_gradients
    }

    /// Compute gradients for a single kernel
    fn compute_kernel_gradients(
        &self,
        kernel: &PerceptionKernel,
        inputs: &[f64],
        activations: &[Vec<f64>],
        output_gradients: &[f64],
    ) -> Vec<Vec<[f64; 16]>> {
        let num_layers = kernel.layers.len();
        let mut all_gradients: Vec<Vec<[f64; 16]>> = Vec::with_capacity(num_layers);

        // Initialize gradient storage
        for layer in &kernel.layers {
            all_gradients.push(vec![[0.0; 16]; layer.num_gates()]);
        }

        // Start with output gradients
        let mut output_grads = output_gradients.to_vec();

        // Backpropagate through layers (from last to first)
        for layer_idx in (0..num_layers).rev() {
            let layer = &kernel.layers[layer_idx];

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
                let (da, db) = self.compute_gate_input_gradients(gate, a, b);
                prev_output_grads[a_idx] += output_grad * da;
                prev_output_grads[b_idx] += output_grad * db;
            }

            output_grads = prev_output_grads;
        }

        all_gradients
    }

    /// Compute gradients of gate output w.r.t. inputs a and b
    fn compute_gate_input_gradients(&self, gate: &ProbabilisticGate, a: f64, b: f64) -> (f64, f64) {
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

    /// Compute gradients w.r.t. input neighborhood values
    ///
    /// Returns gradients for all 9 positions × channels (flat vector)
    pub fn compute_input_gradients(
        &self,
        neighborhood: &NNeighborhood,
        all_activations: &[Vec<Vec<Vec<f64>>>],
        output_gradients: &[f64],
    ) -> Vec<f64> {
        // Initialize input gradients: 9 positions × channels
        let mut input_grads = vec![0.0; 9 * self.channels];

        // Skip center cell gradients from output (they come directly from update module)
        // Center cell is at output positions 0..channels
        for c in 0..self.channels {
            // Center cell gradient comes from the center position of output_gradients
            if c < output_gradients.len() {
                input_grads[4 * self.channels + c] += output_gradients[c];
            }
        }

        // Reconstruct per-kernel per-channel gradients from (c s k) ordering
        let kernel_output_size = self.kernels[0].output_size();
        
        // Build kernel_output_grads[k][c] = Vec<f64> of size kernel_output_size
        let mut kernel_output_grads: Vec<Vec<Vec<f64>>> = 
            vec![vec![vec![0.0; kernel_output_size]; self.channels]; self.num_kernels];
        
        let mut grad_idx = self.channels; // Skip center cell gradients
        for c in 0..self.channels {
            for s in 0..kernel_output_size {
                for k in 0..self.num_kernels {
                    kernel_output_grads[k][c][s] = output_gradients[grad_idx];
                    grad_idx += 1;
                }
            }
        }

        // Process each kernel's contribution to input gradients
        for (k, kernel) in self.kernels.iter().enumerate() {
            for c in 0..self.channels {
                let channel_inputs: Vec<f64> = (0..9)
                    .map(|pos| neighborhood.get(pos, c))
                    .collect();

                let output_grad = &kernel_output_grads[k][c];
                let activations = &all_activations[k][c];

                // Backpropagate through kernel to get input gradients
                let kernel_input_grads = self.compute_kernel_input_gradients(
                    kernel,
                    &channel_inputs,
                    activations,
                    output_grad,
                );

                // Accumulate gradients for this channel's 9 positions
                for pos in 0..9 {
                    input_grads[pos * self.channels + c] += kernel_input_grads[pos];
                }
            }
        }

        input_grads
    }

    /// Compute gradients w.r.t. kernel inputs
    fn compute_kernel_input_gradients(
        &self,
        kernel: &PerceptionKernel,
        inputs: &[f64],
        activations: &[Vec<f64>],
        output_gradients: &[f64],
    ) -> Vec<f64> {
        let num_layers = kernel.layers.len();

        // Start with output gradients
        let mut output_grads = output_gradients.to_vec();

        // Backpropagate through layers (from last to first)
        for layer_idx in (0..num_layers).rev() {
            let layer = &kernel.layers[layer_idx];

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

                // Compute gradients w.r.t. inputs for backprop
                let (da, db) = self.compute_gate_input_gradients(gate, a, b);
                prev_output_grads[a_idx] += output_grad * da;
                prev_output_grads[b_idx] += output_grad * db;
            }

            output_grads = prev_output_grads;
        }

        output_grads
    }
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

/// Trainer for perception module
pub struct PerceptionTrainer {
    pub module: PerceptionModule,
    /// One optimizer per gate: [kernel][layer][gate]
    optimizers: Vec<Vec<Vec<AdamW>>>,
    pub learning_rate: f64,
    pub iteration: usize,
}

impl PerceptionTrainer {
    /// Create a new perception trainer
    pub fn new(module: PerceptionModule, learning_rate: f64) -> Self {
        let optimizers: Vec<Vec<Vec<AdamW>>> = module
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

        Self {
            module,
            optimizers,
            learning_rate,
            iteration: 0,
        }
    }

    /// Train on a single neighborhood example
    ///
    /// Returns the loss for this example
    pub fn train_step(
        &mut self,
        neighborhood: &NNeighborhood,
        target: &[f64],
    ) -> f64 {
        // Forward pass
        let (output, activations) = self.module.forward_soft(neighborhood);

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

        // Backward pass
        let gradients = self.module.compute_gradients(neighborhood, &activations, &output_gradients);

        // Update weights
        // Gradients are accumulated across all channels for each kernel
        for (k, kernel_grads) in gradients.iter().enumerate() {
            // Accumulate gradients across channels
            let num_layers = self.module.kernels[k].layers.len();
            let mut accumulated: Vec<Vec<[f64; 16]>> = Vec::with_capacity(num_layers);

            for layer_idx in 0..num_layers {
                let num_gates = self.module.kernels[k].layers[layer_idx].num_gates();
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
                        *v /= self.module.channels as f64;
                    }
                }

                accumulated.push(layer_grads);
            }

            // Apply updates
            for (layer_idx, layer_grads) in accumulated.iter().enumerate() {
                for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                    // Clip gradients
                    let mut clipped = *gate_grad;
                    for v in clipped.iter_mut() {
                        *v = v.clamp(-100.0, 100.0);
                    }

                    self.optimizers[k][layer_idx][gate_idx].step(
                        &mut self.module.kernels[k].layers[layer_idx].gates[gate_idx].logits,
                        &clipped,
                    );
                    // Invalidate cached probabilities after logit update
                    self.module.kernels[k].layers[layer_idx].gates[gate_idx].invalidate_cache();
                }
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

    // ==================== Connection Tests ====================

    #[test]
    fn test_first_kernel_connections() {
        let wires = first_kernel_connections(0);
        assert_eq!(wires.num_gates(), 8);

        // All b indices should be 4 (center)
        for &b in &wires.b {
            assert_eq!(b, 4);
        }

        // a indices should be 0-3 and 5-8 (all neighbors)
        let mut a_sorted = wires.a.clone();
        a_sorted.sort();
        assert_eq!(a_sorted, vec![0, 1, 2, 3, 5, 6, 7, 8]);
    }

    #[test]
    fn test_unique_connections_8_to_4() {
        let wires = unique_connections(8, 4);
        assert_eq!(wires.num_gates(), 4);

        // Each gate should have distinct input pair
        let mut pairs: Vec<(usize, usize)> = wires.a.iter()
            .zip(wires.b.iter())
            .map(|(&a, &b)| (a, b))
            .collect();

        // All pairs should be valid indices
        for (a, b) in &pairs {
            assert!(*a < 8);
            assert!(*b < 8);
        }
    }

    #[test]
    fn test_unique_connections_4_to_2() {
        let wires = unique_connections(4, 2);
        assert_eq!(wires.num_gates(), 2);
    }

    #[test]
    fn test_unique_connections_2_to_1() {
        let wires = unique_connections(2, 1);
        assert_eq!(wires.num_gates(), 1);
        assert_eq!(wires.a[0], 0);
        assert_eq!(wires.b[0], 1);
    }

    // ==================== Gate Layer Tests ====================

    #[test]
    fn test_gate_layer_forward() {
        let wires = Wires::new(vec![0, 2], vec![1, 3]);
        let layer = GateLayer::new(2, wires);

        let inputs = vec![0.0, 1.0, 0.5, 0.5];
        let outputs = layer.forward_soft(&inputs);

        assert_eq!(outputs.len(), 2);
        // With pass-through initialization, should pass through A
        assert_relative_eq!(outputs[0], 0.0, epsilon = 0.1);
        assert_relative_eq!(outputs[1], 0.5, epsilon = 0.1);
    }

    // ==================== Perception Kernel Tests ====================

    #[test]
    fn test_gol_kernel_architecture() {
        let kernel = PerceptionKernel::gol_kernel();

        assert_eq!(kernel.input_size, 9);
        assert_eq!(kernel.num_layers(), 4);
        assert_eq!(kernel.output_size(), 1);

        // Layer sizes: 8, 4, 2, 1
        assert_eq!(kernel.layers[0].num_gates(), 8);
        assert_eq!(kernel.layers[1].num_gates(), 4);
        assert_eq!(kernel.layers[2].num_gates(), 2);
        assert_eq!(kernel.layers[3].num_gates(), 1);

        // Total gates: 8 + 4 + 2 + 1 = 15
        assert_eq!(kernel.total_gates(), 15);
    }

    #[test]
    fn test_kernel_forward_soft() {
        let kernel = PerceptionKernel::gol_kernel();

        // All zeros input
        let inputs = vec![0.0; 9];
        let activations = kernel.forward_soft(&inputs);

        assert_eq!(activations.len(), 4);
        assert_eq!(activations[0].len(), 8);
        assert_eq!(activations[1].len(), 4);
        assert_eq!(activations[2].len(), 2);
        assert_eq!(activations[3].len(), 1);
    }

    #[test]
    fn test_kernel_forward_hard() {
        let kernel = PerceptionKernel::gol_kernel();

        let inputs = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let output = kernel.forward_hard(&inputs);

        assert_eq!(output.len(), 1);
        assert!(output[0] == 0.0 || output[0] == 1.0);
    }

    // ==================== Perception Module Tests ====================

    #[test]
    fn test_gol_module_architecture() {
        let module = PerceptionModule::gol_module();

        assert_eq!(module.channels, 1);
        assert_eq!(module.num_kernels, 16);
        assert_eq!(module.kernels.len(), 16);

        // Each kernel has 15 gates
        assert_eq!(module.kernels[0].total_gates(), 15);

        // Total: 16 kernels × 15 gates = 240 gates
        assert_eq!(module.total_gates(), 240);

        // Output size: center(1) + 16 kernels × 1 output = 17
        assert_eq!(module.output_size(), 17);
    }

    #[test]
    fn test_module_forward_soft() {
        let module = PerceptionModule::gol_module();
        let neighborhood = NNeighborhood::from_gol_index(0b101010101);

        let (output, activations) = module.forward_soft(&neighborhood);

        // Output: center(1) + 16 kernel outputs = 17
        assert_eq!(output.len(), 17);

        // First value is center cell
        assert_relative_eq!(output[0], 1.0, epsilon = 0.1);

        // Activations: 16 kernels × 1 channel
        assert_eq!(activations.len(), 16);
        assert_eq!(activations[0].len(), 1); // 1 channel
    }

    #[test]
    fn test_module_forward_hard() {
        let module = PerceptionModule::gol_module();
        let neighborhood = NNeighborhood::from_gol_index(0b000010000); // Only center alive

        let output = module.forward_hard(&neighborhood);

        assert_eq!(output.len(), 17);

        // First value is center cell (1.0)
        assert_eq!(output[0], 1.0);

        // All outputs should be 0.0 or 1.0 (hard)
        for v in &output {
            assert!(*v == 0.0 || *v == 1.0);
        }
    }

    #[test]
    fn test_multi_channel_forward() {
        // Test with C=8 (Checkerboard config)
        let module = PerceptionModule::new(
            8,   // 8 channels
            4,   // 4 kernels
            &[9, 8, 4, 2],
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique],
        );

        // Create 8-channel neighborhood
        let mut cells = vec![0.5; 9 * 8];
        // Set center channel 0 to 1.0
        cells[4 * 8] = 1.0;
        let neighborhood = NNeighborhood::new(8, cells);

        let (output, _) = module.forward_soft(&neighborhood);

        // Output: center(8) + 4 kernels × 2 outputs × 8 channels = 8 + 64 = 72
        assert_eq!(output.len(), 8 + 4 * 2 * 8);

        // First 8 values are center cell channels
        assert_relative_eq!(output[0], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_perception_output_ordering_csk() {
        // Test that output ordering matches reference: rearrange(x, 'k c s -> (c s k)')
        // This is critical for correct update module input
        //
        // Reference ordering after center:
        // For each channel c:
        //   For each output bit s:
        //     For each kernel k:
        //       output[c*S*K + s*K + k]

        let channels = 2;
        let num_kernels = 3;
        let kernel_output_size = 2; // Each kernel outputs 2 bits
        
        let module = PerceptionModule::new(
            channels,
            num_kernels,
            &[9, 8, 4, kernel_output_size], // Output 2 bits per kernel
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique],
        );

        // Create neighborhood with distinct channel values
        let mut cells = vec![0.0; 9 * channels];
        // Channel 0: all zeros
        // Channel 1: all ones
        for pos in 0..9 {
            cells[pos * channels + 1] = 1.0;
        }
        let neighborhood = NNeighborhood::new(channels, cells);

        let (output, _) = module.forward_soft(&neighborhood);
        
        // Output size: center(C) + C * S * K = 2 + 2 * 2 * 3 = 14
        assert_eq!(output.len(), channels + channels * kernel_output_size * num_kernels);
        
        // The key test: outputs for channel 0 and channel 1 should be different
        // because their inputs are different (all 0 vs all 1)
        //
        // After center (positions 2..):
        // Ordering is (c s k): c0_s0_k0, c0_s0_k1, c0_s0_k2, c0_s1_k0, ...
        //
        // Channel 0 outputs: positions 2, 3, 4, 5, 6, 7 (first C*S*K/2 positions)
        // Channel 1 outputs: positions 8, 9, 10, 11, 12, 13
        
        // For channel 0 (input all zeros), kernels start as pass-through so output ~0
        // For channel 1 (input all ones), kernels start as pass-through so output ~1
        
        // Check that channel structure is preserved:
        // output[center + c * S * K + s * K + k] is from kernel k, channel c, output bit s
        
        let center_size = channels;
        
        // Get outputs for channel 0, output_bit 0, all kernels (should be similar)
        let c0_s0: Vec<f64> = (0..num_kernels)
            .map(|k| output[center_size + 0 * kernel_output_size * num_kernels + 0 * num_kernels + k])
            .collect();
            
        // Get outputs for channel 1, output_bit 0, all kernels
        let c1_s0: Vec<f64> = (0..num_kernels)
            .map(|k| output[center_size + 1 * kernel_output_size * num_kernels + 0 * num_kernels + k])
            .collect();
        
        // Outputs from channel 0 (all-zero input) and channel 1 (all-one input) 
        // should be meaningfully different due to different inputs
        let c0_mean: f64 = c0_s0.iter().sum::<f64>() / c0_s0.len() as f64;
        let c1_mean: f64 = c1_s0.iter().sum::<f64>() / c1_s0.len() as f64;
        
        // The difference confirms inputs are reaching the right kernel/channel slots
        // Pass-through gates (A) with all-0 input → ~0, with all-1 input → ~1
        assert!(
            (c1_mean - c0_mean).abs() > 0.1,
            "Channel 0 and Channel 1 outputs should differ: c0={:.3}, c1={:.3}",
            c0_mean, c1_mean
        );
    }

    // ==================== Gradient Tests ====================

    #[test]
    fn test_numerical_gradient_kernel() {
        // Use valid GoL architecture: [9, 8, 4, 2, 1] with FirstKernel only for first layer
        let module = PerceptionModule::new(
            1, 1,
            &[9, 8, 4, 2, 1],
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique, ConnectionType::Unique],
        );

        let inputs = vec![0.3, 0.7, 0.2, 0.8, 0.5, 0.6, 0.1, 0.9, 0.4];
        let target = 0.6;
        let epsilon = 1e-5;

        let neighborhood = NNeighborhood::new(1, inputs.clone());
        let (output, activations) = module.forward_soft(&neighborhood);

        let output_grad = vec![2.0 * (output[1] - target)]; // Skip center
        let gradients = module.compute_gradients(&neighborhood, &activations, &[0.0, output_grad[0]]);

        // Numerical gradient check for first layer, first gate, first logit
        let mut module_copy = module.clone();
        let original = module_copy.kernels[0].layers[0].gates[0].logits[0];

        module_copy.kernels[0].layers[0].gates[0].logits[0] = original + epsilon;
        module_copy.kernels[0].layers[0].gates[0].invalidate_cache();
        let (out_plus, _) = module_copy.forward_soft(&neighborhood);
        let loss_plus = (out_plus[1] - target).powi(2);

        module_copy.kernels[0].layers[0].gates[0].logits[0] = original - epsilon;
        module_copy.kernels[0].layers[0].gates[0].invalidate_cache();
        let (out_minus, _) = module_copy.forward_soft(&neighborhood);
        let loss_minus = (out_minus[1] - target).powi(2);

        let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytical_grad = gradients[0][0][0][0][0];

        assert_relative_eq!(analytical_grad, numerical_grad, epsilon = 1e-4, max_relative = 0.01);
    }

    #[test]
    fn test_trainer_loss_decreases() {
        // Use valid GoL architecture with FirstKernel only for first layer (which must output 8)
        let module = PerceptionModule::new(
            1, 1, // Single kernel, single channel
            &[9, 8, 4, 2, 1],
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique, ConnectionType::Unique],
        );

        let mut trainer = PerceptionTrainer::new(module, 0.05);

        // Create a simple training example
        let neighborhood = NNeighborhood::from_gol_index(0b000010111); // Some pattern
        let target = vec![1.0, 0.0]; // Center=1, kernel_output=0

        let initial_loss = trainer.train_step(&neighborhood, &target);

        // Train for a few steps
        for _ in 0..50 {
            trainer.train_step(&neighborhood, &target);
        }

        let final_loss = trainer.train_step(&neighborhood, &target);

        // Loss should decrease
        assert!(final_loss < initial_loss, "Loss should decrease: {} -> {}", initial_loss, final_loss);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_gol_module_on_all_configs() {
        let module = PerceptionModule::gol_module();

        // Test all 512 GoL configurations
        for idx in 0..512 {
            let neighborhood = NNeighborhood::from_gol_index(idx);
            let (output, _) = module.forward_soft(&neighborhood);

            // Should produce 17 outputs
            assert_eq!(output.len(), 17);

            // All outputs should be valid probabilities
            for v in &output {
                assert!(*v >= 0.0 && *v <= 1.0, "Output {} not in [0,1]", v);
            }
        }
    }
}
