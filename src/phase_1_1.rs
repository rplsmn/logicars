//! Phase 1.1: Perception Circuit
//!
//! Implements the perception module for Game of Life:
//! - 3x3 neighborhood → multi-bit feature vector
//! - Multiple parallel kernels (4-16)
//! - "first_kernel" topology from reference implementation
//!
//! Exit criteria: >95% accuracy on all 512 neighborhood configurations

use crate::optimizer::AdamW;
use crate::phase_0_1::{BinaryOp, ProbabilisticGate};

/// A 2D grid of binary cells
#[derive(Debug, Clone)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<bool>,
}

impl Grid {
    /// Create a new grid with all cells dead
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![false; width * height],
        }
    }

    /// Create a grid from a 2D bool array
    pub fn from_cells(width: usize, height: usize, cells: Vec<bool>) -> Self {
        assert_eq!(cells.len(), width * height);
        Self { width, height, cells }
    }

    /// Get cell at (x, y) with wrapping (toroidal boundary)
    #[inline]
    pub fn get(&self, x: isize, y: isize) -> bool {
        let x = x.rem_euclid(self.width as isize) as usize;
        let y = y.rem_euclid(self.height as isize) as usize;
        self.cells[y * self.width + x]
    }

    /// Set cell at (x, y)
    pub fn set(&mut self, x: usize, y: usize, value: bool) {
        self.cells[y * self.width + x] = value;
    }

    /// Extract 3x3 neighborhood around (cx, cy)
    /// Returns [NW, N, NE, W, C, E, SW, S, SE] in reading order
    pub fn neighborhood(&self, cx: usize, cy: usize) -> [bool; 9] {
        let cx = cx as isize;
        let cy = cy as isize;
        [
            self.get(cx - 1, cy - 1), // NW
            self.get(cx, cy - 1),     // N
            self.get(cx + 1, cy - 1), // NE
            self.get(cx - 1, cy),     // W
            self.get(cx, cy),         // C (center)
            self.get(cx + 1, cy),     // E
            self.get(cx - 1, cy + 1), // SW
            self.get(cx, cy + 1),     // S
            self.get(cx + 1, cy + 1), // SE
        ]
    }

    /// Convert neighborhood to 9-bit integer (for indexing truth tables)
    pub fn neighborhood_as_index(&self, cx: usize, cy: usize) -> usize {
        let n = self.neighborhood(cx, cy);
        let mut idx = 0;
        for (i, &cell) in n.iter().enumerate() {
            if cell {
                idx |= 1 << i;
            }
        }
        idx
    }
}

/// 3x3 neighborhood as a value type
#[derive(Debug, Clone, Copy)]
pub struct Neighborhood {
    /// Cells in reading order: [NW, N, NE, W, C, E, SW, S, SE]
    pub cells: [bool; 9],
}

impl Neighborhood {
    /// Create from array
    pub fn new(cells: [bool; 9]) -> Self {
        Self { cells }
    }

    /// Create from 9-bit index
    pub fn from_index(idx: usize) -> Self {
        let mut cells = [false; 9];
        for i in 0..9 {
            cells[i] = (idx >> i) & 1 == 1;
        }
        Self { cells }
    }

    /// Convert to 9-bit index
    pub fn to_index(&self) -> usize {
        let mut idx = 0;
        for (i, &cell) in self.cells.iter().enumerate() {
            if cell {
                idx |= 1 << i;
            }
        }
        idx
    }

    /// Get center cell
    pub fn center(&self) -> bool {
        self.cells[4]
    }

    /// Count alive neighbors (excluding center)
    pub fn neighbor_count(&self) -> u8 {
        let mut count = 0;
        for (i, &cell) in self.cells.iter().enumerate() {
            if i != 4 && cell {
                count += 1;
            }
        }
        count
    }

    /// Apply Game of Life rule
    pub fn gol_next_state(&self) -> bool {
        let count = self.neighbor_count();
        let center = self.center();

        if center {
            // Alive cell survives with 2 or 3 neighbors
            count == 2 || count == 3
        } else {
            // Dead cell becomes alive with exactly 3 neighbors
            count == 3
        }
    }

    /// Get cells as soft values (0.0 or 1.0)
    pub fn soft_cells(&self) -> [f64; 9] {
        let mut soft = [0.0; 9];
        for i in 0..9 {
            soft[i] = if self.cells[i] { 1.0 } else { 0.0 };
        }
        soft
    }
}

/// Connection topology for perception circuits
///
/// The reference implementation uses specific topologies:
/// - "first_kernel": connections that mimic cell interactions
/// - "unique": each gate gets different input pairs
#[derive(Debug, Clone)]
pub struct PerceptionTopology {
    /// For each gate, which two inputs (from 9 neighborhood cells) it receives
    pub connections: Vec<(usize, usize)>,
}

impl PerceptionTopology {
    /// Create a new topology with specified connections
    pub fn new(connections: Vec<(usize, usize)>) -> Self {
        Self { connections }
    }

    /// "first_kernel" topology from reference implementation
    ///
    /// This creates gates that compare adjacent cells in the neighborhood:
    /// - Horizontal pairs: (NW,N), (N,NE), (W,C), (C,E), (SW,S), (S,SE)
    /// - Vertical pairs: (NW,W), (W,SW), (N,C), (C,S), (NE,E), (E,SE)
    /// - Diagonal pairs: (NW,C), (C,SE), (NE,C), (C,SW)
    pub fn first_kernel() -> Self {
        // Index mapping: [NW=0, N=1, NE=2, W=3, C=4, E=5, SW=6, S=7, SE=8]
        let connections = vec![
            // Horizontal pairs
            (0, 1), // NW, N
            (1, 2), // N, NE
            (3, 4), // W, C
            (4, 5), // C, E
            (6, 7), // SW, S
            (7, 8), // S, SE
            // Vertical pairs
            (0, 3), // NW, W
            (3, 6), // W, SW
            (1, 4), // N, C
            (4, 7), // C, S
            (2, 5), // NE, E
            (5, 8), // E, SE
            // Diagonal pairs
            (0, 4), // NW, C
            (4, 8), // C, SE
            (2, 4), // NE, C
            (4, 6), // C, SW
        ];
        Self { connections }
    }

    /// Create a minimal topology with 4 gates for testing
    /// Compares center with each cardinal direction
    pub fn minimal() -> Self {
        // Index mapping: [NW=0, N=1, NE=2, W=3, C=4, E=5, SW=6, S=7, SE=8]
        let connections = vec![
            (4, 1), // C, N
            (4, 3), // C, W
            (4, 5), // C, E
            (4, 7), // C, S
        ];
        Self { connections }
    }

    /// Number of gates in this topology
    pub fn num_gates(&self) -> usize {
        self.connections.len()
    }
}

/// A perception kernel that transforms 3x3 neighborhood into a feature
///
/// Uses a multi-layer circuit of probabilistic gates to compute output.
/// Architecture follows reference: deep network with 128+ hidden units,
/// but we start simpler for Phase 1.1.
pub struct PerceptionKernel {
    /// Layer 1: gates that take pairs from the 9-cell neighborhood
    layer1: Vec<ProbabilisticGate>,
    /// Topology defining which inputs each layer1 gate receives
    topology: PerceptionTopology,
    /// Layer 2: gates that combine layer1 outputs
    layer2: Vec<ProbabilisticGate>,
    /// Layer2 connections: which layer1 outputs to combine
    layer2_connections: Vec<(usize, usize)>,
    /// Final gate that produces single output
    output_gate: ProbabilisticGate,
    /// Output gate connections (from layer2)
    output_connections: (usize, usize),
}

impl PerceptionKernel {
    /// Create a new perception kernel with first_kernel topology
    pub fn new() -> Self {
        Self::with_topology(PerceptionTopology::first_kernel())
    }

    /// Create with custom topology
    pub fn with_topology(topology: PerceptionTopology) -> Self {
        let num_layer1_gates = topology.num_gates();

        // Create layer1 gates
        let layer1: Vec<ProbabilisticGate> = (0..num_layer1_gates)
            .map(|_| ProbabilisticGate::new())
            .collect();

        // Layer2: pair up layer1 outputs
        let num_layer2_gates = num_layer1_gates / 2;
        let layer2: Vec<ProbabilisticGate> = (0..num_layer2_gates)
            .map(|_| ProbabilisticGate::new())
            .collect();

        // Layer2 connections: sequential pairing
        let layer2_connections: Vec<(usize, usize)> = (0..num_layer2_gates)
            .map(|i| (i * 2, i * 2 + 1))
            .collect();

        // Output gate takes first two layer2 outputs
        let output_gate = ProbabilisticGate::new();
        let output_connections = (0, 1.min(num_layer2_gates - 1));

        Self {
            layer1,
            topology,
            layer2,
            layer2_connections,
            output_gate,
            output_connections,
        }
    }

    /// Create a minimal kernel for testing (fewer parameters)
    pub fn minimal() -> Self {
        Self::with_topology(PerceptionTopology::minimal())
    }

    /// Execute in soft mode (for training)
    pub fn execute_soft(&self, neighborhood: &[f64; 9]) -> f64 {
        // Layer 1: apply topology to get inputs for each gate
        let layer1_outputs: Vec<f64> = self.layer1
            .iter()
            .zip(self.topology.connections.iter())
            .map(|(gate, &(a_idx, b_idx))| {
                gate.execute_soft(neighborhood[a_idx], neighborhood[b_idx])
            })
            .collect();

        // Layer 2: combine layer1 outputs
        let layer2_outputs: Vec<f64> = self.layer2
            .iter()
            .zip(self.layer2_connections.iter())
            .map(|(gate, &(a_idx, b_idx))| {
                let a = layer1_outputs.get(a_idx).copied().unwrap_or(0.0);
                let b = layer1_outputs.get(b_idx).copied().unwrap_or(0.0);
                gate.execute_soft(a, b)
            })
            .collect();

        // Output gate
        let (a_idx, b_idx) = self.output_connections;
        let a = layer2_outputs.get(a_idx).copied().unwrap_or(0.0);
        let b = layer2_outputs.get(b_idx).copied().unwrap_or(0.0);
        self.output_gate.execute_soft(a, b)
    }

    /// Execute in hard mode (for inference)
    pub fn execute_hard(&self, neighborhood: &[bool; 9]) -> bool {
        // Layer 1
        let layer1_outputs: Vec<bool> = self.layer1
            .iter()
            .zip(self.topology.connections.iter())
            .map(|(gate, &(a_idx, b_idx))| {
                gate.execute_hard(neighborhood[a_idx], neighborhood[b_idx])
            })
            .collect();

        // Layer 2
        let layer2_outputs: Vec<bool> = self.layer2
            .iter()
            .zip(self.layer2_connections.iter())
            .map(|(gate, &(a_idx, b_idx))| {
                let a = *layer1_outputs.get(a_idx).unwrap_or(&false);
                let b = *layer1_outputs.get(b_idx).unwrap_or(&false);
                gate.execute_hard(a, b)
            })
            .collect();

        // Output gate
        let (a_idx, b_idx) = self.output_connections;
        let a = *layer2_outputs.get(a_idx).unwrap_or(&false);
        let b = *layer2_outputs.get(b_idx).unwrap_or(&false);
        self.output_gate.execute_hard(a, b)
    }

    /// Forward pass returning all intermediate activations (for backprop)
    fn forward_with_activations(&self, neighborhood: &[f64; 9]) -> (Vec<f64>, Vec<f64>, f64) {
        // Layer 1
        let layer1_outputs: Vec<f64> = self.layer1
            .iter()
            .zip(self.topology.connections.iter())
            .map(|(gate, &(a_idx, b_idx))| {
                gate.execute_soft(neighborhood[a_idx], neighborhood[b_idx])
            })
            .collect();

        // Layer 2
        let layer2_outputs: Vec<f64> = self.layer2
            .iter()
            .zip(self.layer2_connections.iter())
            .map(|(gate, &(a_idx, b_idx))| {
                let a = layer1_outputs.get(a_idx).copied().unwrap_or(0.0);
                let b = layer1_outputs.get(b_idx).copied().unwrap_or(0.0);
                gate.execute_soft(a, b)
            })
            .collect();

        // Output
        let (a_idx, b_idx) = self.output_connections;
        let a = layer2_outputs.get(a_idx).copied().unwrap_or(0.0);
        let b = layer2_outputs.get(b_idx).copied().unwrap_or(0.0);
        let output = self.output_gate.execute_soft(a, b);

        (layer1_outputs, layer2_outputs, output)
    }

    /// Number of trainable gates
    pub fn num_gates(&self) -> usize {
        self.layer1.len() + self.layer2.len() + 1
    }

    /// Get mutable reference to a gate by flat index
    pub fn gate_mut(&mut self, idx: usize) -> &mut ProbabilisticGate {
        let n1 = self.layer1.len();
        let n2 = self.layer2.len();

        if idx < n1 {
            &mut self.layer1[idx]
        } else if idx < n1 + n2 {
            &mut self.layer2[idx - n1]
        } else {
            &mut self.output_gate
        }
    }

    /// Get reference to a gate by flat index
    pub fn gate(&self, idx: usize) -> &ProbabilisticGate {
        let n1 = self.layer1.len();
        let n2 = self.layer2.len();

        if idx < n1 {
            &self.layer1[idx]
        } else if idx < n1 + n2 {
            &self.layer2[idx - n1]
        } else {
            &self.output_gate
        }
    }
}

impl Default for PerceptionKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Deep perception kernel with more layers for complex functions like GoL
///
/// Architecture:
/// - Layer 1: 32 gates with comprehensive neighborhood coverage
/// - Layer 2: 16 gates combining layer 1
/// - Layer 3: 8 gates combining layer 2
/// - Layer 4: 4 gates combining layer 3
/// - Layer 5: 2 gates combining layer 4
/// - Output: 1 gate
pub struct DeepPerceptionKernel {
    /// Layers of gates
    layers: Vec<Vec<ProbabilisticGate>>,
    /// Layer 1 connections (from 9 inputs)
    layer1_connections: Vec<(usize, usize)>,
    /// Subsequent layer connections
    layer_connections: Vec<Vec<(usize, usize)>>,
}

impl DeepPerceptionKernel {
    /// Create a new deep perception kernel
    pub fn new() -> Self {
        // Layer 1: All pairs of neighborhood cells (9 choose 2 = 36, we use 32)
        // Plus some critical center-based comparisons
        let layer1_connections: Vec<(usize, usize)> = vec![
            // Center vs all 8 neighbors
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 8),
            // Horizontal pairs
            (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
            // Vertical pairs
            (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8),
            // Diagonal pairs
            (0, 4), (4, 8), (2, 4), (4, 6),
            // Corner pairs
            (0, 8), (2, 6),
            // Edge pairs
            (0, 2), (6, 8), (0, 6), (2, 8),
            // Adjacent to corners
            (1, 3), (1, 5), (7, 3), (7, 5),
        ];
        let n1 = layer1_connections.len();

        // Subsequent layers halve in size
        let n2 = n1 / 2;
        let n3 = n2 / 2;
        let n4 = n3 / 2;
        let n5 = n4 / 2;

        // Create layers
        let mut layers = Vec::new();
        layers.push((0..n1).map(|_| ProbabilisticGate::new()).collect());
        layers.push((0..n2).map(|_| ProbabilisticGate::new()).collect());
        layers.push((0..n3).map(|_| ProbabilisticGate::new()).collect());
        layers.push((0..n4).map(|_| ProbabilisticGate::new()).collect());
        layers.push((0..n5).map(|_| ProbabilisticGate::new()).collect());
        layers.push(vec![ProbabilisticGate::new()]); // Output

        // Sequential pairing connections for each layer
        let mut layer_connections = Vec::new();
        for (i, layer) in layers.iter().skip(1).enumerate() {
            let prev_size = layers[i].len();
            let connections: Vec<(usize, usize)> = (0..layer.len())
                .map(|j| {
                    let a = (j * 2) % prev_size;
                    let b = (j * 2 + 1) % prev_size;
                    (a, b)
                })
                .collect();
            layer_connections.push(connections);
        }

        Self {
            layers,
            layer1_connections,
            layer_connections,
        }
    }

    /// Execute in soft mode
    pub fn execute_soft(&self, neighborhood: &[f64; 9]) -> f64 {
        // Layer 1
        let mut current: Vec<f64> = self.layers[0]
            .iter()
            .zip(self.layer1_connections.iter())
            .map(|(gate, &(a, b))| gate.execute_soft(neighborhood[a], neighborhood[b]))
            .collect();

        // Subsequent layers
        for (layer_idx, layer) in self.layers.iter().skip(1).enumerate() {
            let connections = &self.layer_connections[layer_idx];
            current = layer
                .iter()
                .zip(connections.iter())
                .map(|(gate, &(a, b))| {
                    let va = current.get(a).copied().unwrap_or(0.0);
                    let vb = current.get(b).copied().unwrap_or(0.0);
                    gate.execute_soft(va, vb)
                })
                .collect();
        }

        current[0]
    }

    /// Execute in hard mode
    pub fn execute_hard(&self, neighborhood: &[bool; 9]) -> bool {
        // Layer 1
        let mut current: Vec<bool> = self.layers[0]
            .iter()
            .zip(self.layer1_connections.iter())
            .map(|(gate, &(a, b))| gate.execute_hard(neighborhood[a], neighborhood[b]))
            .collect();

        // Subsequent layers
        for (layer_idx, layer) in self.layers.iter().skip(1).enumerate() {
            let connections = &self.layer_connections[layer_idx];
            current = layer
                .iter()
                .zip(connections.iter())
                .map(|(gate, &(a, b))| {
                    let va = *current.get(a).unwrap_or(&false);
                    let vb = *current.get(b).unwrap_or(&false);
                    gate.execute_hard(va, vb)
                })
                .collect();
        }

        current[0]
    }

    /// Total number of gates
    pub fn num_gates(&self) -> usize {
        self.layers.iter().map(|l| l.len()).sum()
    }

    /// Get mutable reference to gate by flat index
    pub fn gate_mut(&mut self, idx: usize) -> &mut ProbabilisticGate {
        let mut offset = 0;
        for layer in &mut self.layers {
            if idx < offset + layer.len() {
                return &mut layer[idx - offset];
            }
            offset += layer.len();
        }
        panic!("Gate index out of bounds");
    }

    /// Get reference to gate by flat index
    pub fn gate(&self, idx: usize) -> &ProbabilisticGate {
        let mut offset = 0;
        for layer in &self.layers {
            if idx < offset + layer.len() {
                return &layer[idx - offset];
            }
            offset += layer.len();
        }
        panic!("Gate index out of bounds");
    }

    /// Forward with activations for backprop
    fn forward_with_activations(&self, neighborhood: &[f64; 9]) -> Vec<Vec<f64>> {
        let mut activations = Vec::new();

        // Layer 1
        let layer1: Vec<f64> = self.layers[0]
            .iter()
            .zip(self.layer1_connections.iter())
            .map(|(gate, &(a, b))| gate.execute_soft(neighborhood[a], neighborhood[b]))
            .collect();
        activations.push(layer1);

        // Subsequent layers
        for (layer_idx, layer) in self.layers.iter().skip(1).enumerate() {
            let connections = &self.layer_connections[layer_idx];
            let prev = &activations[layer_idx];
            let current: Vec<f64> = layer
                .iter()
                .zip(connections.iter())
                .map(|(gate, &(a, b))| {
                    let va = prev.get(a).copied().unwrap_or(0.0);
                    let vb = prev.get(b).copied().unwrap_or(0.0);
                    gate.execute_soft(va, vb)
                })
                .collect();
            activations.push(current);
        }

        activations
    }
}

impl Default for DeepPerceptionKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Trainer for deep perception kernels
pub struct DeepPerceptionTrainer {
    pub kernel: DeepPerceptionKernel,
    optimizers: Vec<AdamW>,
    pub iteration: usize,
}

impl DeepPerceptionTrainer {
    pub fn new(kernel: DeepPerceptionKernel, learning_rate: f64) -> Self {
        let num_gates = kernel.num_gates();
        let optimizers = (0..num_gates).map(|_| AdamW::new(learning_rate)).collect();
        Self {
            kernel,
            optimizers,
            iteration: 0,
        }
    }

    pub fn default_trainer() -> Self {
        Self::new(DeepPerceptionKernel::new(), 0.05)
    }

    /// Train for one epoch
    pub fn train_epoch(&mut self, truth_table: &GolTruthTable) -> f64 {
        let num_gates = self.kernel.num_gates();
        let mut accumulated_grads: Vec<[f64; 16]> = vec![[0.0; 16]; num_gates];
        let mut total_loss = 0.0;

        for idx in 0..512 {
            let (neighborhood, target) = truth_table.example(idx);
            let soft_input = neighborhood.soft_cells();
            let target_f = if target { 1.0 } else { 0.0 };

            let activations = self.kernel.forward_with_activations(&soft_input);
            let output = activations.last().unwrap()[0];
            let error = output - target_f;
            total_loss += error * error;

            let grads = self.compute_gradients(&soft_input, &activations, target_f);
            for (i, g) in grads.iter().enumerate() {
                for j in 0..16 {
                    accumulated_grads[i][j] += g[j];
                }
            }
        }

        // Update
        let mut gate_idx = 0;
        for layer in &mut self.kernel.layers {
            for gate in layer {
                for j in 0..16 {
                    accumulated_grads[gate_idx][j] /= 512.0;
                }
                self.optimizers[gate_idx].step(&mut gate.logits, &accumulated_grads[gate_idx]);
                // Invalidate cached probabilities after logit update
                gate.invalidate_cache();
                gate_idx += 1;
            }
        }

        self.iteration += 1;
        total_loss / 512.0
    }

    /// Compute gradients using backpropagation
    fn compute_gradients(
        &self,
        input: &[f64; 9],
        activations: &[Vec<f64>],
        target: f64,
    ) -> Vec<[f64; 16]> {
        let num_layers = self.kernel.layers.len();
        let num_gates = self.kernel.num_gates();
        let mut all_grads = vec![[0.0; 16]; num_gates];

        // dL/doutput for MSE
        let output = activations.last().unwrap()[0];
        let d_l_doutput = 2.0 * (output - target);

        // Gradient flowing back
        let mut d_l_dlayer: Vec<f64> = vec![d_l_doutput];

        // Backprop through layers (reverse order)
        let mut gate_offset = num_gates;
        for layer_idx in (0..num_layers).rev() {
            let layer = &self.kernel.layers[layer_idx];
            gate_offset -= layer.len();

            let prev_activations = if layer_idx == 0 {
                input.to_vec()
            } else {
                activations[layer_idx - 1].clone()
            };

            let connections = if layer_idx == 0 {
                &self.kernel.layer1_connections
            } else {
                &self.kernel.layer_connections[layer_idx - 1]
            };

            let mut d_l_dprev = vec![0.0; prev_activations.len()];

            for (gate_i, gate) in layer.iter().enumerate() {
                let (a_idx, b_idx) = connections[gate_i];
                let a = prev_activations.get(a_idx).copied().unwrap_or(0.0);
                let b = prev_activations.get(b_idx).copied().unwrap_or(0.0);
                let d_l_dgate = d_l_dlayer.get(gate_i).copied().unwrap_or(0.0);

                // Gradient w.r.t. logits
                let logit_grads = self.compute_gate_logit_gradients(gate, a, b, d_l_dgate);
                all_grads[gate_offset + gate_i] = logit_grads;

                // Gradient w.r.t. inputs
                if layer_idx > 0 {
                    let (da, db) = self.compute_gate_input_gradients(gate, a, b);
                    if a_idx < d_l_dprev.len() {
                        d_l_dprev[a_idx] += d_l_dgate * da;
                    }
                    if b_idx < d_l_dprev.len() {
                        d_l_dprev[b_idx] += d_l_dgate * db;
                    }
                }
            }

            d_l_dlayer = d_l_dprev;
        }

        all_grads
    }

    fn compute_gate_logit_gradients(
        &self,
        gate: &ProbabilisticGate,
        a: f64,
        b: f64,
        d_l_doutput: f64,
    ) -> [f64; 16] {
        let probs = gate.probabilities();
        let mut gradients = [0.0; 16];

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
            gradients[i] = d_l_doutput * dlogit_i;
        }

        gradients
    }

    fn compute_gate_input_gradients(&self, gate: &ProbabilisticGate, a: f64, b: f64) -> (f64, f64) {
        let probs = gate.probabilities();
        let mut da = 0.0;
        let mut db = 0.0;

        for (i, &op) in BinaryOp::ALL.iter().enumerate() {
            let (op_da, op_db) = PerceptionTrainer::op_input_gradients(op, a, b);
            da += probs[i] * op_da;
            db += probs[i] * op_db;
        }

        (da, db)
    }

    /// Compute hard accuracy
    pub fn compute_accuracy(&self, truth_table: &GolTruthTable) -> f64 {
        let mut correct = 0;
        for idx in 0..512 {
            let (neighborhood, target) = truth_table.example(idx);
            let predicted = self.kernel.execute_hard(&neighborhood.cells);
            if predicted == target {
                correct += 1;
            }
        }
        correct as f64 / 512.0
    }

    /// Train until convergence
    pub fn train(
        &mut self,
        truth_table: &GolTruthTable,
        max_iterations: usize,
        target_loss: f64,
        verbose: bool,
    ) -> PerceptionTrainingResult {
        let mut losses = Vec::new();
        let mut converged = false;

        for i in 0..max_iterations {
            let loss = self.train_epoch(truth_table);
            losses.push(loss);

            if verbose && (i % 100 == 0 || i == max_iterations - 1) {
                let accuracy = self.compute_accuracy(truth_table);
                println!(
                    "Iter {:5}: Loss = {:.6}, Accuracy = {:.2}%",
                    i, loss, accuracy * 100.0
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
        let hard_accuracy = self.compute_accuracy(truth_table);

        PerceptionTrainingResult {
            converged,
            iterations: self.iteration,
            final_loss,
            hard_accuracy,
            losses,
        }
    }
}

/// Truth table for all 512 neighborhood configurations in Game of Life
pub struct GolTruthTable {
    /// For each of 512 configurations, the expected next state of center cell
    pub targets: [bool; 512],
}

impl GolTruthTable {
    /// Create truth table based on Game of Life rules
    pub fn new() -> Self {
        let mut targets = [false; 512];

        for idx in 0..512 {
            let neighborhood = Neighborhood::from_index(idx);
            targets[idx] = neighborhood.gol_next_state();
        }

        Self { targets }
    }

    /// Get target for a specific configuration
    pub fn target(&self, idx: usize) -> bool {
        self.targets[idx]
    }

    /// Get neighborhood and target for an index
    pub fn example(&self, idx: usize) -> (Neighborhood, bool) {
        let neighborhood = Neighborhood::from_index(idx);
        (neighborhood, self.targets[idx])
    }

    /// All examples as (neighborhood, target) pairs
    pub fn all_examples(&self) -> Vec<(Neighborhood, bool)> {
        (0..512).map(|i| self.example(i)).collect()
    }
}

impl Default for GolTruthTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Trainer for perception kernels
pub struct PerceptionTrainer {
    pub kernel: PerceptionKernel,
    /// One optimizer per gate
    optimizers: Vec<AdamW>,
    pub iteration: usize,
}

impl PerceptionTrainer {
    /// Create a new trainer
    pub fn new(kernel: PerceptionKernel, learning_rate: f64) -> Self {
        let num_gates = kernel.num_gates();
        let optimizers: Vec<AdamW> = (0..num_gates)
            .map(|_| AdamW::new(learning_rate))
            .collect();

        Self {
            kernel,
            optimizers,
            iteration: 0,
        }
    }

    /// Create a trainer with default kernel and learning rate
    pub fn default_trainer() -> Self {
        Self::new(PerceptionKernel::new(), 0.05)
    }

    /// Train for one epoch on all 512 configurations
    pub fn train_epoch(&mut self, truth_table: &GolTruthTable) -> f64 {
        let num_gates = self.kernel.num_gates();

        // Accumulate gradients across all examples
        let mut accumulated_grads: Vec<[f64; 16]> = vec![[0.0; 16]; num_gates];
        let mut total_loss = 0.0;

        for idx in 0..512 {
            let (neighborhood, target) = truth_table.example(idx);
            let soft_input = neighborhood.soft_cells();
            let target_f = if target { 1.0 } else { 0.0 };

            // Forward pass
            let (layer1_out, layer2_out, output) = self.kernel.forward_with_activations(&soft_input);

            // Compute loss
            let error = output - target_f;
            total_loss += error * error;

            // Backpropagation
            let grads = self.compute_gradients(
                &soft_input,
                &layer1_out,
                &layer2_out,
                output,
                target_f,
            );

            // Accumulate
            for (gate_idx, gate_grads) in grads.iter().enumerate() {
                for i in 0..16 {
                    accumulated_grads[gate_idx][i] += gate_grads[i];
                }
            }
        }

        // Average gradients and update
        for gate_idx in 0..num_gates {
            for i in 0..16 {
                accumulated_grads[gate_idx][i] /= 512.0;
            }

            self.optimizers[gate_idx].step(
                &mut self.kernel.gate_mut(gate_idx).logits,
                &accumulated_grads[gate_idx],
            );
            // Invalidate cached probabilities after logit update
            self.kernel.gate_mut(gate_idx).invalidate_cache();
        }

        self.iteration += 1;
        total_loss / 512.0
    }

    /// Compute gradients for all gates using backpropagation
    fn compute_gradients(
        &self,
        input: &[f64; 9],
        layer1_out: &[f64],
        layer2_out: &[f64],
        output: f64,
        target: f64,
    ) -> Vec<[f64; 16]> {
        let n1 = self.kernel.layer1.len();
        let n2 = self.kernel.layer2.len();
        let num_gates = n1 + n2 + 1;

        let mut all_grads: Vec<[f64; 16]> = vec![[0.0; 16]; num_gates];

        // dL/doutput for MSE
        let dL_doutput = 2.0 * (output - target);

        // Output gate gradients
        {
            let (a_idx, b_idx) = self.kernel.output_connections;
            let a = layer2_out.get(a_idx).copied().unwrap_or(0.0);
            let b = layer2_out.get(b_idx).copied().unwrap_or(0.0);

            let grads = self.compute_gate_logit_gradients(&self.kernel.output_gate, a, b, dL_doutput);
            all_grads[n1 + n2] = grads;
        }

        // Gradient flowing back to layer2 outputs
        let mut dL_dlayer2: Vec<f64> = vec![0.0; n2];
        {
            let (a_idx, b_idx) = self.kernel.output_connections;
            let a = layer2_out.get(a_idx).copied().unwrap_or(0.0);
            let b = layer2_out.get(b_idx).copied().unwrap_or(0.0);

            let (da, db) = self.compute_gate_input_gradients(&self.kernel.output_gate, a, b);
            if a_idx < n2 {
                dL_dlayer2[a_idx] += dL_doutput * da;
            }
            if b_idx < n2 {
                dL_dlayer2[b_idx] += dL_doutput * db;
            }
        }

        // Layer2 gate gradients
        let mut dL_dlayer1: Vec<f64> = vec![0.0; n1];
        for (gate_idx, gate) in self.kernel.layer2.iter().enumerate() {
            let (a_idx, b_idx) = self.kernel.layer2_connections[gate_idx];
            let a = layer1_out.get(a_idx).copied().unwrap_or(0.0);
            let b = layer1_out.get(b_idx).copied().unwrap_or(0.0);

            let dL_dgate_out = dL_dlayer2[gate_idx];

            // Gradients w.r.t. logits
            let grads = self.compute_gate_logit_gradients(gate, a, b, dL_dgate_out);
            all_grads[n1 + gate_idx] = grads;

            // Gradients w.r.t. inputs (for backprop to layer1)
            let (da, db) = self.compute_gate_input_gradients(gate, a, b);
            if a_idx < n1 {
                dL_dlayer1[a_idx] += dL_dgate_out * da;
            }
            if b_idx < n1 {
                dL_dlayer1[b_idx] += dL_dgate_out * db;
            }
        }

        // Layer1 gate gradients
        for (gate_idx, gate) in self.kernel.layer1.iter().enumerate() {
            let (a_idx, b_idx) = self.kernel.topology.connections[gate_idx];
            let a = input[a_idx];
            let b = input[b_idx];

            let dL_dgate_out = dL_dlayer1[gate_idx];

            let grads = self.compute_gate_logit_gradients(gate, a, b, dL_dgate_out);
            all_grads[gate_idx] = grads;
        }

        all_grads
    }

    /// Compute gradients of loss w.r.t. gate logits
    fn compute_gate_logit_gradients(
        &self,
        gate: &ProbabilisticGate,
        a: f64,
        b: f64,
        dL_doutput: f64,
    ) -> [f64; 16] {
        let probs = gate.probabilities();
        let mut gradients = [0.0; 16];

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
            gradients[i] = dL_doutput * dlogit_i;
        }

        gradients
    }

    /// Compute gradients of gate output w.r.t. inputs
    fn compute_gate_input_gradients(&self, gate: &ProbabilisticGate, a: f64, b: f64) -> (f64, f64) {
        let probs = gate.probabilities();
        let mut da = 0.0;
        let mut db = 0.0;

        for (i, &op) in BinaryOp::ALL.iter().enumerate() {
            let (op_da, op_db) = Self::op_input_gradients(op, a, b);
            da += probs[i] * op_da;
            db += probs[i] * op_db;
        }

        (da, db)
    }

    /// Gradients of operations w.r.t. inputs (copied from phase_0_3)
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

    /// Compute hard accuracy on all 512 configurations
    pub fn compute_accuracy(&self, truth_table: &GolTruthTable) -> f64 {
        let mut correct = 0;

        for idx in 0..512 {
            let (neighborhood, target) = truth_table.example(idx);
            let predicted = self.kernel.execute_hard(&neighborhood.cells);

            if predicted == target {
                correct += 1;
            }
        }

        correct as f64 / 512.0
    }

    /// Train until convergence or max iterations
    pub fn train(
        &mut self,
        truth_table: &GolTruthTable,
        max_iterations: usize,
        target_loss: f64,
        verbose: bool,
    ) -> PerceptionTrainingResult {
        let mut losses = Vec::new();
        let mut converged = false;

        for i in 0..max_iterations {
            let loss = self.train_epoch(truth_table);
            losses.push(loss);

            if verbose && (i % 100 == 0 || i == max_iterations - 1) {
                let accuracy = self.compute_accuracy(truth_table);
                println!(
                    "Iter {:5}: Loss = {:.6}, Accuracy = {:.2}%",
                    i,
                    loss,
                    accuracy * 100.0
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
        let hard_accuracy = self.compute_accuracy(truth_table);

        PerceptionTrainingResult {
            converged,
            iterations: self.iteration,
            final_loss,
            hard_accuracy,
            losses,
        }
    }
}

/// Result of perception training
#[derive(Debug)]
pub struct PerceptionTrainingResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_loss: f64,
    pub hard_accuracy: f64,
    pub losses: Vec<f64>,
}

impl PerceptionTrainingResult {
    /// Check if exit criteria are met (>95% accuracy)
    pub fn meets_exit_criteria(&self) -> bool {
        self.hard_accuracy > 0.95
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_grid_creation() {
        let grid = Grid::new(10, 10);
        assert_eq!(grid.width, 10);
        assert_eq!(grid.height, 10);
        assert_eq!(grid.cells.len(), 100);
    }

    #[test]
    fn test_grid_wrapping() {
        let mut grid = Grid::new(3, 3);
        grid.set(0, 0, true);

        // Test wrapping
        assert!(grid.get(0, 0));
        assert!(grid.get(-3, 0));  // Should wrap to (0, 0)
        assert!(grid.get(3, 0));   // Should wrap to (0, 0)
        assert!(grid.get(0, -3));  // Should wrap to (0, 0)
    }

    #[test]
    fn test_neighborhood_extraction() {
        let mut grid = Grid::new(5, 5);
        // Set center and some neighbors
        grid.set(2, 2, true);  // Center
        grid.set(1, 1, true);  // NW
        grid.set(3, 3, true);  // SE

        let n = grid.neighborhood(2, 2);

        // [NW, N, NE, W, C, E, SW, S, SE]
        assert_eq!(n[0], true);  // NW
        assert_eq!(n[4], true);  // C
        assert_eq!(n[8], true);  // SE
        assert_eq!(n[1], false); // N
    }

    #[test]
    fn test_neighborhood_from_index() {
        // Test all corners
        let n0 = Neighborhood::from_index(0);
        assert!(n0.cells.iter().all(|&c| !c));

        let n511 = Neighborhood::from_index(511);
        assert!(n511.cells.iter().all(|&c| c));

        // Test roundtrip
        for idx in 0..512 {
            let n = Neighborhood::from_index(idx);
            assert_eq!(n.to_index(), idx);
        }
    }

    #[test]
    fn test_gol_rules() {
        // Dead cell with 3 neighbors -> alive
        let n = Neighborhood::from_index(0b000_010_111);  // 3 neighbors in bottom row
        // Wait, let me recalculate. Index: [NW=0, N=1, NE=2, W=3, C=4, E=5, SW=6, S=7, SE=8]
        // For 3 alive neighbors, say NW, N, NE (bits 0,1,2):
        let n = Neighborhood::from_index(0b000000111);  // NW, N, NE alive, center dead
        assert_eq!(n.neighbor_count(), 3);
        assert!(!n.center());
        assert!(n.gol_next_state());  // Should become alive

        // Alive cell with 2 neighbors -> survives
        let n = Neighborhood::from_index(0b000010011);  // NW, N alive, center alive
        assert_eq!(n.neighbor_count(), 2);
        assert!(n.center());
        assert!(n.gol_next_state());  // Should survive

        // Alive cell with 1 neighbor -> dies
        let n = Neighborhood::from_index(0b000010001);  // NW alive, center alive
        assert_eq!(n.neighbor_count(), 1);
        assert!(n.center());
        assert!(!n.gol_next_state());  // Should die

        // Alive cell with 4 neighbors -> dies (overcrowding)
        let n = Neighborhood::from_index(0b000110111);  // NW,N,NE,W alive, center alive
        assert_eq!(n.neighbor_count(), 4);
        assert!(n.center());
        assert!(!n.gol_next_state());  // Should die
    }

    #[test]
    fn test_gol_truth_table() {
        let tt = GolTruthTable::new();

        // Verify some known cases
        // All dead -> dead
        assert!(!tt.target(0));

        // All alive except center (8 neighbors) -> center becomes alive (overcrowding avoidance)
        // Wait, GoL: dead + 3 neighbors -> alive. 8 neighbors != 3
        let idx = 0b111111111 ^ (1 << 4);  // All alive except center
        let n = Neighborhood::from_index(idx);
        assert_eq!(n.neighbor_count(), 8);
        assert!(!n.center());
        assert!(!tt.target(idx));  // 8 neighbors, not 3, so stays dead

        // Count total alive targets
        let alive_count = tt.targets.iter().filter(|&&t| t).count();
        assert!(alive_count > 0);
        assert!(alive_count < 512);
    }

    #[test]
    fn test_perception_topology() {
        let t = PerceptionTopology::first_kernel();
        assert_eq!(t.num_gates(), 16);

        let t_min = PerceptionTopology::minimal();
        assert_eq!(t_min.num_gates(), 4);
    }

    #[test]
    fn test_perception_kernel_creation() {
        let k = PerceptionKernel::new();
        assert_eq!(k.num_gates(), 16 / 2 + 16 + 1);  // layer2 + layer1 + output

        let k_min = PerceptionKernel::minimal();
        // 4 layer1 gates, 2 layer2 gates, 1 output gate
        assert_eq!(k_min.num_gates(), 4 + 2 + 1);
    }

    #[test]
    fn test_perception_kernel_execution() {
        let k = PerceptionKernel::minimal();

        // All zeros
        let input = [0.0; 9];
        let output = k.execute_soft(&input);
        assert!(output >= 0.0 && output <= 1.0);

        // All ones
        let input = [1.0; 9];
        let output = k.execute_soft(&input);
        assert!(output >= 0.0 && output <= 1.0);

        // Hard execution
        let input = [false; 9];
        let _output = k.execute_hard(&input);
    }

    #[test]
    fn test_numerical_gradients_perception() {
        let mut trainer = PerceptionTrainer::new(PerceptionKernel::minimal(), 0.05);
        let truth_table = GolTruthTable::new();

        let epsilon = 1e-5;

        // Test on a few examples
        for idx in [0, 127, 256, 511] {
            let (neighborhood, target) = truth_table.example(idx);
            let soft_input = neighborhood.soft_cells();
            let target_f = if target { 1.0 } else { 0.0 };

            // Compute analytical gradients
            let (layer1_out, layer2_out, output) = trainer.kernel.forward_with_activations(&soft_input);
            let analytical = trainer.compute_gradients(&soft_input, &layer1_out, &layer2_out, output, target_f);

            // Test first gate
            for logit_idx in 0..16 {
                let original = trainer.kernel.gate_mut(0).logits[logit_idx];

                // +epsilon
                trainer.kernel.gate_mut(0).logits[logit_idx] = original + epsilon;
                trainer.kernel.gate_mut(0).invalidate_cache();
                let out_plus = trainer.kernel.execute_soft(&soft_input);
                let loss_plus = (out_plus - target_f).powi(2);

                // -epsilon
                trainer.kernel.gate_mut(0).logits[logit_idx] = original - epsilon;
                trainer.kernel.gate_mut(0).invalidate_cache();
                let out_minus = trainer.kernel.execute_soft(&soft_input);
                let loss_minus = (out_minus - target_f).powi(2);

                // Restore
                trainer.kernel.gate_mut(0).logits[logit_idx] = original;
                trainer.kernel.gate_mut(0).invalidate_cache();

                let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);

                assert_relative_eq!(
                    analytical[0][logit_idx],
                    numerical,
                    epsilon = 1e-3,
                    max_relative = 0.1
                );
            }
        }
    }
}
