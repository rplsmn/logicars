//! Hard Circuit Export and Serialization
//!
//! This module provides functionality to export trained DiffLogicCA models
//! as hard (discrete) circuits with one fixed operation per gate.
//!
//! Key features:
//! - Convert soft (probabilistic) gates to hard (argmax) gates
//! - Count active gates (excluding pass-through operations)
//! - Serialize circuits for later use
//! - Run inference using pure discrete logic

use crate::grid::{BoundaryCondition, NGrid};
use crate::phase_0_1::BinaryOp;
use crate::perception::{GateLayer, PerceptionKernel, PerceptionModule, Wires};
use crate::update::{DiffLogicCA, UpdateModule};
use serde::{Deserialize, Serialize};

/// A single hard (discrete) gate with fixed operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardGate {
    /// The fixed binary operation for this gate
    pub op: u8, // BinaryOp as u8 for serialization
    /// Index of first input wire
    pub wire_a: usize,
    /// Index of second input wire
    pub wire_b: usize,
}

impl HardGate {
    /// Create a new hard gate
    /// NOTE: op should be the INDEX in BinaryOp::ALL (0-15), not the enum discriminant
    pub fn new(op: BinaryOp, wire_a: usize, wire_b: usize) -> Self {
        // Find the index of this operation in ALL
        let index = BinaryOp::ALL.iter().position(|&o| o == op).unwrap_or(0);
        Self {
            op: index as u8,
            wire_a,
            wire_b,
        }
    }

    /// Get the binary operation
    pub fn operation(&self) -> BinaryOp {
        BinaryOp::ALL[self.op as usize]
    }

    /// Execute the gate with boolean inputs
    pub fn execute(&self, inputs: &[bool]) -> bool {
        let a = inputs[self.wire_a];
        let b = inputs[self.wire_b];
        self.operation().execute(a, b)
    }

    /// Check if this gate is a pass-through (A or B)
    pub fn is_pass_through(&self) -> bool {
        // With reference ordering: A = index 3, B = index 5
        self.op == 3 || self.op == 5
    }
}

/// A layer of hard gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardLayer {
    pub gates: Vec<HardGate>,
}

impl HardLayer {
    /// Create from a soft GateLayer by taking argmax of each gate's logits
    pub fn from_soft(layer: &GateLayer) -> Self {
        let gates = layer
            .gates
            .iter()
            .enumerate()
            .map(|(i, gate)| {
                let (op, _prob) = gate.dominant_operation();
                HardGate::new(op, layer.wires.a[i], layer.wires.b[i])
            })
            .collect();
        Self { gates }
    }

    /// Execute the layer
    pub fn execute(&self, inputs: &[bool]) -> Vec<bool> {
        self.gates.iter().map(|g| g.execute(inputs)).collect()
    }

    /// Count active (non-pass-through) gates
    pub fn active_gate_count(&self) -> usize {
        self.gates.iter().filter(|g| !g.is_pass_through()).count()
    }
}

/// A hard kernel (multi-layer network)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardKernel {
    pub layers: Vec<HardLayer>,
    pub input_size: usize,
}

impl HardKernel {
    /// Create from a soft PerceptionKernel
    pub fn from_soft(kernel: &PerceptionKernel) -> Self {
        let layers = kernel.layers.iter().map(HardLayer::from_soft).collect();
        Self {
            layers,
            input_size: kernel.input_size,
        }
    }

    /// Execute the kernel
    pub fn execute(&self, inputs: &[bool]) -> Vec<bool> {
        let mut current = inputs.to_vec();
        for layer in &self.layers {
            current = layer.execute(&current);
        }
        current
    }

    /// Count active gates
    pub fn active_gate_count(&self) -> usize {
        self.layers.iter().map(|l| l.active_gate_count()).sum()
    }

    /// Total gate count
    pub fn total_gate_count(&self) -> usize {
        self.layers.iter().map(|l| l.gates.len()).sum()
    }
}

/// Hard perception module (multiple kernels)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardPerception {
    pub kernels: Vec<HardKernel>,
    pub channels: usize,
}

impl HardPerception {
    /// Create from soft PerceptionModule
    pub fn from_soft(module: &PerceptionModule) -> Self {
        let kernels = module.kernels.iter().map(HardKernel::from_soft).collect();
        Self {
            kernels,
            channels: module.channels,
        }
    }

    /// Execute perception on a neighborhood (9 cells × channels)
    /// Input format: [cell0_ch0, cell0_ch1, ..., cell8_ch0, cell8_ch1, ...]
    /// Returns: [center_ch0, center_ch1, ..., kernel_outputs in (c,s,k) order]
    pub fn execute(&self, neighborhood: &[bool]) -> Vec<bool> {
        let channels = self.channels;
        let num_kernels = self.kernels.len();

        // Determine output size per kernel (from first kernel)
        let kernel_output_size = if !self.kernels.is_empty() {
            // Output size is the number of gates in the final layer
            self.kernels[0].layers.last().map(|l| l.gates.len()).unwrap_or(0)
        } else {
            0
        };

        // Compute kernel outputs for each channel
        // kernel_outputs[k][c] = Vec<bool> of size kernel_output_size
        let mut kernel_outputs: Vec<Vec<Vec<bool>>> =
            vec![vec![Vec::new(); channels]; num_kernels];

        for c in 0..channels {
            // Extract channel c from all 9 cells
            let channel_inputs: Vec<bool> = (0..9)
                .map(|pos| neighborhood[pos * channels + c])
                .collect();

            for (k, kernel) in self.kernels.iter().enumerate() {
                kernel_outputs[k][c] = kernel.execute(&channel_inputs);
            }
        }

        // Build output with correct ordering: (c s k)
        let total_output_size = channels + channels * kernel_output_size * num_kernels;
        let mut output = Vec::with_capacity(total_output_size);

        // First: center cell values (cell 4)
        for c in 0..channels {
            output.push(neighborhood[4 * channels + c]);
        }

        // Then: for each channel, for each output bit, for each kernel
        for c in 0..channels {
            for s in 0..kernel_output_size {
                for k in 0..num_kernels {
                    output.push(kernel_outputs[k][c][s]);
                }
            }
        }

        output
    }

    /// Count active gates
    pub fn active_gate_count(&self) -> usize {
        self.kernels.iter().map(|k| k.active_gate_count()).sum()
    }

    /// Total gate count
    pub fn total_gate_count(&self) -> usize {
        self.kernels.iter().map(|k| k.total_gate_count()).sum()
    }
}

/// Hard update module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardUpdate {
    pub layers: Vec<HardLayer>,
    pub input_size: usize,
    pub output_size: usize,
}

impl HardUpdate {
    /// Create from soft UpdateModule
    pub fn from_soft(module: &UpdateModule) -> Self {
        let layers = module.layers.iter().map(HardLayer::from_soft).collect();
        Self {
            layers,
            input_size: module.input_size,
            output_size: module.output_channels,
        }
    }

    /// Execute update
    pub fn execute(&self, inputs: &[bool]) -> Vec<bool> {
        let mut current = inputs.to_vec();
        for layer in &self.layers {
            current = layer.execute(&current);
        }
        current
    }

    /// Count active gates
    pub fn active_gate_count(&self) -> usize {
        self.layers.iter().map(|l| l.active_gate_count()).sum()
    }

    /// Total gate count
    pub fn total_gate_count(&self) -> usize {
        self.layers.iter().map(|l| l.gates.len()).sum()
    }
}

/// Complete hard circuit (perception + update)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardCircuit {
    pub perception: HardPerception,
    pub update: HardUpdate,
    pub channels: usize,
}

impl HardCircuit {
    /// Create from soft DiffLogicCA model
    pub fn from_soft(model: &DiffLogicCA) -> Self {
        Self {
            perception: HardPerception::from_soft(&model.perception),
            update: HardUpdate::from_soft(&model.update),
            channels: model.perception.channels,
        }
    }

    /// Execute one step on a neighborhood
    /// Input: 9 cells × channels (flattened neighborhood)
    /// Output: channels (next state of center cell)
    pub fn execute(&self, neighborhood: &[bool]) -> Vec<bool> {
        let perception_output = self.perception.execute(neighborhood);
        self.update.execute(&perception_output)
    }

    /// Count active gates (excluding pass-through)
    pub fn active_gate_count(&self) -> usize {
        self.perception.active_gate_count() + self.update.active_gate_count()
    }

    /// Total gate count
    pub fn total_gate_count(&self) -> usize {
        self.perception.total_gate_count() + self.update.total_gate_count()
    }

    /// Get gate distribution (count of each operation type)
    pub fn gate_distribution(&self) -> [usize; 16] {
        let mut counts = [0usize; 16];

        // Count perception gates
        for kernel in &self.perception.kernels {
            for layer in &kernel.layers {
                for gate in &layer.gates {
                    counts[gate.op as usize] += 1;
                }
            }
        }

        // Count update gates
        for layer in &self.update.layers {
            for gate in &layer.gates {
                counts[gate.op as usize] += 1;
            }
        }

        counts
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        std::fs::write(path, json)
    }

    /// Load from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }

    /// Run one step of inference on an entire grid.
    /// Uses non-periodic (zero-padded) boundaries.
    pub fn step(&self, grid: &NGrid) -> NGrid {
        let mut output = NGrid::new(
            grid.width,
            grid.height,
            self.channels,
            BoundaryCondition::NonPeriodic,
        );

        // Process each cell
        for y in 0..grid.height {
            for x in 0..grid.width {
                // Extract 3×3 neighborhood as booleans (threshold at 0.5)
                let neighborhood = self.extract_neighborhood(grid, x, y);

                // Execute the circuit
                let new_state = self.execute(&neighborhood);

                // Write output (convert bool back to f64)
                for c in 0..self.channels {
                    output.set(x, y, c, if new_state[c] { 1.0 } else { 0.0 });
                }
            }
        }

        output
    }

    /// Run multiple steps of inference
    pub fn run_steps(&self, input: &NGrid, steps: usize) -> NGrid {
        let mut current = input.clone();
        for _ in 0..steps {
            current = self.step(&current);
        }
        current
    }

    /// Extract 3×3 neighborhood as boolean vector (9 cells × channels)
    fn extract_neighborhood(&self, grid: &NGrid, x: usize, y: usize) -> Vec<bool> {
        let mut result = Vec::with_capacity(9 * self.channels);

        // Reading order: NW, N, NE, W, C, E, SW, S, SE (matches NNeighborhood)
        let offsets: [(isize, isize); 9] = [
            (-1, -1), (0, -1), (1, -1),  // NW, N, NE
            (-1, 0),  (0, 0),  (1, 0),   // W, C, E
            (-1, 1),  (0, 1),  (1, 1),   // SW, S, SE
        ];

        for (dx, dy) in offsets {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            for c in 0..self.channels {
                let val = grid.get(nx, ny, c);
                result.push(val > 0.5);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::ConnectionType;

    #[test]
    fn test_hard_gate_creation() {
        let gate = HardGate::new(BinaryOp::And, 0, 1);
        assert_eq!(gate.operation(), BinaryOp::And);
        assert!(!gate.is_pass_through());
    }

    #[test]
    fn test_hard_gate_pass_through() {
        let gate_a = HardGate::new(BinaryOp::A, 0, 1);
        let gate_b = HardGate::new(BinaryOp::B, 0, 1);
        let gate_and = HardGate::new(BinaryOp::And, 0, 1);

        assert!(gate_a.is_pass_through());
        assert!(gate_b.is_pass_through());
        assert!(!gate_and.is_pass_through());
    }

    #[test]
    fn test_hard_gate_execute() {
        let gate = HardGate::new(BinaryOp::And, 0, 1);
        let inputs = vec![true, true, false];

        assert!(gate.execute(&inputs)); // true AND true = true
    }

    #[test]
    fn test_hard_layer_from_soft() {
        let wires = Wires {
            a: vec![0, 1],
            b: vec![1, 0],
        };
        let soft_layer = GateLayer::new(2, wires);
        let hard_layer = HardLayer::from_soft(&soft_layer);

        assert_eq!(hard_layer.gates.len(), 2);
    }

    #[test]
    fn test_hard_kernel_from_soft() {
        let soft_kernel = PerceptionKernel::new(
            &[4, 2, 1],
            &[ConnectionType::Unique, ConnectionType::Unique],
        );
        let hard_kernel = HardKernel::from_soft(&soft_kernel);

        assert_eq!(hard_kernel.layers.len(), 2);
        assert_eq!(hard_kernel.total_gate_count(), 3); // 2 + 1
    }

    #[test]
    fn test_hard_circuit_from_model() {
        // Create a small model with proper layer dimensions
        // unique_connections requires out_dim * 2 >= in_dim
        let perception = PerceptionModule::new(
            1,
            2,
            &[9, 8, 4, 2, 1],  // Gradual reduction respecting unique constraint
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique, ConnectionType::Unique],
        );
        let update = crate::update::UpdateModule::new(&[3, 2, 1]);
        let model = DiffLogicCA::new(perception, update);

        let circuit = HardCircuit::from_soft(&model);

        assert_eq!(circuit.channels, 1);
        assert!(circuit.total_gate_count() > 0);
    }

    #[test]
    fn test_hard_circuit_serialization() {
        let perception = PerceptionModule::new(
            1,
            2,
            &[9, 8, 4, 2, 1],
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique, ConnectionType::Unique],
        );
        let update = crate::update::UpdateModule::new(&[3, 2, 1]);
        let model = DiffLogicCA::new(perception, update);
        let circuit = HardCircuit::from_soft(&model);

        // Serialize
        let json = circuit.to_json().unwrap();
        assert!(json.contains("perception"));
        assert!(json.contains("update"));

        // Deserialize
        let loaded = HardCircuit::from_json(&json).unwrap();
        assert_eq!(loaded.channels, circuit.channels);
        assert_eq!(loaded.total_gate_count(), circuit.total_gate_count());
    }

    #[test]
    fn test_gate_distribution() {
        let perception = PerceptionModule::new(
            1,
            2,
            &[9, 8, 4, 2, 1],
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique, ConnectionType::Unique],
        );
        let update = crate::update::UpdateModule::new(&[3, 2, 1]);
        let model = DiffLogicCA::new(perception, update);
        let circuit = HardCircuit::from_soft(&model);

        let dist = circuit.gate_distribution();
        let total: usize = dist.iter().sum();
        assert_eq!(total, circuit.total_gate_count());
    }

    #[test]
    fn test_active_gate_count() {
        // Most gates start as pass-through (A), so active count should be low
        let perception = PerceptionModule::new(
            1,
            2,
            &[9, 8, 4, 2, 1],
            &[ConnectionType::FirstKernel, ConnectionType::Unique, ConnectionType::Unique, ConnectionType::Unique],
        );
        let update = crate::update::UpdateModule::new(&[3, 2, 1]);
        let model = DiffLogicCA::new(perception, update);
        let circuit = HardCircuit::from_soft(&model);

        // Before training, most gates are pass-through (A)
        // So active count should be less than total
        assert!(circuit.active_gate_count() <= circuit.total_gate_count());
    }

    #[test]
    fn test_hard_circuit_multichannel() {
        // Test circuit export with multi-channel model (like checkerboard)
        use crate::checkerboard::{create_small_checkerboard_model};
        
        let model = create_small_checkerboard_model();
        let circuit = HardCircuit::from_soft(&model);

        // Verify multi-channel properties
        assert_eq!(circuit.channels, 8);
        assert!(circuit.total_gate_count() > 0);
        
        // Serialization should work with multi-channel
        let json = circuit.to_json().unwrap();
        let loaded = HardCircuit::from_json(&json).unwrap();
        assert_eq!(loaded.channels, 8);
        assert_eq!(loaded.total_gate_count(), circuit.total_gate_count());
    }

    #[test]
    fn test_hard_circuit_grid_inference() {
        use crate::checkerboard::create_small_checkerboard_model;
        use crate::grid::NGrid;

        let model = create_small_checkerboard_model();
        let circuit = HardCircuit::from_soft(&model);

        // Create a small test grid (4×4, 8 channels)
        let input = NGrid::non_periodic(4, 4, 8);

        // Run one step - should not panic
        let output = circuit.step(&input);
        assert_eq!(output.width, 4);
        assert_eq!(output.height, 4);

        // Run multiple steps
        let output2 = circuit.run_steps(&input, 3);
        assert_eq!(output2.width, 4);
        assert_eq!(output2.height, 4);
    }
}
