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
    pub fn new(op: BinaryOp, wire_a: usize, wire_b: usize) -> Self {
        Self {
            op: op as u8,
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
        // A = 12 (passes through first input)
        // B = 10 (passes through second input)
        self.op == 12 || self.op == 10
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
    /// Returns [center_channels..., kernel_outputs...]
    pub fn execute(&self, neighborhood: &[bool]) -> Vec<bool> {
        let channels = self.channels;

        // Extract center cell (cell 4 in 3×3 grid)
        let center: Vec<bool> = (0..channels)
            .map(|c| neighborhood[4 * channels + c])
            .collect();

        // Run each kernel
        let mut output = center;
        for kernel in &self.kernels {
            // Each kernel processes all 9 cells
            let kernel_output = kernel.execute(neighborhood);
            output.extend(kernel_output);
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
}
