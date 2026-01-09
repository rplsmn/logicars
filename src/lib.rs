//! Logicars - Differentiable Logic Cellular Automata
//!
//! A Rust implementation of Google Research's differentiable logic CA paper.
//!
//! This library is being built from first principles, starting with Phase 0.1.

// Phase 0.1: Single Gate Training
pub mod phase_0_1;
pub mod optimizer;
pub mod trainer;

// Phase 0.2: Gate Layer
pub mod phase_0_2;

// Phase 0.3: Multi-Layer Circuits
pub mod phase_0_3;

// Phase 1.1: Perception Circuit
pub mod phase_1_1;

// N-bit Grid (Phase 1.1 refactor)
pub mod grid;

// Phase 1.2: Perception Module
pub mod perception;

// Phase 1.3: Update Module
pub mod update;

// Phase 1.4: Training Loop
pub mod training;

// Phase 2.1: Checkerboard
pub mod checkerboard;

// Hard Circuit Export (Phase 3.2)
pub mod circuit;

// GPU acceleration (Phase 4.5)
#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export key types
pub use phase_0_1::{BinaryOp, ProbabilisticGate};
pub use optimizer::AdamW;
pub use trainer::{GateTrainer, TrainingResult, TruthTable};
pub use phase_0_2::{GateLayer, LayerTrainer, LayerTrainingResult, LayerTruthTable};
pub use phase_0_3::{Circuit, CircuitTrainer, CircuitTrainingResult, CircuitTruthTable, ConnectionPattern};
pub use phase_1_1::{Grid, Neighborhood, GolTruthTable, PerceptionKernel, PerceptionTrainer, PerceptionTrainingResult, PerceptionTopology, DeepPerceptionKernel, DeepPerceptionTrainer};
pub use grid::{NGrid, NNeighborhood, BoundaryCondition};
pub use perception::{
    PerceptionModule,
    PerceptionKernel as NPerceptionKernel,  // N-bit version
    PerceptionTrainer as NPerceptionTrainer,  // N-bit version
    ConnectionType,
    Wires,
    GateLayer as NbitGateLayer,  // N-bit version
};
pub use update::{
    UpdateModule,
    UpdateTrainer,
    DiffLogicCA,
    DiffLogicCATrainer,
};
pub use training::{
    TrainingConfig,
    TrainingLoop,
    SimpleRng,
    FIRE_RATE,
    GRADIENT_CLIP,
};
pub use checkerboard::{
    create_checkerboard,
    create_random_seed,
    create_checkerboard_perception,
    create_checkerboard_update,
    create_checkerboard_model,
    create_small_checkerboard_model,
    compute_checkerboard_loss,
    compute_checkerboard_accuracy,
    CHECKERBOARD_CHANNELS,
    CHECKERBOARD_KERNELS,
    CHECKERBOARD_GRID_SIZE,
    CHECKERBOARD_SQUARE_SIZE,
    CHECKERBOARD_SYNC_STEPS,
    CHECKERBOARD_ASYNC_STEPS,
};
pub use circuit::{
    HardGate,
    HardLayer,
    HardKernel,
    HardPerception,
    HardUpdate,
    HardCircuit,
};

// Python bindings (Phase 5.1 - not yet implemented)
// When implementing, create src/python_bindings.rs and uncomment cdylib in Cargo.toml
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn logicars(_py: Python, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO: Add Python bindings when Phase 5.1 is implemented
    Ok(())
}