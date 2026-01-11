//! Logicars - Differentiable Logic Cellular Automata
//!
//! A Rust implementation of Google Research's differentiable logic CA paper.

// Core gate primitives (BinaryOp, ProbabilisticGate)
pub mod gates;
pub mod optimizer;

// N-bit Grid
pub mod grid;

// Perception Module
pub mod perception;

// Update Module
pub mod update;

// Training Loop
pub mod training;

// Checkerboard task
pub mod checkerboard;

// Hard Circuit Export
pub mod circuit;

// GPU acceleration
#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export key types
pub use gates::{BinaryOp, ProbabilisticGate};
pub use optimizer::AdamW;
pub use grid::{NGrid, NNeighborhood, BoundaryCondition};
pub use perception::{
    PerceptionModule,
    PerceptionKernel,
    PerceptionTrainer,
    ConnectionType,
    Wires,
    GateLayer,
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