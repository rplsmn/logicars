//! Logicars - Differentiable Logic Cellular Automata
//!
//! A Rust implementation of Google Research's differentiable logic CA paper.
//!
//! This library is being built from first principles, starting with Phase 0.1.

// Phase 0.1: Single Gate Training
pub mod phase_0_1;
pub mod optimizer;
pub mod trainer;

// Re-export key types
pub use phase_0_1::{BinaryOp, ProbabilisticGate};
pub use optimizer::AdamW;
pub use trainer::{GateTrainer, TrainingResult, TruthTable};

// Old modules (from failed attempt - kept for reference)
#[cfg(feature = "python")]
mod python_bindings;

#[cfg(feature = "python")]
use pyo3::prelude::*;

// The PyO3 module definition (only when python feature is enabled)
#[cfg(feature = "python")]
#[pymodule]
fn logicars(_py: Python, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO: Add Python bindings for Phase 0.1 once it's working
    Ok(())
}