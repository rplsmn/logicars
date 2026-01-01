# Archive

This folder contains old and outdated implementations that are kept for documentation and archiving purposes only.

**These files are not actively maintained and should not be used in new development.**

## Archived Files

| File | Description | Last Modified |
|------|-------------|---------------|
| `difflogicca.rs` | Original DiffLogicCA implementation | 2025-03-18 |
| `circuits.rs` | Original circuits implementation | 2025-03-18 |
| `python_bindings.rs` | Python bindings (PyO3) | 2025-03-18 |
| `logic_gates.rs` | Original logic gates implementation | 2025-03-18 |
| `pyproject.toml` | Python project configuration | 2025-03-12 |
| `loss_evolution.png` | Training loss visualization | 2025-03-16 |
| `train_optimized.py` | Python training script | 2025-03-16 |

## Why Archived?

These files have not received any updates since March 2025. The active development has moved to a phased implementation approach (see `src/phase_*.rs` files) which supersedes this original code.

## Current Implementation

For the current active implementation, see:
- `src/phase_0_1.rs` - Single Gate Training
- `src/phase_0_2.rs` - Gate Layer
- `src/phase_0_3.rs` - Multi-Layer Circuits
- `src/phase_1_1.rs` - Perception Circuit
- `src/trainer.rs` - Training infrastructure
- `src/optimizer.rs` - AdamW optimizer

Refer to `claude/plan.md` and `claude/implementation-log.md` for the full development roadmap.
