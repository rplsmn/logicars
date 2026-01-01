# Logicars - Differentiable Logic Cellular Automata

## Getting Started

**IMPORTANT**: Before starting any work on this project, always read the documents in the `claude/` folder if it exists:

1. **`claude/plan.md`** - Full development roadmap and phase requirements
2. **`claude/implementation-log.md`** - Implementation history, learnings, and what's next

This ensures you understand the current project state, what has been accomplished, and what needs to be done next.

## Quick Reference

### Build & Test Commands

```bash
# Run all unit tests
cargo test --lib

# Run tests with output
cargo test --lib -- --nocapture

# Build release binaries
cargo build --release

# Run single gate training demo
cargo run --bin train_gate --release

# Run layer training demo
cargo run --bin train_layer --release
```

### Project Overview

This project implements differentiable logic gates for learning cellular automata rules (particularly Conway's Game of Life) based on the paper [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/).

### Key Implementation Details

- **Soft decoding**: `softmax(weights)` during training (differentiable)
- **Hard decoding**: `argmax(weights)` during inference (discrete)
- **Pass-through gate**: Initialized to logit=10.0 for training stability
- **AdamW optimizer**: LR=0.05, gradient clipping=100.0
- **All 16 binary operations** are available via `BinaryOp` enum

### Current Status

See `claude/implementation-log.md` for detailed progress. The project follows a phased approach from single gates to full CA training.

## Development Workflow

1. Read phase requirements from `claude/plan.md`
2. Create TodoWrite list with specific tasks
3. Write unit tests for core functionality first
4. Implement core logic
5. Run `cargo test --lib` continuously
6. Create integration test binary if needed
7. Verify all exit criteria met
8. Update `claude/implementation-log.md`
9. Commit with detailed message
10. Push to branch and create PR
