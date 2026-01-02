# Logicars - Differentiable Logic Cellular Automata

## Getting Started

**IMPORTANT**: Before starting any work on this project, always read the documents in the `agents/` folder if it exists:

1. **`agents/plan.md`** - Full development roadmap and phase requirements
2. **`agents/implementation-log.md`** - Implementation history, learnings, and what's next

This ensures you understand the current project state, what has been accomplished, and what needs to be done next.

### If the `agents/` folder doesn't exist

Create it and add the two key documents:

```bash
mkdir agents
```

#### 1. Create `agents/plan.md` (Development Roadmap)

This document should contain:

- **Project overview**: What you're building and primary references (papers, implementations)
- **Phased development plan**: Break the work into distinct phases with clear goals
- **Exit criteria for each phase**: Specific, measurable conditions that must be met before proceeding
- **Key implementation details**: Important algorithms, hyperparameters, architectural decisions
- **Risk mitigation strategies**: What could go wrong and how to handle it
- **Critical success factors**: Principles to follow (e.g., "verify before proceeding", "test first")

Structure example:
```markdown
# Project Name Development Roadmap

## Primary References
- [Link to paper/docs]
- [Link to reference implementation]

## Phase 0: Foundation
**Goal**: [What this phase accomplishes]

### 0.1 First Component
- What to implement
- **Exit criteria**: Specific tests/metrics that must pass

### 0.2 Second Component
...

## Phase 1: Core Feature
...
```

#### 2. Create `agents/implementation-log.md` (Progress & Learnings)

This document should contain:

- **Development workflow**: The proven pattern for completing phases
- **Phase completion records**: For each completed phase:
  - Date and status
  - What was implemented (with file locations)
  - Test results and metrics
  - Exit criteria verification (✅ or ❌)
  - Key technical decisions and rationale
  - Important learnings and gotchas
  - Commands for next developer
- **Code organization**: Current project structure
- **Common patterns**: Reusable code patterns (testing, training, etc.)
- **Questions for later**: Issues to address in future phases
- **Next steps**: Clear direction for what comes next

Structure example:
```markdown
# Project Name Implementation Log

## Development Workflow (PROVEN PATTERN)
1. Create Todo List
2. Write unit tests first
...

## Phase 0.1: First Component ✅ COMPLETE

**Date**: YYYY-MM-DD
**Status**: ALL EXIT CRITERIA MET

### What Was Implemented
...

### Test Results
...

### Exit Criteria: ✅ ALL MET
- ✅ Criterion 1
- ✅ Criterion 2
...

### Key Technical Decisions
...

### Important Learnings
...

### Commands for Next Developer
```bash
# How to run tests
# How to verify this phase
```

## Next Steps
**Phase X.X**: Description of what's next
```

#### Why These Documents Matter

- **`plan.md`** is your north star - it prevents scope creep and ensures each phase has clear completion criteria
- **`implementation-log.md`** is institutional memory - it captures what worked, what didn't, and why decisions were made
- Together they enable any developer (human or AI) to pick up the project and continue effectively

#### Maintaining These Documents

As the project progresses, these documents can become unwieldy. **Periodically review and compact them**:

**For `implementation-log.md`:**
- Completed phases can be summarized once stable (keep key learnings, remove verbose details)
- Collapse multiple phase sections into summary tables when appropriate
- Archive very old detailed logs to a separate `agents/archive/` folder if needed
- The most recent 2-3 phases should remain detailed; older phases can be condensed

**For `plan.md`:**
- Remove or update phases that no longer apply due to architectural changes
- Mark completed phases with ✅ and collapse their details
- Update exit criteria if experience shows they were unrealistic or need adjustment
- Remove speculative future phases that are no longer relevant

**When to compact:**
- When documents exceed ~500 lines and become hard to scan
- When major architectural decisions invalidate earlier plans
- When starting a new major phase (good time to archive the old)
- When multiple sessions have added incremental updates that can be consolidated

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

The `archive/` folder contains old, outdated implementations kept for documentation purposes only.

### Key Implementation Details

- **Soft decoding**: `softmax(weights)` during training (differentiable)
- **Hard decoding**: `argmax(weights)` during inference (discrete)
- **Pass-through gate**: Initialized to logit=10.0 for training stability
- **AdamW optimizer**: LR=0.05, gradient clipping=100.0
- **All 16 binary operations** are available via `BinaryOp` enum

### Current Status

See `agents/implementation-log.md` for detailed progress. The project follows a phased approach from single gates to full CA training.

## Development Workflow

1. Read phase requirements from `agents/plan.md`
2. Create TodoWrite list with specific tasks
3. Write unit tests for core functionality first
4. Implement core logic
5. Run `cargo test --lib` continuously
6. Create integration test binary if needed
7. Verify all exit criteria met
8. Update `agents/implementation-log.md`
9. Commit with detailed message
10. Push to branch and create PR

use the gh cli utility to manage interactions with Github