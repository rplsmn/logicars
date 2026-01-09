# Logicars - Differentiable Logic Cellular Automata

## Getting Started

**IMPORTANT**: Before starting any work on this project, always read the documents in the `agents/` folder if it exists:

1. **`agents/INDEX.md`** - **READ FIRST** - Code index with file:line references. Search here before grepping codebase.
2. **`agents/plan.md`** - Full development roadmap and phase requirements
3. **`agents/implementation-log.md`** - Implementation state, learnings, and what's next
4. **`agents/qa-review.md`** - Latest QA review with recommendations and blockers
5. **`plans/INDEX.md`** - Implementation plans (performance, GPU, serialization) - read only when working on specific subsystems

**Token-saving tips**:

- **Code navigation**: Use `agents/INDEX.md` to find functions by name/purpose, then use the file:line references to view directly. Avoid grepping the full codebase unless the index doesn't have what you need.
- **Planning documents**: Use `plans/INDEX.md` to find implementation plans (performance, GPU, serialization). Only read full plan documents when working on that specific subsystem.
- **Completed work**: Don't re-read completed phase plans unless debugging. Check `agents/plan.md` for phase status (✅/🚧/⬜).
- **Current focus**: Always read `agents/implementation-log.md` first - it has the current task and next steps in <150 lines.

This ensures you understand the current project state, what has been accomplished, and what needs to be done next.

#### 2. Create `agents/implementation-log.md` (Progress & Learnings)

Structure example:

```markdown
# Project Name Implementation Log

## Development Workflow (PROVEN PATTERN)
1. Create TodoList and, if on main, create a branch
2. Write unit tests first
3. Work until all success / exit criteria are met and tests all pass
4. Compare implementations to reference intent (paper) and code (reference/.py or reference/*.ipynb)
5. Commit every time something new works, meaning the 4 previous steps are complete, push it and open a PR if not yet opened
...

## Phase 0.1: First Component
### Task : Description of atomic task

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
- Completed phases can be removed once stable (keep key learnings if useful for later phases)
- Collapse multiple phase sections into summary tables when appropriate

**For `plan.md`:**
- Most of the time, doesn't need updatges
- Remove or update phases that no longer apply due to architectural changes
- Mark completed phases with ✅
- Update exit criteria if experience shows they were unrealistic or need adjustment
- Remove speculative future phases that are no longer relevant

**When to compact:**
- When documents exceed ~500 lines and become hard to scan
- When major architectural decisions invalidate earlier plans
- When starting a new major phase
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

### Key Implementation Details

- **Soft decoding**: `softmax(weights)` during training (differentiable)
- **Hard decoding**: `argmax(weights)` during inference (discrete)
- **Pass-through gate**: Initialized to logit=10.0 for training stability
- **AdamW optimizer**: LR=0.05, gradient clipping=100.0
- **All 16 binary operations** are available via `BinaryOp` enum

### Current Status

See `agents/implementation-log.md` for detailed progress. The project follows a phased approach from single gates to full CA training.

## LLM Agent Workflow

**Critical**: These documents go in EVERY context window. Keep them compact, clear, and unambiguous.

### The Development Loop

```
┌─────────────────────────────────────────────────────────┐
│ 1. READ DOCS                                            │
│    └─ AGENTS.md → plan.md → implementation-log.md      │
├─────────────────────────────────────────────────────────┤
│ 2. CREATE PHASE PLAN                                    │
│    └─ plans/phase-X.X-name.md (detailed tasks, tests)  │
├─────────────────────────────────────────────────────────┤
│ 3. IMPLEMENT                                            │
│    └─ TDD: tests first → code → verify exit criteria   │
├─────────────────────────────────────────────────────────┤
│ 4. PUSH PR                                              │
│    └─ Branch → PR → wait for human review              │
├─────────────────────────────────────────────────────────┤
│ 5. UPDATE DOCS (after human approval)                  │
│    ├─ Update implementation-log.md (keep <100 lines)   │
│    ├─ Mark phase complete in plan.md                   │
│    └─ Merge PR                                          │
└─────────────────────────────────────────────────────────┘
        │                                        │
        └────────── Loop back to step 1 ────────┘
```

### Document Maintenance Rules

**`agents/plan.md`** (Can be longer, ~500 lines):

- Full roadmap, all phases, exit criteria
- Update when: phase completes, requirements change, major architectural shift
- Mark phases with ✅/🚧/⬜ status
- Keep technical details (hyperparameters, architectures)

**`agents/implementation-log.md`** (MUST stay <100 lines):

- **Purpose**: Prime LLM on current state and boundaries only
- **NOT a detailed plan** - that goes in `plans/phase-X.X-name.md`
- Update when: phase completes, boundaries change
- Remove: completed phase details (keep only status table)
- Keep: current phase pointer, boundaries (what NOT to do), critical learnings

**After each phase completion**:

1. Update implementation-log.md: mark phase done, update current phase, check line count
2. Update plan.md: mark phase ✅, update status table if needed
3. Archive detailed work in git commits (don't bloat the log)

### Quick Development Steps

1. Read phase requirements from `agents/plan.md`

IF `plans/phase-X.X-name.md` for the current phase DOESN'T EXIST

1. Create detailed implementation plan in `plans/phase-X.X-name.md`
2. Commit with detailed message to plan/ branch (never to main)
3. Push branch and create PR
4. After human approval: update docs (keep log <100 lines), merge PR

IF `plans/phase-X.X-name.md` for the current phase ALREADY EXISTS (committed recently)

1. Create TodoWrite list with specific tasks
2. Write unit tests for core functionality first
3. Implement core logic, run `cargo test --lib` continuously
4. Verify all exit criteria met
5. Commit with detailed message (never to main)
6. Push branch and create PR
7. After human approval: update docs (keep log <100 lines), merge PR

### Long-Running Tasks

For any task that are not generative, e.g. `cargo run`:

1. **Do NOT run it yourself** - it will timeout or block progress
2. **Provide the command** to the human with clear instructions
3. **Wait for feedback** - the human will run it and report results
4. **Complete all your other independant work before handing off**
5. **Continue based on results** - adjust approach if needed

Example:

```
I've committed and pushed the changes. A training run is needed to observe the results.
This training will take ~30 minutes. Please run:

    cargo run --bin train_gol --release

And let me know the final accuracy achieved.
```

### Completion Protocol

Use the gh cli utility to manage interactions with Github.
When working on a new phase / task independant of the previous one, create a new dedicated branch
The human in the loop is responsible for reviewing your work through the PR's
