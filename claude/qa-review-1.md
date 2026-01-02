# QA Review #1: Scope Assessment - Tunnel Vision on GoL

**Date**: 2026-01-01
**Status**: IN PROGRESS
**Branch**: `claude/review-project-scope-9a8sN`

---

## Executive Summary

The project has developed tunnel vision on Game of Life (GoL), when the paper's true value lies in multi-bit experiments (Checkerboard, Growing Lizard, Colored "G"). This QA session addresses the gap before it becomes a costly refactor.

---

## Problem Statement

### User Observation
> "It feels like the Claude sessions are maybe in a tunnel vision on the GoL example when the true usefulness of this project lies in the later cases of the paper, such as the colored image."

### Evidence Found

| Metric | Current | Should Be |
|--------|---------|-----------|
| Experiments in reference/ | 1 (GoL) | 4 |
| Reference code lines | 480 | ~2690 (full notebook) |
| Cell representation | `Vec<bool>` | `Vec<[bool; N]>` or generic |
| Multi-state in plan.md | Phase 4.2 | Phase 1.x |
| Hyperparams documented | GoL only | All 4 experiments |

### The Paper's 4 Experiments

1. **Game of Life** - 1-bit binary, 512 configs, 336 gates
2. **Checkerboard** - 8-bit multi-channel, 20 steps, boundary-invariant generalization
3. **Growing Lizard** - 128-bit state, complex shape, 12 growth steps
4. **Colored "G"** - 64-bit RGB, 8-color palette, 927 gates (most complex)

### Code That Would Break

```rust
// src/phase_1_1.rs - hardcoded to single bool
pub struct Grid {
    pub cells: Vec<bool>,  // Single bit per cell
}

pub struct Neighborhood {
    pub cells: [bool; 9],  // 9 bits, not 9×N bits
}

pub struct GolTruthTable {
    pub targets: [bool; 512],  // Only works for 1-bit output
}
```

---

## Why This Matters

1. **Architectural Debt**: Single-bit assumption is baked into Grid, Neighborhood, and training infrastructure
2. **Wasted Effort**: Fixing Phase 1.1's 81% accuracy ceiling without N-bit support means refactoring twice
3. **Missing Value**: GoL is the "hello world" - colored image generation is the demo that impresses
4. **Reference Gap**: Future Claude sessions will repeat the tunnel vision without complete reference configs

---

## Solution Plan

### Principle: "Make things that work first, add complexity later"

We follow this by:
- Keeping GoL as the first training target (smallest N=1)
- But designing data structures to support N-bits from the start
- N=1 is just a special case of the general architecture

### Task List

- [x] Create this QA review document
- [ ] Fetch original Colab notebook for all experiment configs
- [ ] Update `reference/difflogic_ca.py` with all 4 hyperparams
- [ ] Update `reference/README.md` with all experiment architectures
- [ ] Restructure `claude/plan.md` - move multi-state to Phase 1.x

### NOT Doing Yet

- Refactoring Rust code (that's for the next session after plan approval)
- The plan update will document what needs to change in code

---

## Detailed Changes

### 1. reference/difflogic_ca.py

Add hyperparams for all experiments:
- `GOL_HYPERPARAMS` (already exists)
- `CHECKERBOARD_HYPERPARAMS` (8-bit, 256-node layers)
- `LIZARD_HYPERPARAMS` (128-bit, 512-node layers)
- `COLORED_G_HYPERPARAMS` (64-bit, 512-node layers)

### 2. reference/README.md

Document all 4 experiments with:
- State size (bits per cell)
- Architecture (kernels, layers)
- Training parameters
- Key characteristics

### 3. claude/plan.md Restructuring

**Before:**
```
Phase 1: Game of Life MVP
Phase 4.2: Multi-State CA (too late!)
```

**After:**
```
Phase 1: Core Architecture (N-bit capable)
  1.1: N-bit Grid and Neighborhood (design)
  1.2: Perception Module (parallel kernels)
  1.3: Update Module
  1.4: Training Loop
  1.5: GoL Validation (N=1)
Phase 2: Multi-bit Experiments
  2.1: Checkerboard (N=8)
  2.2: Growing Lizard (N=128)
  2.3: Colored G (N=64)
```

---

## Progress Log

### 2026-01-01 - Session Start

- Identified tunnel vision problem
- Confirmed with evidence from codebase
- Created this tracking document
- Beginning reference extraction

---

## For Future Claude Sessions

If you're continuing this work:

1. Check the task list above - pick up where we left off
2. The goal is to have complete reference configs before touching Rust code
3. The plan.md restructure is critical - it sets direction for all future work
4. GoL remains the first target, but with N-bit-ready architecture

---

**Last Updated**: 2026-01-01
