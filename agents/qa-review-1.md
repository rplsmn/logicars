# QA Review #1: Scope Correction ✅ RESOLVED

**Date**: 2026-01-01 | **Status**: COMPLETE

---

## Problem

Project had tunnel vision on Game of Life (1-bit) when the paper's value lies in multi-bit experiments:

| Experiment | Bits/Cell | Gates | Key Feature |
|------------|-----------|-------|-------------|
| Game of Life | 1 | 336 | Simplest case, validation |
| Checkerboard | 8 | ~400 | Multi-channel, generalization |
| Growing Lizard | 128 | ~600 | Complex shape growth |
| Colored "G" | 64 | 927 | RGB palette, most complex |

**Root cause**: Grid and Neighborhood hardcoded to `Vec<bool>`, multi-state deferred to Phase 4.2.

---

## Solution

**Principle**: N-bit from start. GoL (N=1) is just validation of the general architecture.

### Changes Made

1. **agents/plan.md**: Restructured phases
   - Phase 1.1: N-bit Grid (not single-bit)
   - Phase 1.5: GoL as validation (not goal)
   - Phase 2.x: Multi-bit experiments (Checkerboard, Lizard, Colored G)

2. **reference/difflogic_ca.py**: Added all 5 experiment hyperparams
   - `GOL_HYPERPARAMS`, `CHECKERBOARD_SYNC/ASYNC_HYPERPARAMS`
   - `GROWING_LIZARD_HYPERPARAMS`, `COLORED_G_HYPERPARAMS`

3. **reference/README.md**: Complete documentation of all experiments

---

## Outcome

- ✅ Phase 1.1 implemented with `NGrid<C>` supporting 1-128 channels
- ✅ Phase 1.2 Perception module works for any channel count
- ✅ No refactoring needed for multi-bit experiments
- ✅ GoL remains first target but architecture is N-bit ready

---

## Key Lesson

Design for the general case from start. The simplest case (N=1) validates the architecture; the complex cases (N=64-128) demonstrate the value.
