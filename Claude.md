# Logicars - Differentiable Logic Cellular Automata

## What This Is

A Rust library replicating Google Research's [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/) paper. The goal is learning cellular automata rules (like Conway's Game of Life) using differentiable logic gate networks.

**Status**: Previous implementation attempts failed to converge. Start fresh from first principles.

---

## The Core Idea

Traditional neural networks use continuous weights. This paper uses **discrete logic gates** (AND, OR, XOR, etc.) but makes them trainable:

1. Each "gate" maintains a probability distribution over all 16 possible 2-input binary operations
2. **Training**: compute weighted sum of all gate outputs (soft/differentiable)
3. **Inference**: use only the highest-probability gate (hard/discrete)

This yields circuits that are interpretable, hardware-efficient, and surprisingly capable.

---

## Key Concepts

**16 Binary Operations**: FALSE, AND, A_AND_NOT_B, A, NOT_A_AND_B, B, XOR, OR, NOR, XNOR, NOT_B, A_OR_NOT_B, NOT_A, NOT_A_OR_B, NAND, TRUE

**Soft Gate Output**: `output = sum(probability[i] * gate_i(a, b))` for all 16 gates

**Gradient Flow**: Loss → output → probability distribution updates via chain rule

**Architecture** (per paper):
- Perception circuits: process 3x3 neighborhood → single bit output
- Update circuit: perception outputs + current state → next state
- Multiple perception circuits run in parallel

---

## Development Approach

Each phase should be **complete and verified** before moving to the next:

1. **Single Gate**: Train one gate to learn AND. Verify gradients and convergence.
2. **Gate Layer**: Multiple gates in parallel. Verify independent learning.
3. **Multi-Layer Circuit**: Learn XOR (requires depth). Verify backprop through layers.
4. **Perception Circuit**: Neighborhood → single output. Test with known patterns.
5. **Full CA**: Grid + perception + update. Train on Game of Life.

**Do not skip verification steps.** A bug in phase 1 will cause mysterious failures in phase 5.

---

## Success Criteria

- Training loss decreases monotonically
- Gate probability distributions converge (one gate dominates)
- Hard accuracy >95% on Game of Life (all 512 configurations)
- Generalizes to larger grids than trained on

---

## Technical Stack

- **Core**: Rust (performance, correctness)
- **Bindings**: PyO3 for Python, potentially R via extendr
- **Testing**: Property-based tests for gate correctness, integration tests for training

---

## What to Ignore

The existing `src/` files represent a failed attempt. They may contain:
- Incorrect gradient computations
- Wrong architecture decisions
- Over-complicated abstractions

Treat them as **negative examples** - something didn't work. Design from scratch, then compare to understand what went wrong.

---

## References

- Paper: https://google-research.github.io/self-organising-systems/difflogic-ca/
- Game of Life rules: cell survives with 2-3 neighbors, dead cell born with exactly 3
