# Gradient Clipping: Element-wise vs Global Norm

## The Problem We Had

Our training wasn't converging because we used **element-wise clipping**:

```rust
// WRONG: Clip each gradient independently
clipped[i] = gradient[i].clamp(-100.0, 100.0);
```

The reference uses **global L2 norm clipping**:

```python
optax.clip(100.0)  # Clip by global norm
```

These are fundamentally different operations.

## What's the Difference?

### Element-wise Clipping

Each gradient value is independently clamped to a range. If you have gradients `[50, 200, 30]` and clip to ±100:

```
[50, 200, 30] → [50, 100, 30]
```

**Problem**: This distorts the gradient direction. The relative magnitudes change—the "200" component gets squashed while others stay the same. You're no longer following the true gradient direction.

### Global L2 Norm Clipping

First compute the L2 norm (Euclidean length) of the entire gradient vector:

```
norm = sqrt(50² + 200² + 30²) = sqrt(2500 + 40000 + 900) = 208.2
```

If norm > threshold (100), scale ALL gradients uniformly:

```
scale = 100 / 208.2 = 0.48
[50, 200, 30] × 0.48 = [24, 96, 14.4]
```

**Key property**: The direction is preserved. We're still moving in the same direction, just taking a smaller step. The relative importance of each component stays the same.

## Why Direction Matters

Gradient descent works by following the gradient direction downhill. When you distort the direction with element-wise clipping:

1. You might oscillate instead of converging
2. Some parameters get "stuck" while others move
3. The optimizer's momentum gets confused

With global norm clipping, you guarantee:
- Gradient direction is always correct
- Step size is bounded (prevents explosion)
- Optimizer state (momentum, variance) stays coherent

## Visual Intuition

Imagine you're walking downhill in fog. The gradient tells you which way is steepest.

- **Element-wise clipping**: "Go 10 steps east, but only 2 steps south" — you end up going the wrong way
- **Global norm clipping**: "Go in that direction, but only 10 steps total" — you go the right way, just shorter

## Implementation

```rust
// 1. Compute global norm across ALL parameters
let global_norm = all_gradients.iter()
    .map(|g| g * g)
    .sum::<f64>()
    .sqrt();

// 2. Compute uniform scaling factor
let clip_coef = if global_norm > max_norm {
    max_norm / global_norm
} else {
    1.0  // No clipping needed
};

// 3. Scale ALL gradients uniformly
for g in all_gradients.iter_mut() {
    *g *= clip_coef;
}
```

## When Clipping Activates

With threshold=100 and ~50,000 gradient values (3040 gates × 16 ops), the norm can easily exceed 100 even with small per-element gradients. If each gradient averages 0.1:

```
norm ≈ sqrt(50000 × 0.1²) = sqrt(500) ≈ 22
```

So clipping only activates when gradients are genuinely large, which is exactly what we want—it's a safety valve, not a constant constraint.

## Key Takeaway

**Global norm clipping preserves gradient direction while bounding step size.** Element-wise clipping corrupts the direction, which can prevent learning entirely.
