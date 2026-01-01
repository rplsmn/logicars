# Reference Implementation

This folder contains a snapshot of the reference Python/JAX implementation for comparison and guidance.

## Source

- **Paper**: [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/)
- **Original Notebook**: [Google Colab](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb)
- **Repository**: https://github.com/google-research/self-organising-systems

## License

Copyright 2025 Google LLC - Licensed under Apache License 2.0

## Contents

- `difflogic_ca.py` - Extracted Python code from the Colab notebook

## Key Architecture Details (Game of Life)

### Perception Module
- **16 parallel kernels**
- Each kernel: `[9, 8, 4, 2, 1]` layers (first layer has 8 gates connecting center to neighbors)
- Connection topology: `first_kernel` for layer 1, `unique` for subsequent layers
- Output: 16 feature bits (one per kernel)

### Update Module
- Input: center cell + 16 perception outputs = 17 bits
- Architecture: 23 layers (`[17, 128]*16, [64, 32, 16, 8, 4, 2, 1]`)
- Connection topology: `unique` throughout
- Output: 1 bit (next cell state)

### Training Parameters
- Learning rate: 0.05
- Optimizer: AdamW (weight_decay=1e-2, b1=0.9, b2=0.99)
- Gradient clipping: 100.0
- Fire rate (async): 0.6
- Pass-through gate initialization: index 3 = 10.0

### Total Gates (GoL)
- Perception: ~240 gates (16 kernels × 15 gates each)
- Update: ~100 gates in reduction layers
- **Total active: 336 gates**
