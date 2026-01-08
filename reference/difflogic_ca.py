"""Differentiable Logic Cellular Automata: from Game of Life to Pattern Generation

Extracted from the Google Research Colab notebook for reference.
Copyright 2025 Google LLC - Licensed under Apache License 2.0

Source: https://github.com/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb
"""

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

from collections import namedtuple
from functools import partial

from einops import rearrange
import flax.linen as nn
import jax
from jax.lax import conv_general_dilated_patches
import jax.numpy as jnp
import jax.random as random
import optax

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

PASS_THROUGH_GATE = 3      # Gate index for pass-through (A) operation
DEFAULT_PASS_VALUE = 10.0  # Initial logit for pass-through gate
NUMBER_OF_GATES = 16       # Total binary operations
FIRE_RATE = 0.6            # Probability of cell update in async mode

# ============================================================================
# CORE LOGIC GATE OPERATIONS
# ============================================================================

def bin_op_all_combinations(a, b):
    """Compute all 16 possible binary operations on two inputs.

    Operations (in order):
    0: FALSE, 1: AND, 2: A AND NOT B, 3: A (pass-through)
    4: NOT A AND B, 5: B, 6: XOR, 7: OR
    8: NOR, 9: XNOR, 10: NOT B, 11: A OR NOT B
    12: NOT A, 13: NOT A OR B, 14: NAND, 15: TRUE
    """
    return jnp.stack(
        [
            jnp.zeros_like(a),       # 0: FALSE
            a * b,                    # 1: AND
            a - a * b,                # 2: A AND NOT B
            a,                        # 3: A (pass-through)
            b - a * b,                # 4: NOT A AND B
            b,                        # 5: B
            a + b - 2 * a * b,        # 6: XOR
            a + b - a * b,            # 7: OR
            1 - (a + b - a * b),      # 8: NOR
            1 - (a + b - 2 * a * b),  # 9: XNOR
            1 - b,                    # 10: NOT B
            1 - b + a * b,            # 11: A OR NOT B
            1 - a,                    # 12: NOT A
            1 - a + a * b,            # 13: NOT A OR B
            1 - a * b,                # 14: NAND
            jnp.ones_like(a),         # 15: TRUE
        ],
        axis=-1,
    )


def bin_op_s(a, b, i_s):
    """Apply weighted sum of all possible gate operations.

    Args:
        a: First input (soft value 0-1)
        b: Second input (soft value 0-1)
        i_s: Probability distribution over 16 operations

    Returns:
        Weighted sum of all operation outputs
    """
    combinations = bin_op_all_combinations(a, b)
    result = jax.numpy.sum(combinations * i_s[None, ...], axis=-1)
    return result


def decode_soft(weights):
    """Softmax decoding for training (differentiable)."""
    return nn.softmax(weights, axis=-1)


def decode_hard(weights):
    """One-hot decoding for inference (discrete)."""
    return jax.nn.one_hot(jnp.argmax(weights, axis=-1), 16)


# ============================================================================
# GATE INITIALIZATION AND WIRING
# ============================================================================

def init_gates(n, num_gates=NUMBER_OF_GATES, pass_through_gate=PASS_THROUGH_GATE,
               default_pass_value=DEFAULT_PASS_VALUE):
    """Initialize gate logit matrix with pass-through defaults.

    Args:
        n: Number of gates in layer
        num_gates: Total operation types (16)
        pass_through_gate: Index of pass-through operation (3)
        default_pass_value: Initial logit for pass-through (10.0)

    Returns:
        Gate logits matrix (n x 16)
    """
    gates = jnp.zeros((n, num_gates))
    gates = gates.at[:, pass_through_gate].set(default_pass_value)
    return gates


def get_moore_connections(key):
    """Generate Moore neighborhood connections (center vs 8 neighbors).

    Used for first layer of perception kernels.
    Returns indices for 8 gates, each comparing center (4) to a neighbor.
    """
    neighbors = jnp.array([0, 1, 2, 3, 5, 6, 7, 8])  # All except center (4)
    a = neighbors
    b = jnp.full_like(neighbors, 4)  # All connect to center
    perm = jax.random.permutation(key, neighbors.shape[0])
    a = a[perm]
    b = b[perm]
    return a, b


def get_unique_connections(in_dim, out_dim, key):
    """Generate unique connections ensuring each gate gets different inputs.

    Used for subsequent layers to ensure information mixing.
    """
    assert out_dim * 2 >= in_dim
    x = jnp.arange(in_dim)
    a = x[::2]
    b = x[1::2]
    m = min(a.shape[0], b.shape[0])
    a = a[:m]
    b = b[:m]

    if a.shape[0] < out_dim:
        a_ = x[1::2]
        b_ = x[2::2]
        m = min(a_.shape[0], b_.shape[0])
        a = jnp.concatenate([a, a_[:m]])
        b = jnp.concatenate([b, b_[:m]])

    offset = 2
    while out_dim > a.shape[0] and offset < in_dim:
        a_ = x[:-offset]
        b_ = x[offset:]
        a = jnp.concatenate([a, a_])
        b = jnp.concatenate([b, b_])
        offset += 1

    if a.shape[0] >= out_dim:
        a = a[:out_dim]
        b = b[:out_dim]
    else:
        raise ValueError(
            f'Could not generate enough unique connections: {a.shape[0]} < {out_dim}'
        )

    perm = jax.random.permutation(key, out_dim)
    a = a[perm]
    b = b[perm]

    return a, b


def init_gate_layer(key, in_dim, out_dim, connections):
    """Initialize a single gate layer with specified connection topology.

    Args:
        key: JAX random key
        in_dim: Input dimension
        out_dim: Output dimension (number of gates)
        connections: 'random', 'unique', or 'first_kernel'

    Returns:
        (gate_logits, wires) tuple
    """
    if connections == 'random':
        key1, key2 = jax.random.split(key)
        c = jax.random.permutation(key2, 2 * out_dim) % in_dim
        c = jax.random.permutation(key1, in_dim)[c]
        c = c.reshape(2, out_dim)
        indices_a = c[0, :]
        indices_b = c[1, :]
    elif connections == 'unique':
        indices_a, indices_b = get_unique_connections(in_dim, out_dim, key)
    elif connections == 'first_kernel':
        indices_a, indices_b = get_moore_connections(key)
    else:
        raise ValueError(f'Connection type {connections} not implemented')

    wires = [indices_a, indices_b]
    gate_logits = init_gates(out_dim)
    return gate_logits, wires


# ============================================================================
# NETWORK INITIALIZATION
# ============================================================================

def init_logic_gate_network(hyperparams, params, wires, key):
    """Initialize logic gate network (used for update module)."""
    for i, (in_dim, out_dim) in enumerate(
        zip(hyperparams['layers'][:-1], hyperparams['layers'][1:])
    ):
        key, subkey = jax.random.split(key)
        gate_logits, gate_wires = init_gate_layer(
            subkey, int(in_dim), int(out_dim), hyperparams['connections'][i]
        )
        params.append(gate_logits)
        wires.append(gate_wires)


def init_perceive_network(hyperparams, params, wires, key):
    """Initialize perception network with multiple parallel kernels."""
    for i, (in_dim, out_dim) in enumerate(
        zip(hyperparams['layers'][:-1], hyperparams['layers'][1:])
    ):
        key, subkey = jax.random.split(key)
        gate_logits, gate_wires = init_gate_layer(
            subkey, int(in_dim), int(out_dim), hyperparams['connections'][i]
        )
        # Replicate for n_kernels parallel kernels
        params.append(
            gate_logits.repeat(hyperparams['n_kernels'], axis=0).reshape(
                hyperparams['n_kernels'], out_dim, NUMBER_OF_GATES
            )
        )
        wires.append(gate_wires)


def init_diff_logic_ca(hyperparams, key):
    """Initialize complete Differentiable Logic CA architecture.

    Args:
        hyperparams: Dict with 'update' and 'perceive' hyperparameters
        key: JAX random key

    Returns:
        (params, wires) dicts for update and perceive modules
    """
    key, subkey = jax.random.split(key, 2)

    params = {'update': [], 'perceive': []}
    wires = {'update': [], 'perceive': []}

    init_logic_gate_network(
        hyperparams['update'], params['update'], wires['update'], subkey
    )

    key, subkey = jax.random.split(key)
    init_perceive_network(
        hyperparams['perceive'], params['perceive'], wires['perceive'], subkey
    )

    return params, wires


# ============================================================================
# LAYER EXECUTION
# ============================================================================

def run_layer(logits, wires, x, training):
    """Execute a single logic gate layer.

    Args:
        logits: Gate logit matrix
        wires: [indices_a, indices_b] connection indices
        x: Input activations
        training: 1 for soft decoding, 0 for hard

    Returns:
        Layer output activations
    """
    a = x[..., wires[0]]
    b = x[..., wires[1]]
    logits = jax.lax.cond(training, decode_soft, decode_hard, logits)
    out = bin_op_s(a, b, logits)
    return out


def run_update(params, wires, x, training):
    """Execute update network."""
    for g, c in zip(params, wires):
        x = run_layer(g, c, x, training)
    return x


def run_perceive(params, wires, x, training):
    """Execute perception network with multiple kernels.

    Args:
        params: List of gate logit tensors per layer
        wires: List of connection indices per layer
        x: Input patch (9 cells x channels)
        training: 1 for soft, 0 for hard

    Returns:
        Concatenated [center_cell, kernel_1_out, ..., kernel_n_out]
    """
    run_layer_map = jax.vmap(run_layer, in_axes=(0, None, 0, None))
    x_prev = x
    x = x.T

    # Replicate input for each kernel
    x = jnp.repeat(x[None, ...], params[0].shape[0], axis=0)

    for g, c in zip(params, wires):
        x = run_layer_map(g, c, x, training)

    x = rearrange(x, 'k c s -> (c s k)')

    # CRITICAL: Concatenate center cell with perception outputs
    return jnp.concatenate([x_prev[4, :], x], axis=-1)


def run_circuit(params, wires, x, training):
    """Execute complete circuit: perceive then update."""
    x = run_perceive(params['perceive'], wires['perceive'], x, training)
    x = run_update(params['update'], wires['update'], x, training)
    return x


# ============================================================================
# GRID OPERATIONS
# ============================================================================

@partial(jax.jit, static_argnums=(1, 2))
def get_grid_patches(grid, patch_size, channel_dim, periodic):
    """Extract 3x3 patches from grid using convolution."""
    pad_size = (patch_size - 1) // 2

    padded_grid = jax.lax.cond(
        periodic,
        lambda g: jnp.pad(
            g, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="wrap"
        ),
        lambda g: jnp.pad(
            g, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="constant", constant_values=0,
        ),
        grid,
    )
    padded_grid = jnp.expand_dims(padded_grid, axis=0)
    patches = conv_general_dilated_patches(
        padded_grid,
        filter_shape=(patch_size, patch_size),
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )[0]

    patches = rearrange(patches, "x y (c f) -> (x y) f c", c=channel_dim)
    return patches


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def v_run_circuit_patched(patches, params, wires, training):
    """Apply circuit to batch of patches."""
    run_circuit_patch = jax.vmap(run_circuit, in_axes=(None, None, 0, None))
    return run_circuit_patch(patches, params, wires, training)


@jax.jit
def run_async(grid, params, wires, training, periodic, key):
    """Execute asynchronous update with fire rate masking."""
    patches = get_grid_patches(grid, patch_size=3, channel_dim=grid.shape[-1], periodic=periodic)
    x_new = v_run_circuit_patched(patches, params, wires, training)
    x_new = x_new.reshape(*grid.shape)
    # Fire rate mask: only update ~60% of cells
    update_mask_f32 = (
        jax.random.uniform(key, x_new[..., :1].shape) <= FIRE_RATE
    ).astype(jax.numpy.float32)
    x = grid * (1 - update_mask_f32) + x_new * update_mask_f32
    return x


@jax.jit
def run_sync(grid, params, wires, training, periodic):
    """Execute synchronous update (all cells update together)."""
    patches = get_grid_patches(grid, patch_size=3, channel_dim=grid.shape[-1], periodic=periodic)
    x_new = v_run_circuit_patched(patches, params, wires, training)
    return x_new.reshape(*grid.shape)


@partial(jax.jit, static_argnames=['num_steps', 'periodic', 'async_training'])
def run_iter_nca(grid, params, wires, training, periodic, num_steps, async_training, key):
    """Run NCA for specified number of steps."""
    def body_fn(carry, i):
        grid, key = carry
        if async_training:
            key, subkey = jax.random.split(key)
            x = run_async(grid, params, wires, training, periodic, subkey)
        else:
            x = run_sync(grid, params, wires, training, periodic)
        return (x, key), 0

    (grid, key), _ = jax.lax.scan(body_fn, (grid, key), jnp.arange(0, num_steps, 1))
    return grid


v_run_iter_nca = jax.vmap(run_iter_nca, in_axes=(0, None, None, None, None, None, None, None))


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

TrainState = namedtuple('TrainState', 'param opt_state key')


def loss_f(params, wires, train_x, train_y, periodic, num_steps, async_training, key):
    """Compute MSE loss with soft and hard gate decoding."""
    def eval(params, training):
        y = v_run_iter_nca(
            train_x, params, wires, training, periodic, num_steps, async_training, key
        )
        return jax.numpy.square(y[..., 0] - train_y[..., 0]).sum()

    return eval(params, 1), {'hard': eval(params, 0)}


def init_state(hyperparams, opt, seed):
    """Initialize training state."""
    key = random.PRNGKey(seed)
    key, subkey = random.split(key, 2)
    params, wires = init_diff_logic_ca(hyperparams, subkey)
    opt_state = opt.init(params)
    return TrainState(params, opt_state, key), wires


val_and_grad = jax.value_and_grad(loss_f, argnums=0, has_aux=True)


@partial(jax.jit, static_argnums=(4, 5, 6))
def train_step(train_state, train_x, train_y, wires, periodic, num_steps, async_training):
    """Single training step with gradient update."""
    params, opt_state, key = train_state
    key, k1 = jax.random.split(key, 2)
    (loss, hard), dx = val_and_grad(
        params, wires, train_x, train_y, periodic, num_steps, async_training, k1
    )
    # Optimizer: clip gradients at 100.0, then AdamW
    opt_obj = optax.chain(
        optax.clip(100.0),
        optax.adamw(learning_rate=0.05, b1=0.9, b2=0.99, weight_decay=1e-2),
    )
    dx, opt_state = opt_obj.update(dx, opt_state, params)
    new_params = optax.apply_updates(params, dx)
    return TrainState(new_params, opt_state, key), loss, hard


# ============================================================================
# EXPERIMENT HYPERPARAMETERS
# ============================================================================
# All 4 experiments from the paper, with full training configs.
# Key differences: channels (state bits), architecture size, training steps.

# EXPERIMENT 1: GAME OF LIFE
# Binary CA, 1-bit state, learns all 512 neighborhood configurations
GOL_HYPERPARAMS = {
    'seed': 23,
    'lr': 0.05,
    'batch_size': 20,
    'num_epochs': 3000,
    'num_steps': 1,  # Single step prediction
    'channels': 1,   # 1-bit state
    'periodic': True,
    'async_training': False,
    'grid_size': 128,
    'perceive': {
        'n_kernels': 16,
        'layers': [9, 8, 4, 2, 1],
        'connections': ['first_kernel', 'unique', 'unique', 'unique'],
    },
    'update': {
        # Input: 1 (center) + 16 (kernel outputs) = 17
        # 10 layers of 128, then reduction to 1
        'layers': [17, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                   64, 32, 16, 8, 4, 2, 1],
        'connections': ['unique'] * 17,
    },
}

# EXPERIMENT 2: CHECKERBOARD (SYNCHRONOUS)
# Multi-step pattern generation, 8-bit state, non-periodic boundaries
CHECKERBOARD_SYNC_HYPERPARAMS = {
    'seed': 23,
    'lr': 0.05,
    'batch_size': 2,
    'num_epochs': 500,
    'num_steps': 20,  # 20 growth steps
    'channels': 8,    # 8-bit state
    'periodic': False,
    'async_training': False,
    'grid_size': 16,
    'perceive': {
        'n_kernels': 16,
        'layers': [9, 8, 4, 2],  # Slightly smaller than GoL
        'connections': ['first_kernel', 'unique', 'unique'],
    },
    'update': {
        'layers': [264, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                   128, 64, 32, 16, 8, 8],
        'connections': ['unique'] * 16,
    },
}

# EXPERIMENT 3: CHECKERBOARD (ASYNCHRONOUS)
# Same as sync but with fire_rate masking for fault tolerance
CHECKERBOARD_ASYNC_HYPERPARAMS = {
    'seed': 23,
    'lr': 0.05,
    'batch_size': 1,
    'num_epochs': 800,
    'num_steps': 50,  # More steps for async convergence
    'channels': 8,
    'periodic': False,
    'async_training': True,  # Uses FIRE_RATE = 0.6
    'grid_size': 16,
    'perceive': {
        'n_kernels': 16,
        'layers': [9, 8, 4, 2],
        'connections': ['first_kernel', 'unique', 'unique'],
    },
    'update': {
        # Deeper network for async robustness
        'layers': [264, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                   256, 256, 256, 256, 128, 64, 32, 16, 8, 8],
        'connections': ['unique'] * 20,
    },
}

# EXPERIMENT 4: GROWING LIZARD EMOJI
# Complex pattern generation, 128-bit state, fewer but larger kernels
GROWING_LIZARD_HYPERPARAMS = {
    'seed': 23,
    'lr': 0.06,  # Slightly higher LR
    'batch_size': 1,
    'num_epochs': 3500,
    'num_steps': 12,   # 12 growth steps
    'channels': 128,   # 128-bit state (richest representation)
    'periodic': True,
    'async_training': False,
    'grid_size': 20,   # 20x20 training, generalizes to 40x40
    'perceive': {
        'n_kernels': 4,  # Fewer kernels than GoL (4 vs 16)
        'layers': [9, 8, 4, 2, 1],
        'connections': ['first_kernel', 'unique', 'unique', 'unique'],
    },
    'update': {
        # Input: 128 (center) + 4*128 (kernels) = 640... notebook shows 513
        # 513 = 1 + 4*128 = 513 (treating center as 1 bit? or different calc)
        'layers': [513, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128],
        'connections': ['unique'] * 10,
    },
}

# EXPERIMENT 5: COLORED G (from paper, partial config)
# RGB pattern generation, 64-bit state, 8-color palette
# Note: Full hyperparams not in public notebook, estimated from paper
COLORED_G_HYPERPARAMS = {
    'seed': 23,
    'lr': 0.05,
    'batch_size': 1,
    'num_epochs': 5000,  # Estimated - most complex experiment
    'num_steps': 15,     # 15 steps from paper
    'channels': 64,      # 64-bit state for RGB
    'periodic': False,
    'async_training': False,
    'grid_size': 20,
    'perceive': {
        'n_kernels': 4,
        'layers': [9, 8, 4, 2],
        'connections': ['first_kernel', 'unique', 'unique'],
    },
    'update': {
        # Paper mentions 927 active gates, 11 layers
        'layers': [257, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64],
        'connections': ['unique'] * 11,
    },
}

# Summary of experiments by complexity
EXPERIMENTS = {
    'gol': GOL_HYPERPARAMS,
    'checkerboard_sync': CHECKERBOARD_SYNC_HYPERPARAMS,
    'checkerboard_async': CHECKERBOARD_ASYNC_HYPERPARAMS,
    'lizard': GROWING_LIZARD_HYPERPARAMS,
    'colored_g': COLORED_G_HYPERPARAMS,
}


# ============================================================================
# DATA GENERATION: GAME OF LIFE
# ============================================================================

@jax.jit
def gol_step(board):
    """Applies one step of Conway's Game of Life rules to the board.

    Args:
        board: A 2D array representing the game board (1 = live, 0 = dead)

    Returns:
        Board after one step of the game.
    """
    # Count live neighbors using roll for each of 8 directions
    n = sum(
        jnp.roll(board, d, (0, 1))
        for d in [
            (1, 0),   # Right
            (-1, 0),  # Left
            (0, 1),   # Down
            (0, -1),  # Up
            (1, 1),   # Down-right
            (-1, -1), # Up-left
            (1, -1),  # Down-left
            (-1, 1),  # Up-right
        ]
    )
    # GoL rules: Birth (dead + 3 neighbors), Survive (alive + 2 or 3 neighbors)
    return (n == 3) | (board & (n == 2))


@partial(jax.jit, static_argnums=(1,))
def simulate_gol_batch(boards, steps):
    """Simulate GoL for multiple boards over multiple steps.

    Args:
        boards: Batch of 2D boards [batch, height, width]
        steps: Number of simulation steps

    Returns:
        Trajectories [batch, steps+1, height, width]
    """
    def simulate_one(board):
        states = [board]
        for _ in range(steps):
            board = gol_step(board)
            states.append(board)
        return jnp.stack(states)

    return jax.vmap(simulate_one)(boards)


def generate_all_3x3_neighborhoods():
    """Generate all 512 possible 3x3 binary neighborhoods.

    Returns:
        Tensor of shape [512, 3, 3] with all configurations
    """
    binary_numbers = jnp.arange(512)
    # Convert to binary and pad to 9 bits
    binary_array = (
        (binary_numbers[:, None] & (1 << jnp.arange(8, -1, -1))) > 0
    ).astype(jnp.float32)
    # Reshape to (512, 3, 3)
    return binary_array.reshape(512, 3, 3)


def sample_gol_batch(key, trajectories, batch_size, state_size):
    """Sample a batch from GoL trajectories for training.

    Args:
        key: JAX random key
        trajectories: [n_samples, steps+1, height, width] trajectory data
        batch_size: Number of samples to draw
        state_size: Number of channels (1 for GoL)

    Returns:
        (initial_states, target_states) tuple for training
    """
    n_samples = trajectories.shape[0]
    sample_idx = jax.random.randint(
        key, minval=0, maxval=n_samples, shape=(batch_size,)
    )
    # Create initial states with channel dimension
    init = jnp.zeros(
        (*trajectories[sample_idx, 0].shape, state_size), dtype=jnp.float32
    )
    init = init.at[..., 0].set(trajectories[sample_idx, 0].astype(jnp.float32))

    # Create target states
    target = jnp.zeros(
        (*trajectories[sample_idx, 1].shape, state_size), dtype=jnp.float32
    )
    target = target.at[..., 0].set(trajectories[sample_idx, 1].astype(jnp.float32))

    return init, target


# ============================================================================
# DATA GENERATION: CHECKERBOARD
# ============================================================================

def create_checkerboard(image_size=(64, 64), shape_size=8):
    """Create a checkerboard pattern.

    Args:
        image_size: (height, width) tuple
        shape_size: Size of each square in the pattern

    Returns:
        2D numpy array with checkerboard pattern (0s and 1s)
    """
    import numpy as np
    image = np.zeros(image_size)
    height, width = image_size
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    image = ((x // shape_size) + (y // shape_size)) % 2
    return image.astype(np.uint8)
