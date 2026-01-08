//! Phase 2.1: Checkerboard Pattern Generation
//!
//! Implements checkerboard pattern training for multi-channel CAs.
//! The model learns to generate a checkerboard pattern from random initial seeds.
//!
//! Key features:
//! - 8-bit state (8 channels per cell)
//! - Non-periodic boundaries
//! - 20-step rollout (sync) or 50-step (async)
//! - Pattern stored in channel 0, other channels as working memory
//!
//! Reference: CHECKERBOARD_SYNC_HYPERPARAMS from difflogic_ca.py

use crate::grid::{BoundaryCondition, NGrid};
use crate::perception::{ConnectionType, PerceptionModule};
use crate::training::SimpleRng;

#[cfg(test)]
use crate::training::TrainingConfig;
use crate::update::{DiffLogicCA, UpdateModule};

/// Default checkerboard hyperparameters
pub const CHECKERBOARD_CHANNELS: usize = 8;
pub const CHECKERBOARD_KERNELS: usize = 16;
pub const CHECKERBOARD_GRID_SIZE: usize = 16;
pub const CHECKERBOARD_SQUARE_SIZE: usize = 2;
pub const CHECKERBOARD_SYNC_STEPS: usize = 20;
pub const CHECKERBOARD_ASYNC_STEPS: usize = 50;

/// Create a checkerboard pattern grid.
///
/// The pattern is stored in channel 0, with all other channels set to 0.
///
/// # Arguments
/// * `size` - Grid dimensions (size × size)
/// * `square_size` - Size of each checkerboard square
/// * `channels` - Number of channels (pattern in channel 0)
///
/// # Returns
/// NGrid with checkerboard pattern in channel 0
pub fn create_checkerboard(size: usize, square_size: usize, channels: usize) -> NGrid {
    let mut grid = NGrid::new(size, size, channels, BoundaryCondition::NonPeriodic);

    for y in 0..size {
        for x in 0..size {
            // Checkerboard pattern: ((x / square_size) + (y / square_size)) % 2
            let value = if ((x / square_size) + (y / square_size)) % 2 == 0 {
                0.0
            } else {
                1.0
            };
            grid.set(x, y, 0, value);
        }
    }

    grid
}

/// Create a random seed grid for training.
///
/// All channels are initialized with random binary values (0 or 1).
///
/// # Arguments
/// * `size` - Grid dimensions (size × size)
/// * `channels` - Number of channels
/// * `rng` - Random number generator
pub fn create_random_seed(size: usize, channels: usize, rng: &mut SimpleRng) -> NGrid {
    let mut grid = NGrid::new(size, size, channels, BoundaryCondition::NonPeriodic);

    for y in 0..size {
        for x in 0..size {
            for c in 0..channels {
                let value = if rng.next_bool(0.5) { 1.0 } else { 0.0 };
                grid.set(x, y, c, value);
            }
        }
    }

    grid
}

/// Create perception module for checkerboard experiment.
///
/// Architecture: 16 kernels, [9→8→4→2] (2 output bits per kernel)
pub fn create_checkerboard_perception() -> PerceptionModule {
    PerceptionModule::new(
        CHECKERBOARD_CHANNELS,
        CHECKERBOARD_KERNELS,
        &[9, 8, 4, 2], // 2 output bits per kernel
        &[
            ConnectionType::FirstKernel,
            ConnectionType::Unique,
            ConnectionType::Unique,
        ],
    )
}

/// Create update module for checkerboard experiment.
///
/// Architecture: [264→256×10→128→64→32→16→8→8]
/// Input: 8 (center) + 16 * 2 * 8 = 264
/// Output: 8 channels
pub fn create_checkerboard_update() -> UpdateModule {
    // Input: center(8) + kernels(16) * output_bits(2) * channels(8) = 264
    let input_size = CHECKERBOARD_CHANNELS
        + CHECKERBOARD_KERNELS * 2 * CHECKERBOARD_CHANNELS;

    let mut layer_sizes = vec![input_size]; // 264
    
    // 10 layers of 256
    for _ in 0..10 {
        layer_sizes.push(256);
    }
    
    // Reduction layers
    layer_sizes.extend_from_slice(&[128, 64, 32, 16, 8, CHECKERBOARD_CHANNELS]);

    UpdateModule::new(&layer_sizes)
}

/// Create a smaller update module for faster experimentation.
///
/// Uses fewer hidden layers than full model but enough capacity to learn.
/// Reference uses 10×256 layers; we use 4×256 as a middle ground.
/// Respects the unique_connections constraint: out_dim * 2 >= in_dim
pub fn create_small_checkerboard_update() -> UpdateModule {
    let input_size = CHECKERBOARD_CHANNELS
        + CHECKERBOARD_KERNELS * 2 * CHECKERBOARD_CHANNELS;

    // Architecture: 264→256×4→128→64→32→16→8 (10 layers total)
    // Reference uses 264→256×10→128→64→32→16→8→8 (16 layers)
    // This gives ~2x speedup while maintaining enough capacity
    let layer_sizes = vec![
        input_size, // 264
        256, 256, 256, 256,  // 4 layers of 256 (vs 10 in reference)
        128, 64, 32, 16,     // Reduction layers
        CHECKERBOARD_CHANNELS, // 8
    ];

    UpdateModule::new(&layer_sizes)
}

/// Create complete DiffLogicCA for checkerboard experiment.
pub fn create_checkerboard_model() -> DiffLogicCA {
    let perception = create_checkerboard_perception();
    let update = create_checkerboard_update();
    DiffLogicCA::new(perception, update)
}

/// Create a smaller model for faster experimentation.
pub fn create_small_checkerboard_model() -> DiffLogicCA {
    let perception = create_checkerboard_perception();
    let update = create_small_checkerboard_update();
    DiffLogicCA::new(perception, update)
}

/// Compute MSE loss between prediction and target (channel 0 only for checkerboard).
///
/// # Arguments
/// * `prediction` - Predicted grid after N steps
/// * `target` - Target checkerboard pattern
///
/// # Returns
/// MSE loss over channel 0 cells
pub fn compute_checkerboard_loss(prediction: &NGrid, target: &NGrid) -> f64 {
    assert_eq!(prediction.width, target.width);
    assert_eq!(prediction.height, target.height);
    assert!(prediction.channels >= 1);
    assert!(target.channels >= 1);

    let mut loss = 0.0;
    let num_cells = (prediction.width * prediction.height) as f64;

    for y in 0..prediction.height {
        for x in 0..prediction.width {
            let pred = prediction.get(x as isize, y as isize, 0);
            let tgt = target.get(x as isize, y as isize, 0);
            let diff = pred - tgt;
            loss += diff * diff;
        }
    }

    loss / num_cells
}

/// Compute hard accuracy on channel 0.
pub fn compute_checkerboard_accuracy(prediction: &NGrid, target: &NGrid) -> f64 {
    assert_eq!(prediction.width, target.width);
    assert_eq!(prediction.height, target.height);

    let mut correct = 0;
    let num_cells = prediction.width * prediction.height;

    for y in 0..prediction.height {
        for x in 0..prediction.width {
            let pred: f64 = if prediction.get(x as isize, y as isize, 0) > 0.5 { 1.0 } else { 0.0 };
            let tgt: f64 = if target.get(x as isize, y as isize, 0) > 0.5 { 1.0 } else { 0.0 };
            if (pred - tgt).abs() < 0.01 {
                correct += 1;
            }
        }
    }

    correct as f64 / num_cells as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_checkerboard_basic() {
        let grid = create_checkerboard(4, 1, 8);
        
        assert_eq!(grid.width, 4);
        assert_eq!(grid.height, 4);
        assert_eq!(grid.channels, 8);

        // Check pattern in channel 0: alternating 0,1,0,1 / 1,0,1,0
        assert_eq!(grid.get(0, 0, 0), 0.0);
        assert_eq!(grid.get(1, 0, 0), 1.0);
        assert_eq!(grid.get(0, 1, 0), 1.0);
        assert_eq!(grid.get(1, 1, 0), 0.0);

        // Other channels should be 0
        for c in 1..8 {
            assert_eq!(grid.get(0, 0, c), 0.0);
        }
    }

    #[test]
    fn test_create_checkerboard_square_size_2() {
        let grid = create_checkerboard(8, 2, 8);

        // With square_size=2:
        // (0,0), (1,0), (0,1), (1,1) should all be same (dark)
        // (2,0), (3,0), (2,1), (3,1) should all be same (light)
        assert_eq!(grid.get(0, 0, 0), 0.0);
        assert_eq!(grid.get(1, 0, 0), 0.0);
        assert_eq!(grid.get(0, 1, 0), 0.0);
        assert_eq!(grid.get(1, 1, 0), 0.0);

        assert_eq!(grid.get(2, 0, 0), 1.0);
        assert_eq!(grid.get(3, 0, 0), 1.0);
        assert_eq!(grid.get(2, 1, 0), 1.0);
        assert_eq!(grid.get(3, 1, 0), 1.0);
    }

    #[test]
    fn test_create_random_seed() {
        let mut rng = SimpleRng::new(42);
        let grid = create_random_seed(16, 8, &mut rng);

        assert_eq!(grid.width, 16);
        assert_eq!(grid.height, 16);
        assert_eq!(grid.channels, 8);

        // Should have approximately 50% ones (checking rough distribution)
        let mut ones = 0;
        let total = 16 * 16 * 8;
        for y in 0..16 {
            for x in 0..16 {
                for c in 0..8 {
                    if grid.get(x, y, c) > 0.5 {
                        ones += 1;
                    }
                }
            }
        }
        
        // Should be roughly 50% (with some variance)
        let ratio = ones as f64 / total as f64;
        assert!(ratio > 0.4 && ratio < 0.6, "Expected ~50% ones, got {:.2}%", ratio * 100.0);
    }

    #[test]
    fn test_checkerboard_perception_architecture() {
        let perception = create_checkerboard_perception();

        assert_eq!(perception.channels, 8);
        assert_eq!(perception.num_kernels, 16);
        
        // Output size: 8 (center) + 16 * 2 * 8 = 264
        assert_eq!(perception.output_size(), 264);
    }

    #[test]
    fn test_checkerboard_update_architecture() {
        let update = create_checkerboard_update();

        // Input: 264, output: 8
        assert_eq!(update.input_size, 264);
        assert_eq!(update.output_channels, 8);

        // Should have: 264 -> 256×10 -> 128 -> 64 -> 32 -> 16 -> 8 -> 8
        // That's 17 layer transitions
        assert_eq!(update.layers.len(), 16);
    }

    #[test]
    fn test_checkerboard_model_creation() {
        let model = create_checkerboard_model();

        assert_eq!(model.perception.channels, 8);
        assert_eq!(model.perception.num_kernels, 16);
        assert_eq!(model.update.output_channels, 8);

        // Should have substantial gate count
        let total_gates = model.perception.total_gates() + model.update.total_gates();
        assert!(total_gates > 1000, "Expected >1000 gates, got {}", total_gates);
    }

    #[test]
    fn test_small_checkerboard_model() {
        let model = create_small_checkerboard_model();

        assert_eq!(model.perception.channels, 8);
        assert_eq!(model.update.output_channels, 8);

        // Should have fewer gates than full model
        let small_gates = model.perception.total_gates() + model.update.total_gates();
        let full_model = create_checkerboard_model();
        let full_gates = full_model.perception.total_gates() + full_model.update.total_gates();

        assert!(small_gates < full_gates, "Small model should have fewer gates");
        
        // Verify small model has enough capacity (at least 1000 gates)
        // Old "small" model had 728 gates which was too shallow
        assert!(small_gates >= 1000, "Small model needs enough capacity, got {} gates", small_gates);
    }
    
    #[test]
    fn test_small_model_layer_depth() {
        // Verify small model has sufficient depth for learning
        // Reference uses 16 update layers, small should have at least 8
        let model = create_small_checkerboard_model();
        
        let update_layers = model.update.layers.len();
        assert!(update_layers >= 8, "Small model needs at least 8 update layers, got {}", update_layers);
        
        // Verify architecture: 264→256×4→128→64→32→16→8 = 9 layers
        assert_eq!(model.update.input_size, 264);
        assert_eq!(model.update.output_channels, 8);
    }

    #[test]
    fn test_compute_checkerboard_loss_identical() {
        let grid1 = create_checkerboard(4, 1, 8);
        let grid2 = create_checkerboard(4, 1, 8);

        let loss = compute_checkerboard_loss(&grid1, &grid2);
        assert!((loss - 0.0).abs() < 1e-10, "Identical grids should have 0 loss");
    }

    #[test]
    fn test_compute_checkerboard_loss_different() {
        let grid1 = create_checkerboard(4, 1, 8);
        let mut grid2 = NGrid::new(4, 4, 8, BoundaryCondition::NonPeriodic);
        
        // Set opposite pattern
        for y in 0..4 {
            for x in 0..4 {
                let value = if ((x / 1) + (y / 1)) % 2 == 1 { 0.0 } else { 1.0 };
                grid2.set(x, y, 0, value);
            }
        }

        let loss = compute_checkerboard_loss(&grid1, &grid2);
        // All cells differ by 1.0, so MSE = 1.0
        assert!((loss - 1.0).abs() < 1e-10, "Inverted grids should have loss = 1.0, got {}", loss);
    }

    #[test]
    fn test_compute_checkerboard_accuracy() {
        let target = create_checkerboard(4, 1, 8);
        
        // Perfect prediction
        let pred = create_checkerboard(4, 1, 8);
        let acc = compute_checkerboard_accuracy(&pred, &target);
        assert!((acc - 1.0).abs() < 1e-10, "Perfect prediction should have 100% accuracy");

        // Half wrong
        let mut half_wrong = create_checkerboard(4, 1, 8);
        for y in 0..2usize {
            for x in 0..4usize {
                let current = half_wrong.get(x as isize, y as isize, 0);
                half_wrong.set(x, y, 0, 1.0 - current);
            }
        }
        let acc = compute_checkerboard_accuracy(&half_wrong, &target);
        assert!((acc - 0.5).abs() < 1e-10, "Half wrong should have 50% accuracy");
    }

    #[test]
    fn test_checkerboard_model_forward_pass() {
        use crate::grid::NNeighborhood;

        let model = create_small_checkerboard_model();
        
        // Create a test neighborhood with 8 channels
        let cells = vec![0.5; 9 * 8]; // 9 positions × 8 channels
        let neighborhood = NNeighborhood::new(8, cells);
        
        // Run forward pass - returns (output, perception_activations, update_activations)
        let (output, _, _) = model.forward_soft(&neighborhood);

        // Output should have 8 channels
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_training_config_checkerboard() {
        let config = TrainingConfig::checkerboard_sync();

        assert_eq!(config.num_steps, 20);
        assert!(!config.async_training);
        assert!(!config.periodic);
    }

    #[test]
    fn test_checkerboard_loss_soft_values() {
        // Test loss with soft (probabilistic) values
        let target = create_checkerboard(4, 1, 8);
        
        // Prediction with 0.5 everywhere (maximum uncertainty)
        let mut uncertain = NGrid::new(4, 4, 8, BoundaryCondition::NonPeriodic);
        for y in 0..4 {
            for x in 0..4 {
                uncertain.set(x, y, 0, 0.5);
            }
        }
        
        let loss = compute_checkerboard_loss(&uncertain, &target);
        // Half cells are 0, half are 1. (0.5-0)^2 = 0.25, (0.5-1)^2 = 0.25
        // Average: 0.25
        assert!((loss - 0.25).abs() < 1e-10, "Uncertain prediction should have loss ~0.25, got {}", loss);
    }

    #[test]
    fn test_checkerboard_accuracy_different_sizes() {
        // Test that accuracy works with different grid sizes (generalization test)
        let small_target = create_checkerboard(16, 2, 8);
        let small_pred = create_checkerboard(16, 2, 8);
        let small_acc = compute_checkerboard_accuracy(&small_pred, &small_target);
        assert!((small_acc - 1.0).abs() < 1e-10);

        let large_target = create_checkerboard(64, 2, 8);
        let large_pred = create_checkerboard(64, 2, 8);
        let large_acc = compute_checkerboard_accuracy(&large_pred, &large_target);
        assert!((large_acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_checkerboard_model_input_output_sizes() {
        // Verify the full model has correct input/output sizes matching reference
        let model = create_checkerboard_model();

        // Perception: 16 kernels, [9→8→4→2] = 14 gates per kernel, 16*14 = 224 gates
        assert_eq!(model.perception.num_kernels, 16);
        assert_eq!(model.perception.channels, 8);
        
        // Perception output: center(8) + kernels(16) * output_bits(2) * channels(8) = 264
        assert_eq!(model.perception.output_size(), 264);
        
        // Update input should match perception output
        assert_eq!(model.update.input_size, 264);
        
        // Update output should be 8 channels
        assert_eq!(model.update.output_channels, 8);
    }
}
