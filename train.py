import numpy as np
import matplotlib.pyplot as plt
import time
print("Starting import...")
start_time = time.time()
try:
    import logicars
    print(f"Import completed in {time.time() - start_time:.2f} seconds")
except ImportError:
    print("Import failed!")
    exit(1)

def generate_gol_training_data():
    """
    Generate all 512 possible 3x3 grid configurations and their next states
    according to Game of Life rules.
    """
    def apply_gol_rules(grid):
        """Apply Game of Life rules to determine center cell's next state"""
        center = grid[1, 1]
        # Count live neighbors (excluding center)
        live_neighbors = np.sum(grid) - center
        
        # Apply rules
        if center:  # Cell is alive
            return live_neighbors == 2 or live_neighbors == 3
        else:  # Cell is dead
            return live_neighbors == 3
    
    # Generate all 512 possible 3x3 grid configurations
    configs = []
    next_states = []
    
    for i in range(512):
        # Convert integer to binary representation
        binary = format(i, '09b')
        grid = np.array([int(bit) for bit in binary]).reshape(3, 3).astype(bool)
        
        # Apply GoL rules to get next state of center cell
        next_state = apply_gol_rules(grid)
        
        configs.append(grid)
        next_states.append(next_state)
    
    # Reshape for training
    configs_reshaped = np.zeros((512, 3, 3, 1), dtype=bool)
    next_states_reshaped = np.zeros((512, 3, 3, 1), dtype=bool)
    
    for i in range(512):
        configs_reshaped[i, :, :, 0] = configs[i]
        # Set the center cell's next state
        next_states_reshaped[i, 1, 1, 0] = next_states[i]
    
    return configs_reshaped, next_states_reshaped

def evaluate_model(ca, test_configs, test_targets, n_samples=10):
    """Evaluate trained model performance on sample configurations"""
    correct = 0
    total = 0
    
    # Select random samples
    indices = np.random.choice(len(test_configs), n_samples)
    
    plt.figure(figsize=(15, n_samples*3))
    for i, idx in enumerate(indices):
        # Get sample - only get the 3x3 grid for this specific example
        grid = test_configs[idx].copy()
        target = test_targets[idx, 1, 1, 0]  # Center cell's target state
        
        # Set the CA grid to this configuration - reshape to match CA's expected input
        ca.set_grid(grid)
        
        # Run one step
        ca.step()
        
        # Get result - this returns a 3D array (height, width, channels)
        result_grid = ca.get_grid()
        prediction = result_grid[1, 1, 0]
        
        # Check if correct
        if prediction == target:
            correct += 1
        total += 1
        
        # Visualize
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow(grid[:, :, 0], cmap='binary')
        plt.title(f"Input Grid {idx}")
        plt.axis('off')
        
        plt.subplot(n_samples, 3, i*3 + 2)
        plt.imshow(result_grid[:, :, 0], cmap='binary')
        plt.title(f"Prediction: {prediction}")
        plt.axis('off')
        
        plt.subplot(n_samples, 3, i*3 + 3)
        target_grid = np.zeros((3, 3, 1), dtype=bool)
        target_grid[1, 1, 0] = target
        plt.imshow(target_grid[:, :, 0], cmap='binary')
        plt.title(f"Target: {target}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    accuracy = correct / total * 100
    print(f"Model accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def test_on_larger_grid(ca, size=10):
    """Test the trained model on a larger grid with classic patterns"""
    # Create a new CA with the same parameters but larger grid
    # Note: This assumes the weights are shared between instances
    # In practice, you'd want a way to copy the trained weights
    test_ca = logicars.PyDiffLogicCA(size, size, 1, 16)
    
    # Initialize with a glider pattern
    grid = np.zeros((size, size, 1), dtype=bool)
    
    # Add glider
    glider_pattern = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
    for i, j in glider_pattern:
        grid[i+1, j+1, 0] = True
    
    # Set the grid
    try:
        test_ca.set_grid(grid)
        print("Successfully set grid for larger CA")
    except Exception as e:
        print(f"Error setting larger grid: {e}")
        print("Falling back to original CA size")
    
    # Run simulation
    frames = []
    try:
        # Try with test_ca first
        for i in range(20):
            frames.append(test_ca.get_grid().copy())
            test_ca.step()
    except Exception as e:
        print(f"Error running larger simulation: {e}")
        # Fall back to original CA
        print("Falling back to original CA")
        # Reset to original CA size
        try:
            # Create a 3x3 grid with the glider pattern
            small_grid = np.zeros((3, 3, 1), dtype=bool)
            for i, j in glider_pattern[:3]:  # Use just part of the pattern that fits
                if i < 3 and j < 3:
                    small_grid[i, j, 0] = True
            
            ca.set_grid(small_grid)
            frames = []
            for i in range(20):
                frames.append(ca.get_grid().copy())
                ca.step()
        except Exception as e:
            print(f"Error with fallback solution: {e}")
            frames = [np.zeros((3, 3, 1), dtype=bool)]  # Empty frame as placeholder
    
    # Visualize the evolution
    rows = min(4, len(frames))
    cols = min(5, (len(frames) + rows - 1) // rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    for i, frame in enumerate(frames):
        axes[i].imshow(frame[:, :, 0], cmap='binary')
        axes[i].set_title(f"Step {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def train_gol_model():
    """Train a DiffLogic CA to learn Game of Life rules"""
    print("Generating training data...")
    configs, targets = generate_gol_training_data()
    
    print(f"Training data: {len(configs)} configurations")
    
    # Create a new DiffLogic CA (3x3 grid, 1 bit state, 16 perception circuits)
    print("Creating DiffLogic CA model...")
    ca = logicars.PyDiffLogicCA(3, 3, 1, 16)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 2
    
    print(f"Training for {epochs} epochs...")
    start_time = time.time()
    
    # Train the model - ensure arrays have the right shape
    # Debug shapes
    print(f"Input shapes - configs: {configs.shape}, targets: {targets.shape}")
    
    try:
        ca.train(configs, targets, learning_rate, epochs)
    except Exception as e:
        print(f"Training error: {e}")
        
        # Print more details about shape
        print("\nShape details:")
        print(f"configs.shape = {configs.shape}")
        print(f"targets.shape = {targets.shape}")
        print(f"configs.dtype = {configs.dtype}")
        print(f"targets.dtype = {targets.dtype}")
        
        # Create smaller batch for testing if needed
        print("\nAttempting training with 10 samples...")
        small_configs = configs[:10].copy()
        small_targets = targets[:10].copy()
        try:
            ca.train(small_configs, small_targets, learning_rate, 10)
            print("Training with smaller batch succeeded!")
        except Exception as e:
            print(f"Still failed with error: {e}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(ca, configs, targets)
    
    # Test on a larger grid
    print("Testing on a larger grid...")
    test_on_larger_grid(ca)
    
    return ca

if __name__ == "__main__":
    ca = train_gol_model()