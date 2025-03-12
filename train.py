import numpy as np
import matplotlib.pyplot as plt
try:
    import logicars
except ImportError:
    print("Could not import logicars. Make sure the Rust library is properly built and installed.")
    print("Try running: pip install -e .")
    exit(1)

def generate_training_data():
    """
    Generate training data for the Game of Life rules.
    Returns a set of 3x3 grid configurations and their expected next states.
    """
    # For a complete training set, we'd need all 2^9 = 512 possible configurations
    # Let's build this systematically
    
    # Helper function to apply Game of Life rules
    def apply_gol_rules(grid):
        center = grid[1, 1]
        # Count live neighbors (excluding center)
        live_neighbors = np.sum(grid) - center
        
        # Apply rules
        if center:  # Cell is alive
            return live_neighbors == 2 or live_neighbors == 3
        else:  # Cell is dead
            return live_neighbors == 3
    
    # Generate all possible 3x3 grid configurations
    configs = []
    next_states = []
    
    # Systematically generate all 512 configurations
    for i in range(512):
        # Convert integer to binary representation
        binary = format(i, '09b')
        grid = np.array([int(bit) for bit in binary]).reshape(3, 3).astype(bool)
        
        # Apply GoL rules to determine next state of center cell
        next_state = apply_gol_rules(grid)
        
        # Add to training data
        configs.append(grid)
        next_states.append(next_state)
    
    # Convert to numpy arrays
    configs = np.array(configs)
    next_states = np.array(next_states)
    
    # Reshape for training
    # We need to add an extra dimension for the state size (1 for GoL)
    configs_reshaped = np.zeros((512, 3, 3, 1), dtype=bool)
    next_states_reshaped = np.zeros((512, 3, 3, 1), dtype=bool)
    
    for i in range(512):
        configs_reshaped[i, :, :, 0] = configs[i]
        # Only the center cell's next state matters for training
        next_states_reshaped[i, 1, 1, 0] = next_states[i]
    
    return configs_reshaped, next_states_reshaped

def train_difflogic_ca():
    """Train a differentiable logic CA to learn Game of Life rules."""
    print("Generating training data...")
    initial_states, target_states = generate_training_data()
    
    print(f"Training data shape: {initial_states.shape}")
    print(f"Target data shape: {target_states.shape}")
    
    # Create a new DiffLogic CA
    print("Creating DiffLogic CA model...")
    ca = logicars.PyDiffLogicCA(3, 3, 1, 16)  # 3x3 grid, 1 bit state, 16 perception circuits
    
    # Train the model
    print("Training the model (this may take a while)...")
    # Note: This is where we would call the train method from your Rust library
    # The Rust code has a train method but it seems the PyO3 bindings don't expose it yet
    # ca.train(initial_states, target_states, learning_rate=0.01, epochs=1000)
    
    print("Note: Training functionality is not yet exposed in the Python bindings.")
    print("You need to add a train method to the PyDiffLogicCA class in your Rust code.")
    
    # For now, let's just create a simple visualization of some of the training data
    print("Visualizing some training examples...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            axes[i, j].imshow(initial_states[idx, :, :, 0], cmap='binary')
            axes[i, j].set_title(f"Next state: {target_states[idx, 1, 1, 0]}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return ca

def test_trained_model(ca):
    """Test the trained model on some common Game of Life patterns."""
    # This would be used after training to validate the model
    pass

if __name__ == "__main__":
    ca = train_difflogic_ca()
    # test_trained_model(ca)  # Uncomment after implementing training