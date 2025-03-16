import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
print("Starting import...")
start_time = time.time()
try:
    import logicars
    print(f"Import completed in {time.time() - start_time:.2f} seconds")
except ImportError:
    print("Import failed!")
    exit(1)

def apply_gol_rules(grid, r, c):
    """Apply Game of Life rules to a cell at position (r,c) without wrapping"""
    height, width = grid.shape
    live_neighbors = 0
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Skip the cell itself
                
            # Calculate neighbor position without wrapping
            nr = r + dr
            nc = c + dc
            
            # Only count if within grid boundaries
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc]:
                live_neighbors += 1
    
    # Apply Conway's Game of Life rules
    if grid[r, c]:  # Cell is alive
        return live_neighbors == 2 or live_neighbors == 3
    else:  # Cell is dead
        return live_neighbors == 3

def generate_gol_training_data():
    """
    Generate all 512 possible 3x3 grid configurations and their next states
    according to Game of Life rules.
    """
    
    # Generate all 512 possible 3x3 grid configurations
    configs = []
    next_states = []
    
    for i in range(512):
        binary = format(i, '09b')
        grid = np.array([int(bit) for bit in binary]).reshape(3, 3).astype(bool)
        
        # Only compute next state for CENTER cell (1,1)
        next_center = apply_gol_rules(grid, 1, 1)
        
        configs.append(grid)
        next_states.append(next_center)
    
    # Reshape for training - configs remain 3x3 grids
    configs_reshaped = np.zeros((512, 3, 3, 1), dtype=bool)
    # Next states need to match the expected shape for Rust interface
    next_states_reshaped = np.zeros((512, 3, 3, 1), dtype=bool)
    
    for i in range(512):
        configs_reshaped[i, :, :, 0] = configs[i]
        next_states_reshaped[i, 1, 1, 0] = next_states[i]
    
    return configs_reshaped, next_states_reshaped

def test_gol_rules():
    """Test the Game of Life rules implementation with known patterns."""
    # Test a blinker pattern
    grid1 = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=bool)
    
    expected1 = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=bool)
    
    # Apply rules to each cell
    result1 = np.zeros((3, 3), dtype=bool)
    for r in range(3):
        for c in range(3):
            result1[r, c] = apply_gol_rules(grid1, r, c)
            print(f"Cell ({r},{c}): {grid1[r,c]} -> {result1[r,c]}")
    
    print("Blinker test passed:", np.array_equal(result1, expected1))
    print("Expected:\n", expected1)
    print("Got:\n", result1)

    # Test a glider pattern
    grid2 = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=bool)
    
    expected2 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1]
    ], dtype=bool)
    
    result2 = np.zeros_like(grid2)
    for r in range(3):
        for c in range(3):
            result2[r, c] = apply_gol_rules(grid2, r, c)
    
    print("Glider test passed:", np.array_equal(result2, expected2))
    if not np.array_equal(result2, expected2):
        print("Expected:\n", expected2)
        print("Got:\n", result2)

class LossTracker:
    """Track and visualize loss evolution during training"""
    def __init__(self):
        self.soft_losses = []
        self.hard_losses = []
        self.epochs = []
        
        # Create a live plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.soft_line, = self.ax.plot([], [], 'b-', label='Soft Loss')
        self.hard_line, = self.ax.plot([], [], 'r-', label='Hard Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss Evolution')
        self.ax.legend()
        self.ax.grid(True)
        
    def update(self, epoch, soft_loss, hard_loss):
        self.epochs.append(epoch)
        self.soft_losses.append(soft_loss)
        self.hard_losses.append(hard_loss)
        
        # Update plot
        self.soft_line.set_data(self.epochs, self.soft_losses)
        self.hard_line.set_data(self.epochs, self.hard_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def save(self, filename='loss_evolution.png'):
        plt.ioff()
        plt.figure(figsize=(12, 8))
        plt.plot(self.epochs, self.soft_losses, 'b-', label='Soft Loss')
        plt.plot(self.epochs, self.hard_losses, 'r-', label='Hard Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        print(f"Loss plot saved to {filename}")

def evaluate_model(ca, test_configs, test_targets, n_samples=10, visualize=False):
    """Evaluate trained model performance on sample configurations"""
    correct = 0
    total = 0
    
    # Select random samples
    indices = np.random.choice(len(test_configs), n_samples)
    
    if visualize:
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
        
        # Visualize if requested
        if visualize:
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
    
    if visualize:
        plt.tight_layout()
        plt.show()
    
    accuracy = correct / total * 100
    print(f"Model accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def train_gol_model(epochs=200, learning_rate=0.001, batch_size = 64, temperature = 0.1, l2_strength = 0.001, visualize=False, save_loss_plot=True):
    """Train a DiffLogic CA to learn Game of Life rules"""
    
    # Test the Game of Life rules implementation
    # print("Testing Game of Life rules implementation...")
    # test_gol_rules()
    
    print("Generating training data...")
    configs, targets = generate_gol_training_data()
    
    print(f"Training data: {len(configs)} configurations")
    
    # Create a new DiffLogic CA (3x3 grid, 1 bit state, 16 perception circuits)
    print("Creating DiffLogic CA model...")
    ca = logicars.PyDiffLogicCA(3, 3, 1, 16)
    
    ca.set_batch_size(batch_size)
    ca.set_temperature(temperature)
    ca.set_l2_strength(l2_strength)

    # Initialize loss tracker
    loss_tracker = LossTracker()
    
    min_temperature = 0.5    # Don't go below this
    min_learning_rate = 0.001

    print(f"Training for {epochs} epochs...")
    print(f"Batch size : {batch_size}...")
    print(f"Initial learning rate : {learning_rate}...")
    print(f"Initial temperature : {temperature}...")
    print(f"L2 strength set at : {l2_strength}...")

    start_time = time.time()
    
    # Debug shapes
    print(f"Input shapes - configs: {configs.shape}, targets: {targets.shape}")
    
    try:
        # Train with loss tracking callback
        epoch_start = time.time()
        for epoch in range(epochs):
            
            # Slower initial learning rate with gradual increase 
            # or should it be learning rate decay ??
            current_learning_rate = learning_rate * (1.0 - (epoch / epochs) * 0.8) # Decay 
            current_learning_rate = max(current_learning_rate, min_learning_rate)  # Cap at 0.001         

            # Calculate decaying temperature
            current_temperature = max(
                temperature * (1.0 - epoch / epochs),  # Decay 
                min_temperature
            )

            # Set the temperature for this epoch
            ca.set_temperature(current_temperature)
     
            soft_loss, hard_loss = ca.train_epoch(configs, targets, current_learning_rate, epoch)                      
            
            # Update loss tracker
            loss_tracker.update(epoch, soft_loss, hard_loss)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch}/{epochs}: Soft Loss = {soft_loss:.6f}, Hard Loss = {hard_loss:.6f}, Time: {epoch_time:.2f}s")
                epoch_start = time.time()

        print("Starting cooldown phase...")
        cooldown_epochs = 50
        cooldown_lr = min_learning_rate * 0.1
        fixed_temperature = min_temperature

        for epoch in range(epochs, epochs + cooldown_epochs):
            ca.set_temperature(fixed_temperature) 
            soft_loss, hard_loss = ca.train_epoch(configs, targets, cooldown_lr, epoch)
            loss_tracker.update(epoch, soft_loss, hard_loss)
            
            if epoch % 10 == 0:
                print(f"Cooldown {epoch-epochs}/{cooldown_epochs}: Soft Loss = {soft_loss:.6f}, Hard Loss = {hard_loss:.6f}")

    except AttributeError:
        print("The train_epoch method is not available. Falling back to standard training.")
        try:
            ca.train(configs, targets, current_learning_rate, epochs)
            print("Training completed without loss tracking.")
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
                ca.train(small_configs, small_targets, current_learning_rate, 10)
                print("Training with smaller batch succeeded!")
            except Exception as e:
                print(f"Still failed with error: {e}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    # Save loss plot if tracking was successful
    if save_loss_plot and len(loss_tracker.epochs) > 0:
        loss_tracker.save()
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(ca, configs, targets, visualize=visualize)
    
    return ca

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DiffLogic CA to learn Game of Life rules')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('--tmp', type=float, default=3, help='Temperature')
    parser.add_argument('--l2', type=float, default=0.001, help='L2 regularisation strength')
    parser.add_argument('--visualize', action='store_true', help='Visualize evaluation results')
    parser.add_argument('--no-save-plot', action='store_false', dest='save_plot', help='Do not save loss plot')
    args = parser.parse_args()
    
    ca = train_gol_model(
        epochs=args.epochs, 
        learning_rate=args.lr, 
        batch_size=args.batchsize,
        temperature=args.tmp,
        l2_strength=args.l2,
        visualize=args.visualize,
        save_loss_plot=args.save_plot
    )
