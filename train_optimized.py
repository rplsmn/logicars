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
    
    print(f"Training for {epochs} epochs...")
    start_time = time.time()
    
    # Debug shapes
    print(f"Input shapes - configs: {configs.shape}, targets: {targets.shape}")
    
    try:
        # Train with loss tracking callback
        epoch_start = time.time()
        for epoch in range(epochs):
            
            # Call the train_epoch method (to be implemented in Rust)
            soft_loss, hard_loss = ca.train_epoch(configs, targets, learning_rate)                      
            
            # Update loss tracker
            loss_tracker.update(epoch, soft_loss, hard_loss)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch}/{epochs}: Soft Loss = {soft_loss:.6f}, Hard Loss = {hard_loss:.6f}, Time: {epoch_time:.2f}s")
                epoch_start = time.time()
    
    except AttributeError:
        print("The train_epoch method is not available. Falling back to standard training.")
        try:
            ca.train(configs, targets, learning_rate, epochs)
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
                ca.train(small_configs, small_targets, learning_rate, 10)
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
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.25, help='Learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size')
    parser.add_argument('--tmp', type=float, default=0.05, help='Temperature')
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
