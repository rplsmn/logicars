import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import sys
try:
    import logicars
except ImportError:
    print("Could not import logicars. Make sure the Rust library is properly built and installed.")
    print("Try running: pip install -e .")
    exit(1)

class GameOfLifeSimulation:
    def __init__(self, width=50, height=50):
        """Initialize the Game of Life simulation using the LogicArs library."""
        self.width = width
        self.height = height
        # Create a Game of Life CA
        self.ca = logicars.create_gol(width, height)
        # Add some patterns
        self.add_initial_patterns()
        
    def add_initial_patterns(self):
        """Add some initial patterns to make the simulation interesting."""
        # Add a few gliders
        self.ca.create_glider(5, 5)
        self.ca.create_glider(15, 20)
        self.ca.create_glider(30, 10)
        
        # Add a random pattern as well
        grid = self.ca.get_grid()
        random_grid = np.random.choice([True, False], size=(self.height, self.width, 1), p=[0.2, 0.8])
        self.ca.set_grid(random_grid)
        
    def step(self):
        """Run one step of the CA simulation."""
        self.ca.step()
        
    def get_grid(self):
        """Get the current state of the grid."""
        return self.ca.get_grid()[:, :, 0]  # Extract the first (and only) bit of state
    
    def run_animation(self, num_frames=100, interval=100):
        """Run an animation of the Game of Life simulation."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a colormap (white for dead cells, black for live cells)
        cmap = ListedColormap(['white', 'black'])
        
        # Simple approach: manually update and redraw for each frame
        for frame in range(num_frames):
            ax.clear()
            self.step()
            grid = self.get_grid()
            ax.imshow(grid, cmap=cmap, interpolation='nearest')
            ax.set_title(f'Game of Life - Generation {frame+1}')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.pause(interval/1000)  # Convert ms to seconds
            
        plt.show()
    
    def run_steps(self, steps=100):
        """Run a specific number of steps and show the final state."""
        for _ in range(steps):
            self.step()
        
        # Plot the final state
        plt.figure(figsize=(10, 10))
        plt.imshow(self.get_grid(), cmap='binary', interpolation='nearest')
        plt.title(f'Game of Life after {steps} generations')
        plt.xticks([])
        plt.yticks([])
        plt.show()

if __name__ == "__main__":
    # Create a simulation
    print("Initializing Game of Life simulation...")
    sim = GameOfLifeSimulation(width=50, height=50)
    
    # Run animation
    print("Running animation...")
    #sim.run_animation(num_frames=100, interval=100)
    
    # Alternatively, run a specific number of steps and show the final state
    sim.run_steps(steps=250)