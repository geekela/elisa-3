#!/usr/bin/env python3
"""
Animated Ring Attractor Simulation

This script creates an animated visualization of a ring attractor network showing:
1. Neural activity around the ring (like the image you provided)
2. True head direction vs decoded head direction
3. Real-time bump movement as head direction changes
4. Different movement patterns (drift, rotation, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import torch
from hd_ring_attractor.src.models import RingAttractorNetwork
from hd_ring_attractor.src.utils import angle_to_input

class RingAttractorAnimator:
    """
    Animated visualization of ring attractor network dynamics.
    """
    
    def __init__(self, n_exc=800, n_inh=200, sigma_ee=0.5):
        """
        Initialize the ring attractor animator.
        
        Args:
            n_exc: Number of excitatory neurons
            n_inh: Number of inhibitory neurons  
            sigma_ee: E->E connection width
        """
        self.n_exc = n_exc
        self.device = torch.device('cpu')  # Use CPU for better animation performance
        
        # Create ring attractor network
        self.model = RingAttractorNetwork(
            n_exc=n_exc, 
            n_inh=n_inh, 
            sigma_ee=sigma_ee,
            device=self.device
        )
        self.model.eval()  # Set to evaluation mode
        
        # Neuron positions around the ring (angles in radians)
        self.neuron_angles = np.linspace(0, 2*np.pi, n_exc, endpoint=False)
        
        # Convert to Cartesian coordinates for plotting
        self.neuron_x = np.cos(self.neuron_angles)
        self.neuron_y = np.sin(self.neuron_angles)
        
        # Simulation parameters
        self.dt = 0.05  # Time step for animation
        self.current_time = 0.0
        
        # Head direction trajectory
        self.true_directions = []
        self.decoded_directions = []
        self.times = []
        
        # Setup the plot
        self.setup_plot()
        
    def setup_plot(self):
        """
        Set up the matplotlib figure and axes for animation.
        """
        self.fig, (self.ax_ring, self.ax_comparison) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Ring plot setup
        self.ax_ring.set_xlim(-1.5, 1.5)
        self.ax_ring.set_ylim(-1.5, 1.5)
        self.ax_ring.set_aspect('equal')
        self.ax_ring.set_title('Ring Attractor Activity', fontsize=14, fontweight='bold')
        
        # Draw the ring circle
        circle = Circle((0, 0), 1.0, fill=False, color='gray', linewidth=2, alpha=0.5)
        self.ax_ring.add_patch(circle)
        
        # Initialize activity scatter plot (will be updated)
        self.activity_scatter = self.ax_ring.scatter(
            self.neuron_x, self.neuron_y, 
            s=50, c='blue', cmap='hot', vmin=0, vmax=1, alpha=0.8
        )
        
        # True direction arrow
        self.true_arrow = self.ax_ring.annotate(
            '', xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=3),
            fontsize=12, color='green'
        )
        
        # Decoded direction arrow  
        self.decoded_arrow = self.ax_ring.annotate(
            '', xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=3, linestyle='--'),
            fontsize=12, color='red'
        )
        
        # Add legend for arrows
        self.ax_ring.plot([], [], color='green', linewidth=3, label='True Direction')
        self.ax_ring.plot([], [], color='red', linewidth=3, linestyle='--', label='Decoded Direction')
        self.ax_ring.legend(loc='upper right')
        
        # Remove ticks and labels
        self.ax_ring.set_xticks([])
        self.ax_ring.set_yticks([])
        
        # Comparison plot setup
        self.ax_comparison.set_xlim(0, 20)  # Show last 20 seconds
        self.ax_comparison.set_ylim(-np.pi, np.pi)
        self.ax_comparison.set_xlabel('Time (s)')
        self.ax_comparison.set_ylabel('Head Direction (radians)')
        self.ax_comparison.set_title('True vs Decoded Direction', fontsize=14, fontweight='bold')
        self.ax_comparison.grid(True, alpha=0.3)
        
        # Initialize comparison plot lines
        self.true_line, = self.ax_comparison.plot([], [], 'g-', linewidth=2, label='True Direction')
        self.decoded_line, = self.ax_comparison.plot([], [], 'r--', linewidth=2, label='Decoded Direction')
        self.ax_comparison.legend()
        
        # Add time and error text
        self.time_text = self.ax_ring.text(-1.4, 1.3, '', fontsize=12, fontweight='bold')
        self.error_text = self.ax_ring.text(-1.4, 1.1, '', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
    def generate_head_direction_trajectory(self, pattern='drift', duration=20.0):
        """
        Generate a head direction trajectory for animation.
        
        Args:
            pattern: Type of movement ('drift', 'rotation', 'random_walk', 'figure_eight')
            duration: Duration of the trajectory in seconds
        """
        time_steps = int(duration / self.dt)
        trajectory = np.zeros(time_steps)
        
        if pattern == 'drift':
            # Slow leftward drift (like in your image)
            drift_rate = 0.2  # radians per second
            trajectory = np.linspace(0, duration * drift_rate, time_steps)
            
        elif pattern == 'rotation':
            # Constant rotation
            rotation_rate = 0.5  # radians per second  
            trajectory = np.linspace(0, duration * rotation_rate, time_steps)
            
        elif pattern == 'random_walk':
            # Random walk with momentum
            trajectory[0] = 0
            velocity = 0
            for i in range(1, time_steps):
                # Add random acceleration
                acceleration = np.random.normal(0, 0.1)
                velocity += acceleration * self.dt
                velocity *= 0.95  # Damping
                trajectory[i] = trajectory[i-1] + velocity * self.dt
                
        elif pattern == 'figure_eight':
            # Figure-eight pattern
            t = np.linspace(0, duration, time_steps)
            trajectory = 0.8 * np.sin(2 * np.pi * 0.1 * t) * np.cos(2 * np.pi * 0.05 * t)
            
        elif pattern == 'step':
            # Step changes in direction
            trajectory = np.zeros(time_steps)
            step_duration = int(3.0 / self.dt)  # 3 seconds per step
            directions = [0, np.pi/2, np.pi, 3*np.pi/2, 0]
            for i, direction in enumerate(directions):
                start_idx = i * step_duration
                end_idx = min((i + 1) * step_duration, time_steps)
                if start_idx < time_steps:
                    trajectory[start_idx:end_idx] = direction
        
        # Wrap angles to [0, 2π]
        trajectory = np.mod(trajectory, 2 * np.pi)
        
        return trajectory
        
    def update_ring_attractor(self, true_direction):
        """
        Update the ring attractor network with new head direction input.
        
        Args:
            true_direction: True head direction in radians
            
        Returns:
            decoded_direction: Decoded head direction from network activity
        """
        # Create input for the current direction
        input_pattern = angle_to_input(true_direction, n_exc=self.n_exc)
        input_tensor = torch.tensor(input_pattern, dtype=torch.float32, device=self.device)
        
        # Run network for one step
        with torch.no_grad():
            activity = self.model(input_tensor, steps=1)
            decoded_direction = self.model.decode_angle(activity)
            
        return activity.cpu().numpy(), decoded_direction.cpu().numpy()
        
    def animate_frame(self, frame):
        """
        Animation function called for each frame.
        
        Args:
            frame: Frame number
            
        Returns:
            List of updated plot elements
        """
        # Get current true direction from trajectory
        if frame < len(self.trajectory):
            true_direction = self.trajectory[frame]
        else:
            true_direction = self.trajectory[-1]
            
        # Update ring attractor
        activity, decoded_direction = self.update_ring_attractor(true_direction)
        
        # Update current time
        self.current_time = frame * self.dt
        
        # Store for comparison plot
        self.true_directions.append(true_direction)
        self.decoded_directions.append(decoded_direction)
        self.times.append(self.current_time)
        
        # Update ring activity visualization
        self.activity_scatter.set_array(activity)
        self.activity_scatter.set_sizes(50 + 200 * activity)  # Size proportional to activity
        
        # Update direction arrows
        true_x, true_y = 0.8 * np.cos(true_direction), 0.8 * np.sin(true_direction)
        decoded_x, decoded_y = 0.6 * np.cos(decoded_direction), 0.6 * np.sin(decoded_direction)
        
        # Update arrow positions
        self.true_arrow.set_position((true_x, true_y))
        self.true_arrow.xy = (true_x, true_y)
        self.true_arrow.xytext = (0, 0)
        
        self.decoded_arrow.set_position((decoded_x, decoded_y))
        self.decoded_arrow.xy = (decoded_x, decoded_y)
        self.decoded_arrow.xytext = (0, 0)
        
        # Calculate tracking error
        angle_error = np.abs(np.angle(np.exp(1j * (decoded_direction - true_direction))))
        
        # Update text displays
        self.time_text.set_text(f'Time: {self.current_time:.2f}s')
        self.error_text.set_text(f'Error: {np.degrees(angle_error):.1f}°')
        
        # Update comparison plot
        if len(self.times) > 1:
            # Keep only last 20 seconds of data for comparison plot
            window_size = int(20.0 / self.dt)
            if len(self.times) > window_size:
                times_windowed = self.times[-window_size:]
                true_windowed = self.true_directions[-window_size:]
                decoded_windowed = self.decoded_directions[-window_size:]
            else:
                times_windowed = self.times
                true_windowed = self.true_directions
                decoded_windowed = self.decoded_directions
                
            self.true_line.set_data(times_windowed, true_windowed)
            self.decoded_line.set_data(times_windowed, decoded_windowed)
            
            # Update comparison plot limits
            if len(times_windowed) > 0:
                self.ax_comparison.set_xlim(max(0, times_windowed[0]), times_windowed[-1] + 1)
        
        return [self.activity_scatter, self.true_arrow, self.decoded_arrow, 
                self.time_text, self.error_text, self.true_line, self.decoded_line]
    
    def run_animation(self, pattern='drift', duration=20.0, interval=50, save_gif=False):
        """
        Run the animated simulation.
        
        Args:
            pattern: Movement pattern ('drift', 'rotation', 'random_walk', 'figure_eight', 'step')
            duration: Duration of simulation in seconds
            interval: Animation interval in milliseconds
            save_gif: Whether to save animation as GIF
        """
        print(f"🎬 Starting Ring Attractor Animation")
        print(f"   Pattern: {pattern}")
        print(f"   Duration: {duration} seconds")
        print(f"   Network: {self.n_exc} excitatory, {self.model.n_inh} inhibitory neurons")
        print(f"   σ_EE: {self.model.sigma_ee.item():.3f}")
        
        # Generate trajectory
        self.trajectory = self.generate_head_direction_trajectory(pattern, duration)
        
        # Reset network state
        self.model.reset_state()
        
        # Clear previous data
        self.true_directions = []
        self.decoded_directions = []
        self.times = []
        self.current_time = 0.0
        
        # Create animation
        frames = len(self.trajectory)
        anim = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=frames,
            interval=interval, blit=False, repeat=True
        )
        
        # Save as GIF if requested
        if save_gif:
            print(f"💾 Saving animation as ring_attractor_{pattern}.gif...")
            anim.save(f'ring_attractor_{pattern}.gif', writer='pillow', fps=20)
            print(f"✓ Animation saved!")
        
        plt.show()
        
        return anim

def main():
    """
    Main function to run different animation demos.
    """
    print("Ring Attractor Animation Demo")
    
    
    # Create animator
    animator = RingAttractorAnimator(n_exc=800, n_inh=200, sigma_ee=0.5)
    
    # Choose animation pattern
    patterns = ['drift', 'rotation', 'random_walk', 'figure_eight', 'step']
    
    print("Available patterns:")
    for i, pattern in enumerate(patterns):
        print(f"  {i+1}. {pattern}")
    
    try:
        choice = input("\nSelect pattern (1-5) or press Enter for 'drift': ").strip()
        if choice == '':
            pattern = 'drift'
        else:
            pattern = patterns[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice, using 'drift'")
        pattern = 'drift'
    
    # Ask about saving GIF
    save_gif = input("Save as GIF? (y/n): ").strip().lower() == 'y'
    
    # Run animation
    anim = animator.run_animation(
        pattern=pattern, 
        duration=20.0, 
        interval=50,  # 50ms = 20 FPS
        save_gif=save_gif
    )
    
    return anim

if __name__ == "__main__":
    anim = main() 