#!/usr/bin/env python3
"""
Animated Ring Attractor Simulation

This script creates an animated visualization of the ring attractor network showing:
1. Neural activity bumps forming at head directions
2. Bump movement as head direction changes
3. Real-time comparison of actual vs decoded head direction
4. Different movement patterns (drift, rotation, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import torch
from src.models import RingAttractorNetwork
from src.utils import angle_to_input, generate_trajectory

class RingAttractorAnimator:
    """
    Interactive animated visualization of ring attractor dynamics.
    """
    
    def __init__(self, model=None, n_neurons=64):
        """
        Initialize the ring attractor animator.
        
        Args:
            model: Trained RingAttractorNetwork (optional, will create default if None)
            n_neurons: Number of neurons in the ring
        """
        self.n_neurons = n_neurons
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create or use provided model
        if model is None:
            self.model = RingAttractorNetwork(
                n_exc=n_neurons, 
                n_inh=n_neurons//4, 
                sigma_ee=0.5,
                device=self.device
            )
        else:
            self.model = model
            self.n_neurons = model.n_exc
        
        self.model.eval()  # Set to evaluation mode
        
        # Animation parameters
        self.time_step = 0
        self.max_time_steps = 1000
        self.dt = 0.05  # Time step for simulation
        self.speed_factor = 1.0
        self.is_playing = False
        
        # Create preferred directions for visualization
        self.preferred_dirs = np.linspace(0, 2*np.pi, self.n_neurons, endpoint=False)
        
        # Initialize head direction trajectory
        self.reset_trajectory("slow_drift")
        
        # Set up the figure and animation
        self.setup_figure()
        self.setup_animation()
        
    def reset_trajectory(self, pattern="slow_drift"):
        """
        Generate different head direction patterns.
        """
        time_points = np.arange(self.max_time_steps) * self.dt
        
        if pattern == "slow_drift":
            # Slow leftward drift
            angular_velocity = -0.5  # rad/s
            self.head_directions = np.mod(angular_velocity * time_points, 2*np.pi)
            self.pattern_name = "Slow left drift"
            
        elif pattern == "rotation":
            # Full rotation
            angular_velocity = 1.0  # rad/s
            self.head_directions = np.mod(angular_velocity * time_points, 2*np.pi)
            self.pattern_name = "Clockwise rotation"
            
        elif pattern == "oscillation":
            # Oscillatory movement
            amplitude = np.pi/2
            frequency = 0.3
            self.head_directions = np.pi + amplitude * np.sin(2*np.pi*frequency*time_points)
            self.pattern_name = "Oscillation"
            
        elif pattern == "random_walk":
            # Random walk
            np.random.seed(42)  # For reproducibility
            angular_velocities = np.random.normal(0, 0.2, self.max_time_steps)
            self.head_directions = np.cumsum(angular_velocities * self.dt)
            self.head_directions = np.mod(self.head_directions, 2*np.pi)
            self.pattern_name = "Random walk"
            
        elif pattern == "step_function":
            # Step changes in direction
            steps = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 0])
            step_duration = self.max_time_steps // len(steps)
            self.head_directions = np.repeat(steps, step_duration)[:self.max_time_steps]
            self.pattern_name = "Step changes"
        
        # Initialize tracking arrays
        self.decoded_directions = np.zeros(self.max_time_steps)
        self.tracking_errors = np.zeros(self.max_time_steps)
        self.neural_activities = np.zeros((self.max_time_steps, self.n_neurons))
        
        # Reset model state
        self.model.reset_state()
        self.time_step = 0
        
    def setup_figure(self):
        """
        Set up the matplotlib figure with subplots.
        """
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Ring Attractor Network Simulation', fontsize=16, fontweight='bold')
        
        # Create subplot layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main ring plot (top left, large)
        self.ax_ring = self.fig.add_subplot(gs[0:2, 0:2], projection='polar')
        self.ax_ring.set_title('Neural Activity Ring\n(Red=Actual, Blue=Decoded)', fontsize=12)
        self.ax_ring.set_ylim(0, 1.2)
        self.ax_ring.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Direction comparison plot (top right)
        self.ax_directions = self.fig.add_subplot(gs[0, 2])
        self.ax_directions.set_title('Direction Tracking', fontsize=10)
        self.ax_directions.set_xlabel('Time (s)')
        self.ax_directions.set_ylabel('Angle (degrees)')
        self.ax_directions.set_ylim(0, 360)  # Changed from 0-2π to 0-360 degrees
        self.ax_directions.grid(True, alpha=0.3)
        
        # Error plot (middle right)
        self.ax_error = self.fig.add_subplot(gs[1, 2])
        self.ax_error.set_title('Tracking Error', fontsize=10)
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Error (degrees)')
        self.ax_error.grid(True, alpha=0.3)
        
        # Activity heatmap (bottom)
        self.ax_activity = self.fig.add_subplot(gs[2, :])
        self.ax_activity.set_title('Neural Activity Over Time', fontsize=10)
        self.ax_activity.set_xlabel('Neuron Index')
        self.ax_activity.set_ylabel('Time (s)')
        
        # Initialize empty plots
        self.line_activity, = self.ax_ring.plot([], [], 'g-', linewidth=3, label='Neural Activity')
        self.line_actual, = self.ax_ring.plot([], [], 'ro', markersize=10, label='Actual Direction')
        self.line_decoded, = self.ax_ring.plot([], [], 'bo', markersize=8, label='Decoded Direction')
        
        self.line_dir_actual, = self.ax_directions.plot([], [], 'r-', linewidth=2, label='Actual')
        self.line_dir_decoded, = self.ax_directions.plot([], [], 'b--', linewidth=2, label='Decoded')
        
        self.line_error, = self.ax_error.plot([], [], 'purple', linewidth=2)
        
        # Add legends
        self.ax_ring.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        self.ax_directions.legend()
        
        # Text displays
        self.text_info = self.fig.text(0.02, 0.95, '', fontsize=10, verticalalignment='top')
        self.text_stats = self.fig.text(0.02, 0.85, '', fontsize=9, verticalalignment='top')
        
    def setup_animation(self):
        """
        Set up animation controls and sliders.
        """
        # Add control buttons
        ax_play = plt.axes([0.1, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.2, 0.02, 0.08, 0.04])
        ax_pattern = plt.axes([0.3, 0.02, 0.15, 0.04])
        
        self.btn_play = Button(ax_play, 'Play/Pause')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_pattern = Button(ax_pattern, 'Change Pattern')
        
        # Add speed slider
        ax_speed = plt.axes([0.5, 0.02, 0.2, 0.04])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)
        
        # Connect button callbacks
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_reset.on_clicked(self.reset_animation)
        self.btn_pattern.on_clicked(self.cycle_pattern)
        self.slider_speed.on_changed(self.update_speed)
        
        # Pattern cycle
        self.patterns = ["slow_drift", "rotation", "oscillation", "random_walk", "step_function"]
        self.current_pattern_idx = 0
        
    def toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        
    def reset_animation(self, event):
        """Reset animation to beginning."""
        self.time_step = 0
        self.model.reset_state()
        self.is_playing = False
        
    def cycle_pattern(self, event):
        """Cycle through different movement patterns."""
        self.current_pattern_idx = (self.current_pattern_idx + 1) % len(self.patterns)
        pattern = self.patterns[self.current_pattern_idx]
        self.reset_trajectory(pattern)
        self.time_step = 0
        self.model.reset_state()
        
    def update_speed(self, val):
        """Update animation speed."""
        self.speed_factor = val
        
    def simulate_step(self):
        """
        Simulate one time step of the ring attractor network.
        """
        if self.time_step >= self.max_time_steps:
            return False
            
        # Get current head direction
        current_direction = self.head_directions[self.time_step]
        
        # Convert to input pattern
        input_pattern = angle_to_input(
            torch.tensor(current_direction), 
            n_exc=self.n_neurons,
            input_strength=1.0,
            input_width=0.3
        )
        
        # Run network forward pass
        with torch.no_grad():
            activity = self.model(input_pattern.to(self.device), steps=1)
            
        # Decode direction from network activity
        decoded_direction = self.model.decode_angle(activity)
        
        # Store results
        self.neural_activities[self.time_step] = activity.cpu().numpy()
        self.decoded_directions[self.time_step] = decoded_direction.cpu().numpy()
        
        # Calculate tracking error (circular difference)
        error = np.angle(np.exp(1j * (decoded_direction.cpu().numpy() - current_direction)))
        self.tracking_errors[self.time_step] = np.abs(error)
        
        return True
        
    def update_plots(self):
        """
        Update all plots with current data.
        """
        if self.time_step == 0:
            return
            
        current_time = self.time_step * self.dt
        time_window = min(self.time_step, 200)  # Show last 200 time steps
        
        # Update ring activity plot
        activity = self.neural_activities[self.time_step]
        self.line_activity.set_data(self.preferred_dirs, activity)
        
        # Update direction markers
        actual_dir = self.head_directions[self.time_step]
        decoded_dir = self.decoded_directions[self.time_step]
        
        self.line_actual.set_data([actual_dir], [1.1])
        self.line_decoded.set_data([decoded_dir], [1.05])
        
        # Update direction tracking plot
        start_idx = max(0, self.time_step - time_window)
        time_points = np.arange(start_idx, self.time_step) * self.dt
        
        # Convert directions to degrees for plotting
        actual_dirs_deg = np.degrees(self.head_directions[start_idx:self.time_step])
        decoded_dirs_deg = np.degrees(self.decoded_directions[start_idx:self.time_step])
        
        self.line_dir_actual.set_data(time_points, actual_dirs_deg)
        self.line_dir_decoded.set_data(time_points, decoded_dirs_deg)
        
        # Update error plot
        error_degrees = np.degrees(self.tracking_errors[start_idx:self.time_step])
        self.line_error.set_data(time_points, error_degrees)
        
        # Update activity heatmap
        if self.time_step > 1:
            display_steps = min(self.time_step, 100)
            start_t = max(0, self.time_step - display_steps)
            
            activity_data = self.neural_activities[start_t:self.time_step].T
            
            self.ax_activity.clear()
            im = self.ax_activity.imshow(
                activity_data, 
                aspect='auto', 
                cmap='hot', 
                origin='lower',
                extent=[start_t*self.dt, current_time, 0, self.n_neurons]
            )
            self.ax_activity.set_title('Neural Activity Over Time')
            self.ax_activity.set_xlabel('Time (s)')
            self.ax_activity.set_ylabel('Neuron Index')
        
        # Update axis limits
        if time_points.size > 0:
            self.ax_directions.set_xlim(time_points[0], time_points[-1] + 1)
            self.ax_error.set_xlim(time_points[0], time_points[-1] + 1)
            
            if self.time_step > 10:
                max_error = np.max(self.tracking_errors[start_idx:self.time_step])
                self.ax_error.set_ylim(0, max_error * 1.1)
        
        # Update info text
        self.text_info.set_text(
            f"Pattern: {self.pattern_name}\n"
            f"Time: {current_time:.2f}s\n"
            f"Step: {self.time_step}/{self.max_time_steps}"
        )
        
        # Update statistics
        if self.time_step > 10:
            recent_errors = self.tracking_errors[max(0, self.time_step-50):self.time_step]
            mean_error = np.mean(recent_errors)
            max_error = np.max(recent_errors)
            
            # Convert errors to degrees
            mean_error_deg = np.degrees(mean_error)
            max_error_deg = np.degrees(max_error)
            
            self.text_stats.set_text(
                f"Recent tracking error:\n"
                f"Mean: {mean_error_deg:.1f}°\n"
                f"Max: {max_error_deg:.1f}°\n"
                f"Peak activity: {np.max(activity):.3f}"
            )
            
    def animate(self, frame):
        """
        Animation function called by matplotlib.
        """
        if self.is_playing and self.time_step < self.max_time_steps:
            # Simulate multiple steps based on speed factor
            steps_per_frame = max(1, int(self.speed_factor))
            
            for _ in range(steps_per_frame):
                if not self.simulate_step():
                    break
                self.time_step += 1
                
        self.update_plots()
        
        return (self.line_activity, self.line_actual, self.line_decoded,
                self.line_dir_actual, self.line_dir_decoded, self.line_error)
    
    def run(self, interval=50):
        """
        Start the animation.
        
        Args:
            interval: Animation interval in milliseconds
        """
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, interval=interval, 
            blit=False, repeat=True
        )
        
        plt.show()
        return self.anim

def load_trained_model():
    """
    Load a trained model if available, otherwise use default.
    """
    try:
        # Try to load a trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RingAttractorNetwork(n_exc=800, n_inh=200, device=device)
        
        # You can add model loading here if you have saved models
        # model.load_state_dict(torch.load('trained_model.pth'))
        
        print("Using default ring attractor model")
        return model
        
    except Exception as e:
        print(f"Could not load trained model: {e}")
        print("Using default model")
        return None

if __name__ == "__main__":
    print("🎬 Starting Ring Attractor Animation")
    print("Controls:")
    print("  - Play/Pause: Start/stop animation")
    print("  - Reset: Return to beginning")
    print("  - Change Pattern: Cycle through movement patterns")
    print("  - Speed slider: Adjust animation speed")
    print("\nPatterns available:")
    print("  1. Slow left drift")
    print("  2. Clockwise rotation") 
    print("  3. Oscillation")
    print("  4. Random walk")
    print("  5. Step changes")
    
    # Load model and start animation
    model = load_trained_model()
    animator = RingAttractorAnimator(model=model)
    anim = animator.run()
    
    # Keep the animation running
    plt.show() 