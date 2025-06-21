#!/usr/bin/env python3
"""
Test script specifically for tracking performance with proper bump initialization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import RingAttractorNetwork
from src.utils import angle_to_input, generate_trajectory, compute_error

def test_tracking_with_bump_initialization():
    """
    Test if initializing with an activity bump at the target direction improves tracking.
    """
    print("\n=== Testing Tracking with Bump Initialization ===")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(
        n_exc=200,
        n_inh=50,
        device=device
    )
    
    # Generate a challenging trajectory with sharp transitions
    seq_len = 100
    angles, angular_velocities = generate_trajectory(
        seq_len, 
        dt=0.1,
        angular_velocity_std=0.5  # More dramatic direction changes
    )
    
    # Create inputs from angles
    inputs = angle_to_input(angles, n_exc=model.n_exc, input_strength=1.5)
    
    # Move tensors to the same device as the model
    inputs = inputs.to(device)
    angles = angles.to(device)
    
    # Add batch dimension
    inputs = inputs.unsqueeze(0)  # [1, seq_len, n_exc]
    angles = angles.unsqueeze(0)  # [1, seq_len]
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Initialize bump at the first direction
        first_direction = angles[0, 0].item()
        first_direction_deg = np.degrees(first_direction)
        print(f"Initializing bump at {first_direction_deg:.2f} degrees")
        model.initialize_bump(first_direction, width=0.3, amplitude=1.0)
        
        # Track the trajectory
        hidden_state = model.r_e  # Start with the initialized bump
        predicted_angles = []
        
        for t in range(seq_len):
            input_t = inputs[:, t, :]
            hidden_state = model(input_t, hidden_state, steps=5)  # Run multiple steps for stability
            predicted_angle = model.decode_angle(hidden_state)
            predicted_angles.append(predicted_angle.item())
    
    # Convert to numpy for analysis
    true_angles = angles[0].cpu().numpy()
    predicted_angles = np.array(predicted_angles)
    
    # Calculate tracking error
    errors = np.abs(np.arctan2(np.sin(predicted_angles - true_angles), np.cos(predicted_angles - true_angles)))
    avg_error = np.mean(errors)
    avg_error_deg = np.degrees(avg_error)
    
    print(f"Average tracking error: {avg_error_deg:.2f} degrees")
    
    # Plot tracking results
    plt.figure(figsize=(14, 8))
    
    # Convert to degrees for plotting
    true_angles_deg = np.degrees(true_angles)
    predicted_angles_deg = np.degrees(predicted_angles)
    
    plt.plot(true_angles_deg, label='True Direction', linewidth=2)
    plt.plot(predicted_angles_deg, label='Predicted Direction', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Head Direction (degrees)')
    plt.title('Head Direction Tracking Performance with Bump Initialization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add visualization of tracking error
    plt.figure(figsize=(14, 4))
    error_deg = np.degrees(errors)
    plt.plot(error_deg, color='red', label=f'Tracking Error (avg: {avg_error_deg:.2f}°)')
    plt.axhline(y=avg_error_deg, color='black', linestyle='--', alpha=0.5, 
                label=f'Average Error: {avg_error_deg:.2f}°')
    plt.xlabel('Time Step')
    plt.ylabel('Error (degrees)')
    plt.title('Tracking Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return avg_error, true_angles_deg, predicted_angles_deg

if __name__ == "__main__":
    avg_error, true_angles, predicted_angles = test_tracking_with_bump_initialization()
    
    if avg_error < 0.2:  # Less than ~11 degrees
        print("\n✅ Tracking performance is good!")
    else:
        print("\n⚠️ Tracking performance could be improved.")
        
    # Print some additional statistics
    print(f"First true direction: {true_angles[0]:.2f}°")
    print(f"First predicted direction: {predicted_angles[0]:.2f}°")
    print(f"Final true direction: {true_angles[-1]:.2f}°")
    print(f"Final predicted direction: {predicted_angles[-1]:.2f}°") 