#!/usr/bin/env python3
"""
Script to run tracking with the optimal parameters.
This script loads the pre-trained parameters and runs the tracking test.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models import RingAttractorNetwork
from src.utils import angle_to_input, generate_trajectory, compute_error

def print_parameters(model):
    """
    Print the current parameters of the model.
    """
    print("\nCurrent Model Parameters:")
    print(f"  sigma_ee: {model.sigma_ee.item():.4f}")
    print(f"  g_ee: {model.g_ee.item():.4f}")
    print(f"  g_ei: {model.g_ei.item():.4f}")
    print(f"  g_ie: {model.g_ie.item():.4f}")
    print(f"  g_input: {model.g_input.item():.4f}")
    print(f"  noise_rate_e: {model.noise_rate_e.item():.4f}")
    print(f"  noise_rate_i: {model.noise_rate_i.item():.4f}")

def load_optimal_parameters(model, param_file="optimal_params.pth"):
    """
    Load optimal parameters from a file.
    """
    if os.path.exists(param_file):
        print(f"Loading optimal parameters from {param_file}")
        params = torch.load(param_file)
        
        # Load the full state dict
        model.load_state_dict(params['full_state_dict'])
        
        print("\nLoaded Optimal Parameters:")
        print(f"  sigma_ee: {params['sigma_ee']:.4f}")
        print(f"  g_ee: {params['g_ee']:.4f}")
        print(f"  g_ei: {params['g_ei']:.4f}")
        print(f"  g_ie: {params['g_ie']:.4f}")
        print(f"  g_input: {params['g_input']:.4f}")
        print(f"  noise_rate_e: {params['noise_rate_e']:.4f}")
        print(f"  noise_rate_i: {params['noise_rate_i']:.4f}")
    else:
        print(f"Parameter file {param_file} not found. Using default parameters.")
        # You could set manual parameters here if desired
        
    return model

def run_tracking_test(model):
    """
    Run tracking test with the given model.
    """
    print("\n=== Running Tracking Test with Optimal Parameters ===")
    
    # Generate a challenging trajectory with sharp transitions
    seq_len = 100
    angles, angular_velocities = generate_trajectory(
        seq_len, 
        dt=0.1,
        angular_velocity_std=0.8  # More dramatic direction changes
    )
    
    # Create inputs from angles
    inputs = angle_to_input(angles, n_exc=model.n_exc, input_strength=2.0)
    
    # Add batch dimension
    inputs = inputs.unsqueeze(0)  # [1, seq_len, n_exc]
    angles = angles.unsqueeze(0)  # [1, seq_len]
    
    # Initialize the network with a bump at the first direction
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
    max_error = np.max(errors)
    
    # Convert to degrees
    avg_error_deg = np.degrees(avg_error)
    max_error_deg = np.degrees(max_error)
    
    print(f"Average tracking error: {avg_error_deg:.2f} degrees")
    print(f"Maximum tracking error: {max_error_deg:.2f} degrees")
    
    # Plot tracking results
    plt.figure(figsize=(14, 8))
    
    # Convert to degrees for plotting
    true_angles_deg = np.degrees(true_angles)
    predicted_angles_deg = np.degrees(predicted_angles)
    
    plt.plot(true_angles_deg, label='True Direction', linewidth=2)
    plt.plot(predicted_angles_deg, label='Predicted Direction', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Head Direction (degrees)')
    plt.title('Head Direction Tracking Performance with Optimal Parameters')
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
    
    return true_angles_deg, predicted_angles_deg, avg_error_deg

if __name__ == "__main__":
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(
        n_exc=200,  # Same size as training
        n_inh=50,
        device=device
    )
    
    # Print default parameters
    print("=== Default Model Parameters ===")
    print_parameters(model)
    
    # Load optimal parameters
    model = load_optimal_parameters(model, "optimal_params.pth")
    
    # Run tracking test
    true_angles, predicted_angles, avg_error_deg = run_tracking_test(model)
    
    # Print final evaluation
    if avg_error_deg < 11.5:  # Using degrees instead of radians
        print("\n✅ Tracking performance with optimal parameters is excellent!")
    else:
        print("\n⚠️ Tracking performance could be further improved.") 