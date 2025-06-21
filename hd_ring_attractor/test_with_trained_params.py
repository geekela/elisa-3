#!/usr/bin/env python3
"""
Test script for tracking performance with trained/optimal parameters.
This script loads optimal parameters, sets them in the model, and runs tracking tests.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models import RingAttractorNetwork
from src.enhanced_training import create_training_config, train_ring_attractor_with_adam_cosine
from src.utils import angle_to_input, generate_trajectory, compute_error

def train_optimal_parameters(save_path="optimal_params.pth"):
    """
    Train the model to find optimal parameters and save them.
    """
    print("=== Training Model to Find Optimal Parameters ===")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(
        n_exc=200,  # Larger network for better tracking
        n_inh=50,
        sigma_ee=0.5,
        device=device
    )
    
    # Create config for more thorough training
    config = create_training_config(
        max_epochs=50,
        batch_size=16,
        sequence_length=50,
        n_sequences=100,
        learning_rate=2e-3,
        device=device,
        plot_progress=True,
        log_interval=5
    )
    
    # Train the model
    print("Training model to find optimal parameters...")
    trained_model, history, trainer = train_ring_attractor_with_adam_cosine(model, config)
    
    # Save parameters
    torch.save({
        'sigma_ee': model.sigma_ee.item(),
        'g_ee': model.g_ee.item(),
        'g_ei': model.g_ei.item(),
        'g_ie': model.g_ie.item(),
        'g_input': model.g_input.item(),
        'noise_rate_e': model.noise_rate_e.item(),
        'noise_rate_i': model.noise_rate_i.item(),
        'full_state_dict': model.state_dict()
    }, save_path)
    
    print(f"Optimal parameters saved to {save_path}")
    return model

def set_optimal_parameters(model, load_path=None):
    """
    Set optimal parameters either from a saved file or manually.
    """
    if load_path and os.path.exists(load_path):
        # Load parameters from file
        print(f"Loading optimal parameters from {load_path}")
        params = torch.load(load_path)
        model.load_state_dict(params['full_state_dict'])
        
        # Print loaded parameters
        print("\nLoaded Optimal Parameters:")
        print(f"  sigma_ee: {params['sigma_ee']:.4f}")
        print(f"  g_ee: {params['g_ee']:.4f}")
        print(f"  g_ei: {params['g_ei']:.4f}")
        print(f"  g_ie: {params['g_ie']:.4f}")
        print(f"  g_input: {params['g_input']:.4f}")
        print(f"  noise_rate_e: {params['noise_rate_e']:.4f}")
        print(f"  noise_rate_i: {params['noise_rate_i']:.4f}")
    else:
        # Set parameters manually based on knowledge from previous training
        print("Setting manually optimized parameters")
        with torch.no_grad():
            # These values are based on previous training results
            model.sigma_ee.data = torch.tensor(0.55, device=model.device)
            model.g_ee.data = torch.tensor(1.05, device=model.device)
            model.g_ei.data = torch.tensor(1.35, device=model.device)
            model.g_ie.data = torch.tensor(1.85, device=model.device)
            model.g_input.data = torch.tensor(1.2, device=model.device)
            model.noise_rate_e.data = torch.tensor(0.05, device=model.device)
            model.noise_rate_i.data = torch.tensor(0.1, device=model.device)
        
        # Print the manually set parameters
        print("\nManually Set Optimal Parameters:")
        print(f"  sigma_ee: {model.sigma_ee.item():.4f}")
        print(f"  g_ee: {model.g_ee.item():.4f}")
        print(f"  g_ei: {model.g_ei.item():.4f}")
        print(f"  g_ie: {model.g_ie.item():.4f}")
        print(f"  g_input: {model.g_input.item():.4f}")
        print(f"  noise_rate_e: {model.noise_rate_e.item():.4f}")
        print(f"  noise_rate_i: {model.noise_rate_i.item():.4f}")
    
    return model

def test_tracking_with_optimal_params(model=None, param_file="optimal_params.pth"):
    """
    Test tracking performance with optimal parameters.
    """
    print("\n=== Testing Tracking Performance with Optimal Parameters ===")
    
    # Create model if not provided
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = RingAttractorNetwork(
            n_exc=200,
            n_inh=50,
            device=device
        )
        
        # Set optimal parameters
        model = set_optimal_parameters(model, param_file)
    
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
        print(f"Initializing bump at {first_direction:.2f} radians ({first_direction * 180/np.pi:.2f} degrees)")
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
    
    print(f"Average tracking error: {np.degrees():.2f}°")
    print(f"Maximum tracking error: {np.degrees():.2f}°")
    
    # Plot tracking results
    plt.figure(figsize=(14, 8))
    
    # Convert to degrees for plotting
    true_angles_deg = true_angles * 180 / np.pi
    predicted_angles_deg = predicted_angles * 180 / np.pi
    
    plt.plot(true_angles_deg, label='True Direction', linewidth=2)
    plt.plot(predicted_angles_deg, label='Predicted Direction', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Head Direction (degrees)')
    plt.title('Head Direction Tracking Performance with Optimal Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add visualization of tracking error
    plt.figure(figsize=(14, 4))
    error_deg = errors * 180 / np.pi
    plt.plot(error_deg, color='red', label=f'Tracking Error (avg: {avg_error * 180/np.pi:.2f}°)')
    plt.axhline(y=avg_error * 180/np.pi, color='black', linestyle='--', alpha=0.5, 
                label=f'Average Error: {avg_error * 180/np.pi:.2f}°')
    plt.xlabel('Time Step')
    plt.ylabel('Error (degrees)')
    plt.title('Tracking Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return avg_error, model

if __name__ == "__main__":
    param_file = "optimal_params.pth"
    
    # Check if we need to train or load parameters
    if not os.path.exists(param_file):
        print(f"Parameter file {param_file} not found. Training model to find optimal parameters...")
        model = train_optimal_parameters(param_file)
    else:
        print(f"Using existing parameter file: {param_file}")
        model = None  # Will be created in the test function
    
    # Run tracking test with optimal parameters
    avg_error, trained_model = test_tracking_with_optimal_params(model, param_file)
    
    if avg_error < 0.2:  # Less than ~11 degrees
        print("\n✅ Tracking performance with optimal parameters is excellent!")
    else:
        print("\n⚠️ Tracking performance could be further improved.")
        
    # Print final parameter values
    print("\nFinal Parameter Values:")
    print(f"  sigma_ee: {trained_model.sigma_ee.item():.4f}")
    print(f"  g_ee: {trained_model.g_ee.item():.4f}")
    print(f"  g_ei: {trained_model.g_ei.item():.4f}")
    print(f"  g_ie: {trained_model.g_ie.item():.4f}")
    print(f"  g_input: {trained_model.g_input.item():.4f}")
    print(f"  noise_rate_e: {trained_model.noise_rate_e.item():.4f}")
    print(f"  noise_rate_i: {trained_model.noise_rate_i.item():.4f}") 