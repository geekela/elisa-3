#!/usr/bin/env python3
"""
Script to visualize weight matrices and connectivity patterns in the trained ring attractor network.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from src.models import RingAttractorNetwork

def load_trained_model(param_file="optimal_params.pth"):
    """
    Load a trained model from a saved parameter file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = RingAttractorNetwork(
        n_exc=800,  # Increased for higher resolution visualization
        n_inh=200,  # Increased for higher resolution visualization
        device=device
    )
    
    # Load parameters if file exists
    if os.path.exists(param_file):
        print(f"Loading trained model from {param_file}")
        params = torch.load(param_file)
        
        # Since we're changing the network size, we can't load the state dict directly
        # Instead, we'll set the scalar parameters individually
        with torch.no_grad():
            # Set scalar parameters
            model.sigma_ee.data = torch.tensor(params['sigma_ee'], device=device)
            model.g_ee.data = torch.tensor(params['g_ee'], device=device)
            model.g_ei.data = torch.tensor(params['g_ei'], device=device)
            model.g_ie.data = torch.tensor(params['g_ie'], device=device)
            model.g_input.data = torch.tensor(params['g_input'], device=device)
            model.noise_rate_e.data = torch.tensor(params['noise_rate_e'], device=device)
            model.noise_rate_i.data = torch.tensor(params['noise_rate_i'], device=device)
            
            # The weight matrices and preferred directions will be initialized correctly
            # for the new size by the model's __init__ method
        
        # Print parameters
        print("\nModel Parameters:")
        print(f"  sigma_ee: {model.sigma_ee.item():.4f}")
        print(f"  g_ee: {model.g_ee.item():.4f}")
        print(f"  g_ei: {model.g_ei.item():.4f}")
        print(f"  g_ie: {model.g_ie.item():.4f}")
        print(f"  g_input: {model.g_input.item():.4f}")
        print(f"  noise_rate_e: {model.noise_rate_e.item():.4f}")
        print(f"  noise_rate_i: {model.noise_rate_i.item():.4f}")
        print(f"  Network size: {model.n_exc} excitatory, {model.n_inh} inhibitory neurons")
    else:
        print(f"Parameter file {param_file} not found. Using default parameters.")
        print(f"  Network size: {model.n_exc} excitatory, {model.n_inh} inhibitory neurons")
    
    return model

def visualize_ee_connectivity(model):
    """
    Visualize the E->E connectivity pattern.
    """
    print("\nVisualizing E->E connectivity...")
    
    # Get the E->E weight matrix
    W_EE = model._create_ring_weights(model.sigma_ee).detach().cpu().numpy()
    
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Plot the full weight matrix
    im0 = axes[0].imshow(W_EE, cmap='viridis')
    axes[0].set_title(f'E→E Connectivity Matrix (σ={model.sigma_ee.item():.3f})')
    axes[0].set_xlabel('Postsynaptic Neuron Index')
    axes[0].set_ylabel('Presynaptic Neuron Index')
    
    # Add colorbar
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im0, cax=cax)
    
    # 2. Plot a single row (connectivity pattern for one neuron)
    neuron_idx = model.n_exc // 2  # Middle neuron
    axes[1].plot(W_EE[neuron_idx], 'b-', linewidth=2)
    axes[1].set_title(f'Connectivity Pattern for Neuron {neuron_idx}')
    axes[1].set_xlabel('Postsynaptic Neuron Index')
    axes[1].set_ylabel('Connection Strength')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return W_EE

def visualize_inhibitory_weights(model):
    """
    Visualize the E->I and I->E weight matrices.
    """
    print("\nVisualizing inhibitory weights...")
    
    # Get the weight matrices
    W_EI = model.W_EI.detach().cpu().numpy()  # Shape: [n_exc, n_inh]
    W_IE = model.W_IE.detach().cpu().numpy()  # Shape: [n_inh, n_exc]
    
    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Plot E->I weights (raw)
    im0 = axes[0, 0].imshow(W_EI, cmap='coolwarm')
    axes[0, 0].set_title(f'E→I Weight Matrix (gain={model.g_ei.item():.3f})')
    axes[0, 0].set_xlabel('Inhibitory Neuron Index')
    axes[0, 0].set_ylabel('Excitatory Neuron Index')
    
    # Add colorbar
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im0, cax=cax)
    
    # 2. Plot I->E weights (raw)
    im1 = axes[0, 1].imshow(W_IE, cmap='coolwarm')
    axes[0, 1].set_title(f'I→E Weight Matrix (gain={model.g_ie.item():.3f})')
    axes[0, 1].set_xlabel('Excitatory Neuron Index')
    axes[0, 1].set_ylabel('Inhibitory Neuron Index')
    
    # Add colorbar
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax)
    
    # 3. Plot mean E->I weights by excitatory neuron
    mean_ei = np.mean(W_EI, axis=1)  # Average across inhibitory neurons
    axes[1, 0].plot(mean_ei, 'g-', linewidth=2)
    axes[1, 0].set_title('Average E→I Weight by Excitatory Neuron')
    axes[1, 0].set_xlabel('Excitatory Neuron Index')
    axes[1, 0].set_ylabel('Average Weight')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Plot mean I->E weights by inhibitory neuron
    mean_ie = np.mean(W_IE, axis=1)  # Average across excitatory neurons
    axes[1, 1].plot(mean_ie, 'r-', linewidth=2)
    axes[1, 1].set_title('Average I→E Weight by Inhibitory Neuron')
    axes[1, 1].set_xlabel('Inhibitory Neuron Index')
    axes[1, 1].set_ylabel('Average Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return W_EI, W_IE

def visualize_effective_weights(model):
    """
    Visualize the effective weights (with gains applied).
    """
    print("\nVisualizing effective weights (with gains)...")
    
    # Get the weight matrices with gains applied
    W_EE = model._create_ring_weights(model.sigma_ee).detach().cpu().numpy() * model.g_ee.item()
    W_EI = model.W_EI.detach().cpu().numpy() * model.g_ei.item()  # Apply gain
    W_IE = model.W_IE.detach().cpu().numpy() * model.g_ie.item()  # Apply gain
    
    # Calculate effective inhibition (I->E after E->I)
    # This approximates the "net" inhibitory effect in the ring
    W_effective_inh = np.zeros((model.n_exc, model.n_exc))
    for i in range(model.n_exc):
        for j in range(model.n_exc):
            # For each E->E pair, calculate the inhibition that flows through interneurons
            # This is a simplification - in reality there's complex dynamics
            inhib_contribution = 0
            for k in range(model.n_inh):
                # E_i -> I_k -> E_j pathway
                inhib_contribution += W_EI[i, k] * W_IE[k, j]
            W_effective_inh[i, j] = -inhib_contribution  # Negative because it's inhibition
    
    # Create a figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Plot effective E->E weights
    vmax = np.max([np.abs(W_EE).max(), np.abs(W_effective_inh).max()])
    im0 = axes[0].imshow(W_EE, cmap='Reds', vmin=0, vmax=vmax)
    axes[0].set_title(f'Effective E→E Weights (g_ee={model.g_ee.item():.3f})')
    axes[0].set_xlabel('Postsynaptic Neuron Index')
    axes[0].set_ylabel('Presynaptic Neuron Index')
    
    # Add colorbar
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im0, cax=cax)
    
    # 2. Plot effective inhibition
    im1 = axes[1].imshow(W_effective_inh, cmap='Blues_r', vmin=-vmax, vmax=0)
    axes[1].set_title('Effective Inhibition')
    axes[1].set_xlabel('Postsynaptic Neuron Index')
    axes[1].set_ylabel('Presynaptic Neuron Index')
    
    # Add colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax)
    
    # 3. Plot net connectivity (excitation + inhibition)
    net_connectivity = W_EE + W_effective_inh
    im2 = axes[2].imshow(net_connectivity, cmap='coolwarm', 
                        vmin=-vmax, vmax=vmax)
    axes[2].set_title('Net Connectivity (E→E + Inhibition)')
    axes[2].set_xlabel('Postsynaptic Neuron Index')
    axes[2].set_ylabel('Presynaptic Neuron Index')
    
    # Add colorbar
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    plt.show()
    
    return W_EE, W_effective_inh, net_connectivity

def visualize_weight_profiles(model):
    """
    Visualize the weight profiles from the perspective of a single neuron.
    """
    print("\nVisualizing weight profiles...")
    
    # Calculate the weights
    W_EE = model._create_ring_weights(model.sigma_ee).detach().cpu().numpy() * model.g_ee.item()
    W_EI = model.W_EI.detach().cpu().numpy() * model.g_ei.item()
    W_IE = model.W_IE.detach().cpu().numpy() * model.g_ie.item()
    
    # Calculate effective inhibition for visualization
    W_effective_inh = np.zeros((model.n_exc, model.n_exc))
    for i in range(model.n_exc):
        for j in range(model.n_exc):
            inhib_contribution = 0
            for k in range(model.n_inh):
                inhib_contribution += W_EI[i, k] * W_IE[k, j]
            W_effective_inh[i, j] = -inhib_contribution
    
    # Choose a neuron in the middle
    neuron_idx = model.n_exc // 2
    
    # Get its weight profiles
    ee_profile = W_EE[neuron_idx]
    inh_profile = W_effective_inh[neuron_idx]
    net_profile = ee_profile + inh_profile
    
    # Convert indices to angles for better visualization (in degrees)
    angles = np.linspace(0, 360, model.n_exc, endpoint=False)
    
    # Create a figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Plot weight profiles as functions of neuron index
    axes[0].plot(ee_profile, 'r-', linewidth=2, label='E→E Excitation')
    axes[0].plot(inh_profile, 'b-', linewidth=2, label='Effective Inhibition')
    axes[0].plot(net_profile, 'g-', linewidth=2, label='Net Effect')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].axvline(x=neuron_idx, color='k', linestyle='--', alpha=0.5, 
                   label=f'Neuron {neuron_idx}')
    axes[0].set_title('Weight Profiles from a Single Neuron Perspective (By Index)')
    axes[0].set_xlabel('Target Neuron Index')
    axes[0].set_ylabel('Connection Strength')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Plot weight profiles as functions of angle difference
    axes[1].plot(angles, np.roll(ee_profile, -neuron_idx), 'r-', linewidth=2, 
                label='E→E Excitation')
    axes[1].plot(angles, np.roll(inh_profile, -neuron_idx), 'b-', linewidth=2, 
                label='Effective Inhibition')
    axes[1].plot(angles, np.roll(net_profile, -neuron_idx), 'g-', linewidth=2, 
                label='Net Effect')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5, 
                   label='Reference Direction')
    axes[1].set_title('Weight Profiles from a Single Neuron Perspective (By Angle)')
    axes[1].set_xlabel('Angle Difference (degrees)')
    axes[1].set_ylabel('Connection Strength')
    axes[1].set_xticks(np.arange(0, 361, 45))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ee_profile, inh_profile, net_profile

def visualize_polar_weights(model):
    """
    Visualize the weight profiles in polar coordinates to show the ring structure.
    """
    print("\nVisualizing weight profiles in polar coordinates...")
    
    # Calculate the weights
    W_EE = model._create_ring_weights(model.sigma_ee).detach().cpu().numpy() * model.g_ee.item()
    
    # Calculate effective inhibition for visualization
    W_EI = model.W_EI.detach().cpu().numpy() * model.g_ei.item()
    W_IE = model.W_IE.detach().cpu().numpy() * model.g_ie.item()
    W_effective_inh = np.zeros((model.n_exc, model.n_exc))
    for i in range(model.n_exc):
        for j in range(model.n_exc):
            inhib_contribution = 0
            for k in range(model.n_inh):
                inhib_contribution += W_EI[i, k] * W_IE[k, j]
            W_effective_inh[i, j] = -inhib_contribution
    
    # Choose neurons to visualize
    neuron_indices = [0, model.n_exc//4, model.n_exc//2, 3*model.n_exc//4]
    
    # Create a figure
    fig = plt.figure(figsize=(15, 12))
    
    # Convert to polar coordinates (in radians for polar plot)
    angles = np.linspace(0, 2*np.pi, model.n_exc, endpoint=False)
    # Get corresponding degrees for display
    angles_deg = np.degrees(angles)
    
    # Plot for each selected neuron
    for i, idx in enumerate(neuron_indices):
        # Get neuron's weight profiles
        ee_profile = W_EE[idx]
        inh_profile = W_effective_inh[idx]
        net_profile = ee_profile + inh_profile
        
        # Scale profiles for better visualization
        max_val = max(np.max(np.abs(ee_profile)), np.max(np.abs(inh_profile)), 
                      np.max(np.abs(net_profile)))
        ee_scaled = ee_profile / max_val
        inh_scaled = inh_profile / max_val
        net_scaled = net_profile / max_val
        
        # Create polar subplot
        ax = fig.add_subplot(2, 2, i+1, projection='polar')
        
        # Plot profiles
        ax.plot(angles, ee_scaled, 'r-', linewidth=2, label='E→E')
        ax.plot(angles, inh_scaled, 'b-', linewidth=2, label='Inhibition')
        ax.plot(angles, net_scaled, 'g-', linewidth=2, label='Net')
        
        # Mark the neuron's position
        neuron_angle = angles[idx]
        ax.plot([neuron_angle, neuron_angle], [0, 1.2], 'k--', alpha=0.5)
        
        # Set title and legend
        ax.set_title(f'Neuron at {np.degrees(neuron_angle):.0f}°')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Weight Profiles in Polar Coordinates', fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    # Load trained model
    model = load_trained_model("optimal_params.pth")
    
    # Visualize various aspects of the weight matrices
    W_EE = visualize_ee_connectivity(model)
    W_EI, W_IE = visualize_inhibitory_weights(model)
    W_EE_gain, W_eff_inh, W_net = visualize_effective_weights(model)
    ee_profile, inh_profile, net_profile = visualize_weight_profiles(model)
    visualize_polar_weights(model) 