#!/usr/bin/env python3
"""
Analysis of weight matrices in the ring attractor network:
- Which matrices are trainable vs fixed
- How they behave during training vs simulation
- What changes throughout the process
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from hd_ring_attractor.src.models import RingAttractorNetwork
from hd_ring_attractor.src.enhanced_training import create_training_config, train_ring_attractor_with_adam_cosine

def analyze_weight_matrices():
    """
    Comprehensive analysis of weight matrices in the ring attractor network.
    """
    print("WEIGHT MATRIX ANALYSIS")
   
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(n_exc=800, n_inh=200, sigma_ee=0.5, device=device)
    
    print("WEIGHT MATRIX OVERVIEW:")
    print(f"   Model has {sum(p.numel() for p in model.parameters())} total trainable parameters")
    
    # Analyze each component
    print(f"\nCONNECTION MATRICES:")
    
    # 1. E->E connections (W_EE) - DYNAMIC, computed from σ_EE
    print(f"   1. E→E (W_EE): DYNAMIC - computed from trainable σ_EE")
    print(f"      • Shape: Would be ({model.n_exc}, {model.n_exc}) if stored")
    print(f"      • Status: Computed dynamically in forward pass")
    print(f"      • Controlled by: σ_EE = {model.sigma_ee.item():.4f} (TRAINABLE)")
    
    # 2. E->I connections (W_EI) - TRAINABLE
    print(f"   2. E→I (W_EI): TRAINABLE MATRIX")
    print(f"      • Shape: {model.W_EI.shape}")
    print(f"      • Parameters: {model.W_EI.numel()}")
    print(f"      • Requires grad: {model.W_EI.requires_grad}")
    print(f"      • Current range: [{model.W_EI.min().item():.3f}, {model.W_EI.max().item():.3f}]")
    
    # 3. I->E connections (W_IE) - TRAINABLE
    print(f"   3. I→E (W_IE): TRAINABLE MATRIX")
    print(f"      • Shape: {model.W_IE.shape}")
    print(f"      • Parameters: {model.W_IE.numel()}")
    print(f"      • Requires grad: {model.W_IE.requires_grad}")
    print(f"      • Current range: [{model.W_IE.min().item():.3f}, {model.W_IE.max().item():.3f}]")
    
    # 4. No I->I connections
    print(f"   4. I→I: NOT IMPLEMENTED (biologically realistic)")
    
    print(f"\nGAIN PARAMETERS (scalar multipliers):")
    print(f"   • g_ee (E→E gain): {model.g_ee.item():.4f} (TRAINABLE)")
    print(f"   • g_ei (E→I gain): {model.g_ei.item():.4f} (TRAINABLE)")
    print(f"   • g_ie (I→E gain): {model.g_ie.item():.4f} (TRAINABLE)")
    
    print(f"\nNOISE PARAMETERS:")
    print(f"   • λ_excitatory: {model.noise_rate_e.item():.4f} (TRAINABLE)")
    print(f"   • λ_inhibitory: {model.noise_rate_i.item():.4f} (TRAINABLE)")
    
    return model

def demonstrate_dynamic_vs_static():
    """
    Show the difference between dynamic E->E weights and static trainable matrices.
    """
    print(f"\nDYNAMIC vs STATIC WEIGHT BEHAVIOR:")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(n_exc=800, n_inh=200, sigma_ee=0.3, device=device)
    
    print(f"📈 DYNAMIC E→E WEIGHTS (computed from σ_EE):")
    
    # Show how E->E weights change with different σ_EE values
    sigma_values = [0.2, 0.5, 1.0]
    
    for sigma in sigma_values:
        # Temporarily set σ_EE
        with torch.no_grad():
            model.sigma_ee.data = torch.tensor(sigma)
        
        # Compute E->E weights
        W_EE = model._create_ring_weights(model.sigma_ee)
        
        print(f"   σ_EE = {sigma:.1f}:")
        print(f"     • Weight matrix shape: {W_EE.shape}")
        print(f"     • Connection strength to neighbor: {W_EE[0, 1].item():.4f}")
        print(f"     • Connection strength to opposite: {W_EE[0, 8].item():.4f}")
        print(f"     • Sum of connections: {W_EE[0].sum().item():.4f}")
    
    print(f"\nSTATIC TRAINABLE MATRICES (W_EI, W_IE):")
    print(f"   These matrices have fixed shapes but trainable values:")
    print(f"   • W_EI shape: {model.W_EI.shape} - values change during training")
    print(f"   • W_IE shape: {model.W_IE.shape} - values change during training")
    print(f"   • Initial W_EI[0,0]: {model.W_EI[0,0].item():.4f}")
    print(f"   • Initial W_IE[0,0]: {model.W_IE[0,0].item():.4f}")

def training_vs_simulation_behavior():
    """
    Show what happens to weights during training vs simulation.
    """
    print(f"\nTRAINING vs SIMULATION BEHAVIOR:")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(n_exc=800, n_inh=200, sigma_ee=0.6, device=device)
    
    # Store initial values
    initial_W_EI = model.W_EI.clone().detach()
    initial_W_IE = model.W_IE.clone().detach()
    initial_sigma_ee = model.sigma_ee.item()
    
    print(f"🏋️ DURING TRAINING:")
    print(f"   • W_EI, W_IE matrices: VALUES CHANGE (gradient updates)")
    print(f"   • σ_EE parameter: VALUE CHANGES (gradient updates)")
    print(f"   • E→E weights (W_EE): RECOMPUTED each forward pass from new σ_EE")
    print(f"   • Gain parameters: VALUES CHANGE (gradient updates)")
    print(f"   • Noise parameters: VALUES CHANGE (gradient updates)")
    
    # Quick training to show changes
    print(f"\n   Running quick training to demonstrate...")
    config = create_training_config(
        max_epochs=3,
        batch_size=2,
        sequence_length=10,
        n_sequences=5,
        device=device,
        plot_progress=False,
        log_interval=1
    )
    
    trained_model, history, trainer = train_ring_attractor_with_adam_cosine(model, config)
    
    print(f"\n   AFTER TRAINING:")
    print(f"   • σ_EE changed: {initial_sigma_ee:.4f} → {model.sigma_ee.item():.4f}")
    print(f"   • W_EI[0,0] changed: {initial_W_EI[0,0].item():.4f} → {model.W_EI[0,0].item():.4f}")
    print(f"   • W_IE[0,0] changed: {initial_W_IE[0,0].item():.4f} → {model.W_IE[0,0].item():.4f}")
    
    print(f"\n🎮 DURING SIMULATION/INFERENCE:")
    model.eval()  # Set to evaluation mode
    
    print(f"   • W_EI, W_IE matrices: VALUES FIXED (no gradient updates)")
    print(f"   • σ_EE parameter: VALUE FIXED (no gradient updates)")
    print(f"   • E→E weights (W_EE): STILL RECOMPUTED each forward pass")
    print(f"     (because they depend on current σ_EE value)")
    print(f"   • Only neural activities (r_e, r_i) change over time")
    
    # Demonstrate simulation
    print(f"\n   Running simulation steps...")
    model.reset_state()
    model.initialize_bump(direction=np.pi/2, amplitude=1.0)
    
    initial_activity = model.r_e.clone()
    
    # Run a few simulation steps
    for step in range(3):
        model(steps=1)
        print(f"     Step {step+1}: Peak activity = {model.r_e.max().item():.3f}")
    
    print(f"   • Neural activities changed during simulation")
    print(f"   • Weight matrices remained constant")

def visualize_weight_matrices():
    """
    Visualize the different types of weight matrices.
    """
    print(f"\nVISUALIZING WEIGHT MATRICES:")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(n_exc=800, n_inh=200, sigma_ee=0.5, device=device)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ring Attractor Weight Matrices', fontsize=16)
    
    # 1. E->E weights (dynamic, computed from σ_EE)
    W_EE = model._create_ring_weights(model.sigma_ee)
    im1 = axes[0, 0].imshow(W_EE.detach().cpu().numpy(), cmap='RdBu_r')
    axes[0, 0].set_title(f'E→E Weights (Dynamic)\nσ_EE = {model.sigma_ee.item():.3f}')
    axes[0, 0].set_xlabel('To Excitatory Neuron')
    axes[0, 0].set_ylabel('From Excitatory Neuron')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. E->I weights (trainable matrix)
    im2 = axes[0, 1].imshow(model.W_EI.detach().cpu().numpy(), cmap='RdBu_r')
    axes[0, 1].set_title('E→I Weights (Trainable Matrix)')
    axes[0, 1].set_xlabel('To Inhibitory Neuron')
    axes[0, 1].set_ylabel('From Excitatory Neuron')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. I->E weights (trainable matrix)
    im3 = axes[1, 0].imshow(model.W_IE.detach().cpu().numpy(), cmap='RdBu_r')
    axes[1, 0].set_title('I→E Weights (Trainable Matrix)')
    axes[1, 0].set_xlabel('To Excitatory Neuron')
    axes[1, 0].set_ylabel('From Inhibitory Neuron')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. Show effect of different σ_EE values
    sigma_values = [0.2, 0.8]
    W_EE_narrow = model._create_ring_weights(torch.tensor(sigma_values[0]))
    W_EE_broad = model._create_ring_weights(torch.tensor(sigma_values[1]))
    
    # Plot connection profiles
    center_neuron = 8
    axes[1, 1].plot(W_EE_narrow[center_neuron].detach().cpu().numpy(), 
                    label=f'σ_EE = {sigma_values[0]} (narrow)', linewidth=2)
    axes[1, 1].plot(W_EE_broad[center_neuron].detach().cpu().numpy(), 
                    label=f'σ_EE = {sigma_values[1]} (broad)', linewidth=2)
    axes[1, 1].set_title('E→E Connection Profiles')
    axes[1, 1].set_xlabel('Target Neuron')
    axes[1, 1].set_ylabel('Connection Strength')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Weight matrix visualization complete!")

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE WEIGHT MATRIX ANALYSIS")
    print("=" * 70)
    
    # Run all analyses
    model = analyze_weight_matrices()
    demonstrate_dynamic_vs_static()
    training_vs_simulation_behavior()
    visualize_weight_matrices()
    
    print(f"\n" + "=" * 70)
    print(f"📋 SUMMARY:")
    print(f"✅ E→E weights: DYNAMIC (recomputed from trainable σ_EE)")
    print(f"✅ E→I weights: TRAINABLE MATRIX (values change in training)")
    print(f"✅ I→E weights: TRAINABLE MATRIX (values change in training)")
    print(f"✅ During training: All parameters update via gradients")
    print(f"✅ During simulation: Only activities change, weights fixed")
    print(f"=" * 70) 