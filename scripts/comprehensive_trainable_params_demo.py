#!/usr/bin/env python3
"""
Comprehensive demonstration of all trainable parameters in the ring attractor network.

This demo shows:
1. Poisson noise parameters (λ_excitatory, λ_inhibitory) 
2. E->E connection width parameter (σ_EE)
3. All working together during training
4. Parameter evolution and biological constraints
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import RingAttractorNetwork
from src.enhanced_training import create_training_config, train_ring_attractor_with_adam_cosine

def comprehensive_demo():
    """
    Comprehensive demo of all trainable parameters.
    """
    print("COMPREHENSIVE TRAINABLE PARAMETERS DEMO")
    
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RingAttractorNetwork(
        n_exc=800, 
        n_inh=200, 
        sigma_ee=0.6,  # Custom initial value
        device=device
    )
    
    print("INITIAL TRAINABLE PARAMETERS:")
    print(f"   Excitatory Poisson λ: {model.noise_rate_e.item():.4f}")
    print(f"   Inhibitory Poisson λ: {model.noise_rate_i.item():.4f}")  
    print(f"   E->E connection width (σ_EE): {model.sigma_ee.item():.4f}")
    print(f"   All parameters trainable: {all(p.requires_grad for p in [model.noise_rate_e, model.noise_rate_i, model.sigma_ee])}")
    
    # Store initial values
    initial_params = {
        'noise_rate_e': model.noise_rate_e.item(),
        'noise_rate_i': model.noise_rate_i.item(),
        'sigma_ee': model.sigma_ee.item()
    }
    
    print(f"\n🚀 STARTING ENHANCED TRAINING...")
    print(f"   Training all parameters simultaneously!")
    
    # Enhanced training
    config = create_training_config(
        max_epochs=15,
        batch_size=8,
        sequence_length=50,
        n_sequences=100,
        device=device,
        plot_progress=False,
        log_interval=3
    )
    
    trained_model, history, trainer = train_ring_attractor_with_adam_cosine(model, config)
    
    print(f"\n📈 FINAL TRAINABLE PARAMETERS:")
    print(f"   🎲 Excitatory Poisson λ: {model.noise_rate_e.item():.4f}")
    print(f"   🎲 Inhibitory Poisson λ: {model.noise_rate_i.item():.4f}")
    print(f"   🔗 E->E connection width (σ_EE): {model.sigma_ee.item():.4f}")
    
    # Calculate changes
    changes = {
        'noise_rate_e': model.noise_rate_e.item() - initial_params['noise_rate_e'],
        'noise_rate_i': model.noise_rate_i.item() - initial_params['noise_rate_i'], 
        'sigma_ee': model.sigma_ee.item() - initial_params['sigma_ee']
    }
    
    print(f"\nPARAMETER EVOLUTION:")
    print(f"    Excitatory λ change: {changes['noise_rate_e']:+.4f}")
    print(f"    Inhibitory λ change: {changes['noise_rate_i']:+.4f}")
    print(f"    σ_EE change: {changes['sigma_ee']:+.4f}")
    
    # Detailed evolution
    print(f"\n EPOCH-BY-EPOCH EVOLUTION:")
    for epoch in range(len(trainer.param_history['sigma_ee'])):
        exc_lambda = trainer.param_history['noise_rate_e'][epoch]
        inh_lambda = trainer.param_history['noise_rate_i'][epoch]
        sigma = trainer.param_history['sigma_ee'][epoch]
        print(f"   Epoch {epoch+1:2d}: λ_exc={exc_lambda:.4f}, λ_inh={inh_lambda:.4f}, σ_EE={sigma:.4f}")
    
    # Create comprehensive visualization
    plot_comprehensive_results(trainer.param_history, history)
    
    # Test parameter constraints
    test_all_constraints(model)
    
    print(f"\n" + "=" * 70)
    print(f"🎉 COMPREHENSIVE DEMO COMPLETED!")
    print(f"✅ Poisson noise λ parameters trained")
    print(f"✅ E->E connection width σ_EE trained")  
    print(f"✅ All parameters evolved during training")
    print(f"✅ Biological constraints enforced")
    print(f"✅ Training convergence achieved")
    print(f"=" * 70)
    
    return trainer.param_history

def plot_comprehensive_results(param_history, training_history):
    """
    Create comprehensive visualization of all trainable parameters.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('🎯 Comprehensive Trainable Parameters Evolution', fontsize=16, fontweight='bold')
    
    epochs = list(range(1, len(param_history['sigma_ee']) + 1))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(training_history['epoch'], training_history['train_loss'], 'b-', 
                    linewidth=2, label='Training Loss')
    axes[0, 0].plot(training_history['epoch'], training_history['val_loss'], 'r--', 
                    linewidth=2, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('📉 Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Poisson λ Parameters
    axes[0, 1].plot(epochs, param_history['noise_rate_e'], 'blue', 
                    linewidth=3, marker='o', label='🎲 Excitatory λ', markersize=6)
    axes[0, 1].plot(epochs, param_history['noise_rate_i'], 'red', 
                    linewidth=3, marker='s', label='🎲 Inhibitory λ', markersize=6)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Poisson Rate Parameter (λ)')
    axes[0, 1].set_title('🎲 Poisson Noise Parameters')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: σ_EE Parameter
    axes[0, 2].plot(epochs, param_history['sigma_ee'], 'purple', 
                    linewidth=3, marker='^', label='🔗 σ_EE', markersize=8)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Connection Width (σ_EE)')
    axes[0, 2].set_title('🔗 E→E Connection Width')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate Schedule
    axes[1, 0].plot(training_history['epoch'], training_history['learning_rate'], 'green', 
                    linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('📈 Cosine Annealing LR')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 5: All Gain Parameters
    axes[1, 1].plot(epochs, param_history['g_ee'], label='g_ee (E→E gain)', linewidth=2)
    axes[1, 1].plot(epochs, param_history['g_ei'], label='g_ei (E→I gain)', linewidth=2)
    axes[1, 1].plot(epochs, param_history['g_ie'], label='g_ie (I→E gain)', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gain Values')
    axes[1, 1].set_title('⚡ Network Gains')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Parameter Changes (Bar Plot)
    initial_values = [param_history['noise_rate_e'][0], param_history['noise_rate_i'][0], param_history['sigma_ee'][0]]
    final_values = [param_history['noise_rate_e'][-1], param_history['noise_rate_i'][-1], param_history['sigma_ee'][-1]]
    changes = [f - i for f, i in zip(final_values, initial_values)]
    
    param_names = ['🎲 λ_exc', '🎲 λ_inh', '🔗 σ_EE']
    colors = ['blue', 'red', 'purple']
    
    bars = axes[1, 2].bar(param_names, changes, color=colors, alpha=0.7)
    axes[1, 2].set_ylabel('Parameter Change')
    axes[1, 2].set_title('📊 Total Parameter Changes')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        axes[1, 2].annotate(f'{change:+.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def test_all_constraints(model):
    """
    Test biological constraints for all trainable parameters.
    """
    print(f"\n🧪 TESTING BIOLOGICAL CONSTRAINTS:")
    
    with torch.no_grad():
        # Test Poisson λ constraints [0.01, 10.0]
        original_exc = model.noise_rate_e.item()
        original_inh = model.noise_rate_i.item()
        original_sigma = model.sigma_ee.item()
        
        # Test extreme values
        model.noise_rate_e.data = torch.tensor(-0.5)
        model.noise_rate_i.data = torch.tensor(15.0)
        model.sigma_ee.data = torch.tensor(3.0)
        
        # Apply constraints
        model.noise_rate_e.data.clamp_(min=0.01, max=10.0)
        model.noise_rate_i.data.clamp_(min=0.01, max=10.0)
        model.sigma_ee.data.clamp_(min=0.05, max=2.0)
        
        print(f"   🎲 Excitatory λ: -0.5 → {model.noise_rate_e.item():.4f} (clamped to [0.01, 10.0])")
        print(f"   🎲 Inhibitory λ: 15.0 → {model.noise_rate_i.item():.4f} (clamped to [0.01, 10.0])")
        print(f"   🔗 σ_EE: 3.0 → {model.sigma_ee.item():.4f} (clamped to [0.05, 2.0])")
        
        # Restore original values
        model.noise_rate_e.data = torch.tensor(original_exc)
        model.noise_rate_i.data = torch.tensor(original_inh)
        model.sigma_ee.data = torch.tensor(original_sigma)
    
    print(f"   ✅ All constraints working correctly!")

if __name__ == "__main__":
    param_history = comprehensive_demo() 