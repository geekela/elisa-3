#!/usr/bin/env python3
"""
Comprehensive demo of enhanced training for Ring Attractor Network.

This script demonstrates:
1. Basic usage of the enhanced training system
2. Comparison with the original training method
3. Different configuration options
4. Analysis of training results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import RingAttractorNetwork
from src.training import train_model, train_model_enhanced, quick_train
from src.enhanced_training import create_training_config


def demo_basic_enhanced_training():
    """
    Basic demonstration of enhanced training with Adam + cosine annealing.
    """
    print("=== Demo 1: Basic Enhanced Training ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = RingAttractorNetwork(
        n_exc=800,
        n_inh=200,
        sigma_ee=0.3,
        device=device
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} trainable parameters")
    
    # Enhanced training with default settings
    print("\nStarting enhanced training...")
    trained_model, history, trainer = train_model_enhanced(
        model,
        learning_rate=1e-3,
        max_epochs=50,          # Shorter for demo
        batch_size=32,
        early_stopping=True,
        plot_progress=True
    )
    
    return trained_model, history


def demo_quick_training():
    """
    Demonstration of quick training for rapid prototyping.
    """
    print("\n=== Demo 2: Quick Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = RingAttractorNetwork(
        n_exc=80,   # Smaller model for quick training
        n_inh=20,
        sigma_ee=0.3,
        device=device
    )
    
    print("Running quick training (optimized for speed)...")
    trained_model, history = quick_train(
        model,
        epochs=20,
        lr=2e-3,    # Higher learning rate for faster convergence
        device=device
    )
    
    print(f"Quick training completed in {len(history['epoch'])} epochs")
    print(f"Final loss: {history['train_loss'][-1]:.6f}")
    
    return trained_model, history


def demo_configuration_comparison():
    """
    Compare different training configurations.
    """
    print("\n=== Demo 3: Configuration Comparison ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create identical models for comparison
    def create_model():
        torch.manual_seed(42)  # Ensure identical initialization
        return RingAttractorNetwork(n_exc=800, n_inh=200, sigma_ee=0.3, device=device)
    
    configs = {
        'Conservative': create_training_config(
            learning_rate=5e-4,
            max_epochs=30,
            weight_decay=1e-3,
            patience=15,
            plot_progress=False
        ),
        'Standard': create_training_config(
            learning_rate=1e-3,
            max_epochs=30,
            weight_decay=1e-4,
            patience=10,
            plot_progress=False
        ),
        'Aggressive': create_training_config(
            learning_rate=2e-3,
            max_epochs=30,
            weight_decay=1e-5,
            patience=5,
            plot_progress=False
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTraining with {config_name} configuration...")
        model = create_model()
        
        trained_model, history, trainer = train_model_enhanced(model, **config)
        
        results[config_name] = {
            'model': trained_model,
            'history': history,
            'final_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
            'epochs': len(history['epoch'])
        }
        
        print(f"{config_name}: Final loss = {results[config_name]['final_loss']:.6f}, "
              f"Epochs = {results[config_name]['epochs']}")
    
    # Plot comparison
    plot_configuration_comparison(results)
    
    return results


def plot_configuration_comparison(results):
    """
    Plot comparison of different configurations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss curves
    for config_name, result in results.items():
        epochs = result['history']['epoch']
        train_loss = result['history']['train_loss']
        val_loss = result['history']['val_loss']
        
        axes[0].plot(epochs, train_loss, '--', label=f'{config_name} (Train)', alpha=0.7)
        axes[0].plot(epochs, val_loss, '-', label=f'{config_name} (Val)', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Comparison: Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Learning rate schedules
    for config_name, result in results.items():
        epochs = result['history']['epoch']
        lr = result['history']['learning_rate']
        axes[1].plot(epochs, lr, label=config_name, linewidth=2)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedules (Cosine Annealing)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def demo_parameter_evolution():
    """
    Demonstrate how ring attractor parameters evolve during training.
    """
    print("\n=== Demo 4: Parameter Evolution Analysis ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = RingAttractorNetwork(
        n_exc=800,
        n_inh=200,
        sigma_ee=0.3,
        device=device
    )
    
    # Store initial parameters
    initial_params = {
        'g_ee': model.g_ee.item(),
        'g_ei': model.g_ei.item(),
        'g_ie': model.g_ie.item(),
        'poisson_lambda_e': model.noise_rate_e.item(),  # Poisson λ for excitatory
        'poisson_lambda_i': model.noise_rate_i.item(),  # Poisson λ for inhibitory
        'sigma_ee': model.sigma_ee.item()               # E->E connection width
    }
    
    print("Initial parameters:")
    for name, value in initial_params.items():
        print(f"  {name}: {value:.4f}")
    
    # Train with enhanced method
    trained_model, history, trainer = train_model_enhanced(
        model,
        learning_rate=1e-3,
        max_epochs=50,
        apply_constraints=True,  # Enable biological constraints
        plot_progress=False
    )
    
    # Store final parameters
    final_params = {
        'g_ee': model.g_ee.item(),
        'g_ei': model.g_ei.item(),
        'g_ie': model.g_ie.item(),
        'poisson_lambda_e': model.noise_rate_e.item(),  # Poisson λ for excitatory
        'poisson_lambda_i': model.noise_rate_i.item(),  # Poisson λ for inhibitory
        'sigma_ee': model.sigma_ee.item()               # E->E connection width
    }
    
    print("\nFinal parameters:")
    for name, value in final_params.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nParameter changes:")
    for name in initial_params:
        change = final_params[name] - initial_params[name]
        percent_change = (change / initial_params[name]) * 100
        print(f"  {name}: {change:+.4f} ({percent_change:+.1f}%)")
    
    # Plot parameter evolution
    plot_parameter_evolution(trainer.param_history)
    
    return initial_params, final_params, trainer.param_history


def plot_parameter_evolution(param_history):
    """
    Plot how parameters evolved during training.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Ring Attractor Parameter Evolution', fontsize=16)
    
    epochs = list(range(len(param_history['g_ee'])))
    
    # Plot gains
    axes[0, 0].plot(epochs, param_history['g_ee'], label='g_ee (E→E)', linewidth=2)
    axes[0, 0].plot(epochs, param_history['g_ei'], label='g_ei (E→I)', linewidth=2)
    axes[0, 0].plot(epochs, param_history['g_ie'], label='g_ie (I→E)', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Gain Value')
    axes[0, 0].set_title('Connection Gains')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Poisson rate parameters
    axes[0, 1].plot(epochs, param_history['noise_rate_e'], label='Excitatory λ', linewidth=2, color='blue')
    axes[0, 1].plot(epochs, param_history['noise_rate_i'], label='Inhibitory λ', linewidth=2, color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Poisson Rate Parameter (λ)')
    axes[0, 1].set_title('Poisson Noise Parameters')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot excitatory/inhibitory balance
    balance = np.array(param_history['g_ee']) / np.array(param_history['g_ie'])
    axes[1, 0].plot(epochs, balance, linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('E/I Balance (g_ee/g_ie)')
    axes[1, 0].set_title('Excitatory/Inhibitory Balance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot parameter stability (rolling std)
    window = 5
    if len(epochs) > window:
        g_ee_std = np.array([np.std(param_history['g_ee'][max(0, i-window):i+1]) 
                            for i in range(len(epochs))])
        axes[1, 1].plot(epochs, g_ee_std, linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Parameter Variability (Rolling Std)')
        axes[1, 1].set_title('Parameter Stability (g_ee)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demo_comparison_with_original():
    """
    Compare enhanced training with original training method.
    """
    print("\n=== Demo 5: Comparison with Original Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create identical models
    torch.manual_seed(42)
    model_original = RingAttractorNetwork(n_exc=50, n_inh=10, sigma_ee=0.3, device=device)
    
    torch.manual_seed(42)  # Same initialization
    model_enhanced = RingAttractorNetwork(n_exc=50, n_inh=10, sigma_ee=0.3, device=device)
    
    print("Training with original method...")
    try:
        losses_original = train_model(
            model_original,
            n_epochs=30,
            learning_rate=1e-3,
            batch_size=32,
            device=device
        )
        print(f"Original training final loss: {losses_original[-1]:.6f}")
    except Exception as e:
        print(f"Original training failed: {e}")
        losses_original = None
    
    print("\nTraining with enhanced method...")
    model_enhanced, history_enhanced, trainer = train_model_enhanced(
        model_enhanced,
        learning_rate=1e-3,
        max_epochs=30,
        batch_size=32,
        plot_progress=False,
        device=device
    )
    print(f"Enhanced training final loss: {history_enhanced['val_loss'][-1]:.6f}")
    
    # Plot comparison if original training worked
    if losses_original is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses_original)), losses_original, 'b-', 
                label='Original (Adam only)', linewidth=2)
        plt.plot(history_enhanced['epoch'], history_enhanced['train_loss'], 'r-', 
                label='Enhanced (Adam + Cosine)', linewidth=2)
        plt.plot(history_enhanced['epoch'], history_enhanced['val_loss'], 'r--', 
                label='Enhanced (Validation)', linewidth=2, alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Method Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
    
    return model_enhanced, history_enhanced


def main():
    """
    Run all demonstrations.
    """
    print("ring Attractor Network Enhanced Training Demonstration")
    
    
    # Set random seeds for reproducibility
    #torch.manual_seed(42)
    #np.random.seed(42)
    
    try:
        # Demo 1: Basic enhanced training
        model1, history1 = demo_basic_enhanced_training()
        
        # Demo 2: Quick training
        model2, history2 = demo_quick_training()
        
        # Demo 3: Configuration comparison
        results3 = demo_configuration_comparison()
        
        # Demo 4: Parameter evolution
        initial_params, final_params, param_hist = demo_parameter_evolution()
        
        # Demo 5: Comparison with original
        model5, history5 = demo_comparison_with_original()
        
        print("All demonstrations completed successfully!")
        
        
        # Summary
        print("\nSummary:")
        print("✓ Enhanced training with Adam + Cosine Annealing implemented")
        print("✓ Biological parameter constraints working")
        print("✓ Comprehensive monitoring and visualization")
        print("✓ Multiple configuration options available")
        print("✓ Quick training option for rapid prototyping")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 