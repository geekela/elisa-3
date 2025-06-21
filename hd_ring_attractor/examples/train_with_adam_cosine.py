#!/usr/bin/env python3
"""
Example script showing how to train your Ring Attractor Network 
with Adam optimizer and cosine annealing learning rate schedule.

This script demonstrates:
1. How to initialize your ring attractor network
2. How to configure the enhanced training setup
3. How to run training with Adam + cosine annealing
4. How to analyze and visualize the results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from hd_ring_attractor.src.models import RingAttractorNetwork  # Your existing model
from src.enhanced_training import train_ring_attractor_with_adam_cosine, create_training_config


def main():
    """
    Main training function demonstrating Adam + Cosine Annealing setup.
    """
    
    print("=== Ring Attractor Network Training with Adam + Cosine Annealing ===")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    #torch.manual_seed(42)
    #np.random.seed(42)
    
    # Initialize your ring attractor network
    model = RingAttractorNetwork(
        n_exc=800,             
        n_inh=200,              
        sigma_ee=0.5,           # Width of excitatory connections
        device=device
    )
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print("\n=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        print(f"{name:15} | Shape: {str(param.shape):15} | Params: {param.numel():6}")
    
    # Create training configuration
    config = create_training_config(
        # Optimizer settings
        learning_rate=1e-3,         # Initial learning rate
        weight_decay=1e-4,          # L2 regularization
        betas=(0.9, 0.999),         # Adam momentum parameters
        
        # Scheduler settings  
        max_epochs=100,             # Total training epochs
        min_lr=1e-6,               # Minimum learning rate for cosine annealing
        
        # Training settings
        batch_size=32,              # Batch size
        sequence_length=100,        # Sequence length
        n_sequences=1000,           # Number of training sequences
        validation_split=0.2,       # 20% for validation
        
        # Regularization
        clip_gradients=True,        # Enable gradient clipping
        max_grad_norm=1.0,         # Maximum gradient norm
        apply_constraints=True,     # Apply biological constraints
        
        # Monitoring
        log_interval=10,            # Log every 10 epochs
        save_checkpoints=True,      # Save best model
        plot_progress=True,         # Plot training curves
        
        # Early stopping
        early_stopping=True,
        patience=20,                # Stop if no improvement for 20 epochs
        min_delta=1e-6,            # Minimum improvement threshold
        
        # Device
        device=device
    )
    
    print("✓ Training configuration created")
    
    # Display initial parameter values
    print("\n=== Initial Ring Attractor Parameters ===")
    print(f"Excitatory-Excitatory gain (g_ee): {model.g_ee.item():.4f}")
    print(f"Excitatory-Inhibitory gain (g_ei): {model.g_ei.item():.4f}")
    print(f"Inhibitory-Excitatory gain (g_ie): {model.g_ie.item():.4f}")
    print(f"Excitatory Poisson λ (noise_rate_e): {model.noise_rate_e.item():.4f}")
    print(f"Inhibitory Poisson λ (noise_rate_i): {model.noise_rate_i.item():.4f}")
    print(f"E->E connection width (σ_EE): {model.sigma_ee.item():.4f}")
    
    # Train the model
    print("\n" + "="*60)
    print("Starting enhanced training...")
    print("="*60)
    
    trained_model, history, trainer = train_ring_attractor_with_adam_cosine(model, config)
    
    # Print final results
    print("\n=== Training Results ===")
    if history['train_loss']:
        print(f"Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Final learning rate: {history['learning_rate'][-1]:.2e}")
        print(f"Total epochs completed: {len(history['epoch'])}")
    
    # Print learned parameters
    print("\n=== Learned Ring Attractor Parameters ===")
    print(f"Excitatory-Excitatory gain (g_ee): {model.g_ee.item():.4f}")
    print(f"Excitatory-Inhibitory gain (g_ei): {model.g_ei.item():.4f}")
    print(f"Inhibitory-Excitatory gain (g_ie): {model.g_ie.item():.4f}")
    print(f"Excitatory noise rate: {model.noise_rate_e.item():.4f}")
    print(f"Inhibitory noise rate: {model.noise_rate_i.item():.4f}")
    
    # Analyze parameter changes
    if trainer.param_history['g_ee']:
        print("\n=== Parameter Evolution ===")
        initial_g_ee = trainer.param_history['g_ee'][0]
        final_g_ee = trainer.param_history['g_ee'][-1]
        print(f"g_ee change: {initial_g_ee:.4f} → {final_g_ee:.4f} (Δ={final_g_ee-initial_g_ee:+.4f})")
        
        initial_g_ei = trainer.param_history['g_ei'][0]
        final_g_ei = trainer.param_history['g_ei'][-1]
        print(f"g_ei change: {initial_g_ei:.4f} → {final_g_ei:.4f} (Δ={final_g_ei-initial_g_ei:+.4f})")
        
        initial_g_ie = trainer.param_history['g_ie'][0]
        final_g_ie = trainer.param_history['g_ie'][-1]
        print(f"g_ie change: {initial_g_ie:.4f} → {final_g_ie:.4f} (Δ={final_g_ie-initial_g_ie:+.4f})")
    
    # Save final model
    model_path = 'trained_ring_attractor_adam_cosine.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_config': config,
        'training_history': history,
        'final_loss': history['val_loss'][-1] if history['val_loss'] else float('inf')
    }, model_path)
    print(f"\n✓ Model saved as '{model_path}'")
    
    # Demonstrate model usage after training
    print("\n=== Testing Trained Model ===")
    test_model_performance(model, device)
    
    return trained_model, history, trainer


def test_model_performance(model, device):
    """
    Quick test of the trained model's performance on a simple trajectory.
    """
    from src.utils import generate_trajectory, angle_to_input, compute_error
    
    model.eval()
    
    # Generate a test trajectory
    test_length = 50
    angles, _ = generate_trajectory(test_length, dt=0.1, angular_velocity_std=0.1)
    inputs = angle_to_input(angles, n_exc=model.n_exc)
    
    # Move to device
    inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension
    angles = angles.to(device)
    
    # Test the model
    with torch.no_grad():
        h = None
        predicted_angles = []
        errors = []
        
        for t in range(test_length):
            input_t = inputs[:, t, :]
            true_angle = angles[t]
            
            # Forward pass
            h = model(input_t, h, steps=1)
            predicted_angle = model.decode_angle(h).squeeze()
            
            # Compute error
            error = compute_error(predicted_angle, true_angle)
            
            predicted_angles.append(predicted_angle.item())
            errors.append(error.item())
    
    # Print performance statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    
    print(f"Test trajectory performance:")
    print(f"  Mean error: {mean_error:.4f} radians ({np.degrees(mean_error):.2f}°)")
    print(f"  Std error:  {std_error:.4f} radians ({np.degrees(std_error):.2f}°)")
    print(f"  Max error:  {max_error:.4f} radians ({np.degrees(max_error):.2f}°)")
    
    return mean_error, predicted_angles, errors


def demonstrate_configuration_options():
    """
    Show different configuration options for training.
    """
    print("\n=== Example Configuration Options ===")
    
    # Conservative training (slower but more stable)
    conservative_config = create_training_config(
        learning_rate=5e-4,         # Lower learning rate
        max_epochs=200,             # More epochs
        batch_size=16,              # Smaller batches
        patience=30,                # More patience
        weight_decay=5e-4,          # Higher regularization
    )
    print("Conservative config: Lower LR, more epochs, higher regularization")
    
    # Aggressive training (faster but potentially less stable)
    aggressive_config = create_training_config(
        learning_rate=2e-3,         # Higher learning rate
        max_epochs=50,              # Fewer epochs
        batch_size=64,              # Larger batches
        patience=10,                # Less patience
        weight_decay=1e-5,          # Lower regularization
    )
    print("Aggressive config: Higher LR, fewer epochs, lower regularization")
    
    # Research-focused (detailed monitoring)
    research_config = create_training_config(
        learning_rate=1e-3,
        max_epochs=100,
        log_interval=5,             # More frequent logging
        save_checkpoints=True,
        plot_progress=True,
        early_stopping=False,       # Train for full duration
    )
    print("Research config: Detailed monitoring, no early stopping")
    
    return conservative_config, aggressive_config, research_config


if __name__ == "__main__":
    # Run the main training example
    model, history, trainer = main()
    
    # Show configuration options
    demonstrate_configuration_options()
    
    print("\n" + "="*60)
    print("Training complete! Check the plots for detailed analysis.")
    print("="*60) 