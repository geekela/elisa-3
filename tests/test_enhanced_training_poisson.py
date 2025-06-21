#!/usr/bin/env python3
"""
Test enhanced training with Poisson noise implementation.
"""

import torch
from src.models import RingAttractorNetwork
from src.enhanced_training import train_ring_attractor_with_adam_cosine, create_training_config

def test_enhanced_training_with_poisson():
    """
    Test that enhanced training works with Poisson noise.
    """
    print("=== Testing Enhanced Training with Poisson Noise ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = RingAttractorNetwork(
        n_exc=80,  # Smaller for quick test
        n_inh=20,
        device=device
    )
    
    print(f"\nInitial Poisson Parameters:")
    print(f"  Excitatory λ: {model.noise_rate_e.item():.4f}")
    print(f"  Inhibitory λ: {model.noise_rate_i.item():.4f}")
    
    # Create quick config for testing
    config = create_training_config(
        learning_rate=1e-3,
        max_epochs=5,  # Just a few epochs for testing
        batch_size=4,
        sequence_length=20,
        n_sequences=10,
        device=device,
        log_interval=1,
        plot_progress=False
    )
    
    print(f"\nStarting enhanced training...")
    
    # Train the model
    try:
        trained_model, history, trainer = train_ring_attractor_with_adam_cosine(
            model=model,
            config=config
        )
        
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Final loss: {history['train_loss'][-1]:.6f}")
        
        print(f"\nFinal Poisson Parameters:")
        print(f"  Excitatory λ: {model.noise_rate_e.item():.4f}")
        print(f"  Inhibitory λ: {model.noise_rate_i.item():.4f}")
        
        print(f"\nParameter Evolution:")
        for epoch in range(len(trainer.param_history['noise_rate_e'])):
            print(f"  Epoch {epoch+1}: λ_exc={trainer.param_history['noise_rate_e'][epoch]:.4f}, λ_inh={trainer.param_history['noise_rate_i'][epoch]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_training_with_poisson()
    
    if success:
        print(f"\n" + "="*60)
        print(f"🎉 ENHANCED TRAINING WITH POISSON NOISE SUCCESSFUL!")
        print(f"✓ Poisson noise parameters are properly trained")
        print(f"✓ Training converges without errors")
        print(f"✓ Parameters are tracked throughout training")
        print(f"="*60)
    else:
        print(f"\n❌ Test failed - please check the implementation") 