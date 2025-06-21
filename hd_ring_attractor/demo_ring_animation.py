#!/usr/bin/env python3
"""
Demo: Ring Attractor Animation with Trained Model

This script demonstrates how to use the ring attractor animation
with your trained models that have Poisson noise and trainable parameters.
"""

import torch
import os
from ring_attractor_animation import RingAttractorAnimator
from src.models import RingAttractorNetwork

def load_best_trained_model():
    """
    Load the best trained model if available.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try to find saved models
    model_paths = [
        'trained_ring_attractor.pth',
        'models/trained_ring_attractor.pth',
        'checkpoints/best_model.pth',
        'src/trained_model.pth'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Loading trained model from: {model_path}")
                
                # Create model with your trained parameters
                model = RingAttractorNetwork(
                    n_exc=800, 
                    n_inh=200, 
                    sigma_ee=0.6,  # Your trained value
                    device=device
                )
                
                # Load trained weights
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                    print(f"   Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
                    
                    # Print trained parameter values
                    if hasattr(model, 'noise_rate_e'):
                        print(f"   Trained noise_rate_e: {model.noise_rate_e.item():.4f}")
                    if hasattr(model, 'noise_rate_i'):
                        print(f"   Trained noise_rate_i: {model.noise_rate_i.item():.4f}")
                    if hasattr(model, 'sigma_ee'):
                        print(f"   Trained sigma_ee: {model.sigma_ee.item():.4f}")
                        
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights")
                
                model.eval()
                return model
                
            except Exception as e:
                print(f"Could not load {model_path}: {e}")
                continue
    
    # If no trained model found, create default with realistic parameters
    print("No trained model found, creating default model")
    print("   (To use your trained model, save it as 'trained_ring_attractor.pth')")
    
    model = RingAttractorNetwork(
        n_exc=800, 
        n_inh=200, 
        sigma_ee=0.6,
        device=device
    )
    
    # Set reasonable trained-like parameters for demonstration
    if hasattr(model, 'noise_rate_e'):
        model.noise_rate_e.data = torch.tensor(0.08)  # Reduced from 0.1
    if hasattr(model, 'noise_rate_i'):
        model.noise_rate_i.data = torch.tensor(0.05)  # Stable
        
    model.eval()
    return model

def main():
    """
    Run the ring attractor animation demo.
    """
    print("🎬 Ring Attractor Animation Demo")
    print("=" * 50)
    
    # Load trained model
    model = load_best_trained_model()
    
    print("\n🎯 Starting animation...")
    print("Features demonstrated:")
    print("  • Real-time bump formation and tracking")
    print("  • Multiple head direction patterns")
    print("  • Actual vs decoded direction comparison")
    print("  • Tracking error analysis")
    print("  • Neural activity heatmap over time")
    print("  • Interactive controls (play/pause/speed/pattern)")
    
    print("\nAnimation controls:")
    print("  🎮 Play/Pause: Start/stop the simulation")
    print("  🔄 Reset: Return to beginning")
    print("  🔀 Change Pattern: Cycle through movement types")
    print("  ⚡ Speed Slider: Adjust playback speed")
    
    # Create and run animator
    animator = RingAttractorAnimator(model=model)
    
    print(f"\n🚀 Model loaded with {model.n_exc} excitatory neurons")
    print("   Close the animation window to exit")
    print("=" * 50)
    
    # Start the animation
    anim = animator.run()

if __name__ == "__main__":
    main() 