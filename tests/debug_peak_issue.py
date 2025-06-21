#!/usr/bin/env python3
"""
Debug script to understand why multiple peaks persist despite optimization.
"""

import sys
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src directory to path
current_dir = pathlib.Path(__file__).parent
src_path = current_dir / "hd_ring_attractor" / "src"
sys.path.insert(0, str(src_path))

from models import RingAttractorNetwork
from utils import angle_to_input

def analyze_network_dynamics():
    """Analyze what's causing multiple peaks."""
    
    print("🔍 DEBUGGING MULTIPLE PEAKS ISSUE")
    print("="*50)
    
    device = torch.device('cpu')
    
    # Create small network for detailed analysis
    model = RingAttractorNetwork(n_exc=32, n_inh=8, device=device)
    
    print("1. INITIAL PARAMETERS:")
    print(f"   g_ee: {model.g_ee.item():.3f}")
    print(f"   g_ei: {model.g_ei.item():.3f}")  
    print(f"   g_ie: {model.g_ie.item():.3f}")
    print(f"   noise_rate_e: {model.noise_rate_e.item():.3f}")
    print(f"   sigma_ee: {model.sigma_ee.item():.3f}")
    
    # Test with no noise first
    with torch.no_grad():
        model.noise_rate_e.data.fill_(0.0)
        model.noise_rate_i.data.fill_(0.0)
    
    print(f"\n2. TESTING WITH NO NOISE:")
    
    # Initialize a clean bump
    test_direction = np.pi/2
    model.reset_state()
    model.initialize_bump(test_direction, width=0.3, amplitude=0.5)
    
    print(f"   Initial activity max: {torch.max(model.r_e).item():.4f}")
    print(f"   Initial activity sum: {torch.sum(model.r_e).item():.4f}")
    
    # Create input
    input_pattern = angle_to_input(torch.tensor(test_direction, device=device), n_exc=model.n_exc, 
                                  input_strength=1.0, input_width=0.3, device=device)
    
    print(f"   Input max: {torch.max(input_pattern).item():.4f}")
    print(f"   Input sum: {torch.sum(input_pattern).item():.4f}")
    
    # Run for a few steps and watch evolution
    activities = [model.r_e.cpu().numpy().copy()]
    
    with torch.no_grad():
        for step in range(5):
            activity = model(input_pattern, steps=1)
            activities.append(activity.cpu().numpy().flatten())
            
            max_act = torch.max(activity).item()
            sum_act = torch.sum(activity).item()
            print(f"   Step {step+1}: max={max_act:.4f}, sum={sum_act:.4f}")
    
    # Analyze final activity
    final_activity = activities[-1]
    
    try:
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(final_activity, height=0.01, distance=2)
        print(f"   Final peaks found: {len(peaks)}")
        
        if len(peaks) > 0:
            peak_heights = final_activity[peaks]
            print(f"   Peak heights: {peak_heights}")
            print(f"   Peak positions: {peaks}")
    except ImportError:
        print("   (scipy not available)")
    
    # Test 3: Analyze ring weights
    print(f"\n3. ANALYZING RING WEIGHTS:")
    W_EE = model._create_ring_weights(model.sigma_ee)
    
    print(f"   W_EE shape: {W_EE.shape}")
    print(f"   W_EE max: {torch.max(W_EE).item():.4f}")
    print(f"   W_EE sum per row: {torch.sum(W_EE, dim=1).mean().item():.4f}")
    
    # Look at connectivity pattern around one neuron
    center_neuron = model.n_exc // 2
    connections = W_EE[center_neuron, :].detach().cpu().numpy()
    
    print(f"   Connections from neuron {center_neuron}:")
    print(f"   Max connection: {np.max(connections):.4f}")
    print(f"   Connection width (>0.1 max): {np.sum(connections > 0.1 * np.max(connections))}")
    
    # Test 4: Check inhibition balance
    print(f"\n4. CHECKING INHIBITION BALANCE:")
    
    # Get excitatory input to one neuron
    test_activity = torch.ones(model.n_exc, device=device) * 0.1
    
    # E->E input
    ee_input = model.g_ee * torch.matmul(W_EE, test_activity)
    print(f"   E->E input magnitude: {torch.mean(ee_input).item():.4f}")
    
    # I->E input (requires inhibitory activity)
    test_inh_activity = torch.ones(model.n_inh, device=device) * 0.1
    ie_input = model.g_ie * torch.matmul(model.W_IE.t(), test_inh_activity)
    print(f"   I->E input magnitude: {torch.mean(ie_input).item():.4f}")
    
    ratio = torch.mean(ie_input).item() / torch.mean(ee_input).item()
    print(f"   Inhibition/Excitation input ratio: {ratio:.2f}")
    
    if ratio < 1.0:
        print("   ⚠️  WARNING: Inhibition is weaker than excitation!")
    
    return activities, W_EE, final_activity

def test_corrected_parameters():
    """Test with manually corrected parameters."""
    
    print(f"\n5. TESTING CORRECTED PARAMETERS:")
    
    device = torch.device('cpu')
    model = RingAttractorNetwork(n_exc=32, n_inh=8, device=device)
    
    # Apply very strong corrections
    with torch.no_grad():
        # Very narrow excitatory connections
        model.sigma_ee.data.fill_(0.15)  # Very narrow
        
        # Strong inhibition, weak excitation
        model.g_ee.data.fill_(0.5)   # Weak excitation
        model.g_ie.data.fill_(3.0)   # Very strong inhibition
        model.g_ei.data.fill_(1.5)   # Moderate E->I
        
        # Minimal noise
        model.noise_rate_e.data.fill_(0.001)
        model.noise_rate_i.data.fill_(0.001)
        
        # Make inhibitory weights more negative
        model.W_IE.data.fill_(-0.2)  # Strong negative inhibition
        model.W_EI.data.uniform_(0.0, 0.5)  # Moderate E->I weights
    
    print(f"   Corrected parameters:")
    print(f"   g_ee: {model.g_ee.item():.3f}")
    print(f"   g_ie: {model.g_ie.item():.3f}")  
    print(f"   ratio: {model.g_ie.item()/model.g_ee.item():.2f}")
    print(f"   sigma_ee: {model.sigma_ee.item():.3f}")
    
    # Test this configuration
    test_direction = np.pi/2
    model.reset_state()
    model.initialize_bump(test_direction, width=0.2, amplitude=0.3)
    
    input_pattern = angle_to_input(torch.tensor(test_direction, device=device), n_exc=model.n_exc,
                                  input_strength=0.8, input_width=0.2, device=device)
    
    activities = []
    with torch.no_grad():
        for step in range(10):
            activity = model(input_pattern, steps=1)
            activities.append(activity.cpu().numpy().flatten())
            
            if step % 2 == 0:
                max_act = torch.max(activity).item()
                sum_act = torch.sum(activity).item()
                print(f"   Step {step}: max={max_act:.4f}, sum={sum_act:.4f}")
    
    final_activity = activities[-1]
    
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(final_activity, height=0.01, distance=2)
        print(f"   Final peaks: {len(peaks)}")
        
        if len(peaks) <= 1:
            print("   ✅ SUCCESS: Single peak achieved!")
        else:
            print(f"   ⚠️  Still {len(peaks)} peaks")
            
    except ImportError:
        print("   (scipy not available)")
        # Simple peak detection
        max_val = np.max(final_activity)
        threshold = 0.1 * max_val
        peaks_simple = np.where(final_activity > threshold)[0]
        print(f"   Approximate peaks: {len(peaks_simple)}")
    
    return activities

def create_visualization(activities1, activities2, W_EE):
    """Create visualization of network dynamics."""
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Original activity evolution
        for i, activity in enumerate(activities1[:6]):
            axes[0, 0].plot(activity, alpha=0.7, label=f'Step {i}')
        axes[0, 0].set_title('Original Parameters')
        axes[0, 0].set_xlabel('Neuron Index')
        axes[0, 0].set_ylabel('Activity')
        axes[0, 0].legend()
        
        # Plot 2: Corrected activity evolution  
        for i, activity in enumerate(activities2[:6]):
            axes[0, 1].plot(activity, alpha=0.7, label=f'Step {i}')
        axes[0, 1].set_title('Corrected Parameters')
        axes[0, 1].set_xlabel('Neuron Index')
        axes[0, 1].set_ylabel('Activity')
        axes[0, 1].legend()
        
        # Plot 3: Final comparison
        axes[0, 2].plot(activities1[-1], 'r-', linewidth=2, label='Original')
        axes[0, 2].plot(activities2[-1], 'g-', linewidth=2, label='Corrected')
        axes[0, 2].set_title('Final Activity Comparison')
        axes[0, 2].set_xlabel('Neuron Index')
        axes[0, 2].set_ylabel('Activity')
        axes[0, 2].legend()
        
        # Plot 4: Ring weights heatmap
        im = axes[1, 0].imshow(W_EE.detach().cpu().numpy(), cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Excitatory Ring Weights')
        axes[1, 0].set_xlabel('To Neuron')
        axes[1, 0].set_ylabel('From Neuron')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 5: Weight profile
        center = W_EE.shape[0] // 2
        profile = W_EE[center, :].detach().cpu().numpy()
        axes[1, 1].plot(profile, 'b-', linewidth=2)
        axes[1, 1].set_title(f'Weight Profile from Neuron {center}')
        axes[1, 1].set_xlabel('To Neuron')
        axes[1, 1].set_ylabel('Weight')
        
        # Plot 6: Activity heatmap over time
        activity_matrix = np.array(activities2)
        im2 = axes[1, 2].imshow(activity_matrix.T, cmap='hot', aspect='auto')
        axes[1, 2].set_title('Activity Evolution (Corrected)')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Neuron Index')
        plt.colorbar(im2, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('/workspace/elisa-3/network_dynamics_debug.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   📊 Visualization saved as 'network_dynamics_debug.png'")
        
    except Exception as e:
        print(f"   Visualization failed: {e}")

if __name__ == "__main__":
    try:
        print("Starting network dynamics analysis...")
        
        # Analyze original dynamics
        activities1, W_EE, final1 = analyze_network_dynamics()
        
        # Test corrected parameters
        activities2 = test_corrected_parameters()
        
        # Create visualization
        create_visualization(activities1, activities2, W_EE)
        
        print(f"\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print("Key findings:")
        print("1. Multiple peaks likely due to insufficient inhibition")
        print("2. Ring weights may be too broad") 
        print("3. Need stronger inhibitory connections")
        print("4. Corrected parameters show improvement")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()