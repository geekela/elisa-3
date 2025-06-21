#!/usr/bin/env python3
"""
Test the improved single-peak ring attractor model.
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

from single_peak_model import create_single_peak_model
from utils import angle_to_input

def test_improved_model():
    """Test the improved single-peak model."""
    
    print("🧠 TESTING IMPROVED SINGLE-PEAK MODEL")
    print("="*50)
    
    device = torch.device('cpu')
    
    # Create improved model
    model = create_single_peak_model(n_exc=64, n_inh=16, device=device)
    
    print(f"\n1. STATIC DIRECTION TEST:")
    
    # Test static direction holding
    test_direction = np.pi/2
    model.reset_state()
    model.initialize_bump(test_direction, width=0.2, amplitude=0.3)
    
    print(f"   Initial peaks: {model.get_peak_count()}")
    
    # Hold direction with input
    input_pattern = angle_to_input(torch.tensor(test_direction, device=device), n_exc=model.n_exc,
                                  input_strength=1.0, input_width=0.25, device=device)
    
    activities = []
    peak_counts = []
    
    with torch.no_grad():
        for step in range(20):
            activity = model(input_pattern, steps=1)
            activities.append(activity.cpu().numpy().flatten())
            
            peaks = model.get_peak_count()
            peak_counts.append(peaks)
            
            if step % 5 == 0:
                max_act = torch.max(activity).item()
                print(f"   Step {step}: peaks={peaks}, max_activity={max_act:.4f}")
    
    final_peaks = peak_counts[-5:]  # Last 5 steps
    print(f"   Final peak count: {np.mean(final_peaks):.1f}")
    
    print(f"\n2. DIRECTION TRACKING TEST:")
    
    # Test tracking different directions
    test_directions = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    tracking_errors = []
    tracking_peaks = []
    
    for direction in test_directions:
        model.reset_state()
        input_pattern = angle_to_input(torch.tensor(direction, device=device), n_exc=model.n_exc,
                                      input_strength=1.0, input_width=0.25, device=device)
        
        with torch.no_grad():
            # Let it settle
            for _ in range(10):
                activity = model(input_pattern, steps=1)
            
            # Check final state
            decoded = model.decode_angle(activity).item()
            error = abs(decoded - direction)
            error = min(error, 2*np.pi - error)
            tracking_errors.append(np.degrees(error))
            
            peaks = model.get_peak_count()
            tracking_peaks.append(peaks)
    
    print(f"   Mean tracking error: {np.mean(tracking_errors):.1f}°")
    print(f"   Max tracking error: {np.max(tracking_errors):.1f}°")
    print(f"   Mean peak count: {np.mean(tracking_peaks):.1f}")
    print(f"   Single peak ratio: {np.mean(np.array(tracking_peaks) <= 1):.1%}")
    
    print(f"\n3. PERSISTENCE WITHOUT INPUT:")
    
    # Test persistence without input
    model.reset_state()
    model.initialize_bump(np.pi/2, width=0.2, amplitude=0.4)
    
    persistence_peaks = []
    persistence_activities = []
    
    with torch.no_grad():
        for step in range(30):  # 3 seconds
            activity = model(external_input=None, steps=1)
            persistence_activities.append(activity.cpu().numpy().flatten())
            
            peaks = model.get_peak_count()
            persistence_peaks.append(peaks)
            
            if step % 10 == 0:
                max_act = torch.max(activity).item()
                print(f"   Step {step}: peaks={peaks}, activity={max_act:.4f}")
    
    print(f"   Mean persistence peaks: {np.mean(persistence_peaks):.1f}")
    print(f"   Persistence single peak ratio: {np.mean(np.array(persistence_peaks) <= 1):.1%}")
    
    print(f"\n4. TURN DYNAMICS TEST:")
    
    # Test smooth turn
    model.reset_state()
    start_dir = 0.0
    end_dir = np.pi
    n_steps = 20
    directions = np.linspace(start_dir, end_dir, n_steps)
    
    turn_peaks = []
    turn_errors = []
    
    with torch.no_grad():
        for i, direction in enumerate(directions):
            input_pattern = angle_to_input(torch.tensor(direction, device=device), n_exc=model.n_exc,
                                          input_strength=1.0, input_width=0.25, device=device)
            
            activity = model(input_pattern, steps=1)
            
            peaks = model.get_peak_count()
            turn_peaks.append(peaks)
            
            # Check tracking
            decoded = model.decode_angle(activity).item()
            error = abs(decoded - direction)
            error = min(error, 2*np.pi - error)
            turn_errors.append(np.degrees(error))
            
            if i % 5 == 0:
                print(f"   Turn step {i}: direction={np.degrees(direction):.0f}°, peaks={peaks}, error={np.degrees(error):.1f}°")
    
    print(f"   Turn peak count: {np.mean(turn_peaks):.1f}")
    print(f"   Turn tracking error: {np.mean(turn_errors):.1f}°")
    
    # Create visualization
    create_test_visualization(activities, persistence_activities, tracking_peaks, turn_peaks)
    
    # Overall assessment
    print(f"\n" + "="*50)
    print("IMPROVED MODEL ASSESSMENT")
    print("="*50)
    
    overall_peaks = tracking_peaks + turn_peaks + persistence_peaks
    overall_single_ratio = np.mean(np.array(overall_peaks) <= 1)
    
    print(f"Overall single peak ratio: {overall_single_ratio:.1%}")
    print(f"Mean tracking error: {np.mean(tracking_errors + turn_errors):.1f}°")
    
    if overall_single_ratio >= 0.8:
        print("✅ EXCELLENT: Improved model successfully maintains single peaks!")
    elif overall_single_ratio >= 0.6:
        print("✅ GOOD: Significant improvement in single-peak stability")
    else:
        print("⚠️  NEEDS WORK: Further optimization required")
    
    return {
        'single_peak_ratio': overall_single_ratio,
        'tracking_error': np.mean(tracking_errors + turn_errors),
        'static_peaks': np.mean(final_peaks),
        'tracking_peaks': np.mean(tracking_peaks),
        'persistence_peaks': np.mean(persistence_peaks),
        'turn_peaks': np.mean(turn_peaks)
    }

def create_test_visualization(static_activities, persistence_activities, tracking_peaks, turn_peaks):
    """Create visualization of test results."""
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Static activity evolution
        static_matrix = np.array(static_activities)
        im1 = axes[0, 0].imshow(static_matrix.T, cmap='hot', aspect='auto')
        axes[0, 0].set_title('Static Direction Activity')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Neuron Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Final static activity
        axes[0, 1].plot(static_activities[-1], 'b-', linewidth=2)
        axes[0, 1].set_title('Final Static Activity Pattern')
        axes[0, 1].set_xlabel('Neuron Index')
        axes[0, 1].set_ylabel('Activity')
        axes[0, 1].grid(True)
        
        # Plot 3: Persistence activity evolution
        persist_matrix = np.array(persistence_activities)
        im2 = axes[0, 2].imshow(persist_matrix.T, cmap='hot', aspect='auto')
        axes[0, 2].set_title('Persistence Without Input')
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_ylabel('Neuron Index')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Plot 4: Peak count evolution
        axes[1, 0].plot(tracking_peaks, 'ro-', label='Tracking', alpha=0.7)
        axes[1, 0].plot(turn_peaks, 'bo-', label='Turning', alpha=0.7)
        axes[1, 0].axhline(y=1, color='g', linestyle='--', label='Single Peak')
        axes[1, 0].set_title('Peak Count During Tests')
        axes[1, 0].set_xlabel('Test Condition')
        axes[1, 0].set_ylabel('Number of Peaks')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Final persistence activity
        axes[1, 1].plot(persistence_activities[-1], 'g-', linewidth=2)
        axes[1, 1].set_title('Final Persistence Activity')
        axes[1, 1].set_xlabel('Neuron Index')
        axes[1, 1].set_ylabel('Activity')
        axes[1, 1].grid(True)
        
        # Plot 6: Summary statistics
        categories = ['Tracking', 'Turning', 'Persistence']
        peak_means = [np.mean(tracking_peaks), np.mean(turn_peaks), 
                     np.mean([model.get_peak_count() for _ in range(len(persistence_activities))])]
        
        colors = ['green' if x <= 1.2 else 'orange' if x <= 2.0 else 'red' for x in peak_means]
        bars = axes[1, 2].bar(categories, peak_means, color=colors, alpha=0.7)
        axes[1, 2].axhline(y=1, color='g', linestyle='--', label='Target')
        axes[1, 2].set_title('Mean Peak Count by Test')
        axes[1, 2].set_ylabel('Mean Peaks')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # Add value labels on bars
        for bar, value in zip(bars, peak_means):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/workspace/elisa-3/improved_model_test.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   📊 Test visualization saved as 'improved_model_test.png'")
        
    except Exception as e:
        print(f"   Visualization failed: {e}")

if __name__ == "__main__":
    try:
        result = test_improved_model()
        print(f"\nTest completed successfully!")
        print(f"Key metrics:")
        print(f"  Single peak ratio: {result['single_peak_ratio']:.1%}")
        print(f"  Tracking error: {result['tracking_error']:.1f}°")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()