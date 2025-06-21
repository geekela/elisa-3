#!/usr/bin/env python3
"""
Test script for single-peak optimization of ring attractor networks.

This script tests the optimized training configurations and validates that
multiple peaks are eliminated while maintaining good tracking performance.
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add src directory to path
current_dir = pathlib.Path(__file__).parent
src_path = current_dir / "hd_ring_attractor" / "src"
sys.path.insert(0, str(src_path))

from models import RingAttractorNetwork
from single_peak_trainer import train_single_peak_network, SinglePeakTrainer
from optimized_training import (monitor_peak_count, diagnose_multiple_peaks,
                               create_single_peak_config, apply_single_peak_constraints)
from utils import angle_to_input

def test_parameter_optimization():
    """Test different parameter optimization strategies."""
    
    print("="*60)
    print("TESTING PARAMETER OPTIMIZATION STRATEGIES")
    print("="*60)
    
    device = torch.device('cpu')  # Use CPU for stability
    results = {}
    
    # Test different difficulty levels
    difficulties = ['easy', 'standard', 'challenging']
    
    for difficulty in difficulties:
        print(f"\n--- Testing {difficulty.upper()} configuration ---")
        
        # Create model
        model = RingAttractorNetwork(n_exc=200, n_inh=50, device=device)  # Smaller for faster testing
        
        # Create trainer
        trainer = SinglePeakTrainer(model, difficulty=difficulty, device=device)
        trainer.config['max_epochs'] = 20  # Short training for testing
        trainer.config['n_sequences'] = 100  # Fewer sequences
        
        # Quick training
        print("Training for 20 epochs...")
        trained_model, history = trainer.train(verbose=False)
        
        # Evaluate peak count
        peak_stats = monitor_peak_count(trained_model)
        
        results[difficulty] = {
            'single_peak_ratio': peak_stats['single_peak_ratio'],
            'mean_peaks': peak_stats['mean_peaks'],
            'final_loss': history['train_loss'][-1],
            'parameters': {
                'g_ee': trained_model.g_ee.item(),
                'g_ei': trained_model.g_ei.item(),
                'g_ie': trained_model.g_ie.item(),
                'noise_rate_e': trained_model.noise_rate_e.item(),
                'sigma_ee': trained_model.sigma_ee.item(),
            }
        }
        
        print(f"Results:")
        print(f"  Single peak ratio: {peak_stats['single_peak_ratio']:.1%}")
        print(f"  Mean peaks: {peak_stats['mean_peaks']:.2f}")
        print(f"  Final loss: {history['train_loss'][-1]:.6f}")
    
    return results

def test_constraint_effectiveness():
    """Test the effectiveness of different constraint strengths."""
    
    print("\n" + "="*60)
    print("TESTING CONSTRAINT EFFECTIVENESS")
    print("="*60)
    
    device = torch.device('cpu')
    constraint_strengths = ['weak', 'medium', 'strong']
    results = {}
    
    for strength in constraint_strengths:
        print(f"\n--- Testing {strength.upper()} constraints ---")
        
        # Create model with problematic parameters (likely to cause multiple peaks)
        model = RingAttractorNetwork(n_exc=200, n_inh=50, device=device)
        
        # Set problematic initial parameters
        with torch.no_grad():
            model.g_ee.data.fill_(1.5)      # High excitation
            model.g_ie.data.fill_(1.0)      # Low inhibition
            model.noise_rate_e.data.fill_(0.3)  # High noise
            model.sigma_ee.data.fill_(0.8)  # Wide connections
        
        print("Before constraints:")
        peak_stats_before = monitor_peak_count(model)
        print(f"  Single peak ratio: {peak_stats_before['single_peak_ratio']:.1%}")
        
        # Apply constraints
        apply_single_peak_constraints(model, strength=strength)
        
        print("After constraints:")
        peak_stats_after = monitor_peak_count(model)
        print(f"  Single peak ratio: {peak_stats_after['single_peak_ratio']:.1%}")
        
        results[strength] = {
            'before': peak_stats_before['single_peak_ratio'],
            'after': peak_stats_after['single_peak_ratio'],
            'improvement': peak_stats_after['single_peak_ratio'] - peak_stats_before['single_peak_ratio']
        }
    
    return results

def test_network_stability():
    """Test network stability under different conditions."""
    
    print("\n" + "="*60)
    print("TESTING NETWORK STABILITY")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create optimized model
    model = RingAttractorNetwork(n_exc=200, n_inh=50, device=device)
    trainer = SinglePeakTrainer(model, difficulty='standard', device=device)
    
    # Apply optimal constraints
    apply_single_peak_constraints(model, strength='medium')
    
    # Test different scenarios
    scenarios = {
        'static_hold': lambda: test_static_direction(model),
        'slow_turn': lambda: test_slow_turn(model),
        'fast_turn': lambda: test_fast_turn(model),
        'no_input': lambda: test_no_input_persistence(model)
    }
    
    results = {}
    
    for scenario_name, test_func in scenarios.items():
        print(f"\n--- Testing {scenario_name.replace('_', ' ').title()} ---")
        try:
            result = test_func()
            results[scenario_name] = result
            print(f"  Single peak ratio: {result['single_peak_ratio']:.1%}")
            print(f"  Mean tracking error: {result.get('mean_error', 0):.2f}°")
        except Exception as e:
            print(f"  Test failed: {e}")
            results[scenario_name] = {'single_peak_ratio': 0.0, 'error': str(e)}
    
    return results

def test_static_direction(model):
    """Test static direction holding."""
    model.eval()
    
    # Test multiple static directions
    test_directions = np.linspace(0, 2*np.pi, 8, endpoint=False)
    peak_ratios = []
    
    for direction in test_directions:
        model.reset_state()
        model.initialize_bump(direction, width=0.3, amplitude=0.1)
        
        # Hold direction for 2 seconds
        input_pattern = angle_to_input(torch.tensor(direction), n_exc=model.n_exc)
        
        with torch.no_grad():
            for _ in range(20):  # 20 * 0.1s = 2s
                _ = model(input_pattern, steps=1)
        
        # Check peak count
        activity = model.r_e.cpu().numpy()
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(activity, height=0.05 * np.max(activity), distance=5)
        peak_ratios.append(1.0 if len(peaks) <= 1 else 0.0)
    
    return {'single_peak_ratio': np.mean(peak_ratios)}

def test_slow_turn(model):
    """Test slow head direction turn."""
    model.eval()
    model.reset_state()
    
    # Slow turn from 0 to π over 4 seconds
    n_steps = 40
    directions = np.linspace(0, np.pi, n_steps)
    
    peak_ratios = []
    tracking_errors = []
    
    with torch.no_grad():
        for i, direction in enumerate(directions):
            input_pattern = angle_to_input(torch.tensor(direction), n_exc=model.n_exc)
            activity = model(input_pattern, steps=1)
            
            # Check peaks
            activity_np = activity.cpu().numpy().flatten()
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(activity_np, height=0.05 * np.max(activity_np), distance=5)
            peak_ratios.append(1.0 if len(peaks) <= 1 else 0.0)
            
            # Check tracking
            decoded = model.decode_angle(activity).item()
            error = abs(decoded - direction)
            error = min(error, 2*np.pi - error)
            tracking_errors.append(np.degrees(error))
    
    return {
        'single_peak_ratio': np.mean(peak_ratios),
        'mean_error': np.mean(tracking_errors)
    }

def test_fast_turn(model):
    """Test fast head direction turn."""
    model.eval()
    model.reset_state()
    
    # Fast turn from 0 to π over 1 second
    n_steps = 10
    directions = np.linspace(0, np.pi, n_steps)
    
    peak_ratios = []
    tracking_errors = []
    
    with torch.no_grad():
        for direction in directions:
            input_pattern = angle_to_input(torch.tensor(direction), n_exc=model.n_exc)
            activity = model(input_pattern, steps=1)
            
            # Check peaks
            activity_np = activity.cpu().numpy().flatten()
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(activity_np, height=0.05 * np.max(activity_np), distance=5)
            peak_ratios.append(1.0 if len(peaks) <= 1 else 0.0)
            
            # Check tracking
            decoded = model.decode_angle(activity).item()
            error = abs(decoded - direction)
            error = min(error, 2*np.pi - error)
            tracking_errors.append(np.degrees(error))
    
    return {
        'single_peak_ratio': np.mean(peak_ratios),
        'mean_error': np.mean(tracking_errors)
    }

def test_no_input_persistence(model):
    """Test persistence without input."""
    model.eval()
    model.reset_state()
    
    # Initialize bump
    initial_direction = np.pi/2
    model.initialize_bump(initial_direction, width=0.3, amplitude=0.1)
    
    peak_ratios = []
    
    with torch.no_grad():
        for _ in range(30):  # 3 seconds without input
            activity = model(external_input=None, steps=1)
            
            # Check peaks
            activity_np = activity.cpu().numpy().flatten()
            if np.max(activity_np) > 0.01:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(activity_np, height=0.05 * np.max(activity_np), distance=5)
                peak_ratios.append(1.0 if len(peaks) <= 1 else 0.0)
            else:
                peak_ratios.append(0.0)  # No activity = failed
    
    return {'single_peak_ratio': np.mean(peak_ratios)}

def create_comparison_plot(original_results, optimized_results):
    """Create comparison plot showing improvements."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = list(original_results.keys())
    original_ratios = [original_results[s]['single_peak_ratio'] for s in scenarios]
    optimized_ratios = [optimized_results[s]['single_peak_ratio'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    axes[0].bar(x - width/2, original_ratios, width, label='Original', alpha=0.7, color='red')
    axes[0].bar(x + width/2, optimized_ratios, width, label='Optimized', alpha=0.7, color='green')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Single Peak Ratio')
    axes[0].set_title('Single Peak Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Show improvement
    improvements = [opt - orig for orig, opt in zip(original_ratios, optimized_ratios)]
    axes[1].bar(scenarios, improvements, color=['green' if imp > 0 else 'red' for imp in improvements])
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Improvement')
    axes[1].set_title('Single Peak Ratio Improvement')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Overall statistics
    orig_mean = np.mean(original_ratios)
    opt_mean = np.mean(optimized_ratios)
    
    axes[2].bar(['Original', 'Optimized'], [orig_mean, opt_mean], 
               color=['red', 'green'], alpha=0.7)
    axes[2].set_ylabel('Mean Single Peak Ratio')
    axes[2].set_title('Overall Performance')
    axes[2].grid(True, alpha=0.3)
    
    for i, v in enumerate([orig_mean, opt_mean]):
        axes[2].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    """Run comprehensive single-peak optimization tests."""
    
    print("🧠 RING ATTRACTOR SINGLE-PEAK OPTIMIZATION TEST")
    print("=" * 60)
    
    # Test 1: Parameter optimization strategies
    param_results = test_parameter_optimization()
    
    # Test 2: Constraint effectiveness  
    constraint_results = test_constraint_effectiveness()
    
    # Test 3: Network stability
    stability_results = test_network_stability()
    
    # Summary
    print("\n" + "="*60)
    print("OPTIMIZATION TEST SUMMARY")
    print("="*60)
    
    print("\n1. Parameter Optimization Results:")
    for difficulty, result in param_results.items():
        print(f"   {difficulty.capitalize():12}: {result['single_peak_ratio']:.1%} single peaks")
    
    print("\n2. Constraint Effectiveness:")
    for strength, result in constraint_results.items():
        print(f"   {strength.capitalize():8}: {result['improvement']:+.1%} improvement")
    
    print("\n3. Network Stability:")
    for scenario, result in stability_results.items():
        if 'single_peak_ratio' in result:
            print(f"   {scenario.replace('_', ' ').title():15}: {result['single_peak_ratio']:.1%} single peaks")
    
    # Overall assessment
    all_ratios = []
    for results in [param_results, stability_results]:
        for result in results.values():
            if isinstance(result, dict) and 'single_peak_ratio' in result:
                all_ratios.append(result['single_peak_ratio'])
    
    if all_ratios:
        overall_performance = np.mean(all_ratios)
        print(f"\n🎯 OVERALL SINGLE-PEAK PERFORMANCE: {overall_performance:.1%}")
        
        if overall_performance >= 0.9:
            print("✅ EXCELLENT: Optimization successfully eliminates multiple peaks!")
        elif overall_performance >= 0.7:
            print("✅ GOOD: Significant improvement in single-peak stability")
        elif overall_performance >= 0.5:
            print("⚠️  MODERATE: Some improvement, but further optimization needed")
        else:
            print("❌ POOR: Multiple peaks still prevalent, review parameters")
    
    print("\n" + "="*60)
    print("Test complete! Check results above for optimization effectiveness.")
    
    return param_results, constraint_results, stability_results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)