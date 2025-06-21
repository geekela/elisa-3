#!/usr/bin/env python3
"""
Test if using 10-degree bins solves the discretization issue.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("="*70)
print("TESTING 10-DEGREE BINS VS 20-DEGREE BINS")
print("="*70)

# Parameters
n_neurons = 800
tested_directions = 36  # Testing every 10 degrees

# Test both bin sizes
bin_configs = [
    {"bins": 18, "width": 20, "color": "green", "label": "20° bins (current)"},
    {"bins": 36, "width": 10, "color": "blue", "label": "10° bins (proposed)"},
    {"bins": 40, "width": 9, "color": "orange", "label": "9° bins (divides evenly!)"},
    {"bins": 16, "width": 22.5, "color": "red", "label": "22.5° bins (divides evenly!)"}
]

# Analyze each configuration
print("\nDISCRETIZATION ANALYSIS:")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, config in enumerate(bin_configs):
    n_bins = config["bins"]
    
    print(f"\n{config['label']}:")
    print(f"  {n_neurons} neurons ÷ {n_bins} bins = {n_neurons/n_bins:.4f} neurons/bin")
    
    quotient = n_neurons // n_bins
    remainder = n_neurons % n_bins
    
    if remainder == 0:
        print(f"  ✓ DIVIDES EVENLY! Each bin gets exactly {quotient} neurons")
        bin_counts = [quotient] * n_bins
    else:
        print(f"  ✗ Uneven: {n_bins - remainder} bins get {quotient}, {remainder} bins get {quotient + 1}")
        bin_counts = [quotient] * (n_bins - remainder) + [quotient + 1] * remainder
    
    # Simulate with measurement resolution
    # Create perfectly uniform neurons
    neuron_angles = np.linspace(0, 360, n_neurons, endpoint=False)
    
    # Test at 10° intervals (36 directions)
    test_dirs = np.linspace(0, 360, tested_directions, endpoint=False)
    
    # Assign each neuron to nearest test direction
    assigned_dirs = []
    for angle in neuron_angles:
        # Find nearest test direction
        diffs = np.abs(test_dirs - angle)
        diffs[diffs > 180] = 360 - diffs[diffs > 180]  # Handle wraparound
        nearest_idx = np.argmin(diffs)
        assigned_dirs.append(test_dirs[nearest_idx])
    
    # Histogram with this bin configuration
    hist, bins = np.histogram(assigned_dirs, bins=n_bins, range=(0, 360))
    
    # Calculate uniformity metrics
    expected = n_neurons / n_bins
    chi2 = np.sum((hist - expected)**2 / expected)
    std_dev = np.std(hist)
    
    print(f"  After 10° measurement resolution:")
    print(f"    Range: {np.min(hist)} - {np.max(hist)} neurons per bin")
    print(f"    Std deviation: {std_dev:.2f}")
    print(f"    Chi-square: {chi2:.2f}")
    
    # Plot
    ax = axes[idx]
    x_pos = np.arange(n_bins)
    bars = ax.bar(x_pos, hist, alpha=0.7, color=config["color"], edgecolor='black')
    ax.axhline(expected, color='red', linestyle='--', linewidth=2, 
               label=f'Expected: {expected:.1f}')
    
    # Highlight variation
    ax.fill_between(x_pos, expected - std_dev, expected + std_dev, 
                    alpha=0.2, color='gray', label=f'±1 SD')
    
    ax.set_xlabel('Bin number')
    ax.set_ylabel('Neurons per bin')
    ax.set_title(f'{config["label"]}\nχ² = {chi2:.1f}, SD = {std_dev:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-limits for comparison
    ax.set_ylim(0, 60)

plt.tight_layout()
plt.savefig('bin_size_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("\n1. MATHEMATICAL PERFECTION:")
print("   - 10° bins: 800÷36 = 22.22... (still fractional!)")
print("   - 9° bins: 800÷40 = 20 exactly ✓")
print("   - 22.5° bins: 800÷16 = 50 exactly ✓")

print("\n2. MEASUREMENT ALIGNMENT:")
print("   - 10° bins align perfectly with 10° measurement intervals")
print("   - This reduces quantization error")
print("   - But doesn't eliminate discretization completely")

print("\n3. RECOMMENDATION:")
print("   ✓ 10° bins are BETTER for your setup because:")
print("     - Aligns with measurement resolution (test every 10°)")
print("     - Reduces quantization artifacts")
print("     - More intuitive (each test direction = 1 bin)")
print("   ")
print("   ✓ For perfect uniformity, you'd need:")
print("     - 40 bins (9° each) or 16 bins (22.5° each)")
print("     - Or adjust to 720 neurons (divides by both 18 and 36)")

# Additional analysis: perfect numbers
print("\n4. PERFECT NEURON COUNTS:")
perfect_counts = []
for n in range(700, 900):
    if n % 36 == 0 and n % 18 == 0:
        perfect_counts.append(n)

print(f"   Neuron counts that divide evenly by both 18 and 36:")
print(f"   {perfect_counts}")
print(f"   Closest to 800: {min(perfect_counts, key=lambda x: abs(x-800))} neurons") 