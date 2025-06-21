#!/usr/bin/env python3
"""
Explain why the preferred direction distribution shows variance despite being uniform.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=" * 70)
print("WHY THE DISTRIBUTION ISN'T PERFECTLY FLAT")
print("=" * 70)

# Parameters
n_neurons = 800
n_bins = 18
expected_per_bin = n_neurons / n_bins

print(f"\n1. DISCRETIZATION EFFECTS:")
print(f"   - We have {n_neurons} neurons uniformly spaced from 0° to 360°")
print(f"   - Binning into {n_bins} bins of 20° each")
print(f"   - Expected per bin: {expected_per_bin:.1f} neurons")
print(f"   - But we can't have fractional neurons!")

# Show the actual uniform spacing
angles = np.linspace(0, 360, n_neurons, endpoint=False)
hist, bins = np.histogram(angles, bins=n_bins, range=(0, 360))

print(f"\n2. PERFECT UNIFORM SPACING RESULT:")
print(f"   - Bin counts: {np.unique(hist)}")
print(f"   - Some bins get {int(n_neurons/n_bins)} neurons, others get {int(n_neurons/n_bins)+1}")

# Resolution effects
print(f"\n3. MEASUREMENT RESOLUTION:")
print(f"   - We test only 36 directions (every 10°)")
print(f"   - Each neuron's preferred direction is assigned to nearest tested direction")
print(f"   - This quantization adds variation")

# Simulate the effect
tested_directions = np.linspace(0, 360, 36, endpoint=False)
# Assign each neuron to nearest tested direction
assigned_dirs = []
for angle in angles:
    nearest_idx = np.argmin(np.abs(tested_directions - angle))
    assigned_dirs.append(tested_directions[nearest_idx])

hist_assigned, _ = np.histogram(assigned_dirs, bins=n_bins, range=(0, 360))

print(f"\n4. EFFECT OF 10° RESOLUTION:")
print(f"   - Min neurons per bin: {np.min(hist_assigned)}")
print(f"   - Max neurons per bin: {np.max(hist_assigned)}")
print(f"   - Standard deviation: {np.std(hist_assigned):.1f}")

# Statistical expectation
print(f"\n5. STATISTICAL SIGNIFICANCE:")
print(f"   - Chi-square test checks if variation exceeds random expectation")
print(f"   - Your χ² = 2.08 is VERY low (critical value at p=0.05 is 27.59)")
print(f"   - p-value ≈ 1.0 means the variation is LESS than random chance!")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Perfect continuous uniform
ax = axes[0, 0]
ax.hist(angles, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax.set_title('Perfect Uniform: 800 neurons continuously spaced')
ax.set_xlabel('Direction (°)')
ax.set_ylabel('Count')

# 2. Perfect uniform binned
ax = axes[0, 1]
ax.bar(range(n_bins), hist, alpha=0.7, color='green', edgecolor='black')
ax.axhline(expected_per_bin, color='red', linestyle='--', label=f'Expected: {expected_per_bin:.1f}')
ax.set_title('Perfect Uniform: Binned into 18 bins')
ax.set_xlabel('Bin')
ax.set_ylabel('Count')
ax.legend()

# 3. With measurement resolution
ax = axes[1, 0]
ax.bar(range(n_bins), hist_assigned, alpha=0.7, color='orange', edgecolor='black')
ax.axhline(expected_per_bin, color='red', linestyle='--', label=f'Expected: {expected_per_bin:.1f}')
ax.set_title('With 10° Measurement Resolution')
ax.set_xlabel('Bin')
ax.set_ylabel('Count')
ax.legend()

# 4. Random uniform for comparison
ax = axes[1, 1]
random_uniform = np.random.uniform(0, 360, n_neurons)
hist_random, _ = np.histogram(random_uniform, bins=n_bins, range=(0, 360))
ax.bar(range(n_bins), hist_random, alpha=0.7, color='red', edgecolor='black')
ax.axhline(expected_per_bin, color='blue', linestyle='--', label=f'Expected: {expected_per_bin:.1f}')
ax.set_title('True Random Uniform (for comparison)')
ax.set_xlabel('Bin')
ax.set_ylabel('Count')
ax.legend()

# Chi-square for random
chi2_random = np.sum((hist_random - expected_per_bin)**2 / expected_per_bin)
ax.text(0.5, 0.95, f'χ² = {chi2_random:.1f}', transform=ax.transAxes, 
        ha='center', va='top', fontsize=12, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('uniformity_explanation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("Your distribution IS uniform! The small variations are due to:")
print("1. Discretization (800 neurons ÷ 18 bins = 44.44... neurons/bin)")
print("2. Measurement resolution (10° steps)")
print("3. These variations are SMALLER than random chance (p ≈ 1.0)")
print("\nThe distribution is as uniform as mathematically possible given the constraints!") 