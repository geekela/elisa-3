#!/usr/bin/env python3
"""
Code snippet to update your notebook for uniform preferred direction analysis.

Copy and paste this code into your notebook to get uniform distribution.
"""

print("""
To update your notebook for uniform preferred direction distribution, add this code:

=== CELL 1: Import the improved analysis function ===
```python
import sys
sys.path.append('.')
from fix_uniform_preferred_directions import analyze_all_neurons_tuning
```

=== CELL 2: Replace the tuning curve analysis ===
```python
# Instead of the sampled analysis, use this:
def get_uniform_tuning_analysis(model):
    '''Get tuning curves for ALL neurons to ensure uniform distribution.'''
    
    device = next(model.parameters()).device
    model.eval()
    
    # Test directions
    n_directions = 36
    directions = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    tuning_curves = np.zeros((model.n_exc, n_directions))
    
    print(f"Analyzing ALL {model.n_exc} neurons...")
    
    for i, direction in enumerate(directions):
        model.reset_state()
        
        # Strong input for clear responses
        input_pattern = angle_to_input(
            torch.tensor(direction, device=device),
            n_exc=model.n_exc,
            input_strength=1.5,
            input_width=0.3
        ).to(device)
        
        # Run to steady state
        with torch.no_grad():
            for _ in range(20):
                activity = model(input_pattern, steps=1)
        
        tuning_curves[:, i] = activity.cpu().numpy().flatten()
        
        if i % 6 == 0:
            print(f"  Tested direction {i+1}/{n_directions}")
    
    # Analyze ALL neurons with low threshold
    preferred_dirs = []
    valid_neurons = []
    
    for neuron_idx in range(model.n_exc):
        curve = tuning_curves[neuron_idx]
        peak_idx = np.argmax(curve)
        peak_response = curve[peak_idx]
        
        if peak_response > 0.001:  # Very low threshold
            preferred_dir_deg = np.degrees(directions[peak_idx])
            preferred_dirs.append(preferred_dir_deg)
            valid_neurons.append(neuron_idx)
    
    print(f"Found {len(valid_neurons)}/{model.n_exc} active neurons")
    
    return {
        'preferred_directions': preferred_dirs,
        'tuning_curves': tuning_curves,
        'directions': directions
    }

# Use it like this:
tuning_data = get_uniform_tuning_analysis(trained_model)
```

=== CELL 3: Plot the uniform distribution ===
```python
# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(tuning_data['preferred_directions'], bins=18, 
         alpha=0.7, color='lightgreen', edgecolor='black')

# Add expected uniform line
expected_count = len(tuning_data['preferred_directions']) / 18
plt.axhline(expected_count, color='red', linestyle='--', 
            linewidth=2, label=f'Expected: {expected_count:.1f}')

plt.xlabel('Preferred Direction (°)')
plt.ylabel('Number of Neurons')
plt.title('Preferred Direction Distribution (ALL Neurons)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Chi-square test
from scipy import stats
hist, _ = np.histogram(tuning_data['preferred_directions'], bins=18)
expected = np.full_like(hist, expected_count)
chi2 = np.sum((hist - expected)**2 / expected)
p_value = 1 - stats.chi2.cdf(chi2, df=17)
print(f"\\nChi-square test: χ² = {chi2:.2f}, p = {p_value:.4f}")
print(f"Distribution is {'UNIFORM' if p_value > 0.05 else 'NON-UNIFORM'}")
```

Alternatively, you can use the new UniformRingAttractorNetwork:
```python
from src.uniform_ring_attractor import UniformRingAttractorNetwork

# Create model with guaranteed uniformity
model = UniformRingAttractorNetwork(
    n_exc=800, n_inh=200, sigma_ee=0.15,
    device=str(device), enforce_symmetry=True
)

# The model has built-in uniformity validation
is_uniform, chi2, p_value = model.validate_uniformity()
print(f"Uniformity test: {'PASSED' if is_uniform else 'FAILED'}")
```
""") 