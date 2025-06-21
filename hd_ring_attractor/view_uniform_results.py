#!/usr/bin/env python3
"""
Display the uniform preferred direction distribution results.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Check which images exist
images = {
    'uniform_preferred_directions_analysis.png': 'Complete Uniformity Analysis',
    'uniform_model_validation.png': 'Model Validation Results',
    'bump_stability_test.png': 'Bump Stability Test'
}

existing_images = []
for img_file, title in images.items():
    if os.path.exists(img_file):
        existing_images.append((img_file, title))
        print(f"✓ Found: {img_file}")
    else:
        print(f"✗ Not found: {img_file}")

if not existing_images:
    print("\nNo result images found! Please run test_uniform_model.py first.")
    exit(1)

print(f"\nDisplaying {len(existing_images)} result image(s)...")

# Display the images
for img_file, title in existing_images:
    img = mpimg.imread(img_file)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
print("\nClose the image windows to exit.")
plt.show() 