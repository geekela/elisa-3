# Enhanced Training for Ring Attractor Network

This guide explains how to use the enhanced training system with **Adam optimizer** and **cosine annealing learning rate schedule** for your Ring Attractor Network.

## 🚀 Quick Start

### Basic Usage

```python
from src.models import RingAttractorNetwork
from src.training import train_model_enhanced

# Create your model
model = RingAttractorNetwork(n_exc=800, n_inh=200, sigma_ee=0.3)

# Train with enhanced method (Adam + cosine annealing)
trained_model, history, trainer = train_model_enhanced(
    model,
    learning_rate=1e-3,
    max_epochs=100,
    early_stopping=True,
    plot_progress=True
)
```

### Quick Training (for rapid prototyping)

```python
from src.training import quick_train

# Fast training with sensible defaults
model, history = quick_train(model, epochs=50, lr=1e-3)
```

## 🔧 Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-3` | Initial learning rate for Adam optimizer |
| `max_epochs` | `100` | Maximum number of training epochs |
| `min_lr` | `1e-6` | Minimum learning rate for cosine annealing |
| `weight_decay` | `1e-4` | L2 regularization strength |
| `batch_size` | `32` | Training batch size |
| `early_stopping` | `True` | Enable early stopping |
| `patience` | `20` | Early stopping patience |

### Example Configurations

#### Conservative Training (stable, slower)
```python
config = {
    'learning_rate': 5e-4,
    'max_epochs': 200,
    'weight_decay': 1e-3,
    'patience': 30,
    'batch_size': 16
}
model, history, trainer = train_model_enhanced(model, **config)
```

#### Aggressive Training (faster, potentially less stable)
```python
config = {
    'learning_rate': 2e-3,
    'max_epochs': 50,
    'weight_decay': 1e-5,
    'patience': 10,
    'batch_size': 64
}
model, history, trainer = train_model_enhanced(model, **config)
```

## 📊 Key Features

### 1. Adam Optimizer with Cosine Annealing
- **Adam**: Adaptive learning rates per parameter with momentum
- **Cosine Annealing**: Smooth learning rate decay for better convergence
- **Biological Constraints**: Keeps ring attractor parameters in valid ranges

### 2. Comprehensive Monitoring
- Training and validation loss tracking
- Learning rate evolution
- Gradient norms
- Ring attractor specific parameters (gains, noise rates)

### 3. Automatic Visualization
```python
# Training automatically shows plots when complete
train_model_enhanced(model, plot_progress=True)
```

### 4. Checkpointing and Early Stopping
- Automatic model saving at best validation loss
- Early stopping to prevent overfitting
- Resume training from checkpoints

## 🧠 Ring Attractor Specific Features

### Biological Parameter Constraints
The training automatically applies biologically plausible constraints:

```python
# Gains remain positive
g_ee, g_ei, g_ie ∈ [0.1, 10.0]

# Noise rates remain positive  
noise_rate_e, noise_rate_i ∈ [0.1, 20.0]

# Inhibitory weights properly constrained
W_IE ≤ 0.5  # Mostly inhibitory
W_EI ∈ [-0.5, 2.0]  # Can be excitatory to inhibitory
```

### Parameter Evolution Tracking
Monitor how your ring attractor parameters evolve:

```python
# Access parameter history
print(f"Initial g_ee: {trainer.param_history['g_ee'][0]:.4f}")
print(f"Final g_ee: {trainer.param_history['g_ee'][-1]:.4f}")
```

## 🏃‍♂️ Running Examples

### 1. Run the comprehensive demo
```bash
cd hd_ring_attractor
python demo_enhanced_training.py
```

### 2. Run the example script
```bash
cd hd_ring_attractor
python examples/train_with_adam_cosine.py
```

### 3. Use in Jupyter notebooks
```python
# In your notebook
%run demo_enhanced_training.py
```

## 📈 Understanding the Results

### Training Curves
- **Training Loss**: Should decrease smoothly
- **Validation Loss**: Should track training loss (no overfitting)
- **Learning Rate**: Follows cosine annealing (smooth decay)

### Parameter Evolution
- **Gain Parameters**: Should stabilize to biologically plausible values
- **Noise Parameters**: Should optimize for task performance
- **E/I Balance**: Critical for stable ring attractor dynamics

## 🛠 Advanced Usage

### Custom Configuration
```python
from src.enhanced_training import create_training_config, AdamCosineTrainer

# Create custom configuration
config = create_training_config(
    learning_rate=1e-3,
    max_epochs=150,
    # ... other parameters
)

# Create trainer directly for maximum control
trainer = AdamCosineTrainer(model, config)
history = trainer.train()
```

### Loading and Resuming Training
```python
# Save checkpoint during training (automatic)
trainer.save_checkpoint(epoch, 'my_checkpoint.pth')

# Resume training
start_epoch = trainer.load_checkpoint('my_checkpoint.pth')
```

## 🔍 Troubleshooting

### Common Issues

1. **Training too slow**: Increase learning rate or use aggressive config
2. **Training unstable**: Decrease learning rate or use conservative config  
3. **Poor convergence**: Check biological constraints, try longer training
4. **Memory issues**: Reduce batch size or model size

### Performance Tips

1. **Use GPU**: Set `device='cuda'` if available
2. **Adjust batch size**: Larger batches for stable gradients
3. **Monitor gradients**: Check for vanishing/exploding gradients
4. **Validate constraints**: Ensure parameters stay in valid ranges

## 📚 References

- **Adam Optimizer**: [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- **Cosine Annealing**: [Loshchilov & Hutter, 2016](https://arxiv.org/abs/1608.03983)
- **Ring Attractors**: [Zhang, 1996](https://doi.org/10.1523/JNEUROSCI.16-21-06889.1996)

---

## 🎯 Summary

The enhanced training system provides:

✅ **Better Optimization**: Adam + cosine annealing for superior convergence  
✅ **Biological Realism**: Automatic parameter constraints  
✅ **Comprehensive Monitoring**: Track everything that matters  
✅ **Easy to Use**: Simple interface with sensible defaults  
✅ **Flexible**: Multiple configuration options for different needs  

Start with the basic usage and experiment with different configurations to find what works best for your specific ring attractor network application! 