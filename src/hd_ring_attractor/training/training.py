import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

try:
    from ..utils.utils import angle_to_input, generate_trajectory, compute_error
    from .enhanced_training import AdamCosineTrainer, create_training_config
except ImportError:
    # Fallback for when module is imported directly (e.g., in Jupyter notebooks)
    try:
        from hd_ring_attractor.utils.utils import angle_to_input, generate_trajectory, compute_error
        from hd_ring_attractor.training.enhanced_training import AdamCosineTrainer, create_training_config
    except ImportError:
        from utils import angle_to_input, generate_trajectory, compute_error
        from enhanced_training import AdamCosineTrainer, create_training_config


def create_dataset(n_sequences=1000, sequence_length=100, dt=0.1, n_exc=800):
    all_angles = []
    all_inputs = []
    
    for _ in range(n_sequences):
        angles, _ = generate_trajectory(sequence_length, dt=dt)
        inputs = angle_to_input(angles, n_exc=n_exc)
        
        all_angles.append(angles)
        all_inputs.append(inputs)
    
    angles_tensor = torch.stack(all_angles)
    inputs_tensor = torch.stack(all_inputs)
    
    return inputs_tensor, angles_tensor


def train_model(model, n_epochs=100, batch_size=32, learning_rate=1e-3, 
                n_sequences=1000, sequence_length=100, device='cpu'):
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    inputs, angles = create_dataset(n_sequences, sequence_length, n_exc=model.n_exc)
    dataset = TensorDataset(inputs, angles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        epoch_loss = 0.0
        
        for batch_inputs, batch_angles in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_angles = batch_angles.to(device)
            
            optimizer.zero_grad()
            
            batch_size, seq_len, _ = batch_inputs.shape
            h = None
            total_loss = 0.0
            
            for t in range(seq_len):
                input_t = batch_inputs[:, t, :]
                angle_t = batch_angles[:, t]
                
                h = model(input_t, h, steps=1)
                predicted_angle = model.decode_angle(h)
                
                loss = compute_error(predicted_angle, angle_t).mean()
                total_loss += loss
            
            avg_loss = total_loss / seq_len
            avg_loss.backward()
            optimizer.step()
            
            epoch_loss += avg_loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    return losses


def evaluate_model(model, test_sequences=100, sequence_length=200, device='cpu', n_exc=800):
    model = model.to(device)
    model.eval()
    
    all_errors = []
    
    with torch.no_grad():
        for _ in range(test_sequences):
            angles, _ = generate_trajectory(sequence_length)
            inputs = angle_to_input(angles, n_exc=n_exc)
            
            inputs = inputs.unsqueeze(0).to(device)
            angles = angles.to(device)
            
            h = None
            errors = []
            
            for t in range(sequence_length):
                input_t = inputs[:, t, :]
                angle_t = angles[t]
                
                h = model(input_t, h, steps=1)
                predicted_angle = model.decode_angle(h).squeeze()
                
                error = compute_error(predicted_angle, angle_t)
                errors.append(error.item())
            
            all_errors.extend(errors)
    
    return np.array(all_errors)


def train_model_enhanced(model, **kwargs):
    """
    Enhanced version of train_model using Adam optimizer with cosine annealing.
    
    This function provides advanced optimization capabilities for your ring attractor network:
    - Adam optimizer with proper defaults
    - Cosine annealing learning rate schedule
    - Biological parameter constraints
    - Comprehensive monitoring and visualization
    - Early stopping and checkpointing
    
    Args:
        model: Your RingAttractorNetwork instance
        **kwargs: Training configuration parameters (see create_training_config for options)
    
    Returns:
        tuple: (trained_model, training_history, trainer_instance)
    
    Example usage:
        model, history, trainer = train_model_enhanced(
            model,
            learning_rate=1e-3,
            max_epochs=100,
            batch_size=32,
            early_stopping=True,
            plot_progress=True
        )
    """
    
    # Create configuration from kwargs
    config = create_training_config(**kwargs)
    
    # Create and run enhanced trainer
    trainer = AdamCosineTrainer(model, config)
    history = trainer.train()
    
    return model, history, trainer


def quick_train(model, epochs=50, lr=1e-3, device='cpu'):
    """
    Quick training function for rapid prototyping and testing.
    Uses the enhanced trainer with sensible defaults.
    
    Args:
        model: Your RingAttractorNetwork instance
        epochs: Number of training epochs
        lr: Learning rate
        device: Training device ('cpu' or 'cuda')
    
    Returns:
        tuple: (trained_model, training_history)
    """
    
    config = create_training_config(
        learning_rate=lr,
        max_epochs=epochs,
        batch_size=32,
        n_sequences=500,        # Smaller dataset for quick training
        validation_split=0.2,
        early_stopping=True,
        patience=10,
        plot_progress=False,    # No plots for quick training
        log_interval=max(1, epochs // 5),  # Log 5 times during training
        device=device
    )
    
    model, history, trainer = train_model_enhanced(model, **config)
    
    return model, history