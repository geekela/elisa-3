import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict

try:
    from .utils import angle_to_input, generate_trajectory, compute_error
except ImportError:
    # Fallback for when module is imported directly (e.g., in Jupyter notebooks)
    from utils import angle_to_input, generate_trajectory, compute_error


class AdamCosineTrainer:
    """
    Enhanced trainer for Ring Attractor Network using Adam optimizer 
    with cosine annealing learning rate schedule.
    
    This class provides comprehensive training capabilities with:
    - Adam optimizer with configurable parameters
    - Cosine annealing learning rate scheduling
    - Biological constraints enforcement
    - Detailed monitoring and logging
    - Training visualization
    """
    
    def __init__(self, model, config=None):
        """
        Initialize the enhanced trainer for your ring attractor network.
        
        Args:
            model: Your RingAttractorNetwork instance
            config: Configuration dictionary with training parameters
        """
        self.model = model
        self.config = self._setup_default_config(config)
        self.device = self.config['device']
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self._setup_optimization()
        
        # Setup biological constraints
        self.constraint_fn = self._create_biological_constraints()
        
        # Training history tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'parameter_norm': [],
            'time_per_epoch': []
        }
        
        # Parameter-specific tracking for your ring attractor
        self.param_history = {
            'g_ee': [],
            'g_ei': [],
            'g_ie': [],
            'noise_rate_e': [],  # Poisson lambda for excitatory neurons
            'noise_rate_i': [],  # Poisson lambda for inhibitory neurons
            'sigma_ee': []       # E->E connection width parameter
        }
        
        print("=== Ring Attractor Network Trainer Initialized ===")
        print(f"Device: {self.device}")
        print(f"Initial Learning Rate: {self.config['learning_rate']:.1e}")
        print(f"Max Epochs: {self.config['max_epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Weight Decay: {self.config['weight_decay']:.1e}")
        print(f"Noise Type: Poisson-distributed")
        print(f"Initial Poisson λ (Excitatory): {self.model.noise_rate_e.item():.4f}")
        print(f"Initial Poisson λ (Inhibitory): {self.model.noise_rate_i.item():.4f}")
        print(f"Initial σ_EE (E->E width): {self.model.sigma_ee.item():.4f}")
        
    def _setup_default_config(self, user_config):
        """
        Setup default configuration optimized for ring attractor networks.
        """
        default_config = {
            # Optimizer settings
            'learning_rate': 1e-3,      # Initial learning rate
            'weight_decay': 1e-4,       # L2 regularization
            'betas': (0.9, 0.999),      # Adam momentum parameters
            'eps': 1e-8,                # Adam numerical stability
            
            # Scheduler settings
            'max_epochs': 100,          # Total training epochs
            'min_lr': 1e-6,            # Minimum learning rate for cosine annealing
            
            # Training settings
            'batch_size': 32,           # Batch size
            'sequence_length': 100,     # Sequence length for training
            'n_sequences': 1000,        # Number of training sequences
            'validation_split': 0.2,    # Fraction of data for validation
            
            # Regularization
            'clip_gradients': True,     # Enable gradient clipping
            'max_grad_norm': 1.0,      # Maximum gradient norm
            'apply_constraints': True,  # Apply biological constraints
            
            # Monitoring
            'log_interval': 10,         # Log every N epochs
            'save_checkpoints': True,   # Save model checkpoints
            'plot_progress': True,      # Plot training progress
            
            # Device
            'device': 'cpu',  # Fixed to use CPU to avoid CUDA errors
            
            # Early stopping
            'early_stopping': True,
            'patience': 20,             # Epochs to wait for improvement
            'min_delta': 1e-6,         # Minimum improvement threshold
        }
        
        # Update with user configuration
        if user_config:
            default_config.update(user_config)
            
        return default_config
    
    def _setup_optimization(self):
        """
        Setup Adam optimizer with cosine annealing learning rate schedule.
        """
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create parameter groups with different learning rates
        # This ensures all parameters get properly trained
        param_groups = [
            # Main network parameters
            {'params': [self.model.W_EI, self.model.W_IE, self.model.g_ee], 
             'lr': self.config['learning_rate']},
            
            # Gain parameters that need more training
            {'params': [self.model.g_ei, self.model.g_ie], 
             'lr': self.config['learning_rate'] * 2.0},
            
            # Noise parameters that need special attention
            {'params': [self.model.noise_rate_e, self.model.noise_rate_i],
             'lr': self.config['learning_rate'] * 3.0},
            
            # Connection width parameter
            {'params': [self.model.sigma_ee],
             'lr': self.config['learning_rate'] * 1.5}
        ]
        
        # Setup Adam optimizer with parameter groups
        optimizer = torch.optim.Adam(
            param_groups,
            betas=self.config['betas'],
            eps=self.config['eps'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['max_epochs'],
            eta_min=self.config['min_lr']
        )
        
        print("✓ Adam optimizer configured with parameter-specific learning rates")
        print("✓ Cosine annealing scheduler configured")
        
        return optimizer, scheduler
    
    def _create_biological_constraints(self):
        """
        Create biological constraint function for ring attractor parameters.
        """
        def apply_constraints():
            """
            Apply biological constraints to ensure parameters remain plausible.
            """
            with torch.no_grad():
                # Ensure positive gains (biological neurons can't have negative gains)
                self.model.g_ee.data.clamp_(min=0.1, max=10.0)
                self.model.g_ei.data.clamp_(min=0.1, max=10.0) 
                self.model.g_ie.data.clamp_(min=0.1, max=10.0)
                
                # Ensure positive Poisson rate parameters (λ > 0 required for Poisson distribution)
                # Use smaller range to encourage more significant changes during training
                self.model.noise_rate_e.data.clamp_(min=0.01, max=5.0)  # Excitatory Poisson lambda
                self.model.noise_rate_i.data.clamp_(min=0.01, max=5.0)  # Inhibitory Poisson lambda
                
                # Ensure positive sigma_ee (connection width must be positive, reasonable range)
                self.model.sigma_ee.data.clamp_(min=0.1, max=20.0)  # E->E connection width parameter (expanded range from 0.1 to 20 degrees)
                
                # Constrain inhibitory weights (they should be mostly negative or small positive)
                # W_IE: inhibitory to excitatory (should be negative for inhibition)
                self.model.W_IE.data.clamp_(max=0.5)  # Allow small positive values
                
                # W_EI: excitatory to inhibitory (can be positive)
                self.model.W_EI.data.clamp_(min=-0.5, max=2.0)
                
                # Optional: Apply sparsity to maintain biological plausibility
                if hasattr(self.model, 'apply_sparsity') and self.model.apply_sparsity:
                    self._apply_sparsity_constraints()
        
        return apply_constraints
    
    def _apply_sparsity_constraints(self):
        """
        Apply sparsity constraints to maintain biological connectivity patterns.
        """
        with torch.no_grad():
            # Keep only strongest connections (top 20% for example)
            for param_name in ['W_EI', 'W_IE']:
                if hasattr(self.model, param_name):
                    param = getattr(self.model, param_name)
                    threshold = torch.quantile(torch.abs(param.data), 0.8)
                    mask = torch.abs(param.data) > threshold
                    param.data = param.data * mask.float()
    
    def create_enhanced_dataset(self):
        """
        Create enhanced dataset with train/validation split for ring attractor training.
        """
        print("Creating enhanced dataset...")
        
        # Generate training data
        all_angles = []
        all_inputs = []
        
        for _ in tqdm(range(self.config['n_sequences']), desc="Generating sequences"):
            # Generate more diverse trajectories
            angles, _ = generate_trajectory(
                self.config['sequence_length'], 
                dt=0.1,
                angular_velocity_std=np.random.uniform(0.05, 0.3)  # Varying difficulty
            )
            inputs = angle_to_input(angles, n_exc=self.model.n_exc)
            
            all_angles.append(angles)
            all_inputs.append(inputs)
        
        # Convert to tensors
        angles_tensor = torch.stack(all_angles)
        inputs_tensor = torch.stack(all_inputs)
        
        # Train/validation split
        n_train = int(len(angles_tensor) * (1 - self.config['validation_split']))
        
        # Training data
        train_inputs = inputs_tensor[:n_train]
        train_angles = angles_tensor[:n_train]
        train_dataset = TensorDataset(train_inputs, train_angles)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        # Validation data
        val_inputs = inputs_tensor[n_train:]
        val_angles = angles_tensor[n_train:]
        val_dataset = TensorDataset(val_inputs, val_angles)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        print(f"✓ Created {len(train_dataset)} training sequences")
        print(f"✓ Created {len(val_dataset)} validation sequences")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch with detailed monitoring.
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Track gradients for monitoring
        total_grad_norm = 0.0
        
        for batch_inputs, batch_angles in train_loader:
            # Move to device
            batch_inputs = batch_inputs.to(self.device)
            batch_angles = batch_angles.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass through ring attractor network
            batch_size, seq_len, _ = batch_inputs.shape
            total_loss = 0.0
            
            # Initialize hidden state for each sequence in the batch
            hidden_states = None  # Let the model handle the initialization
            
            # Process sequence step by step
            for t in range(seq_len):
                input_t = batch_inputs[:, t, :]
                angle_t = batch_angles[:, t]
                
                # Forward through network, passing previous hidden state
                hidden_states = self.model(input_t, hidden_states, steps=1)
                predicted_angle = self.model.decode_angle(hidden_states)
                
                # Compute error using circular distance
                loss = compute_error(predicted_angle, angle_t).mean()
                total_loss += loss
            
            # Average loss over sequence
            avg_loss = total_loss / seq_len
            
            # Add regularization terms to encourage parameter updates
            # This helps ensure all parameters get trained
            param_reg_loss = 0.0
            
            # Add small regularization for Poisson noise parameters
            param_reg_loss += 0.001 * (self.model.noise_rate_i - 0.1).abs() 
            param_reg_loss += 0.001 * (self.model.g_ei - 1.0).abs()
            
            # Combined loss
            combined_loss = avg_loss + param_reg_loss
            
            # Backward pass
            combined_loss.backward()
            
            # Compute gradient norm for monitoring
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm
            
            # Gradient clipping if enabled
            if self.config['clip_gradients']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['max_grad_norm']
                )
            
            # Optimizer step (Adam update)
            self.optimizer.step()
            
            # Apply biological constraints
            if self.config['apply_constraints']:
                self.constraint_fn()
            
            # Accumulate loss
            epoch_loss += avg_loss.item()  # Only track the actual tracking loss, not the regularization
            num_batches += 1
        
        return epoch_loss / num_batches, total_grad_norm / num_batches
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch.
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_angles in val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_angles = batch_angles.to(self.device)
                
                batch_size, seq_len, _ = batch_inputs.shape
                hidden_states = None  # Let the model handle the initialization
                total_loss = 0.0
                
                for t in range(seq_len):
                    input_t = batch_inputs[:, t, :]
                    angle_t = batch_angles[:, t]
                    
                    # Forward through network, passing previous hidden state
                    hidden_states = self.model(input_t, hidden_states, steps=1)
                    predicted_angle = self.model.decode_angle(hidden_states)
                    
                    loss = compute_error(predicted_angle, angle_t).mean()
                    total_loss += loss
                
                avg_loss = total_loss / seq_len
                val_loss += avg_loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def train(self):
        """
        Complete training procedure with Adam + cosine annealing.
        """
        print("\n=== Starting Enhanced Training ===")
        
        # Create dataset
        train_loader, val_loader = self.create_enhanced_dataset()
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        print("\nEpoch | Train Loss | Val Loss  | LR        | Grad Norm | Time")
        print("-" * 70)
        
        for epoch in range(self.config['max_epochs']):
            epoch_start_time = time.time()
            
            # Training
            train_loss, grad_norm = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Compute parameter norm
            param_norm = 0.0
            for param in self.model.parameters():
                param_norm += param.data.norm(2).item() ** 2
            param_norm = param_norm ** 0.5
            
            # Time tracking
            epoch_time = time.time() - epoch_start_time
            
            # Store history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['gradient_norm'].append(grad_norm)
            self.history['parameter_norm'].append(param_norm)
            self.history['time_per_epoch'].append(epoch_time)
            
            # Store ring attractor specific parameters
            self.param_history['g_ee'].append(self.model.g_ee.item())
            self.param_history['g_ei'].append(self.model.g_ei.item())
            self.param_history['g_ie'].append(self.model.g_ie.item())
            self.param_history['noise_rate_e'].append(self.model.noise_rate_e.item())
            self.param_history['noise_rate_i'].append(self.model.noise_rate_i.item())
            self.param_history['sigma_ee'].append(self.model.sigma_ee.item())
            
            # Logging
            if epoch % self.config['log_interval'] == 0:
                print(f"{epoch:5d} | {train_loss:.6f} | {val_loss:.6f} | "
                      f"{current_lr:.2e} | {grad_norm:.4f} | {epoch_time:.1f}s")
            
            # Step scheduler (cosine annealing)
            self.scheduler.step()
            
            # Early stopping check
            if self.config['early_stopping']:
                if val_loss < best_val_loss - self.config['min_delta']:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.config['save_checkpoints']:
                        self.save_checkpoint(epoch, 'best_model.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['patience']:
                    print(f"\n✓ Early stopping at epoch {epoch} (patience: {self.config['patience']})")
                    break
        
        print("\n=== Training Complete ===")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Plot results if requested
        if self.config['plot_progress']:
            self.plot_training_results()
        
        return self.history
    
    def plot_training_results(self):
        """
        Plot comprehensive training results for ring attractor network.
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Ring Attractor Network Training Results', fontsize=16)
        
        epochs = self.history['epoch']
        
        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Learning rate schedule (Cosine Annealing)
        axes[0, 1].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Cosine Annealing Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Gradient norm
        axes[1, 0].plot(epochs, self.history['gradient_norm'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter evolution (Ring Attractor specific)
        axes[1, 1].plot(epochs, self.param_history['g_ee'], label='g_ee (E→E gain)', linewidth=2)
        axes[1, 1].plot(epochs, self.param_history['g_ei'], label='g_ei (E→I gain)', linewidth=2)
        axes[1, 1].plot(epochs, self.param_history['g_ie'], label='g_ie (I→E gain)', linewidth=2)
        # Plot sigma_ee on secondary y-axis since it has different units
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(epochs, self.param_history['sigma_ee'], 'purple', linestyle='--', 
                     label='σ_EE (connection width)', linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gain Values')
        ax_twin.set_ylabel('σ_EE (Connection Width)', color='purple')
        axes[1, 1].set_title('Network Parameters Evolution')
        
        # Combine legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Poisson noise parameters
        axes[2, 0].plot(epochs, self.param_history['noise_rate_e'], label='Excitatory Poisson λ', linewidth=2)
        axes[2, 0].plot(epochs, self.param_history['noise_rate_i'], label='Inhibitory Poisson λ', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Poisson Rate Parameter (λ)')
        axes[2, 0].set_title('Poisson Noise Parameters')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Training efficiency
        if len(self.history['time_per_epoch']) > 0:
            axes[2, 1].plot(epochs, self.history['time_per_epoch'], 'orange', linewidth=2)
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Time per Epoch (s)')
            axes[2, 1].set_title('Training Speed')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_checkpoint(self, epoch, filename):
        """
        Save model checkpoint with training state.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'param_history': self.param_history,
            'config': self.config
        }
        torch.save(checkpoint, filename)
        print(f"✓ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename):
        """
        Load model checkpoint and resume training.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.param_history = checkpoint['param_history']
        
        print(f"✓ Checkpoint loaded: {filename}")
        return checkpoint['epoch']


def create_training_config(**kwargs):
    """
    Helper function to create training configuration for your ring attractor network.
    
    Usage:
        config = create_training_config(
            learning_rate=1e-3,
            max_epochs=150,
            batch_size=64
        )
    """
    return kwargs


def train_ring_attractor_with_adam_cosine(model, config=None):
    """
    Complete example of training your ring attractor network with Adam + Cosine Annealing.
    
    Args:
        model: Your RingAttractorNetwork instance
        config: Optional configuration dictionary
    
    Returns:
        Trained model and training history
    """
    
    # Create trainer
    trainer = AdamCosineTrainer(model, config)
    
    # Train the model
    history = trainer.train()
    
    return model, history, trainer 