"""
Single Peak Trainer - Optimized training to eliminate multiple peaks in ring attractor networks.

This module provides a specialized trainer that ensures single, coherent activity bumps
through optimized parameters, constraints, and loss functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict

try:
    from .utils import angle_to_input, generate_trajectory, compute_error
    from .optimized_training import (create_single_peak_config, apply_single_peak_constraints,
                                   initialize_for_single_peak, monitor_peak_count, 
                                   diagnose_multiple_peaks, SinglePeakLoss)
except ImportError:
    from utils import angle_to_input, generate_trajectory, compute_error
    from optimized_training import (create_single_peak_config, apply_single_peak_constraints,
                                  initialize_for_single_peak, monitor_peak_count, 
                                  diagnose_multiple_peaks, SinglePeakLoss)


class SinglePeakTrainer:
    """
    Specialized trainer for ensuring single-peak ring attractor dynamics.
    
    This trainer implements advanced techniques to eliminate multiple peaks:
    - Optimized parameter initialization
    - Strict biological constraints
    - Custom loss function with peak penalties
    - Real-time peak monitoring
    - Adaptive training strategies
    """
    
    def __init__(self, model, difficulty='standard', device='cpu'):
        """
        Initialize the single-peak trainer.
        
        Args:
            model: RingAttractorNetwork instance
            difficulty: 'easy', 'standard', or 'challenging'
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        self.difficulty = difficulty
        
        # Get optimized configuration
        self.config = create_single_peak_config(difficulty)
        self.config['device'] = device
        
        # Initialize model with optimized parameters
        initialize_for_single_peak(self.model, self.config)
        
        # Setup optimizer with custom parameter groups
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup custom loss function
        self.criterion = SinglePeakLoss(peak_penalty=1.0, sparsity_penalty=0.1)
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'tracking_loss': [],
            'peak_penalty': [],
            'sparsity_loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'peak_statistics': [],
            'single_peak_ratio': [],
            'time_per_epoch': []
        }
        
        # Parameter tracking
        self.param_history = {
            'g_ee': [], 'g_ei': [], 'g_ie': [],
            'noise_rate_e': [], 'noise_rate_i': [],
            'sigma_ee': []
        }
        
        print("=== Single Peak Ring Attractor Trainer Initialized ===")
        print(f"Difficulty: {difficulty}")
        print(f"Device: {device}")
        print(f"Learning Rate: {self.config['learning_rate']:.1e}")
        print(f"Max Epochs: {self.config['max_epochs']}")
        print(f"Constraint Strength: {self.config['constraint_strength']}")
        
        # Display initial parameters
        self._print_current_parameters()
        
    def _setup_optimizer(self):
        """Setup optimizer with parameter-specific learning rates."""
        
        # Create parameter groups with different learning rates
        param_groups = [
            # Gain parameters - moderate learning rate
            {'params': [self.model.g_ee, self.model.g_ei, self.model.g_ie],
             'lr': self.config['learning_rate'] * 0.8},
            
            # Noise parameters - slower learning (these are critical for stability)
            {'params': [self.model.noise_rate_e, self.model.noise_rate_i],
             'lr': self.config['learning_rate'] * 0.3},
            
            # Connection width - careful tuning needed
            {'params': [self.model.sigma_ee],
             'lr': self.config['learning_rate'] * 0.5},
            
            # Inhibitory weights - critical for single peak
            {'params': [self.model.W_IE, self.model.W_EI],
             'lr': self.config['learning_rate'] * 0.6}
        ]
        
        optimizer = torch.optim.Adam(
            param_groups,
            betas=self.config['betas'],
            eps=self.config['eps'],
            weight_decay=self.config['weight_decay']
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['max_epochs'],
            eta_min=self.config['min_lr']
        )
    
    def _print_current_parameters(self):
        """Print current network parameters."""
        print(f"\nCurrent Parameters:")
        print(f"  g_ee: {self.model.g_ee.item():.4f}")
        print(f"  g_ei: {self.model.g_ei.item():.4f}")
        print(f"  g_ie: {self.model.g_ie.item():.4f}")
        print(f"  noise_rate_e: {self.model.noise_rate_e.item():.4f}")
        print(f"  noise_rate_i: {self.model.noise_rate_i.item():.4f}")
        print(f"  sigma_ee: {self.model.sigma_ee.item():.4f}")
        
        # Check balance
        ratio = self.model.g_ie.item() / self.model.g_ee.item()
        print(f"  Inhibition/Excitation ratio: {ratio:.2f} {'✓' if ratio >= 1.5 else '⚠'}")
    
    def create_dataset(self):
        """Create training dataset with controlled difficulty."""
        print("Creating optimized dataset...")
        
        all_angles = []
        all_inputs = []
        
        # Generate trajectories with controlled complexity
        for _ in tqdm(range(self.config['n_sequences']), desc="Generating sequences"):
            if self.difficulty == 'easy':
                # Slow, smooth trajectories
                angular_velocity_std = np.random.uniform(0.02, 0.08)
            elif self.difficulty == 'standard':
                # Moderate complexity
                angular_velocity_std = np.random.uniform(0.05, 0.15)
            else:  # challenging
                # More complex trajectories
                angular_velocity_std = np.random.uniform(0.08, 0.25)
            
            angles, _ = generate_trajectory(
                self.config['sequence_length'],
                dt=0.1,
                angular_velocity_std=angular_velocity_std
            )
            
            inputs = angle_to_input(angles, n_exc=self.model.n_exc, 
                                  input_strength=1.0, input_width=0.3)
            
            all_angles.append(angles)
            all_inputs.append(inputs)
        
        # Convert to tensors
        angles_tensor = torch.stack(all_angles)
        inputs_tensor = torch.stack(all_inputs)
        
        # Train/validation split
        n_train = int(len(angles_tensor) * (1 - self.config['validation_split']))
        
        train_angles = angles_tensor[:n_train]
        train_inputs = inputs_tensor[:n_train]
        val_angles = angles_tensor[n_train:]
        val_inputs = inputs_tensor[n_train:]
        
        print(f"✓ Created {len(train_angles)} training sequences")
        print(f"✓ Created {len(val_angles)} validation sequences")
        
        return (train_angles, train_inputs), (val_angles, val_inputs)
    
    def train_epoch(self, train_data):
        """Train for one epoch."""
        self.model.train()
        train_angles, train_inputs = train_data
        
        total_loss = 0.0
        total_tracking_loss = 0.0
        total_peak_penalty = 0.0
        total_sparsity_loss = 0.0
        n_batches = 0
        
        # Create batches
        n_samples = len(train_angles)
        indices = torch.randperm(n_samples, device=self.device)
        
        for i in range(0, n_samples, self.config['batch_size']):
            batch_indices = indices[i:i + self.config['batch_size']]
            batch_angles = train_angles[batch_indices].to(self.device)
            batch_inputs = train_inputs[batch_indices].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass through sequences
            batch_loss = 0.0
            batch_tracking = 0.0
            batch_peak = 0.0
            batch_sparsity = 0.0
            
            for t in range(batch_angles.shape[1]):
                # Get inputs and targets for this timestep
                inputs_t = batch_inputs[:, t, :]
                targets_t = batch_angles[:, t]
                
                # Forward pass
                activity = self.model(inputs_t, steps=1)
                
                # Compute loss with peak penalties
                loss, loss_components = self.criterion(activity, targets_t, self.model)
                
                batch_loss += loss
                batch_tracking += loss_components['tracking_loss']
                batch_peak += loss_components['peak_penalty']
                batch_sparsity += loss_components['sparsity_loss']
            
            # Average over sequence length
            batch_loss /= batch_angles.shape[1]
            batch_tracking /= batch_angles.shape[1]
            batch_peak /= batch_angles.shape[1]
            batch_sparsity /= batch_angles.shape[1]
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping
            if self.config.get('clip_gradients', True):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('max_grad_norm', 0.2)
                )
            
            self.optimizer.step()
            
            # Apply constraints after each update
            apply_single_peak_constraints(
                self.model, 
                strength=self.config['constraint_strength']
            )
            
            # Accumulate losses
            total_loss += batch_loss.item()
            total_tracking_loss += batch_tracking
            total_peak_penalty += batch_peak
            total_sparsity_loss += batch_sparsity
            n_batches += 1
        
        return {
            'total_loss': total_loss / n_batches,
            'tracking_loss': total_tracking_loss / n_batches,
            'peak_penalty': total_peak_penalty / n_batches,
            'sparsity_loss': total_sparsity_loss / n_batches
        }
    
    def validate_epoch(self, val_data):
        """Validate for one epoch."""
        self.model.eval()
        val_angles, val_inputs = val_data
        
        total_loss = 0.0
        total_tracking_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            n_samples = len(val_angles)
            
            for i in range(0, n_samples, self.config['batch_size']):
                end_idx = min(i + self.config['batch_size'], n_samples)
                batch_angles = val_angles[i:end_idx].to(self.device)
                batch_inputs = val_inputs[i:end_idx].to(self.device)
                
                batch_loss = 0.0
                batch_tracking = 0.0
                
                for t in range(batch_angles.shape[1]):
                    inputs_t = batch_inputs[:, t, :]
                    targets_t = batch_angles[:, t]
                    
                    activity = self.model(inputs_t, steps=1)
                    loss, loss_components = self.criterion(activity, targets_t, self.model)
                    
                    batch_loss += loss.item()
                    batch_tracking += loss_components['tracking_loss']
                
                batch_loss /= batch_angles.shape[1]
                batch_tracking /= batch_angles.shape[1]
                
                total_loss += batch_loss
                total_tracking_loss += batch_tracking
                n_batches += 1
        
        return {
            'val_loss': total_loss / n_batches,
            'val_tracking_loss': total_tracking_loss / n_batches
        }
    
    def train(self, save_best=True, verbose=True):
        """
        Main training loop with single-peak optimization.
        
        Args:
            save_best: Whether to save the best model
            verbose: Whether to print progress
            
        Returns:
            Trained model and training history
        """
        
        print("=== Starting Single-Peak Training ===")
        
        # Create dataset
        train_data, val_data = self.create_dataset()
        
        best_val_loss = float('inf')
        best_single_peak_ratio = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_data)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_data)
            
            # Monitor peak statistics
            peak_stats = monitor_peak_count(self.model)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Compute gradient norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            
            # Record history
            epoch_time = time.time() - start_time
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['tracking_loss'].append(train_metrics['tracking_loss'])
            self.history['peak_penalty'].append(train_metrics['peak_penalty'])
            self.history['sparsity_loss'].append(train_metrics['sparsity_loss'])
            self.history['learning_rate'].append(current_lr)
            self.history['gradient_norm'].append(grad_norm)
            self.history['peak_statistics'].append(peak_stats)
            self.history['single_peak_ratio'].append(peak_stats['single_peak_ratio'])
            self.history['time_per_epoch'].append(epoch_time)
            
            # Record parameters
            self.param_history['g_ee'].append(self.model.g_ee.item())
            self.param_history['g_ei'].append(self.model.g_ei.item())
            self.param_history['g_ie'].append(self.model.g_ie.item())
            self.param_history['noise_rate_e'].append(self.model.noise_rate_e.item())
            self.param_history['noise_rate_i'].append(self.model.noise_rate_i.item())
            self.param_history['sigma_ee'].append(self.model.sigma_ee.item())
            
            # Check for improvement
            if val_metrics['val_loss'] < best_val_loss and peak_stats['single_peak_ratio'] >= 0.8:
                best_val_loss = val_metrics['val_loss']
                best_single_peak_ratio = peak_stats['single_peak_ratio']
                patience_counter = 0
                
                if save_best:
                    torch.save(self.model.state_dict(), 'best_single_peak_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and (epoch % self.config.get('log_interval', 10) == 0 or epoch < 5):
                print(f"Epoch {epoch:3d} | "
                      f"Loss: {train_metrics['total_loss']:.6f} | "
                      f"Val: {val_metrics['val_loss']:.6f} | "
                      f"Track: {train_metrics['tracking_loss']:.6f} | "
                      f"Peaks: {peak_stats['mean_peaks']:.1f} | "
                      f"Single: {peak_stats['single_peak_ratio']:.1%} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")
                
                if epoch % 20 == 0:
                    self._print_current_parameters()
            
            # Early stopping
            if (self.config.get('early_stop_on_multiple_peaks', False) and 
                peak_stats['single_peak_ratio'] < 0.5 and epoch > 20):
                print(f"Early stopping: Single peak ratio too low ({peak_stats['single_peak_ratio']:.1%})")
                break
            
            if patience_counter >= self.config.get('patience', 30):
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
        
        print(f"\n=== Training Complete ===")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best single peak ratio: {best_single_peak_ratio:.1%}")
        
        # Final diagnosis
        diagnosis = diagnose_multiple_peaks(self.model)
        if diagnosis['issues']:
            print(f"\nRemaining issues:")
            for issue in diagnosis['issues']:
                print(f"  • {issue}")
            print(f"Recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  • {rec}")
        else:
            print("✓ No parameter issues detected!")
        
        return self.model, self.history
    
    def plot_training_progress(self):
        """Plot comprehensive training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = self.history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss components
        axes[0, 1].plot(epochs, self.history['tracking_loss'], label='Tracking')
        axes[0, 1].plot(epochs, self.history['peak_penalty'], label='Peak Penalty')
        axes[0, 1].plot(epochs, self.history['sparsity_loss'], label='Sparsity')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Single peak ratio
        axes[0, 2].plot(epochs, self.history['single_peak_ratio'], 'g-', linewidth=2)
        axes[0, 2].axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
        axes[0, 2].set_title('Single Peak Ratio')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Ratio')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Parameters evolution
        axes[1, 0].plot(epochs, self.param_history['g_ee'], label='g_ee')
        axes[1, 0].plot(epochs, self.param_history['g_ei'], label='g_ei')
        axes[1, 0].plot(epochs, self.param_history['g_ie'], label='g_ie')
        axes[1, 0].set_title('Gain Parameters')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Noise parameters
        axes[1, 1].plot(epochs, self.param_history['noise_rate_e'], label='Excitatory')
        axes[1, 1].plot(epochs, self.param_history['noise_rate_i'], label='Inhibitory')
        axes[1, 1].set_title('Noise Rates')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Connection width
        axes[1, 2].plot(epochs, self.param_history['sigma_ee'], 'b-', linewidth=2)
        axes[1, 2].set_title('Connection Width (σ_EE)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('σ_EE')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()


def train_single_peak_network(n_exc=800, n_inh=200, difficulty='standard', 
                            device='cpu', epochs=None):
    """
    Convenience function to train a single-peak ring attractor network.
    
    Args:
        n_exc: Number of excitatory neurons
        n_inh: Number of inhibitory neurons  
        difficulty: 'easy', 'standard', or 'challenging'
        device: Device to use
        epochs: Number of epochs (if None, uses config default)
        
    Returns:
        Trained model and history
    """
    
    # Import here to avoid circular imports
    from models import RingAttractorNetwork
    
    # Create model
    model = RingAttractorNetwork(n_exc=n_exc, n_inh=n_inh, device=device)
    
    # Create trainer
    trainer = SinglePeakTrainer(model, difficulty=difficulty, device=device)
    
    # Override epochs if specified
    if epochs is not None:
        trainer.config['max_epochs'] = epochs
    
    # Train
    trained_model, history = trainer.train()
    
    # Plot results
    trainer.plot_training_progress()
    
    # Final evaluation
    final_stats = monitor_peak_count(trained_model)
    print(f"\nFinal Evaluation:")
    print(f"  Single peak ratio: {final_stats['single_peak_ratio']:.1%}")
    print(f"  Mean activity: {final_stats['mean_activity']:.4f}")
    print(f"  Max peaks observed: {final_stats['max_peaks']}")
    
    if final_stats['single_peak_ratio'] >= 0.8:
        print("✅ SUCCESS: Network maintains single peaks!")
    else:
        print("⚠️  Warning: Multiple peaks still present")
    
    return trained_model, history, trainer