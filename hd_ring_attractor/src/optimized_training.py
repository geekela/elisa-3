"""
Optimized training configuration to eliminate multiple peaks in ring attractor networks.

This module provides enhanced training configurations and constraint functions
specifically designed to maintain single, coherent activity bumps.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

def create_single_peak_config(difficulty='standard'):
    """
    Create optimized training configuration to ensure single peak formation.
    
    Args:
        difficulty: 'easy', 'standard', or 'challenging'
        
    Returns:
        Dictionary with optimized training parameters
    """
    
    if difficulty == 'easy':
        return {
            # Optimizer settings - more conservative
            'learning_rate': 5e-4,
            'weight_decay': 5e-4,  # Stronger regularization
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            
            # Scheduler settings
            'max_epochs': 150,
            'min_lr': 1e-7,
            
            # Training settings - smaller, more controlled
            'batch_size': 16,
            'sequence_length': 80,
            'n_sequences': 800,
            'validation_split': 0.2,
            
            # Enhanced constraints for single peak
            'apply_strict_constraints': True,
            'constraint_strength': 'strong',
            'monitor_peaks': True,
            'early_stop_on_multiple_peaks': True,
            
            # Network initialization - optimized for stability
            'init_noise_rate_e': 0.02,  # Much lower noise
            'init_noise_rate_i': 0.01,
            'init_sigma_ee': 0.25,       # Narrower connections
            'init_g_ee': 0.8,
            'init_g_ei': 1.2,
            'init_g_ie': 2.5,            # Stronger inhibition
            
            # Training dynamics
            'clip_gradients': True,
            'max_grad_norm': 0.1,        # Stricter clipping
            'use_adaptive_lr': True,
        }
    
    elif difficulty == 'standard':
        return {
            # Optimizer settings
            'learning_rate': 8e-4,
            'weight_decay': 2e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            
            # Scheduler settings
            'max_epochs': 120,
            'min_lr': 5e-7,
            
            # Training settings
            'batch_size': 24,
            'sequence_length': 100,
            'n_sequences': 1000,
            'validation_split': 0.2,
            
            # Enhanced constraints
            'apply_strict_constraints': True,
            'constraint_strength': 'medium',
            'monitor_peaks': True,
            'early_stop_on_multiple_peaks': True,
            
            # Network initialization
            'init_noise_rate_e': 0.05,
            'init_noise_rate_i': 0.02,
            'init_sigma_ee': 0.35,
            'init_g_ee': 0.9,
            'init_g_ei': 1.4,
            'init_g_ie': 2.2,
            
            # Training dynamics
            'clip_gradients': True,
            'max_grad_norm': 0.2,
            'use_adaptive_lr': True,
        }
    
    else:  # challenging
        return {
            # Optimizer settings
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            
            # Scheduler settings
            'max_epochs': 100,
            'min_lr': 1e-6,
            
            # Training settings
            'batch_size': 32,
            'sequence_length': 120,
            'n_sequences': 1200,
            'validation_split': 0.2,
            
            # Enhanced constraints
            'apply_strict_constraints': True,
            'constraint_strength': 'medium',
            'monitor_peaks': True,
            'early_stop_on_multiple_peaks': False,  # Let it train through
            
            # Network initialization
            'init_noise_rate_e': 0.08,
            'init_noise_rate_i': 0.04,
            'init_sigma_ee': 0.45,
            'init_g_ee': 1.0,
            'init_g_ei': 1.5,
            'init_g_ie': 2.0,
            
            # Training dynamics
            'clip_gradients': True,
            'max_grad_norm': 0.3,
            'use_adaptive_lr': True,
        }


def apply_single_peak_constraints(model, strength='medium'):
    """
    Apply optimized constraints to ensure single peak formation.
    
    Args:
        model: RingAttractorNetwork instance
        strength: 'weak', 'medium', or 'strong'
    """
    
    with torch.no_grad():
        if strength == 'strong':
            # Very strict constraints for guaranteed single peaks
            model.g_ee.data.clamp_(min=0.5, max=1.2)      # Moderate excitation
            model.g_ei.data.clamp_(min=1.0, max=2.0)      # Strong E→I
            model.g_ie.data.clamp_(min=1.8, max=3.5)      # Very strong I→E (key for single peak)
            
            # Very low noise to prevent fragmentation
            model.noise_rate_e.data.clamp_(min=0.005, max=0.05)
            model.noise_rate_i.data.clamp_(min=0.001, max=0.02)
            
            # Narrow connections for focused bumps
            model.sigma_ee.data.clamp_(min=0.15, max=0.4)
            
        elif strength == 'medium':
            # Balanced constraints
            model.g_ee.data.clamp_(min=0.4, max=1.5)
            model.g_ei.data.clamp_(min=0.8, max=2.2)
            model.g_ie.data.clamp_(min=1.5, max=3.0)
            
            # Moderate noise levels
            model.noise_rate_e.data.clamp_(min=0.01, max=0.1)
            model.noise_rate_i.data.clamp_(min=0.005, max=0.05)
            
            # Reasonable connection width
            model.sigma_ee.data.clamp_(min=0.2, max=0.6)
            
        else:  # weak
            # Minimal constraints
            model.g_ee.data.clamp_(min=0.3, max=2.0)
            model.g_ei.data.clamp_(min=0.5, max=3.0)
            model.g_ie.data.clamp_(min=1.0, max=4.0)
            
            model.noise_rate_e.data.clamp_(min=0.005, max=0.2)
            model.noise_rate_i.data.clamp_(min=0.001, max=0.1)
            
            model.sigma_ee.data.clamp_(min=0.1, max=1.0)
        
        # Always constrain inhibitory weights for proper function
        model.W_IE.data.clamp_(max=0.3)  # Inhibitory connections should be small/negative
        model.W_EI.data.clamp_(min=-0.3, max=1.5)
        
        # Enforce excitation/inhibition balance for single peak
        # Key insight: I→E must be stronger than E→E to prevent multiple peaks
        if model.g_ie.data < model.g_ee.data * 1.5:
            model.g_ie.data = model.g_ee.data * 1.8  # Force stronger inhibition


def initialize_for_single_peak(model, config):
    """
    Initialize network parameters optimized for single peak formation.
    
    Args:
        model: RingAttractorNetwork instance
        config: Configuration dictionary
    """
    
    with torch.no_grad():
        # Set optimized initial values
        model.noise_rate_e.data.fill_(config.get('init_noise_rate_e', 0.05))
        model.noise_rate_i.data.fill_(config.get('init_noise_rate_i', 0.02))
        model.sigma_ee.data.fill_(config.get('init_sigma_ee', 0.35))
        model.g_ee.data.fill_(config.get('init_g_ee', 0.9))
        model.g_ei.data.fill_(config.get('init_g_ei', 1.4))
        model.g_ie.data.fill_(config.get('init_g_ie', 2.2))
        
        # Initialize inhibitory weights for proper balance
        # W_IE should be mostly negative (inhibitory connections)
        model.W_IE.data.normal_(mean=-0.1, std=0.05)
        model.W_IE.data.clamp_(max=0.2)
        
        # W_EI can be positive (excitatory to inhibitory)
        model.W_EI.data.normal_(mean=0.1, std=0.05)
        model.W_EI.data.clamp_(min=0.0, max=1.0)


def monitor_peak_count(model, input_directions=None, threshold=0.1):
    """
    Monitor the number of activity peaks in the network.
    
    Args:
        model: RingAttractorNetwork instance
        input_directions: List of directions to test (if None, uses random)
        threshold: Minimum peak height relative to maximum
        
    Returns:
        Dictionary with peak statistics
    """
    
    model.eval()
    
    if input_directions is None:
        # Test at 8 different directions
        input_directions = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    peak_counts = []
    max_activities = []
    
    with torch.no_grad():
        for direction in input_directions:
            model.reset_state()
            model.initialize_bump(direction, width=0.3, amplitude=0.1)
            
            # Let network settle
            for _ in range(10):
                _ = model(external_input=None, steps=1)
            
            # Analyze final activity
            activity = model.r_e.cpu().numpy()
            max_activity = np.max(activity)
            max_activities.append(max_activity)
            
            if max_activity > 0.01:  # Only analyze if there's significant activity
                # Find peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(activity, height=threshold * max_activity, distance=10)
                peak_counts.append(len(peaks))
            else:
                peak_counts.append(0)
    
    return {
        'peak_counts': peak_counts,
        'mean_peaks': np.mean(peak_counts),
        'max_peaks': np.max(peak_counts),
        'single_peak_ratio': np.mean(np.array(peak_counts) <= 1),
        'max_activities': max_activities,
        'mean_activity': np.mean(max_activities)
    }


def diagnose_multiple_peaks(model):
    """
    Diagnose why multiple peaks are occurring.
    
    Args:
        model: RingAttractorNetwork instance
        
    Returns:
        Dictionary with diagnostic information
    """
    
    diagnosis = {
        'parameter_values': {
            'g_ee': model.g_ee.item(),
            'g_ei': model.g_ei.item(), 
            'g_ie': model.g_ie.item(),
            'noise_rate_e': model.noise_rate_e.item(),
            'noise_rate_i': model.noise_rate_i.item(),
            'sigma_ee': model.sigma_ee.item(),
        },
        'issues': [],
        'recommendations': []
    }
    
    # Check excitation/inhibition balance
    g_ee = model.g_ee.item()
    g_ie = model.g_ie.item()
    
    if g_ie < g_ee * 1.5:
        diagnosis['issues'].append(f"Insufficient inhibition: g_ie ({g_ie:.3f}) should be >1.5x g_ee ({g_ee:.3f})")
        diagnosis['recommendations'].append(f"Increase g_ie to at least {g_ee * 1.8:.3f}")
    
    # Check noise levels
    noise_e = model.noise_rate_e.item()
    if noise_e > 0.1:
        diagnosis['issues'].append(f"High excitatory noise: {noise_e:.3f} may fragment bumps")
        diagnosis['recommendations'].append("Reduce noise_rate_e to <0.1")
    
    # Check connection width
    sigma = model.sigma_ee.item()
    if sigma > 0.6:
        diagnosis['issues'].append(f"Wide connections: σ_EE={sigma:.3f} may support multiple bumps")
        diagnosis['recommendations'].append("Reduce sigma_ee to <0.6")
    
    # Check for proper constraints
    if not (0.1 <= g_ee <= 2.0):
        diagnosis['issues'].append(f"g_ee ({g_ee:.3f}) outside reasonable range [0.1, 2.0]")
    
    return diagnosis


class SinglePeakLoss(nn.Module):
    """
    Custom loss function that penalizes multiple peaks while maintaining tracking accuracy.
    """
    
    def __init__(self, peak_penalty=1.0, sparsity_penalty=0.1):
        super().__init__()
        self.peak_penalty = peak_penalty
        self.sparsity_penalty = sparsity_penalty
        
    def forward(self, predicted_activity, target_angles, model):
        """
        Compute loss with single-peak enforcement.
        
        Args:
            predicted_activity: Network activity output (batch_size, n_exc)
            target_angles: Target head directions (batch_size,)
            model: RingAttractorNetwork instance
            
        Returns:
            Total loss with peak penalties
        """
        
        # Basic tracking loss
        decoded_angles = model.decode_angle(predicted_activity)
        tracking_loss = torch.mean((decoded_angles - target_angles)**2)
        
        # Peak count penalty
        peak_penalty = 0.0
        for i in range(predicted_activity.shape[0]):
            activity = predicted_activity[i]
            
            # Compute second derivative to find peaks
            activity_padded = torch.cat([activity[-1:], activity, activity[:1]])
            second_deriv = activity_padded[:-2] - 2*activity_padded[1:-1] + activity_padded[2:]
            
            # Count negative second derivatives (peaks)
            peaks = (second_deriv < -0.01).float().sum()
            
            # Penalize having more than 1 peak
            if peaks > 1:
                peak_penalty += self.peak_penalty * (peaks - 1)**2
        
        peak_penalty /= predicted_activity.shape[0]
        
        # Sparsity penalty to encourage focused activity
        sparsity_loss = self.sparsity_penalty * torch.mean(predicted_activity**2)
        
        return tracking_loss + peak_penalty + sparsity_loss, {
            'tracking_loss': tracking_loss.item(),
            'peak_penalty': peak_penalty.item() if isinstance(peak_penalty, torch.Tensor) else peak_penalty,
            'sparsity_loss': sparsity_loss.item()
        }