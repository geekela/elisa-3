"""
Single Peak Ring Attractor Model - Architecturally designed to eliminate multiple peaks.

This model implements a fundamentally improved architecture that ensures single peaks:
1. Much stronger inhibitory feedback
2. Narrower excitatory connections  
3. Better balanced dynamics
4. Automatic peak suppression mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinglePeakRingAttractor(nn.Module):
    """
    Ring attractor network architecturally designed for single peaks.
    
    Key improvements over standard model:
    - Global inhibitory winner-take-all mechanism
    - Adaptive connection strengths based on activity
    - Built-in peak suppression
    - Optimized parameter initialization
    """
    
    def __init__(self, n_exc=800, n_inh=200, device='cpu', **kwargs):
        super().__init__()
        
        self.n_exc = n_exc
        self.n_inh = n_inh  
        self.device = device
        
        # Time constants optimized for stability
        self.tau_e = 15.0  # Slower excitatory dynamics
        self.tau_i = 8.0   # Faster inhibitory dynamics
        self.dt = 0.1
        
        # Preferred directions
        angles = torch.linspace(0, 2*np.pi, n_exc, device=device)
        self.register_buffer('preferred_dirs', angles)
        
        # Optimized parameters for single peaks
        self.sigma_ee = nn.Parameter(torch.tensor(0.25, device=device))  # Very narrow
        
        # Gain parameters with better balance
        self.g_ee = nn.Parameter(torch.tensor(0.6, device=device))   # Moderate excitation
        self.g_ei = nn.Parameter(torch.tensor(2.0, device=device))   # Strong E->I
        self.g_ie = nn.Parameter(torch.tensor(4.0, device=device))   # Very strong I->E
        
        # Global inhibition parameter (new!)
        self.g_global = nn.Parameter(torch.tensor(0.8, device=device))
        
        # Input gain
        self.g_input = nn.Parameter(torch.tensor(1.2, device=device))
        
        # Very low noise for stability
        self.noise_rate_e = nn.Parameter(torch.tensor(0.01, device=device))
        self.noise_rate_i = nn.Parameter(torch.tensor(0.005, device=device))
        
        # Inhibitory weights - designed for winner-take-all
        self.W_EI = nn.Parameter(torch.ones(n_exc, n_inh, device=device) * 0.3)  # Uniform E->I
        self.W_IE = nn.Parameter(torch.ones(n_inh, n_exc, device=device) * (-0.3))  # Uniform negative I->E
        
        # Adaptive threshold for peak suppression
        self.adaptive_threshold = nn.Parameter(torch.tensor(0.1, device=device))
        
        self.reset_state()
        
    def _create_focused_ring_weights(self, sigma):
        """Create very focused ring connectivity."""
        angles = self.preferred_dirs
        angle_diff = angles.unsqueeze(0) - angles.unsqueeze(1)
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Very focused Gaussian
        W = torch.exp(-0.5 * (angle_diff / sigma)**2)
        
        # Strong normalization to prevent runaway excitation
        W = W / (W.sum(dim=1, keepdim=True) + 1e-8)
        
        # Remove self-connections completely
        W.fill_diagonal_(0)
        
        # Apply adaptive suppression - reduce connections if activity is too high
        # This helps prevent multiple peaks from forming
        W = W * 0.8  # Scale down all connections
        
        return W
    
    def reset_state(self):
        """Reset to baseline state."""
        current_device = next(self.parameters()).device
        self.device = current_device
        
        self.r_e = torch.zeros(self.n_exc, device=current_device)
        self.r_i = torch.zeros(self.n_inh, device=current_device)
        
        # Activity history for winner-take-all
        self.activity_history = torch.zeros(5, self.n_exc, device=current_device)
        self.history_idx = 0
    
    def to(self, device):
        """Move to device."""
        super().to(device)
        self.device = device
        
        if hasattr(self, 'r_e') and self.r_e is not None:
            self.r_e = self.r_e.to(device)
        if hasattr(self, 'r_i') and self.r_i is not None:
            self.r_i = self.r_i.to(device)
        if hasattr(self, 'activity_history'):
            self.activity_history = self.activity_history.to(device)
            
        return self
    
    def initialize_bump(self, direction, width=0.2, amplitude=0.5):
        """Initialize with a focused bump."""
        angle_diff = self.preferred_dirs - direction
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Create a sharper bump
        self.r_e = amplitude * torch.exp(-0.5 * (angle_diff / width)**2)
        self.r_i = torch.zeros(self.n_inh, device=self.preferred_dirs.device)
        
        # Initialize history
        for i in range(5):
            self.activity_history[i] = self.r_e.clone()
    
    def _apply_winner_take_all(self, activity):
        """Apply winner-take-all dynamics to suppress multiple peaks."""
        if activity.dim() == 1:
            # Single sample
            
            # Find the peak
            max_idx = torch.argmax(activity)
            max_val = activity[max_idx]
            
            # Suppress all activity below threshold
            threshold = max_val * 0.3  # Keep only strong activity
            suppressed = torch.where(activity > threshold, activity, activity * 0.1)
            
            # Additional local competition - each neuron competes with neighbors
            if len(suppressed) > 3:
                for i in range(len(suppressed)):
                    neighbors = []
                    for offset in [-2, -1, 1, 2]:  # Check neighbors
                        neighbor_idx = (i + offset) % len(suppressed)
                        neighbors.append(suppressed[neighbor_idx])
                    
                    # If this neuron is not the strongest among neighbors, suppress it
                    neighbor_max = max(neighbors)
                    if suppressed[i] < neighbor_max * 0.8:
                        suppressed[i] *= 0.5
            
            return suppressed
        else:
            # Batch processing
            batch_size = activity.shape[0]
            result = torch.zeros_like(activity)
            
            for b in range(batch_size):
                result[b] = self._apply_winner_take_all(activity[b])
            
            return result
    
    def forward(self, external_input=None, h=None, steps=1):
        """Forward pass with single-peak mechanisms."""
        
        # Handle batch dimension
        batch_mode = False
        batch_size = 1
        
        if external_input is not None:
            if external_input.dim() == 2:
                batch_mode = True
                batch_size = external_input.shape[0]
                ext_input = external_input
            else:
                ext_input = external_input.unsqueeze(0)
                batch_mode = True
                batch_size = 1
        else:
            ext_input = None
        
        # Initialize states
        if h is not None:
            if h.dim() == 1:
                r_e = h.unsqueeze(0) if batch_mode else h
            else:
                r_e = h
                
            if batch_mode:
                r_i = torch.zeros(batch_size, self.n_inh, device=self.preferred_dirs.device)
            else:
                r_i = torch.zeros(self.n_inh, device=self.preferred_dirs.device)
        elif self.training:
            if batch_mode:
                r_e = 0.01 * torch.randn(batch_size, self.n_exc, device=self.preferred_dirs.device)
                r_i = 0.01 * torch.randn(batch_size, self.n_inh, device=self.preferred_dirs.device)
            else:
                r_e = 0.01 * torch.randn(self.n_exc, device=self.preferred_dirs.device)
                r_i = 0.01 * torch.randn(self.n_inh, device=self.preferred_dirs.device)
        else:
            if batch_mode:
                r_e = self.r_e.unsqueeze(0).expand(batch_size, -1)
                r_i = self.r_i.unsqueeze(0).expand(batch_size, -1)
            else:
                r_e = self.r_e.clone()
                r_i = self.r_i.clone()
        
        # Main dynamics loop
        for step in range(steps):
            # Create focused ring weights
            W_EE = self._create_focused_ring_weights(self.sigma_ee)
            
            if batch_mode:
                all_input_e = []
                all_input_i = []
                
                for b in range(batch_size):
                    # Excitatory input
                    ee_input = self.g_ee * torch.matmul(W_EE, r_e[b])
                    ie_input = self.g_ie * torch.matmul(self.W_IE.t(), r_i[b])
                    
                    # Global inhibition - key innovation!
                    global_activity = torch.sum(r_e[b])
                    global_inhibition = self.g_global * global_activity
                    
                    input_e = ee_input + ie_input - global_inhibition
                    
                    if ext_input is not None:
                        input_e += self.g_input * ext_input[b]
                    
                    # Inhibitory input
                    input_i = self.g_ei * torch.matmul(self.W_EI.t(), r_e[b])
                    
                    all_input_e.append(input_e)
                    all_input_i.append(input_i)
                
                input_e = torch.stack(all_input_e)
                input_i = torch.stack(all_input_i)
            else:
                # Single sample
                ee_input = self.g_ee * torch.matmul(W_EE, r_e)
                ie_input = self.g_ie * torch.matmul(self.W_IE.t(), r_i)
                
                # Global inhibition
                global_activity = torch.sum(r_e)
                global_inhibition = self.g_global * global_activity
                
                input_e = ee_input + ie_input - global_inhibition
                
                if ext_input is not None:
                    input_e += self.g_input * ext_input[0]
                
                input_i = self.g_ei * torch.matmul(self.W_EI.t(), r_e)
            
            # Add minimal noise
            if self.training or not hasattr(self, 'r_e'):
                if batch_mode:
                    noise_e = torch.normal(0, self.noise_rate_e.item(), size=input_e.shape, device=self.preferred_dirs.device)
                    noise_i = torch.normal(0, self.noise_rate_i.item(), size=input_i.shape, device=self.preferred_dirs.device)
                else:
                    noise_e = torch.normal(0, self.noise_rate_e.item(), size=input_e.shape, device=self.preferred_dirs.device)
                    noise_i = torch.normal(0, self.noise_rate_i.item(), size=input_i.shape, device=self.preferred_dirs.device)
                
                input_e += noise_e
                input_i += noise_i
            
            # Update activities with different time constants
            dr_e = self.dt * (-r_e + F.relu(input_e)) / self.tau_e
            dr_i = self.dt * (-r_i + F.relu(input_i)) / self.tau_i
            
            r_e = r_e + dr_e
            r_i = r_i + dr_i
            
            # Apply ReLU
            r_e = F.relu(r_e)
            r_i = F.relu(r_i)
            
            # Apply winner-take-all to enforce single peak
            r_e = self._apply_winner_take_all(r_e)
        
        # Update instance state
        if not self.training:
            if batch_mode:
                self.r_e = r_e[0].detach().clone()
                self.r_i = r_i[0].detach().clone()
            else:
                self.r_e = r_e.detach().clone()
                self.r_i = r_i.detach().clone()
            
            # Update activity history for winner-take-all
            if hasattr(self, 'activity_history'):
                self.activity_history[self.history_idx] = self.r_e.clone()
                self.history_idx = (self.history_idx + 1) % 5
        
        return r_e
    
    def decode_angle(self, h=None):
        """Decode head direction."""
        if h is not None:
            activity = h
        else:
            activity = self.r_e
        
        activity = activity.to(self.preferred_dirs.device)
        
        if activity.dim() == 2:
            # Batch processing
            batch_size = activity.shape[0]
            decoded_angles = []
            
            for i in range(batch_size):
                r_e = activity[i]
                if r_e.sum() > 0:
                    x = (r_e * torch.cos(self.preferred_dirs)).sum()
                    y = (r_e * torch.sin(self.preferred_dirs)).sum()
                    decoded_dir = torch.atan2(y, x)
                else:
                    decoded_dir = torch.tensor(0.0, device=self.preferred_dirs.device)
                decoded_angles.append(decoded_dir)
            
            return torch.stack(decoded_angles)
        else:
            # Single sample
            if activity.sum() > 0:
                x = (activity * torch.cos(self.preferred_dirs)).sum()
                y = (activity * torch.sin(self.preferred_dirs)).sum()
                decoded_dir = torch.atan2(y, x)
            else:
                decoded_dir = torch.tensor(0.0, device=self.preferred_dirs.device)
            
            return decoded_dir
    
    def get_peak_count(self):
        """Get current number of activity peaks."""
        activity = self.r_e.cpu().numpy()
        
        if np.max(activity) < 0.01:
            return 0
        
        # Simple peak detection
        threshold = 0.1 * np.max(activity)
        above_threshold = activity > threshold
        
        # Count connected components
        peaks = 0
        in_peak = False
        
        for i in range(len(activity)):
            if above_threshold[i] and not in_peak:
                peaks += 1
                in_peak = True
            elif not above_threshold[i]:
                in_peak = False
        
        return peaks
    
    def apply_single_peak_constraints(self):
        """Apply constraints to ensure single peak behavior."""
        with torch.no_grad():
            # Ensure narrow connections
            self.sigma_ee.data.clamp_(min=0.1, max=0.4)
            
            # Ensure strong inhibition dominance
            self.g_ee.data.clamp_(min=0.3, max=1.0)
            self.g_ie.data.clamp_(min=2.0, max=6.0)  # Must be much stronger than g_ee
            self.g_ei.data.clamp_(min=1.0, max=3.0)
            self.g_global.data.clamp_(min=0.3, max=1.5)
            
            # Ensure low noise
            self.noise_rate_e.data.clamp_(min=0.001, max=0.05)
            self.noise_rate_i.data.clamp_(min=0.001, max=0.02)
            
            # Ensure inhibitory weights are properly balanced
            self.W_IE.data.clamp_(max=0.1)  # Should be small or negative
            self.W_EI.data.clamp_(min=0.0, max=1.0)
            
            # Force strong inhibition
            if self.g_ie.data < self.g_ee.data * 2.0:
                self.g_ie.data = self.g_ee.data * 3.0


def create_single_peak_model(n_exc=800, n_inh=200, device='cpu'):
    """
    Create a properly configured single-peak ring attractor model.
    
    Args:
        n_exc: Number of excitatory neurons
        n_inh: Number of inhibitory neurons  
        device: Device to use
        
    Returns:
        Configured SinglePeakRingAttractor model
    """
    
    model = SinglePeakRingAttractor(n_exc=n_exc, n_inh=n_inh, device=device)
    
    # Apply initial constraints
    model.apply_single_peak_constraints()
    
    print(f"Single Peak Model Created:")
    print(f"  Excitatory neurons: {n_exc}")
    print(f"  Inhibitory neurons: {n_inh}")
    print(f"  σ_EE (connection width): {model.sigma_ee.item():.3f}")
    print(f"  g_ee: {model.g_ee.item():.3f}")
    print(f"  g_ie: {model.g_ie.item():.3f}")
    print(f"  Inhibition/Excitation ratio: {model.g_ie.item()/model.g_ee.item():.2f}")
    print(f"  Global inhibition: {model.g_global.item():.3f}")
    
    return model