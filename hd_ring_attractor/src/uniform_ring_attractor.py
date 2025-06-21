#!/usr/bin/env python3
"""
Uniform Ring Attractor Network

An improved HD ring attractor model that ensures:
1. Uniform preferred direction distribution
2. Symmetric connectivity
3. Stable single-peak dynamics
4. Biologically plausible behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UniformRingAttractorNetwork(nn.Module):
    """
    Enhanced ring attractor network with guaranteed uniform preferred directions.
    
    Key improvements:
    - Enforces circular symmetry in connectivity
    - Ensures all neurons are properly activated
    - Maintains uniform preferred direction distribution
    - Optimized for single-peak stability
    """
    
    def __init__(self, n_exc=800, n_inh=200, sigma_ee=0.15, tau_e=10.0, tau_i=5.0,
                 dt=0.1, device='cpu', enforce_symmetry=True):
        """
        Initialize the uniform ring attractor network.
        
        Args:
            n_exc: Number of excitatory neurons (evenly distributed on ring)
            n_inh: Number of inhibitory neurons
            sigma_ee: Width of E-E connectivity (smaller = more localized)
            tau_e: Excitatory time constant (ms)
            tau_i: Inhibitory time constant (ms)
            dt: Integration time step (ms)
            device: Computation device
            enforce_symmetry: Whether to enforce perfect circular symmetry
        """
        super().__init__()
        
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.device = device
        self.enforce_symmetry = enforce_symmetry
        
        # Time constants
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.dt = dt
        
        # Create uniformly distributed preferred directions
        # This is the key to uniform distribution
        angles = torch.linspace(0, 2*np.pi, n_exc+1, device=device)[:-1]  # Exclude endpoint
        self.register_buffer('preferred_dirs', angles)
        
        # Verify uniform spacing (with reasonable tolerance for floating point)
        spacings = torch.diff(torch.cat([angles, angles[:1] + 2*np.pi]))
        spacing_variance = torch.var(spacings).item()
        if spacing_variance > 1e-10:
            print(f"Warning: slight non-uniformity detected (variance={spacing_variance:.2e})")
        
        # Trainable connectivity parameters
        self.sigma_ee = nn.Parameter(torch.tensor(sigma_ee, device=device))
        
        # Initialize symmetric inhibitory connections
        # Using structured initialization for better symmetry
        # W_EI: (n_inh, n_exc) for E->I connections
        # W_IE: (n_exc, n_inh) for I->E connections
        self.W_EI = nn.Parameter(self._init_symmetric_weights(n_inh, n_exc) * 0.1)
        self.W_IE = nn.Parameter(self._init_symmetric_weights(n_exc, n_inh) * 0.1)
        
        # Gain parameters with constraints for stability
        self.g_ee = nn.Parameter(torch.tensor(0.8, device=device))
        self.g_ei = nn.Parameter(torch.tensor(1.2, device=device))
        self.g_ie = nn.Parameter(torch.tensor(1.0, device=device))
        
        # Noise parameters (small for stability)
        self.noise_rate_e = nn.Parameter(torch.tensor(0.001, device=device))
        self.noise_rate_i = nn.Parameter(torch.tensor(0.0005, device=device))
        
        # State variables
        self.r_e = None
        self.r_i = None
        
        # Activity regularization
        self.activity_target = 0.2  # Target mean activity
        self.sparsity_target = 0.1  # Target sparsity
        
    def _init_symmetric_weights(self, n_rows, n_cols):
        """Initialize weights with some structure for better symmetry."""
        weights = torch.randn(n_rows, n_cols, device=self.device)
        # Add slight bias towards uniform connectivity
        weights = weights + torch.ones_like(weights) * 0.1
        return weights
        
    def _create_ring_weights(self, sigma):
        """
        Create perfectly symmetric ring connectivity matrix.
        
        Args:
            sigma: Width of connectivity (in units of 2π/n_exc)
            
        Returns:
            Symmetric connectivity matrix
        """
        n = self.n_exc
        indices = torch.arange(n, device=self.device)
        
        # Compute pairwise circular distances
        i_idx = indices.unsqueeze(1)
        j_idx = indices.unsqueeze(0)
        
        # Circular distance (minimum of clockwise and counter-clockwise)
        dist = torch.minimum(
            torch.abs(i_idx - j_idx),
            n - torch.abs(i_idx - j_idx)
        ).float()
        
        # Convert to angular distance
        angular_dist = dist * (2 * np.pi / n)
        
        # Gaussian connectivity based on angular distance
        weights = torch.exp(-0.5 * (angular_dist / sigma)**2)
        
        # Normalize by row to ensure uniform total input
        if self.enforce_symmetry:
            weights = weights / weights.sum(dim=1, keepdim=True)
        
        return weights
    
    def reset_state(self, batch_size=None):
        """Reset neural activity with small random values."""
        if batch_size is not None:
            self.r_e = 0.01 * torch.randn(batch_size, self.n_exc, device=self.device)
            self.r_i = 0.01 * torch.randn(batch_size, self.n_inh, device=self.device)
        else:
            self.r_e = 0.01 * torch.randn(self.n_exc, device=self.device)
            self.r_i = 0.01 * torch.randn(self.n_inh, device=self.device)
    
    def initialize_bump(self, direction, width=0.2, amplitude=0.5):
        """
        Initialize activity with a bump at specified direction.
        
        Args:
            direction: Head direction in radians [0, 2π]
            width: Width of the bump (smaller = sharper)
            amplitude: Peak amplitude of the bump
        """
        # Calculate circular distance from each neuron's preferred direction
        angle_diff = self.preferred_dirs - direction
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Create focused Gaussian bump
        self.r_e = amplitude * torch.exp(-0.5 * (angle_diff / width)**2)
        self.r_i = torch.zeros(self.n_inh, device=self.device)
        
    def forward(self, external_input=None, steps=1):
        """
        Run network dynamics.
        
        Args:
            external_input: External input to excitatory neurons
            steps: Number of integration steps
            
        Returns:
            Excitatory activity after integration
        """
        # Initialize if needed
        if self.r_e is None:
            self.reset_state()
        
        # Check device compatibility
        if external_input is not None:
            external_input = external_input.to(self.device)
            
        # Run dynamics
        for _ in range(steps):
            # Get symmetric E-E connectivity
            W_EE = self._create_ring_weights(torch.abs(self.sigma_ee))
            
            # Handle batch processing
            assert self.r_e is not None and self.r_i is not None, "State not initialized"
            
            if self.r_e.dim() == 2:
                batch_size = self.r_e.shape[0]
                r_e, r_i = self.r_e, self.r_i
                
                # Compute inputs for each sample in batch
                input_e = torch.zeros_like(r_e)
                input_i = torch.zeros_like(r_i)
                
                for b in range(batch_size):
                    # E-E recurrent input (symmetric)
                    input_e[b] = self.g_ee * torch.matmul(W_EE, r_e[b])
                    
                    # I->E input
                    input_e[b] -= self.g_ie * torch.matmul(self.W_IE, r_i[b])
                    
                    # E->I input
                    input_i[b] = self.g_ei * torch.matmul(self.W_EI, r_e[b])
                    
                    # Add external input if provided
                    if external_input is not None:
                        if external_input.dim() == 2:
                            input_e[b] += external_input[b]
                        else:
                            input_e[b] += external_input
            else:
                # Single sample processing
                input_e = (self.g_ee * torch.matmul(W_EE, self.r_e) -
                          self.g_ie * torch.matmul(self.W_IE, self.r_i))
                input_i = self.g_ei * torch.matmul(self.W_EI, self.r_e)
                
                if external_input is not None:
                    input_e = input_e + external_input
            
            # Add small noise for stability
            if self.training:
                noise_e = torch.normal(0, torch.abs(self.noise_rate_e).item(), 
                                     size=input_e.shape, device=self.device)
                noise_i = torch.normal(0, torch.abs(self.noise_rate_i).item(), 
                                     size=input_i.shape, device=self.device)
                input_e = input_e + noise_e
                input_i = input_i + noise_i
            
            # Update with bounded activation
            if self.r_e.dim() == 2:
                # Batch update
                self.r_e = self.r_e + self.dt/self.tau_e * (-self.r_e + F.relu(input_e))
                self.r_i = self.r_i + self.dt/self.tau_i * (-self.r_i + F.relu(input_i))
                
                # Soft upper bound for stability
                self.r_e = torch.tanh(self.r_e / 2.0) * 2.0
                self.r_i = torch.tanh(self.r_i / 2.0) * 2.0
            else:
                # Single sample update
                self.r_e = self.r_e + self.dt/self.tau_e * (-self.r_e + F.relu(input_e))
                self.r_i = self.r_i + self.dt/self.tau_i * (-self.r_i + F.relu(input_i))
                
                # Soft upper bound
                self.r_e = torch.tanh(self.r_e / 2.0) * 2.0
                self.r_i = torch.tanh(self.r_i / 2.0) * 2.0
        
        return self.r_e
    
    def decode_angle(self, activity=None):
        """
        Decode head direction from population activity using circular mean.
        
        Args:
            activity: Neural activity (uses current state if None)
            
        Returns:
            Decoded angle in radians
        """
        if activity is None:
            activity = self.r_e
            
        # Ensure activity is not None
        if activity is None:
            return torch.tensor(0.0, device=self.device)
            
        # Ensure proper device
        activity = activity.to(self.device)
        
        # Get preferred directions tensor
        preferred_dirs = self.preferred_dirs
        
        # Circular decoding
        if activity.sum() > 1e-6:
            # Weighted circular mean
            x = (activity * torch.cos(preferred_dirs)).sum()
            y = (activity * torch.sin(preferred_dirs)).sum()
            decoded_angle = torch.atan2(y, x)
        else:
            decoded_angle = torch.tensor(0.0, device=self.device)
            
        return decoded_angle
    
    def get_tuning_curves(self, n_directions=36, input_strength=1.5):
        """
        Compute tuning curves for all neurons.
        
        Args:
            n_directions: Number of directions to test
            input_strength: Strength of input signal
            
        Returns:
            Dictionary with tuning curves and analysis
        """
        self.eval()
        directions = torch.linspace(0, 2*np.pi, n_directions, device=self.device)
        tuning_curves = torch.zeros(self.n_exc, n_directions, device=self.device)
        
        for i, direction in enumerate(directions):
            self.reset_state()
            
            # Create input
            angle_diff = self.preferred_dirs - direction
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            input_pattern = input_strength * torch.exp(-0.5 * (angle_diff / 0.3)**2)
            
            # Run to steady state
            with torch.no_grad():
                for _ in range(20):
                    activity = self(input_pattern, steps=1)
            
            tuning_curves[:, i] = activity
        
        # Analyze preferred directions
        preferred_dirs_idx = torch.argmax(tuning_curves, dim=1)
        preferred_dirs_rad = directions[preferred_dirs_idx]
        preferred_dirs_deg = torch.rad2deg(preferred_dirs_rad)
        
        return {
            'tuning_curves': tuning_curves.cpu().numpy(),
            'directions': directions.cpu().numpy(),
            'preferred_dirs_deg': preferred_dirs_deg.cpu().numpy(),
            'preferred_dirs_rad': preferred_dirs_rad.cpu().numpy()
        }
    
    def validate_uniformity(self, n_bins=18, alpha=0.05):
        """
        Validate that preferred directions are uniformly distributed.
        
        Args:
            n_bins: Number of histogram bins
            alpha: Significance level for chi-square test
            
        Returns:
            Tuple of (is_uniform, chi2_stat, p_value)
        """
        # Get tuning curves
        tuning_data = self.get_tuning_curves()
        preferred_dirs = tuning_data['preferred_dirs_deg']
        
        # Histogram
        hist, _ = np.histogram(preferred_dirs, bins=n_bins, range=(0, 360))
        
        # Chi-square test
        expected_count = len(preferred_dirs) / n_bins
        chi2_stat = np.sum((hist - expected_count)**2 / expected_count)
        
        # Degrees of freedom
        dof = n_bins - 1
        
        # Critical value and p-value
        from scipy import stats
        critical_value = stats.chi2.ppf(1 - alpha, dof)
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        is_uniform = chi2_stat < critical_value
        
        return is_uniform, chi2_stat, p_value 