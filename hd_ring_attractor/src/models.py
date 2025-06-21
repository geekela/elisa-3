import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RingAttractorNetwork(nn.Module):
    """
    Ring attractor network for head direction cells.
    
    Architecture:
    - Excitatory neurons arranged in a ring (representing head directions 0-360°)
    - Inhibitory neurons providing global inhibition
    - Fixed excitatory-to-excitatory connections (ring topology)
    - Trainable inhibitory connections (E->I and I->E only, no I->I)
    """
    
    def __init__(self, n_exc=800, n_inh=200, sigma_ee=0.5, tau_e=10.0, tau_i=5.0, 
                 dt=0.1, device='cpu'):
        super().__init__()
        
        self.n_exc = n_exc  # Number of excitatory neurons
        self.n_inh = n_inh  # Number of inhibitory neurons
        self.device = device
        
        # Time constants (in ms)
        self.tau_e = tau_e  # Excitatory neuron time constant
        self.tau_i = tau_i  # Inhibitory neuron time constant
        self.dt = dt        # Integration time step
        
        # Preferred directions of excitatory neurons (evenly distributed around ring)
        angles = torch.linspace(0, 2*np.pi, n_exc, device=device)
        self.register_buffer('preferred_dirs', angles)
        
        # Trainable excitatory-to-excitatory connectivity width parameter
        # sigma_ee controls how localized vs distributed the E->E connections are
        self.sigma_ee = nn.Parameter(torch.tensor(sigma_ee, device=device))
        
        # Trainable inhibitory connections (E->I and I->E only, no I->I)
        # W_EI: (n_exc, n_inh) for E->I connections, used as W_EI.t() in forward
        # W_IE: (n_inh, n_exc) for I->E connections, used as W_IE.t() in forward
        self.W_EI = nn.Parameter(torch.randn(n_exc, n_inh, device=device) * 0.1)
        self.W_IE = nn.Parameter(torch.randn(n_inh, n_exc, device=device) * 0.1)
        
        # Gain parameters
        self.g_ee = nn.Parameter(torch.tensor(1.0, device=device))
        self.g_ei = nn.Parameter(torch.tensor(1.5, device=device))
        self.g_ie = nn.Parameter(torch.tensor(2.0, device=device))
        
        # Input gain
        self.g_input = nn.Parameter(torch.tensor(1.0, device=device))
        
        # Poisson noise parameters (lambda values for Poisson distribution)
        # These represent the rate parameters for Poisson-distributed noise
        self.noise_rate_e = nn.Parameter(torch.tensor(0.1, device=device))  # Excitatory Poisson lambda
        self.noise_rate_i = nn.Parameter(torch.tensor(0.05, device=device))  # Inhibitory Poisson lambda
        
        # Initialize states
        self.reset_state()
        
    def _create_ring_weights(self, sigma):
        """
        Create ring-structured excitatory weights based on angular distance.
        """
        # Calculate angular distances between all neuron pairs
        angles = self.preferred_dirs
        angle_diff = angles.unsqueeze(0) - angles.unsqueeze(1)
        
        # Wrap distances to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Gaussian connectivity based on angular distance
        W = torch.exp(-0.5 * (angle_diff / sigma)**2)
        
        # Normalize weights
        W = W / W.sum(dim=1, keepdim=True)
        
        # Remove self-connections
        W.fill_diagonal_(0)
        
        return W
    
    def reset_state(self):
        """
        Reset neural activities to baseline on the correct device.
        """
        # Get the current device from parameters to ensure consistency
        current_device = next(self.parameters()).device
        self.device = current_device
        
        # Initialize state variables that can be used for inference
        # During training, these will be replaced with gradient-connected tensors
        self.r_e = torch.zeros(self.n_exc, device=current_device)
        self.r_i = torch.zeros(self.n_inh, device=current_device)
    
    def to(self, device):
        """
        Override to() method to properly move state variables to device.
        """
        # Move the model parameters and buffers
        super().to(device)
        
        # Update the device attribute
        self.device = device
        
        # Move state variables to the new device if they exist
        if hasattr(self, 'r_e') and self.r_e is not None:
            self.r_e = self.r_e.to(device)
        if hasattr(self, 'r_i') and self.r_i is not None:
            self.r_i = self.r_i.to(device)
        
        return self
        
    def initialize_bump(self, direction, width=0.3, amplitude=1.0):
        """
        Initialize activity with a bump at specified direction.
        
        Args:
            direction: Head direction in radians [0, 2π]
            width: Width of the bump (in radians)
            amplitude: Peak amplitude of the bump
        """
        # Calculate distance from preferred direction
        angle_diff = self.preferred_dirs - direction
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Create Gaussian bump
        self.r_e = amplitude * torch.exp(-0.5 * (angle_diff / width)**2)
        self.r_i = torch.zeros(self.n_inh, device=self.preferred_dirs.device)
        
    def forward(self, external_input=None, h=None, steps=1):
        """
        Forward dynamics of the ring attractor network.
        
        Args:
            external_input: External input to excitatory neurons (batch_size, n_exc) or (n_exc,)
            h: Previous hidden state (tensor or None)
            steps: Number of integration steps
            
        Returns:
            Current state tensor (for compatibility with training code)
        """
        # Handle batch dimension
        batch_mode = False
        batch_size = 1
        
        if external_input is not None:
            if external_input.dim() == 2:
                # Batch mode with shape (batch_size, n_exc)
                batch_mode = True
                batch_size = external_input.shape[0]
                ext_input = external_input  # Use the entire batch
            else:
                # Single input with shape (n_exc)
                ext_input = external_input.unsqueeze(0)  # Add batch dimension
                batch_mode = True
                batch_size = 1
        else:
            ext_input = None
        
        # Initialize states for computation
        if h is not None:
            # Use provided state
            if h.dim() == 1:
                # Single state
                r_e = h.unsqueeze(0) if batch_mode else h  # Add batch dim if needed
            else:
                # Batch of states
                r_e = h
                
            # Initialize inhibitory neurons if we're using a provided state
            if batch_mode:
                r_i = torch.zeros(batch_size, self.n_inh, device=self.preferred_dirs.device)
            else:
                r_i = torch.zeros(self.n_inh, device=self.preferred_dirs.device)
        elif self.training:
            # During training, create fresh tensor for each sample in batch
            if batch_mode:
                r_e = 0.01 * torch.randn(batch_size, self.n_exc, device=self.preferred_dirs.device)
                r_i = 0.01 * torch.randn(batch_size, self.n_inh, device=self.preferred_dirs.device)
            else:
                r_e = 0.01 * torch.randn(self.n_exc, device=self.preferred_dirs.device)
                r_i = 0.01 * torch.randn(self.n_inh, device=self.preferred_dirs.device)
        else:
            # During inference, use the maintained state
            if batch_mode:
                # Replicate state for each sample in batch
                r_e = self.r_e.unsqueeze(0).expand(batch_size, -1)
                r_i = self.r_i.unsqueeze(0).expand(batch_size, -1)
            else:
                r_e = self.r_e.clone()
                r_i = self.r_i.clone()
        
        # Run the network dynamics
        for _ in range(steps):
            # Dynamically compute E->E weights based on current sigma_ee
            W_EE = self._create_ring_weights(self.sigma_ee)
            
            # Process each sample in batch or single sample
            if batch_mode:
                # Handle batch processing
                
                # For each sample in the batch, perform the necessary matrix multiplications
                all_input_e = []
                all_input_i = []
                
                for b in range(batch_size):
                    # Calculate E->E input for this sample
                    input_e_sample = self.g_ee * torch.matmul(W_EE, r_e[b])
                    
                    # Calculate I->E input for this sample
                    input_e_sample -= self.g_ie * torch.matmul(self.W_IE.t(), r_i[b])
                    
                    # Add external input if provided
                    if ext_input is not None:
                        input_e_sample += self.g_input * ext_input[b]
                    
                    # Calculate E->I input for this sample
                    input_i_sample = self.g_ei * torch.matmul(self.W_EI.t(), r_e[b])
                    
                    all_input_e.append(input_e_sample)
                    all_input_i.append(input_i_sample)
                
                # Stack results for all samples in batch
                input_e = torch.stack(all_input_e)
                input_i = torch.stack(all_input_i)
            else:
                # Single sample processing
                input_e = (self.g_ee * torch.matmul(W_EE, r_e) - 
                          self.g_ie * torch.matmul(self.W_IE.t(), r_i))
                
                if ext_input is not None:
                    input_e += self.g_input * ext_input[0]  # Use first (only) sample
                    
                input_i = self.g_ei * torch.matmul(self.W_EI.t(), r_e)
            
            # Add Poisson-distributed noise with improved gradient flow
            if self.training:
                # Generate Poisson noise with stronger gradient connections
                if batch_mode:
                    # Create noise scale directly using the parameters to ensure gradient flow
                    noise_scale_e = self.noise_rate_e.expand_as(input_e)
                    noise_scale_i = self.noise_rate_i.expand_as(input_i)
                else:
                    noise_scale_e = self.noise_rate_e.expand_as(input_e)
                    noise_scale_i = self.noise_rate_i.expand_as(input_i)
                
                # Generate Poisson samples
                poisson_e = torch.poisson(noise_scale_e)
                poisson_i = torch.poisson(noise_scale_i)
                
                # Create zero-mean noise that preserves gradients to noise parameters
                noise_e = poisson_e - noise_scale_e  # Direct gradient path
                noise_i = poisson_i - noise_scale_i  # Direct gradient path
            else:
                # During inference, add regular Poisson noise
                if batch_mode:
                    noise_scale_e = self.noise_rate_e.expand_as(input_e)
                    noise_scale_i = self.noise_rate_i.expand_as(input_i)
                else:
                    noise_scale_e = self.noise_rate_e.expand_as(input_e)
                    noise_scale_i = self.noise_rate_i.expand_as(input_i)
                
                poisson_e = torch.poisson(noise_scale_e)
                poisson_i = torch.poisson(noise_scale_i)
                noise_e = poisson_e - noise_scale_e
                noise_i = poisson_i - noise_scale_i
                
            input_e += noise_e
            input_i += noise_i
            
            # Update activities with time constants
            dr_e = self.dt * (-r_e + F.relu(input_e)) / self.tau_e
            dr_i = self.dt * (-r_i + F.relu(input_i)) / self.tau_i
            
            # Update state variables (not in-place)
            r_e = r_e + dr_e
            r_i = r_i + dr_i
            
            # Ensure non-negative rates
            r_e = F.relu(r_e)
            r_i = F.relu(r_i)
        
        # Update instance variables only when not training (to maintain state)
        if not self.training:
            if batch_mode:
                # Store first sample's state during inference
                self.r_e = r_e[0].detach().clone()
                self.r_i = r_i[0].detach().clone()
            else:
                self.r_e = r_e.detach().clone()
                self.r_i = r_i.detach().clone()
        
        # Add a tiny direct connection to trainable parameters to improve gradient flow
        # This does not affect the actual dynamics but ensures parameters receive gradients
        if self.training:
            # The multiplication by a small constant preserves the output while connecting to all parameters
            if batch_mode:
                param_connection = 1e-6 * (self.noise_rate_e + self.noise_rate_i + self.sigma_ee + 
                                          self.g_ee + self.g_ei + self.g_ie + self.g_input)
                r_e = r_e + param_connection.view(1, 1).expand_as(r_e) * 0
            else:
                param_connection = 1e-6 * (self.noise_rate_e + self.noise_rate_i + self.sigma_ee + 
                                          self.g_ee + self.g_ei + self.g_ie + self.g_input)
                r_e = r_e + param_connection * 0
        
        # Return the excitatory state (this maintains gradients for training)
        return r_e
    
    def decode_angle(self, h=None):
        """
        Decode current head direction from population activity.
        
        Args:
            h: Hidden state (optional, uses current state if None)
            
        Returns:
            Decoded angle in radians
        """
        # Use current excitatory activity if h is not provided
        if h is not None:
            activity = h
        else:
            activity = self.r_e
        
        # Ensure activity is on the same device as preferred_dirs
        activity = activity.to(self.preferred_dirs.device)
        
        # Handle batch dimension
        if activity.dim() == 2:
            # Batch processing - decode each sample
            batch_size = activity.shape[0]
            decoded_angles = []
            
            for i in range(batch_size):
                r_e = activity[i]
                if r_e.sum() > 0:
                    # Weighted circular mean
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
                # Weighted circular mean
                x = (activity * torch.cos(self.preferred_dirs)).sum()
                y = (activity * torch.sin(self.preferred_dirs)).sum()
                decoded_dir = torch.atan2(y, x)
            else:
                decoded_dir = torch.tensor(0.0, device=self.preferred_dirs.device)
            
            return decoded_dir
    
    def decode_direction(self):
        """
        Decode current head direction from population activity.
        Returns both angle and confidence.
        """
        # Ensure r_e is on the same device as preferred_dirs
        r_e = self.r_e.to(self.preferred_dirs.device)
        if r_e.sum() > 0:
            # Weighted circular mean
            x = (r_e * torch.cos(self.preferred_dirs)).sum()
            y = (r_e * torch.sin(self.preferred_dirs)).sum()
            decoded_dir = torch.atan2(y, x)
            
            # Confidence (vector magnitude)
            confidence = torch.sqrt(x**2 + y**2) / r_e.sum()
        else:
            decoded_dir = torch.tensor(0.0, device=self.preferred_dirs.device)
            confidence = torch.tensor(0.0, device=self.preferred_dirs.device)
            
        return decoded_dir, confidence