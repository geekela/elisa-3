import torch
import numpy as np


def angle_to_input(angle, n_exc=800, input_strength=1.0, input_width=0.5, device=None):
    """
    Convert head direction angle(s) to ring attractor input.
    
    Args:
        angle: Head direction in radians (can be tensor or single value)
        n_exc: Number of excitatory neurons in the ring
        input_strength: Strength of the input signal
        input_width: Width of the input tuning curve (in radians)
    
    Returns:
        Input tensor of shape (..., n_exc) matching the ring structure
    """
    # Ensure angle is a tensor
    if not isinstance(angle, torch.Tensor):
        target_device = device if device is not None else 'cpu'
        angle = torch.tensor(angle, device=target_device)
    
    # Create preferred directions for excitatory neurons
    target_device = device if device is not None else (angle.device if isinstance(angle, torch.Tensor) else 'cpu')
    preferred_dirs = torch.linspace(0, 2*np.pi, n_exc, device=target_device)
    
    # Handle both single angles and sequences
    if angle.dim() == 0:
        # Single angle
        angle_diff = preferred_dirs - angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        input_pattern = input_strength * torch.exp(-0.5 * (angle_diff / input_width)**2)
        return input_pattern
    else:
        # Multiple angles (sequence or batch)
        original_shape = angle.shape
        angle_flat = angle.view(-1)
        
        # Calculate input for each angle
        inputs = []
        for a in angle_flat:
            angle_diff = preferred_dirs - a
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            input_pattern = input_strength * torch.exp(-0.5 * (angle_diff / input_width)**2)
            inputs.append(input_pattern)
        
        # Stack and reshape to match original shape + n_exc dimension
        inputs = torch.stack(inputs)
        return inputs.view(*original_shape, n_exc)


def generate_trajectory(n_steps, dt=0.1, angular_velocity_std=0.1, device='cpu'):
    """
    Generate a random trajectory of head directions.
    
    Args:
        n_steps: Number of time steps
        dt: Time step size
        angular_velocity_std: Standard deviation of angular velocity
    
    Returns:
        angles: Head directions over time (radians)
        angular_velocities: Angular velocities
    """
    angles = torch.zeros(n_steps, device=device)
    angular_velocities = torch.randn(n_steps, device=device) * angular_velocity_std
    
    for t in range(1, n_steps):
        angles[t] = angles[t-1] + angular_velocities[t] * dt
    
    # Keep angles in [0, 2π] range
    angles = torch.remainder(angles, 2 * np.pi)
    return angles, angular_velocities


def circular_distance(angle1, angle2):
    """
    Calculate circular distance between two angles.
    """
    diff = angle1 - angle2
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def compute_error(predicted_angle, true_angle):
    """
    Compute circular error between predicted and true angles.
    """
    error = circular_distance(predicted_angle, true_angle)
    return torch.abs(error)