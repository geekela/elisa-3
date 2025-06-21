"""
Enhanced Single Peak Ring Attractor Model - Full Biological Implementation

This model implements the complete solution to the multiple peaks problem based on:
1. Reference documentation analysis
2. Biological constraints from HD cell physiology  
3. Mathematical formulation from neural dynamics theory
4. Comprehensive parameter optimization

Key innovations:
- Biologically constrained architecture
- Multiple winner-take-all mechanisms
- Adaptive parameter regulation
- Comprehensive validation framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class EnhancedSinglePeakRingAttractor(nn.Module):
    """
    Enhanced ring attractor with comprehensive single-peak mechanisms.
    
    Based on biological constraints and theoretical analysis:
    - HD cells are excitatory only (no inhibitory HD cells)
    - Sharp directional tuning (~90° width)
    - Persistent activity during stationary periods
    - Winner-take-all dynamics prevent multiple peaks
    - Biologically plausible parameters
    """
    
    def __init__(self, 
                 n_exc: int = 800, 
                 n_inh: int = 200, 
                 device: str = 'cpu',
                 enforce_biological_constraints: bool = True,
                 **kwargs):
        super().__init__()
        
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.device = device
        self.enforce_biological = enforce_biological_constraints
        
        # Biological time constants (ms converted to simulation units)
        self.tau_e = nn.Parameter(torch.tensor(15.0, device=device))  # Slower excitatory
        self.tau_i = nn.Parameter(torch.tensor(8.0, device=device))   # Faster inhibitory
        self.dt = 0.1  # Simulation time step
        
        # Preferred directions for ring topology
        angles = torch.linspace(0, 2*np.pi, n_exc, device=device)
        self.register_buffer('preferred_dirs', angles)
        
        # CRITICAL PARAMETERS FOR SINGLE PEAKS
        # (Based on sensitivity analysis and biological constraints)
        
        # Connection width - MUST be narrow for single peaks
        self.sigma_ee = nn.Parameter(torch.tensor(0.2, device=device))  # Very focused
        
        # Gain parameters - Critical I/E balance
        self.g_ee = nn.Parameter(torch.tensor(0.5, device=device))   # Moderate excitation
        self.g_ei = nn.Parameter(torch.tensor(2.5, device=device))   # Strong E->I drive
        self.g_ie = nn.Parameter(torch.tensor(5.0, device=device))   # Very strong I->E (10:1 ratio)
        
        # Global inhibition - Winner-take-all mechanism
        self.g_global = nn.Parameter(torch.tensor(1.0, device=device))
        
        # Local competition strength
        self.g_local_competition = nn.Parameter(torch.tensor(0.5, device=device))
        
        # Input processing
        self.g_input = nn.Parameter(torch.tensor(1.5, device=device))
        
        # Noise parameters - MUST be low for stability
        self.noise_rate_e = nn.Parameter(torch.tensor(0.005, device=device))
        self.noise_rate_i = nn.Parameter(torch.tensor(0.002, device=device))
        
        # Adaptive thresholds
        self.adaptive_threshold = nn.Parameter(torch.tensor(0.05, device=device))
        self.peak_suppression_strength = nn.Parameter(torch.tensor(0.3, device=device))
        
        # Inhibitory connection matrices
        # E->I: Broad connectivity (all excitatory cells drive inhibition)
        self.W_EI = nn.Parameter(
            torch.normal(0.4, 0.1, size=(n_exc, n_inh), device=device).clamp(0.0, 1.0)
        )
        
        # I->E: Uniform strong inhibition (global inhibitory pool)
        self.W_IE = nn.Parameter(
            torch.full((n_inh, n_exc), -0.4, device=device)  # Negative weights
        )
        
        # Activity history for temporal dynamics
        self.activity_history = None
        self.history_length = 5
        self.history_idx = 0
        
        # Validation metrics
        self.last_peak_count = 0
        self.activity_statistics = {}
        
        self.reset_state()
        
        # Apply biological constraints
        if self.enforce_biological:
            self.apply_biological_constraints()
    
    def apply_biological_constraints(self):
        """Apply biological constraints to ensure realistic parameters."""
        with torch.no_grad():
            # 1. CRITICAL: Inhibition must dominate excitation
            # Based on reference: I/E ratio should be 6-10:1
            min_ie_ratio = 6.0
            if self.g_ie.data < self.g_ee.data * min_ie_ratio:
                self.g_ie.data = self.g_ee.data * min_ie_ratio
            
            # Clamp to biological ranges
            self.g_ee.data.clamp_(min=0.2, max=1.0)
            self.g_ie.data.clamp_(min=3.0, max=10.0)
            self.g_ei.data.clamp_(min=1.5, max=4.0)
            
            # 2. CRITICAL: Connection width must be narrow
            # Based on HD cell tuning width (~90° = π/2 radians)
            self.sigma_ee.data.clamp_(min=0.1, max=0.35)
            
            # 3. CRITICAL: Noise must be minimal
            # High noise fragments coherent bumps
            self.noise_rate_e.data.clamp_(min=0.001, max=0.02)
            self.noise_rate_i.data.clamp_(min=0.001, max=0.01)
            
            # 4. Global inhibition for winner-take-all
            self.g_global.data.clamp_(min=0.5, max=2.0)
            
            # 5. Time constants (biological realism)
            self.tau_e.data.clamp_(min=10.0, max=25.0)  # 10-25 ms
            self.tau_i.data.clamp_(min=5.0, max=15.0)   # 5-15 ms
            
            # 6. Inhibitory weights
            self.W_IE.data.clamp_(max=-0.1)  # Must be negative
            self.W_EI.data.clamp_(min=0.0, max=1.0)  # Must be positive
            
            # 7. Input strength
            self.g_input.data.clamp_(min=0.5, max=3.0)
    
    def _create_optimized_ring_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """Create highly optimized ring connectivity for single peaks."""
        angles = self.preferred_dirs
        
        # Angular differences with proper wraparound
        angle_diff = angles.unsqueeze(0) - angles.unsqueeze(1)
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Very focused Gaussian connectivity
        W = torch.exp(-0.5 * (angle_diff / sigma)**2)
        
        # Critical normalizations for stability
        # 1. Remove self-connections (prevent runaway activity)
        W.fill_diagonal_(0)
        
        # 2. Normalize to prevent explosive dynamics
        row_sums = W.sum(dim=1, keepdim=True) + 1e-8
        W = W / row_sums
        
        # 3. Scale down to ensure inhibition can dominate
        W = W * 0.7  # Prevent excitatory dominance
        
        # 4. Add distance-dependent decay
        distance_weight = torch.exp(-0.1 * torch.abs(angle_diff))
        W = W * distance_weight
        
        return W
    
    def _apply_multiple_winner_take_all(self, activity: torch.Tensor) -> torch.Tensor:
        """Apply multiple winner-take-all mechanisms to ensure single peaks."""
        
        if activity.dim() == 1:
            # Single sample processing
            return self._single_sample_wta(activity)
        else:
            # Batch processing
            batch_size = activity.shape[0]
            result = torch.zeros_like(activity)
            
            for b in range(batch_size):
                result[b] = self._single_sample_wta(activity[b])
            
            return result
    
    def _single_sample_wta(self, activity: torch.Tensor) -> torch.Tensor:
        """Apply winner-take-all to single activity vector."""
        
        if torch.max(activity) < 1e-6:
            return activity  # No significant activity
        
        # 1. Global winner-take-all: suppress weak activity
        max_activity = torch.max(activity)
        threshold = max_activity * 0.25  # Keep only strong activity (25% of max)
        
        # Strong suppression of sub-threshold activity
        suppressed = torch.where(activity > threshold, 
                                activity, 
                                activity * 0.05)  # 95% suppression
        
        # 2. Local competition: each neuron competes with neighbors
        if len(suppressed) > 5:
            for i in range(len(suppressed)):
                # Check local neighborhood
                neighbors = []
                for offset in [-3, -2, -1, 1, 2, 3]:
                    neighbor_idx = (i + offset) % len(suppressed)
                    neighbors.append(suppressed[neighbor_idx])
                
                # Local competition rule
                max_neighbor = max(neighbors) if neighbors else 0
                if suppressed[i] < max_neighbor * 0.7:  # Must be within 70% of strongest neighbor
                    suppressed[i] *= 0.3  # Strong suppression
        
        # 3. Adaptive suppression based on activity distribution
        # If activity is too spread out, increase suppression
        activity_std = torch.std(suppressed)
        activity_mean = torch.mean(suppressed)
        
        if activity_std > activity_mean * 0.5:  # Too much spread
            # Find peak and suppress everything else more strongly
            peak_idx = torch.argmax(suppressed)
            for i in range(len(suppressed)):
                if i != peak_idx:
                    distance_from_peak = min(abs(i - peak_idx), 
                                           len(suppressed) - abs(i - peak_idx))
                    if distance_from_peak > 3:  # Far from peak
                        suppressed[i] *= 0.1  # Very strong suppression
        
        # 4. Final normalization to prevent drift
        total_activity = torch.sum(suppressed)
        if total_activity > 0:
            # Maintain reasonable total activity level
            target_total = 0.8
            if total_activity > target_total * 2:
                suppressed = suppressed * (target_total / total_activity)
        
        return suppressed
    
    def forward(self, 
                external_input: Optional[torch.Tensor] = None, 
                h: Optional[torch.Tensor] = None, 
                steps: int = 1) -> torch.Tensor:
        """
        Forward pass with comprehensive single-peak mechanisms.
        
        Args:
            external_input: External input pattern [batch_size, n_exc] or [n_exc]
            h: Hidden state [batch_size, n_exc] or [n_exc]
            steps: Number of integration steps
            
        Returns:
            Updated excitatory activity
        """
        
        # Batch handling
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
        elif self.training:
            # Training mode: small random initialization
            if batch_mode:
                r_e = 0.01 * torch.randn(batch_size, self.n_exc, device=self.preferred_dirs.device)
                r_i = 0.01 * torch.randn(batch_size, self.n_inh, device=self.preferred_dirs.device)
            else:
                r_e = 0.01 * torch.randn(self.n_exc, device=self.preferred_dirs.device)
                r_i = 0.01 * torch.randn(self.n_inh, device=self.preferred_dirs.device)
        else:
            # Inference mode: use stored state
            if batch_mode:
                r_e = self.r_e.unsqueeze(0).expand(batch_size, -1)
                r_i = self.r_i.unsqueeze(0).expand(batch_size, -1)
            else:
                r_e = self.r_e.clone()
                r_i = self.r_i.clone()
        
        # Initialize inhibitory state if not done
        if 'r_i' not in locals():
            if batch_mode:
                r_i = torch.zeros(batch_size, self.n_inh, device=self.preferred_dirs.device)
            else:
                r_i = torch.zeros(self.n_inh, device=self.preferred_dirs.device)
        
        # Apply biological constraints during forward pass
        if self.enforce_biological:
            self.apply_biological_constraints()
        
        # Main dynamics loop
        for step in range(steps):
            # Create optimized ring weights
            W_EE = self._create_optimized_ring_weights(self.sigma_ee)
            
            if batch_mode:
                all_input_e = []
                all_input_i = []
                
                for b in range(batch_size):
                    input_e, input_i = self._compute_single_step_dynamics(
                        r_e[b], r_i[b], W_EE, 
                        ext_input[b] if ext_input is not None else None
                    )
                    all_input_e.append(input_e)
                    all_input_i.append(input_i)
                
                input_e = torch.stack(all_input_e)
                input_i = torch.stack(all_input_i)
            else:
                input_e, input_i = self._compute_single_step_dynamics(
                    r_e, r_i, W_EE,
                    ext_input[0] if ext_input is not None else None
                )
            
            # Add noise (minimal for stability)
            if self.training or torch.rand(1) < 0.1:  # Occasional noise in inference
                if batch_mode:
                    noise_e = torch.normal(0, self.noise_rate_e.item(), 
                                         size=input_e.shape, device=self.preferred_dirs.device)
                    noise_i = torch.normal(0, self.noise_rate_i.item(), 
                                         size=input_i.shape, device=self.preferred_dirs.device)
                else:
                    noise_e = torch.normal(0, self.noise_rate_e.item(), 
                                         size=input_e.shape, device=self.preferred_dirs.device)
                    noise_i = torch.normal(0, self.noise_rate_i.item(), 
                                         size=input_i.shape, device=self.preferred_dirs.device)
                
                input_e += noise_e
                input_i += noise_i
            
            # Update dynamics with proper time constants
            dr_e = self.dt * (-r_e + F.relu(input_e)) / self.tau_e
            dr_i = self.dt * (-r_i + F.relu(input_i)) / self.tau_i
            
            r_e = r_e + dr_e
            r_i = r_i + dr_i
            
            # Apply activation functions
            r_e = F.relu(r_e)
            r_i = F.relu(r_i)
            
            # CRITICAL: Apply winner-take-all mechanisms
            r_e = self._apply_multiple_winner_take_all(r_e)
        
        # Update instance state (for single sample mode)
        if not self.training and not batch_mode:
            self.r_e = r_e.detach().clone()
            self.r_i = r_i.detach().clone()
            
            # Update activity history
            if self.activity_history is None:
                self.activity_history = torch.zeros(self.history_length, self.n_exc, 
                                                   device=self.preferred_dirs.device)
            
            self.activity_history[self.history_idx] = self.r_e.clone()
            self.history_idx = (self.history_idx + 1) % self.history_length
            
            # Update statistics
            self.last_peak_count = self.get_peak_count()
            self._update_activity_statistics()
        
        return r_e
    
    def _compute_single_step_dynamics(self, 
                                    r_e: torch.Tensor, 
                                    r_i: torch.Tensor, 
                                    W_EE: torch.Tensor,
                                    external_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute single step dynamics for excitatory and inhibitory populations."""
        
        # Excitatory dynamics
        ee_input = self.g_ee * torch.matmul(W_EE, r_e)
        ie_input = self.g_ie * torch.matmul(self.W_IE.t(), r_i)
        
        # Global inhibition (critical for single peaks)
        global_activity = torch.sum(r_e)
        global_inhibition = self.g_global * global_activity
        
        # Local competition (additional mechanism)
        local_competition = self.g_local_competition * torch.mean(r_e)
        
        # Total excitatory input
        input_e = ee_input + ie_input - global_inhibition - local_competition
        
        # Add external input if provided
        if external_input is not None:
            input_e += self.g_input * external_input
        
        # Inhibitory dynamics (driven by excitatory activity)
        input_i = self.g_ei * torch.matmul(self.W_EI.t(), r_e)
        
        return input_e, input_i
    
    def reset_state(self):
        """Reset network to baseline state."""
        current_device = next(self.parameters()).device
        self.device = current_device
        
        self.r_e = torch.zeros(self.n_exc, device=current_device)
        self.r_i = torch.zeros(self.n_inh, device=current_device)
        
        # Reset activity history
        self.activity_history = None
        self.history_idx = 0
        
        # Reset statistics
        self.last_peak_count = 0
        self.activity_statistics = {}
    
    def initialize_bump(self, direction: float, width: float = 0.15, amplitude: float = 0.6):
        """Initialize a focused activity bump at specified direction."""
        angle_diff = self.preferred_dirs - direction
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # Create sharp, focused bump
        self.r_e = amplitude * torch.exp(-0.5 * (angle_diff / width)**2)
        self.r_i = torch.zeros(self.n_inh, device=self.preferred_dirs.device)
        
        # Initialize history with current state
        if self.activity_history is None:
            self.activity_history = torch.zeros(self.history_length, self.n_exc, 
                                               device=self.preferred_dirs.device)
        
        for i in range(self.history_length):
            self.activity_history[i] = self.r_e.clone()
    
    def get_peak_count(self, activity: Optional[torch.Tensor] = None) -> int:
        """Count number of activity peaks with enhanced detection."""
        if activity is None:
            activity = self.r_e
        
        activity_np = activity.cpu().numpy()
        
        if np.max(activity_np) < 0.001:
            return 0
        
        # Enhanced peak detection
        threshold = 0.05 * np.max(activity_np)  # Lower threshold for better detection
        above_threshold = activity_np > threshold
        
        # Count connected components (peaks)
        peaks = 0
        in_peak = False
        min_peak_width = 2  # Minimum width for a valid peak
        current_peak_width = 0
        
        for i in range(len(activity_np)):
            if above_threshold[i]:
                if not in_peak:
                    in_peak = True
                    current_peak_width = 1
                else:
                    current_peak_width += 1
            else:
                if in_peak:
                    # End of peak - check if it was wide enough
                    if current_peak_width >= min_peak_width:
                        peaks += 1
                    in_peak = False
                    current_peak_width = 0
        
        # Check if we ended in a peak
        if in_peak and current_peak_width >= min_peak_width:
            peaks += 1
        
        return peaks
    
    def decode_angle(self, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode head direction from activity pattern."""
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
                if r_e.sum() > 1e-6:
                    x = (r_e * torch.cos(self.preferred_dirs)).sum()
                    y = (r_e * torch.sin(self.preferred_dirs)).sum()
                    decoded_dir = torch.atan2(y, x)
                else:
                    decoded_dir = torch.tensor(0.0, device=self.preferred_dirs.device)
                decoded_angles.append(decoded_dir)
            
            return torch.stack(decoded_angles)
        else:
            # Single sample
            if activity.sum() > 1e-6:
                x = (activity * torch.cos(self.preferred_dirs)).sum()
                y = (activity * torch.sin(self.preferred_dirs)).sum()
                decoded_dir = torch.atan2(y, x)
            else:
                decoded_dir = torch.tensor(0.0, device=self.preferred_dirs.device)
            
            return decoded_dir
    
    def _update_activity_statistics(self):
        """Update activity statistics for monitoring."""
        if hasattr(self, 'r_e'):
            self.activity_statistics = {
                'max_activity': torch.max(self.r_e).item(),
                'total_activity': torch.sum(self.r_e).item(),
                'activity_std': torch.std(self.r_e).item(),
                'peak_count': self.last_peak_count,
                'inhibition_ratio': self.g_ie.item() / self.g_ee.item(),
                'connection_width': self.sigma_ee.item(),
                'noise_level': self.noise_rate_e.item()
            }
    
    def get_model_health_report(self) -> Dict[str, Any]:
        """Get comprehensive model health report."""
        self._update_activity_statistics()
        
        health_report = {
            'parameters': {
                'g_ee': self.g_ee.item(),
                'g_ie': self.g_ie.item(),
                'g_ei': self.g_ei.item(),
                'sigma_ee': self.sigma_ee.item(),
                'g_global': self.g_global.item(),
                'noise_rate_e': self.noise_rate_e.item(),
                'inhibition_excitation_ratio': self.g_ie.item() / self.g_ee.item()
            },
            'activity_stats': self.activity_statistics,
            'biological_constraints': {
                'narrow_connections': self.sigma_ee.item() <= 0.35,
                'strong_inhibition': (self.g_ie.item() / self.g_ee.item()) >= 6.0,
                'low_noise': self.noise_rate_e.item() <= 0.02,
                'global_inhibition': self.g_global.item() >= 0.5
            },
            'single_peak_indicators': {
                'current_peak_count': self.last_peak_count,
                'single_peak_achieved': self.last_peak_count <= 1,
                'activity_concentration': self.activity_statistics.get('max_activity', 0) / 
                                        (self.activity_statistics.get('total_activity', 1e-8) + 1e-8)
            }
        }
        
        # Overall health assessment
        constraints_met = sum(health_report['biological_constraints'].values())
        health_report['overall_health'] = {
            'constraints_met': f"{constraints_met}/4",
            'health_score': constraints_met / 4.0,
            'status': 'EXCELLENT' if constraints_met == 4 else 
                     'GOOD' if constraints_met >= 3 else 
                     'NEEDS_WORK' if constraints_met >= 2 else 'POOR'
        }
        
        return health_report


def create_enhanced_single_peak_model(n_exc: int = 800, 
                                    n_inh: int = 200, 
                                    device: str = 'cpu',
                                    enforce_biological: bool = True) -> EnhancedSinglePeakRingAttractor:
    """
    Create enhanced single-peak ring attractor with optimal configuration.
    
    Args:
        n_exc: Number of excitatory neurons
        n_inh: Number of inhibitory neurons
        device: Computing device
        enforce_biological: Whether to enforce biological constraints
        
    Returns:
        Configured enhanced model
    """
    
    model = EnhancedSinglePeakRingAttractor(
        n_exc=n_exc, 
        n_inh=n_inh, 
        device=device,
        enforce_biological_constraints=enforce_biological
    )
    
    # Apply final parameter optimization
    with torch.no_grad():
        # Set optimal parameters based on analysis
        model.g_ee.data.fill_(0.4)   # Moderate excitation
        model.g_ie.data.fill_(6.0)   # Strong inhibition (15:1 ratio)
        model.g_ei.data.fill_(2.5)   # Strong E->I drive
        model.sigma_ee.data.fill_(0.18)  # Very narrow connections
        model.g_global.data.fill_(1.2)   # Strong global inhibition
        model.noise_rate_e.data.fill_(0.003)  # Very low noise
        model.noise_rate_i.data.fill_(0.001)  # Even lower inhibitory noise
        
        # Optimize inhibitory weights
        model.W_IE.data.fill_(-0.5)  # Strong uniform inhibition
        model.W_EI.data.uniform_(0.3, 0.6)  # Moderate E->I connectivity
    
    # Final constraint application
    if enforce_biological:
        model.apply_biological_constraints()
    
    print(f"Enhanced Single Peak Model Created:")
    print(f"  Excitatory neurons: {n_exc}")
    print(f"  Inhibitory neurons: {n_inh}")
    print(f"  σ_EE (connection width): {model.sigma_ee.item():.3f}")
    print(f"  g_ee: {model.g_ee.item():.3f}")
    print(f"  g_ie: {model.g_ie.item():.3f}")
    print(f"  Inhibition/Excitation ratio: {model.g_ie.item()/model.g_ee.item():.1f}:1")
    print(f"  Global inhibition: {model.g_global.item():.3f}")
    print(f"  Noise level: {model.noise_rate_e.item():.4f}")
    print(f"  Biological constraints: {'ENABLED' if enforce_biological else 'DISABLED'}")
    
    return model


def validate_single_peak_model(model: EnhancedSinglePeakRingAttractor, 
                              n_tests: int = 8,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive validation of single-peak model.
    
    Args:
        model: Model to validate
        n_tests: Number of test directions
        verbose: Whether to print detailed results
        
    Returns:
        Validation results dictionary
    """
    
    if verbose:
        print(f"🔍 VALIDATING ENHANCED SINGLE-PEAK MODEL")
        print(f"=" * 50)
    
    # Test directions
    test_directions = np.linspace(0, 2*np.pi, n_tests, endpoint=False)
    
    peak_counts = []
    tracking_errors = []
    max_activities = []
    
    for i, direction in enumerate(test_directions):
        model.reset_state()
        model.initialize_bump(direction, width=0.15, amplitude=0.5)
        
        # Create input
        from utils import angle_to_input
        device = next(model.parameters()).device
        input_pattern = angle_to_input(
            torch.tensor(direction, device=device),
            n_exc=model.n_exc,
            input_strength=1.5,
            input_width=0.2,
            device=device
        )
        
        # Let settle
        with torch.no_grad():
            for _ in range(20):  # More settling time
                activity = model(input_pattern, steps=1)
        
        # Analyze results
        peaks = model.get_peak_count()
        peak_counts.append(peaks)
        
        decoded = model.decode_angle(activity).item()
        error = abs(decoded - direction)
        error = min(error, 2*np.pi - error)
        tracking_errors.append(np.degrees(error))
        
        max_act = torch.max(activity).item()
        max_activities.append(max_act)
        
        if verbose and i < 4:
            print(f"  Direction {np.degrees(direction):3.0f}°: {peaks} peak(s), "
                  f"error={np.degrees(error):5.1f}°, activity={max_act:.4f}")
    
    # Calculate metrics
    single_peak_ratio = np.mean(np.array(peak_counts) <= 1)
    mean_error = np.mean(tracking_errors)
    mean_activity = np.mean(max_activities)
    
    # Get model health
    health_report = model.get_model_health_report()
    
    results = {
        'single_peak_ratio': single_peak_ratio,
        'mean_tracking_error': mean_error,
        'mean_activity_level': mean_activity,
        'peak_counts': peak_counts,
        'tracking_errors': tracking_errors,
        'health_report': health_report,
        'test_directions': test_directions
    }
    
    if verbose:
        print(f"\\n📊 VALIDATION RESULTS:")
        print(f"  Single peak ratio: {single_peak_ratio:.1%}")
        print(f"  Mean tracking error: {mean_error:.1f}°")
        print(f"  Mean activity level: {mean_activity:.4f}")
        print(f"  Model health: {health_report['overall_health']['status']}")
        
        if single_peak_ratio >= 0.95:
            print(f"  ✅ OUTSTANDING: Consistently achieves single peaks!")
        elif single_peak_ratio >= 0.9:
            print(f"  ✅ EXCELLENT: Successfully maintains single peaks!")
        elif single_peak_ratio >= 0.8:
            print(f"  ✅ GOOD: Mostly single peaks")
        else:
            print(f"  ⚠️  NEEDS WORK: Multiple peaks still occurring")
    
    return results