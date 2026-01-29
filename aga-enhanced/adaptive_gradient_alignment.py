"""
Adaptive Gradient Alignment (AGA) Module for FG-GLT

This module implements the core contribution for ICML submission:
1. Distribution shift detection using KL divergence between cached and current gradients
2. Automatic cache refresh when shift exceeds threshold
3. Selective alignment focusing on top-k% most influential parameters
4. Layer-wise adaptive weighting based on gradient magnitude

Theoretical Foundation:
- Theorem 1: Convergence guarantee under bounded gradient staleness
- Theorem 2: Optimal refresh frequency derived from spectral properties

Author: Saeedeh (RMIT University)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math


class AlignmentMetric(Enum):
    """Supported gradient alignment metrics"""
    COSINE = "cosine"
    L2 = "l2"
    KL = "kl"
    WASSERSTEIN = "wasserstein"


@dataclass
class AdaptiveAlignmentConfig:
    """Configuration for Adaptive Gradient Alignment"""
    # Distribution shift detection
    kl_threshold: float = 0.3  # Threshold for cache refresh (cosine distance: 0.3 = 0.7 similarity)
    shift_window_size: int = 10  # Window for computing moving average of shift
    
    # Selective alignment
    top_k_ratio: float = 0.5  # Fraction of parameters to align (most influential)
    layer_adaptive: bool = True  # Enable layer-wise adaptive weighting
    
    # Cache management
    initial_cache_epoch: int = 0  # Epoch from which to cache gradients
    min_refresh_interval: int = 20  # Minimum epochs between cache refreshes (increased for stability)
    max_cache_age: int = 50  # Maximum epochs before forced refresh
    
    # Low-rank compression (for memory efficiency)
    use_low_rank: bool = False
    low_rank_ratio: float = 0.1  # Keep top 10% singular values
    
    # Alignment loss configuration
    alignment_metric: AlignmentMetric = AlignmentMetric.COSINE
    alignment_weight_decay: float = 0.99  # Decay alignment weight over epochs
    
    # Theoretical bounds
    lipschitz_constant: float = 1.0  # L for convergence analysis
    smoothness_constant: float = 1.0  # β for smoothness assumption


class GradientDistributionAnalyzer:
    """
    Analyzes gradient distributions for staleness detection.
    
    Key metrics:
    - KL divergence between cached and current gradient distributions
    - Cosine similarity decay over time
    - Layer-wise gradient magnitude changes
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.history: List[Dict[str, torch.Tensor]] = []
        self.shift_history: List[float] = []
        
    def compute_kl_divergence(
        self, 
        cached_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
        eps: float = 1e-8
    ) -> float:
        """
        Compute KL divergence between gradient distributions.
        
        We treat gradient magnitudes as probability distributions after normalization.
        
        KL(P || Q) = Σ P(x) log(P(x) / Q(x))
        
        where P = normalized |cached_grads|, Q = normalized |current_grads|
        """
        kl_per_param = []
        
        for name, g_cached in cached_grads.items():
            if name not in current_grads or current_grads[name] is None:
                continue
                
            g_current = current_grads[name].to(self.device)
            g_cached = g_cached.to(self.device)
            
            # Flatten and take absolute values
            p = torch.abs(g_cached.view(-1)) + eps
            q = torch.abs(g_current.view(-1)) + eps
            
            # Normalize to probability distributions
            p = p / p.sum()
            q = q / q.sum()
            
            # KL divergence
            kl = (p * (torch.log(p) - torch.log(q))).sum()
            kl_per_param.append(kl.item())
        
        if len(kl_per_param) == 0:
            return 0.0
            
        return np.mean(kl_per_param)
    
    def compute_cosine_similarity(
        self,
        cached_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
        eps: float = 1e-8
    ) -> float:
        """Compute average cosine similarity between gradient vectors"""
        similarities = []
        
        for name, g_cached in cached_grads.items():
            if name not in current_grads or current_grads[name] is None:
                continue
                
            g_current = current_grads[name].to(self.device)
            g_cached = g_cached.to(self.device)
            
            # Flatten
            v1 = g_cached.view(-1)
            v2 = g_current.view(-1)
            
            # Cosine similarity
            sim = (v1 @ v2) / (v1.norm() * v2.norm() + eps)
            similarities.append(sim.item())
        
        if len(similarities) == 0:
            return 1.0
            
        return np.mean(similarities)
    
    def compute_wasserstein_distance(
        self,
        cached_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute 1-Wasserstein distance between gradient distributions.
        
        For 1D distributions, W_1(P, Q) = ∫|F_P(x) - F_Q(x)|dx
        """
        distances = []
        
        for name, g_cached in cached_grads.items():
            if name not in current_grads or current_grads[name] is None:
                continue
                
            g_current = current_grads[name].to(self.device)
            g_cached = g_cached.to(self.device)
            
            # Sort gradient values
            v1_sorted = torch.sort(g_cached.view(-1))[0]
            v2_sorted = torch.sort(g_current.view(-1))[0]
            
            # Wasserstein-1 distance (Earth Mover's Distance)
            w1 = torch.mean(torch.abs(v1_sorted - v2_sorted)).item()
            distances.append(w1)
        
        if len(distances) == 0:
            return 0.0
            
        return np.mean(distances)
    
    def compute_distribution_shift(
        self,
        cached_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
        metric: AlignmentMetric = AlignmentMetric.COSINE  # Changed default to COSINE for gradient alignment
    ) -> float:
        """
        Compute distribution shift between cached and current gradients.
        
        Returns a scalar indicating how much the gradient distribution has shifted.
        Higher values indicate more staleness.
        """
        if metric == AlignmentMetric.KL:
            return self.compute_kl_divergence(cached_grads, current_grads)
        elif metric == AlignmentMetric.COSINE:
            # Convert similarity to distance
            return 1.0 - self.compute_cosine_similarity(cached_grads, current_grads)
        elif metric == AlignmentMetric.WASSERSTEIN:
            return self.compute_wasserstein_distance(cached_grads, current_grads)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def update_history(self, shift_value: float):
        """Track shift values over time"""
        self.shift_history.append(shift_value)
    
    def get_moving_average_shift(self, window_size: int = 10) -> float:
        """Get moving average of distribution shift"""
        if len(self.shift_history) == 0:
            return 0.0
        window = self.shift_history[-window_size:]
        return np.mean(window)


class LowRankGradientCompressor:
    """
    Compress gradients using low-rank approximation to reduce memory.
    
    Uses SVD: G ≈ U Σ V^T, keeping top-k singular values.
    
    This addresses reviewer concern about gradient cache overhead.
    """
    
    def __init__(self, rank_ratio: float = 0.1, device: torch.device = None):
        self.rank_ratio = rank_ratio
        self.device = device or torch.device('cpu')
        
    def compress(
        self, 
        grads: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compress gradients using truncated SVD.
        
        Returns dict mapping param_name -> (U, S, V) where G ≈ U @ diag(S) @ V.T
        """
        compressed = {}
        
        for name, g in grads.items():
            if g is None or g.ndim < 2:
                # For 1D tensors, store directly
                compressed[name] = (g.clone(), None, None)
                continue
            
            original_shape = g.shape
            
            # Reshape to 2D for SVD
            if g.ndim > 2:
                g_2d = g.view(g.shape[0], -1)
            else:
                g_2d = g
            
            # Compute SVD
            try:
                U, S, Vh = torch.linalg.svd(g_2d, full_matrices=False)
            except:
                # Fallback for numerical issues
                compressed[name] = (g.clone(), None, None)
                continue
            
            # Keep top-k singular values
            k = max(1, int(self.rank_ratio * min(g_2d.shape)))
            
            U_k = U[:, :k].clone()
            S_k = S[:k].clone()
            Vh_k = Vh[:k, :].clone()
            
            # Store compressed representation with shape info
            compressed[name] = (U_k, S_k, Vh_k, original_shape)
        
        return compressed
    
    def decompress(
        self, 
        compressed: Dict[str, Tuple]
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct gradients from compressed representation"""
        decompressed = {}
        
        for name, data in compressed.items():
            if len(data) == 3 and data[1] is None:
                # Was stored directly (1D tensor)
                decompressed[name] = data[0]
                continue
            
            U_k, S_k, Vh_k, original_shape = data
            
            # Reconstruct: G ≈ U @ diag(S) @ V^T
            g_reconstructed = U_k @ torch.diag(S_k) @ Vh_k
            
            # Reshape to original
            if len(original_shape) > 2:
                g_reconstructed = g_reconstructed.view(original_shape)
            
            decompressed[name] = g_reconstructed
        
        return decompressed
    
    def compute_compression_ratio(
        self, 
        original: Dict[str, torch.Tensor],
        compressed: Dict[str, Tuple]
    ) -> float:
        """Compute memory compression ratio"""
        original_size = sum(g.numel() for g in original.values() if g is not None)
        
        compressed_size = 0
        for name, data in compressed.items():
            if len(data) == 3 and data[1] is None:
                compressed_size += data[0].numel()
            else:
                U_k, S_k, Vh_k, _ = data
                compressed_size += U_k.numel() + S_k.numel() + Vh_k.numel()
        
        if original_size == 0:
            return 1.0
            
        return compressed_size / original_size


class SelectiveGradientAligner:
    """
    Performs selective gradient alignment on most influential parameters.
    
    Key insight: Not all parameters contribute equally to optimization dynamics.
    Aligning only the top-k% most influential parameters can:
    1. Reduce computational overhead
    2. Focus alignment on parameters that matter most
    3. Prevent over-constraining the sparse model
    """
    
    def __init__(self, top_k_ratio: float = 0.5, device: torch.device = None):
        self.top_k_ratio = top_k_ratio
        self.device = device or torch.device('cpu')
        self.importance_scores: Dict[str, float] = {}
        
    def compute_parameter_importance(
        self,
        grads: Dict[str, torch.Tensor],
        method: str = "gradient_magnitude"
    ) -> Dict[str, float]:
        """
        Compute importance scores for each parameter.
        
        Methods:
        - gradient_magnitude: |∇θ|_2
        - gradient_variance: Var(∇θ)
        - fisher_information: E[(∇θ)²] (approximated)
        """
        importance = {}
        
        for name, g in grads.items():
            if g is None:
                importance[name] = 0.0
                continue
                
            if method == "gradient_magnitude":
                importance[name] = g.norm().item()
            elif method == "gradient_variance":
                importance[name] = g.var().item()
            elif method == "fisher_information":
                importance[name] = (g ** 2).mean().item()
            else:
                importance[name] = g.norm().item()
        
        self.importance_scores = importance
        return importance
    
    def select_top_k_parameters(
        self,
        grads: Dict[str, torch.Tensor]
    ) -> List[str]:
        """Select top-k% most important parameters for alignment"""
        importance = self.compute_parameter_importance(grads)
        
        # Sort by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top-k%
        k = max(1, int(self.top_k_ratio * len(sorted_params)))
        selected = [name for name, _ in sorted_params[:k]]
        
        return selected
    
    def compute_selective_alignment_loss(
        self,
        cached_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
        selected_params: List[str] = None,
        metric: AlignmentMetric = AlignmentMetric.COSINE,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute alignment loss only for selected parameters.
        
        L_align = (1/|S|) Σ_{θ ∈ S} d(∇θ_sparse, ∇θ_dense)
        
        where S is the set of selected parameters.
        """
        if selected_params is None:
            selected_params = self.select_top_k_parameters(current_grads)
        
        losses = []
        
        for name in selected_params:
            if name not in cached_grads or name not in current_grads:
                continue
            if cached_grads[name] is None or current_grads[name] is None:
                continue
            
            g_cached = cached_grads[name].to(self.device)
            g_current = current_grads[name].to(self.device)
            
            if metric == AlignmentMetric.COSINE:
                # Cosine distance: 1 - cos(g_cached, g_current)
                v1 = g_cached.view(-1)
                v2 = g_current.view(-1)
                cos_sim = (v1 @ v2) / (v1.norm() * v2.norm() + eps)
                loss = 1.0 - cos_sim
            elif metric == AlignmentMetric.L2:
                # Normalized L2 distance
                loss = F.mse_loss(g_current, g_cached)
            else:
                # Default to cosine
                v1 = g_cached.view(-1)
                v2 = g_current.view(-1)
                cos_sim = (v1 @ v2) / (v1.norm() * v2.norm() + eps)
                loss = 1.0 - cos_sim
            
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        
        return torch.stack(losses).mean()


class LayerWiseAdaptiveWeighting:
    """
    Compute layer-wise adaptive weights for gradient alignment.
    
    Different layers may require different alignment strengths:
    - Early layers: More important for feature extraction
    - Later layers: More task-specific
    
    We weight alignment loss by layer sensitivity.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.layer_weights: Dict[str, float] = {}
        
    def compute_layer_sensitivity(
        self,
        model: nn.Module,
        grads: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute sensitivity of each layer to pruning.
        
        Sensitivity = ||∇W||_F / ||W||_F
        """
        sensitivity = {}
        
        for name, param in model.named_parameters():
            if name not in grads or grads[name] is None:
                continue
            
            grad_norm = grads[name].norm().item()
            param_norm = param.norm().item() + 1e-8
            
            sensitivity[name] = grad_norm / param_norm
        
        # Normalize to [0, 1]
        if len(sensitivity) > 0:
            max_sens = max(sensitivity.values())
            if max_sens > 0:
                sensitivity = {k: v / max_sens for k, v in sensitivity.items()}
        
        self.layer_weights = sensitivity
        return sensitivity
    
    def get_weighted_alignment_loss(
        self,
        cached_grads: Dict[str, torch.Tensor],
        current_grads: Dict[str, torch.Tensor],
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute layer-weighted alignment loss"""
        losses = []
        weights = []
        
        for name, g_cached in cached_grads.items():
            if name not in current_grads or current_grads[name] is None:
                continue
            
            g_current = current_grads[name].to(self.device)
            g_cached = g_cached.to(self.device)
            
            # Cosine distance
            v1 = g_cached.view(-1)
            v2 = g_current.view(-1)
            cos_sim = (v1 @ v2) / (v1.norm() * v2.norm() + eps)
            loss = 1.0 - cos_sim
            
            # Get layer weight (default to 1.0 if not computed)
            weight = self.layer_weights.get(name, 1.0)
            
            losses.append(loss)
            weights.append(weight)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        
        losses = torch.stack(losses)
        weights = torch.tensor(weights, device=self.device)
        weights = weights / weights.sum()  # Normalize
        
        return (losses * weights).sum()


class AdaptiveGradientCache:
    """
    Main class managing the adaptive gradient cache.
    
    Features:
    1. Automatic refresh when distribution shift exceeds threshold
    2. Low-rank compression for memory efficiency
    3. Selective parameter tracking
    4. Theoretical convergence monitoring
    """
    
    def __init__(
        self,
        config: AdaptiveAlignmentConfig,
        device: torch.device = None
    ):
        self.config = config
        self.device = device or torch.device('cpu')
        
        # Initialize components
        self.analyzer = GradientDistributionAnalyzer(self.device)
        self.compressor = LowRankGradientCompressor(
            rank_ratio=config.low_rank_ratio,
            device=self.device
        ) if config.use_low_rank else None
        self.aligner = SelectiveGradientAligner(
            top_k_ratio=config.top_k_ratio,
            device=self.device
        )
        self.layer_weighter = LayerWiseAdaptiveWeighting(self.device)
        
        # Cache state
        self.cached_grads: Dict[str, torch.Tensor] = {}
        self.compressed_cache: Dict[str, Tuple] = {}
        self.cache_epoch: int = -1
        self.last_refresh_epoch: int = -1
        self.refresh_count: int = 0
        
        # Statistics
        self.shift_at_refresh: List[float] = []
        self.convergence_estimates: List[float] = []
        
    def initialize_cache(
        self,
        model: nn.Module,
        data,
        loss_fn,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize gradient cache from model.
        
        This should be called after dense pretraining.
        """
        model.train()
        model.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        
        # Extract gradients
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().detach()
        
        # Store cache
        self.cached_grads = grads
        self.cache_epoch = epoch
        self.last_refresh_epoch = epoch
        
        # Optionally compress
        if self.compressor is not None:
            self.compressed_cache = self.compressor.compress(grads)
        
        # Compute layer sensitivity for weighting
        self.layer_weighter.compute_layer_sensitivity(model, grads)
        
        return grads
    
    def should_refresh(
        self,
        current_grads: Dict[str, torch.Tensor],
        current_epoch: int
    ) -> Tuple[bool, float, str]:
        """
        Determine if cache should be refreshed.
        
        Returns: (should_refresh, shift_value, reason)
        """
        # Get effective cached gradients
        if self.compressor is not None and len(self.compressed_cache) > 0:
            cached = self.compressor.decompress(self.compressed_cache)
        else:
            cached = self.cached_grads
        
        # Compute distribution shift - use COSINE for gradient alignment (not KL)
        # KL measures distribution shape, cosine measures gradient direction alignment
        shift = self.analyzer.compute_distribution_shift(
            cached, current_grads, AlignmentMetric.COSINE  # Force cosine for gradient alignment
        )
        self.analyzer.update_history(shift)
        
        # Check conditions
        cache_age = current_epoch - self.cache_epoch
        epochs_since_refresh = current_epoch - self.last_refresh_epoch
        
        # Condition 1: Shift exceeds threshold
        if shift > self.config.kl_threshold and epochs_since_refresh >= self.config.min_refresh_interval:
            return True, shift, "shift_threshold_exceeded"
        
        # Condition 2: Cache too old
        if cache_age > self.config.max_cache_age:
            return True, shift, "max_age_exceeded"
        
        # Condition 3: Moving average shift is high
        avg_shift = self.analyzer.get_moving_average_shift(self.config.shift_window_size)
        if avg_shift > self.config.kl_threshold * 1.5 and epochs_since_refresh >= self.config.min_refresh_interval:
            return True, shift, "moving_average_high"
        
        return False, shift, "no_refresh_needed"
    
    def refresh_cache(
        self,
        new_grads: Dict[str, torch.Tensor],
        current_epoch: int,
        shift_value: float = None
    ):
        """Update the cache with new gradients"""
        self.cached_grads = {k: v.clone().detach() for k, v in new_grads.items() if v is not None}
        self.cache_epoch = current_epoch
        self.last_refresh_epoch = current_epoch
        self.refresh_count += 1
        
        if shift_value is not None:
            self.shift_at_refresh.append(shift_value)
        
        # Update compressed cache
        if self.compressor is not None:
            self.compressed_cache = self.compressor.compress(self.cached_grads)
    
    def compute_alignment_loss(
        self,
        current_grads: Dict[str, torch.Tensor],
        current_epoch: int,
        model: nn.Module = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute adaptive gradient alignment loss.
        
        Returns: (loss, info_dict)
        """
        # Get effective cached gradients
        if self.compressor is not None and len(self.compressed_cache) > 0:
            cached = self.compressor.decompress(self.compressed_cache)
        else:
            cached = self.cached_grads
        
        # Check if refresh needed
        should_refresh, shift, reason = self.should_refresh(current_grads, current_epoch)
        
        info = {
            "shift": shift,
            "should_refresh": should_refresh,
            "refresh_reason": reason,
            "cache_age": current_epoch - self.cache_epoch,
            "refresh_count": self.refresh_count
        }
        
        # Track gradient norms for theoretical bound computation
        grad_norms = [g.norm().item() for g in current_grads.values() if g is not None]
        if not hasattr(self, '_gradient_norms'):
            self._gradient_norms = []
        self._gradient_norms.extend(grad_norms)
        # Keep only recent history (last 100 epochs)
        if len(self._gradient_norms) > 100:
            self._gradient_norms = self._gradient_norms[-100:]
        
        # Optionally refresh
        if should_refresh:
            self.refresh_cache(current_grads, current_epoch, shift)
            cached = self.cached_grads
            info["refreshed"] = True
        else:
            info["refreshed"] = False
        
        # Select parameters for alignment
        selected_params = self.aligner.select_top_k_parameters(current_grads)
        info["num_aligned_params"] = len(selected_params)
        
        # Compute alignment loss based on configuration
        if self.config.layer_adaptive and model is not None:
            # Layer-wise weighted loss
            self.layer_weighter.compute_layer_sensitivity(model, current_grads)
            loss = self.layer_weighter.get_weighted_alignment_loss(cached, current_grads)
        else:
            # Selective alignment loss
            loss = self.aligner.compute_selective_alignment_loss(
                cached, current_grads, selected_params, self.config.alignment_metric
            )
        
        # Apply decay based on cache age (older cache = weaker alignment)
        cache_age = current_epoch - self.cache_epoch
        decay_factor = self.config.alignment_weight_decay ** cache_age
        loss = loss * decay_factor
        info["decay_factor"] = decay_factor
        
        return loss, info
    
    def get_theoretical_bound(self, current_epoch: int) -> float:
        """
        Compute theoretical convergence bound based on Theorem 1.
        
        Bound = L * σ² * τ / (2 * √T)
        
        where:
        - L: Lipschitz constant
        - σ²: gradient variance (estimated from gradient norms)
        - τ: cache staleness
        - T: total iterations
        """
        L = self.config.lipschitz_constant
        
        # Estimate gradient variance from actual gradient norms (not shift history)
        if hasattr(self, '_gradient_norms') and len(self._gradient_norms) > 0:
            # Use actual gradient variance from gradient norms
            sigma_sq = np.var(self._gradient_norms)
        else:
            # Default estimate if no history yet
            sigma_sq = 0.01
        
        tau = current_epoch - self.cache_epoch  # staleness
        T = max(1, current_epoch)  # total iterations
        
        bound = L * sigma_sq * tau / (2 * np.sqrt(T))
        self.convergence_estimates.append(bound)
        
        return bound
    
    def get_statistics(self) -> Dict:
        """Get cache statistics for logging"""
        return {
            "cache_epoch": self.cache_epoch,
            "refresh_count": self.refresh_count,
            "avg_shift": self.analyzer.get_moving_average_shift(),
            "shift_history_len": len(self.analyzer.shift_history),
            "compression_ratio": (
                self.compressor.compute_compression_ratio(
                    self.cached_grads, self.compressed_cache
                ) if self.compressor is not None else 1.0
            ),
            "latest_bound": self.convergence_estimates[-1] if self.convergence_estimates else None
        }


# =============================================================================
# THEORETICAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_optimal_refresh_frequency(
    lipschitz_L: float,
    smoothness_beta: float,
    gradient_variance: float,
    target_accuracy: float
) -> int:
    """
    Compute optimal cache refresh frequency based on Theorem 2.
    
    From our theoretical analysis:
    τ* = √(ε / (L * σ²))
    
    where:
    - τ*: optimal refresh interval
    - ε: target convergence accuracy
    - L: Lipschitz constant
    - σ²: gradient variance
    """
    if gradient_variance <= 0 or lipschitz_L <= 0:
        return 10  # default
    
    tau_opt = np.sqrt(target_accuracy / (lipschitz_L * gradient_variance))
    return max(1, int(tau_opt))


def compute_convergence_rate(
    T: int,
    tau: int,
    L: float,
    sigma_sq: float,
    eta: float
) -> float:
    """
    Compute expected convergence rate with gradient alignment.
    
    From Theorem 1:
    E[||∇f(θ_T)||²] ≤ (f(θ_0) - f*) / (η*T) + η*L*σ² + L²*τ²*σ²
    
    The last term represents the error from stale gradients.
    """
    # Simplified bound assuming f(θ_0) - f* ≈ 1
    term1 = 1.0 / (eta * T)  # convergence from optimization
    term2 = eta * L * sigma_sq  # variance from SGD
    term3 = L**2 * tau**2 * sigma_sq  # staleness error
    
    return term1 + term2 + term3


def estimate_spectral_threshold(
    laplacian_eigenvalues: torch.Tensor,
    target_sparsity: float
) -> float:
    """
    Connect cache refresh threshold to graph spectral properties.
    
    From Theorem 2: The optimal threshold depends on the spectral gap
    of the graph Laplacian.
    
    λ_threshold ∝ (1 - target_sparsity) * spectral_gap
    """
    # Sort eigenvalues
    sorted_eigs = torch.sort(laplacian_eigenvalues)[0]
    
    # Spectral gap = λ_2 - λ_1 (assuming λ_1 ≈ 0 for connected graphs)
    if len(sorted_eigs) > 1:
        spectral_gap = sorted_eigs[1].item()
    else:
        spectral_gap = 0.1
    
    # Threshold proportional to sparsity level and spectral gap
    threshold = (1 - target_sparsity) * spectral_gap * 0.1
    
    return threshold
