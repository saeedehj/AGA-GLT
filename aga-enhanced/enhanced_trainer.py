"""
Enhanced Trainer with Adaptive Gradient Alignment for FG-GLT

This module extends the original trainer with:
1. Adaptive gradient caching with automatic refresh
2. Selective parameter alignment
3. Layer-wise adaptive weighting
4. Theoretical convergence monitoring
5. Low-rank gradient compression

For ICML 2026 submission.

Author: Saeedeh (RMIT University)
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from pruner import DifferentiablePruner, ElementwiseGate
from functools import reduce
import operator
from torch.nn.utils import parametrize
import copy
import json
import time
from contextlib import nullcontext
import statistics as _st
from pathlib import Path
import datetime
import numpy as np

from adaptive_gradient_alignment import (
    AdaptiveGradientCache,
    AdaptiveAlignmentConfig,
    AlignmentMetric,
    GradientDistributionAnalyzer,
    compute_optimal_refresh_frequency,
    compute_convergence_rate
)


def _cosine(a, b, eps=1e-8):
    a = a.view(-1)
    b = b.view(-1).to(a.device)
    return 1.0 - (a @ b) / (a.norm() * b.norm() + eps)


class EnhancedDenseModelTrainer:
    """
    Enhanced trainer with Adaptive Gradient Alignment.
    
    Key improvements over original trainer:
    1. Adaptive cache refresh based on distribution shift
    2. Selective alignment on most influential parameters
    3. Layer-wise adaptive weighting
    4. Theoretical convergence monitoring
    5. Memory-efficient gradient compression
    """

    def __init__(self, model, data, dataset_name, device, lr=0.01, weight_decay=5e-4):
        self.model = model.to(device)
        self.data = data.to(device)
        self.dataset_name = dataset_name
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.gradients = {}
        self._weight_parametrizations = {}

        module_dict = dict(self.model.named_modules())

        for real_name, p in list(self.model.named_parameters()):
            if not isinstance(p, torch.nn.Parameter):
                continue
            if (not p.requires_grad) or (p.ndim < 2) or real_name.endswith("bias"):
                continue

            mod_name, param_field = real_name.rsplit(".", 1)
            owning_module = module_dict[mod_name]

            gate = ElementwiseGate(torch.ones_like(p))
            parametrize.register_parametrization(
                owning_module, param_field, gate)
            canonical_name = f"{mod_name}.parametrizations.{param_field}.original"
            self._weight_parametrizations[canonical_name] = gate
        
        # Initialize adaptive gradient cache (will be configured later)
        self.adaptive_cache = None
        self.training_stats = {
            "shift_history": [],
            "refresh_epochs": [],
            "convergence_bounds": [],
            "alignment_losses": []
        }

    def train(self, epochs=200):
        """Standard dense training (unchanged)"""
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.nll_loss(out[self.data.train_mask],
                              self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

            train_acc = self.evaluate(self.data.train_mask)
            val_acc = self.evaluate(self.data.val_mask)

            print(
                f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    @torch.no_grad()
    def evaluate(self, mask, edge_index=None, edge_weight=None):
        """Evaluate model, optionally on pruned graph"""
        self.model.eval()
        if edge_index is None:
            edge_index = self.data.edge_index
        logits = self.model(self.data.x, edge_index, edge_weight=edge_weight)
        pred = logits[mask].max(dim=1)[1]
        correct = pred.eq(self.data.y[mask]).sum().item()
        return correct / mask.sum().item()

    @torch.no_grad()
    def evaluate_prune(self, mask, pruner, use_hard=False, thresh=0.5, beta=0.1):
        self.model.eval()

        if use_hard:
            # Use gate probabilities for hard pruning (not edge_probs which can be negative)
            edge_probs, _ = pruner(self.data.x, self.data.edge_index, beta=beta)
            edge_gate_probs = pruner.edge_gate_probs(beta=beta)
            hard_mask = (edge_gate_probs > thresh)  # Use gate_probs, not edge_probs!
            ei = self.data.edge_index[:, hard_mask]
            ew = edge_probs[hard_mask]  # Use edge_probs as weights for kept edges
            logits = self.model(self.data.x, ei, edge_weight=ew)
        else:
            edge_probs, _ = pruner(self.data.x, self.data.edge_index, beta=beta)
            logits = self.model(
                self.data.x, self.data.edge_index, edge_weight=edge_probs)

        pred = logits[mask].max(dim=1)[1]
        correct = pred.eq(self.data.y[mask]).sum().item()
        return correct / mask.sum().item()

    def save_model(self):
        os.makedirs("./dense_models_nofeat", exist_ok=True)
        save_path = f'./dense_models_nofeat/{self.dataset_name.lower()}_dense_gcn.pth'
        torch.save(self.model.state_dict(), save_path)
        print(f"Dense model saved explicitly at {save_path}")

    def compute_and_save_gradients(self):
        """Compute gradients from dense model for caching"""
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4)
        optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(
            out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()

        dense_grads = {name: param.grad.clone().detach()
                       for name, param in self.model.named_parameters()}
        os.makedirs("./dense_gradients_GCN", exist_ok=True)
        grad_save_path = f'./dense_gradients_GCN/{self.dataset_name.lower()}_dense_grads.pth'
        torch.save(dense_grads, grad_save_path)
        print(f"Dense gradients explicitly saved at {grad_save_path}")

    @torch.no_grad()
    def test(self):
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        pred = logits[self.data.test_mask].max(dim=1)[1]
        correct = pred.eq(self.data.y[self.data.test_mask]).sum().item()
        test_acc = correct / self.data.test_mask.sum().item()
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc

    def _current_grads(self, loss):
        """Compute current gradients w.r.t. loss"""
        grads = torch.autograd.grad(
            loss, 
            [p for p in self.model.parameters() if p.requires_grad], 
            retain_graph=True, 
            create_graph=False
        )
        out = {}
        i = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                out[name] = grads[i].detach()
                i += 1
        return out

    def _sync_cuda(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _forward_for_timing(self, edge_index=None, edge_weight=None):
        x = self.data.x
        if edge_index is None:
            edge_index = self.data.edge_index
        return self.model(x, edge_index, edge_weight=edge_weight)

    @torch.no_grad()
    def measure_inference_time(self, edge_index=None, edge_weight=None,
                               n_warmup: int = 10, n_runs: int = 200):
        self.model.eval()

        for _ in range(n_warmup):
            _ = self._forward_for_timing(
                edge_index=edge_index, edge_weight=edge_weight)
        self._sync_cuda()

        times = []
        for _ in range(n_runs):
            self._sync_cuda()
            t0 = time.perf_counter()
            _ = self._forward_for_timing(
                edge_index=edge_index, edge_weight=edge_weight)
            self._sync_cuda()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        mean_ms = sum(times) / len(times) if times else 0.0
        std_ms = _st.pstdev(times) if len(times) > 1 else 0.0
        return mean_ms, std_ms

    def _grad_matching_loss_static(self, dense_grads_dict, current_grads_dict):
        """Original static gradient matching (for comparison)"""
        per_param = []
        for name, g_cur in current_grads_dict.items():
            if (name in dense_grads_dict) and (dense_grads_dict[name] is not None):
                per_param.append(_cosine(g_cur, dense_grads_dict[name]))
        if len(per_param) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(per_param).mean()

    def _l1_sparsity_loss(self, edge_probs, weight_probs_dict):
        loss = edge_probs.abs().mean()
        if weight_probs_dict:
            loss = loss + torch.stack([w.abs().mean()
                                      for w in weight_probs_dict.values()]).mean()
        return loss

    @torch.no_grad()
    def _edge_sparsity_metrics(self, edge_probs, thresh=0.5):
        total = edge_probs.numel()
        exp_kept = edge_probs.clamp(0, 1).sum().item()
        exp_sparsity = 1.0 - exp_kept / total
        hard_kept = (edge_probs > thresh).sum().item()
        hard_sparsity = 1.0 - hard_kept / total
        return exp_sparsity, hard_sparsity, hard_kept, total

    @torch.no_grad()
    def _weight_sparsity_metrics(self, weight_probs_dict, thresh=0.5):
        total = 0
        exp_kept = 0.0
        hard_kept = 0
        for w in weight_probs_dict.values():
            total += w.numel()
            exp_kept += w.clamp(0, 1).sum().item()
            hard_kept += (w > thresh).sum().item()
        if total == 0:
            return 0.0, 0.0, 0, 0
        exp_sparsity = 1.0 - exp_kept / total
        hard_sparsity = 1.0 - hard_kept / total
        return exp_sparsity, hard_sparsity, hard_kept, total

    @torch.no_grad()
    def _project_topk(self, probs_1d, target_sparsity):
        flat = probs_1d.view(-1)
        if target_sparsity is None:
            return (flat > 0.5)

        k_keep = int(round((1 - target_sparsity) * flat.numel()))
        if k_keep <= 0:
            return torch.zeros_like(flat, dtype=torch.bool)
        if k_keep >= flat.numel():
            return torch.ones_like(flat, dtype=torch.bool)

        thresh = torch.topk(flat, k_keep, sorted=True).values[-1]
        mask = (flat >= thresh)

        extra = int(mask.sum().item() - k_keep)
        if extra > 0:
            eq_idx = (flat == thresh).nonzero(as_tuple=False).view(-1)
            drop = eq_idx[-extra:]
            mask[drop] = False

        return mask

    def train_sparse_with_adaptive_grad_matching(
        self,
        dense_grads_path: str,
        # Adaptive Gradient Alignment config
        aga_config: AdaptiveAlignmentConfig = None,
        # Standard hyperparameters
        alpha: float = 0.2,           # λ_gm (gradient matching weight)
        beta: float = 2e-4,           # λ_sp (sparsity regularization weight)
        tau: float = 0.1,             # τ (Concrete temperature)
        epochs: int = 200,
        pruner_lr: float = 1e-2,
        target_edge_sparsity: float = 0.5,
        target_weight_sparsity: float = 0.5,
        project_every: int = 10,
        lc_lambda: float = 0.0,
        hetero: bool = False,
        # Comparison mode
        use_adaptive: bool = True,    # Enable/disable adaptive alignment
        use_static_baseline: bool = False,  # Run both for comparison
        # Retraining option
        retrain_after_pruning: bool = False,  # Retrain on hard-pruned graph
        retrain_epochs: int = 50,      # Epochs for retraining
        retrain_lr: float = 0.01       # Learning rate for retraining
    ):
        """
        Train with Adaptive Gradient Alignment.
        
        Key differences from original:
        1. Uses AdaptiveGradientCache for smart caching
        2. Automatic cache refresh when distribution shift detected
        3. Selective alignment on top-k% parameters
        4. Layer-wise adaptive weighting
        5. Theoretical convergence monitoring
        """
        
        # Setup adaptive alignment config
        if aga_config is None:
            aga_config = AdaptiveAlignmentConfig(
                kl_threshold=0.3,  # Increased: cosine distance threshold (0.3 = 0.7 similarity)
                top_k_ratio=0.5,
                layer_adaptive=True,
                use_low_rank=False,  # Enable for memory efficiency
                low_rank_ratio=0.2,
                min_refresh_interval=20,  # Increased: allow stable alignment period
                max_cache_age=50,
                alignment_metric=AlignmentMetric.COSINE
            )
        
        # Load initial dense gradients
        dense_grads = torch.load(dense_grads_path)
        
        # Initialize adaptive cache
        self.adaptive_cache = AdaptiveGradientCache(aga_config, self.device)
        self.adaptive_cache.cached_grads = dense_grads
        self.adaptive_cache.cache_epoch = 0
        self.adaptive_cache.last_refresh_epoch = 0
        
        # Initialize pruner
        num_edges = self.data.edge_index.size(1)
        node_feat_dim = self.data.x.size(1)

        pruner = DifferentiablePruner(
            num_edges=num_edges,
            model=self.model,
            node_feat_dim=node_feat_dim,
            device=self.device
        ).to(self.device)

        # Handle edge pruning disabled case
        if target_edge_sparsity is None or target_edge_sparsity == 0:
            with torch.no_grad():
                pruner.edge_log_alpha.data.fill_(10.0)
        
        if target_weight_sparsity is None:
            for gate in self._weight_parametrizations.values():
                gate.set_gate(torch.ones_like(gate.gate))
            for safe, param in pruner.weight_log_alpha.items():
                param.data.fill_(10.0)

        # Joint optimizer
        joint_params = list(self.model.parameters()) + list(pruner.parameters())
        optimizer = optim.Adam(
            joint_params, lr=self.optimizer.param_groups[0]['lr'], weight_decay=0.0)
        pruner_opt = optim.Adam(pruner.parameters(), lr=pruner_lr)

        best_val = 0.0
        
        # Training loop with adaptive gradient alignment
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            pruner_opt.zero_grad()

            # Pruner forward
            edge_probs, weight_probs = pruner(self.data.x, self.data.edge_index)
            edge_gate_probs = pruner.edge_gate_probs(beta=tau).detach()
            edge_probs = pruner.edge_probs(
                self.data.x, self.data.edge_index, beta=tau
            )

            # Push weight gates
            for real_name, gate_module in self._weight_parametrizations.items():
                gate_tensor = weight_probs.get(real_name, None)
                if gate_tensor is not None:
                    gate_module.set_gate(gate_tensor.detach())

            # Model forward
            out = self.model(self.data.x, self.data.edge_index, edge_weight=edge_probs)

            # Classification loss
            loss_cls = F.nll_loss(
                out[self.data.train_mask], self.data.y[self.data.train_mask])

            # Compute current gradients
            cur_grads = self._current_grads(loss_cls)

            # =====================================================
            # ADAPTIVE GRADIENT ALIGNMENT (Key contribution)
            # =====================================================
            if use_adaptive:
                loss_gm, align_info = self.adaptive_cache.compute_alignment_loss(
                    cur_grads, epoch, self.model
                )
                
                # Log statistics
                self.training_stats["shift_history"].append(align_info["shift"])
                self.training_stats["alignment_losses"].append(loss_gm.item())
                
                if align_info["refreshed"]:
                    self.training_stats["refresh_epochs"].append(epoch)
                    print(f"  [AGA] Cache refreshed at epoch {epoch}, reason: {align_info['refresh_reason']}, shift: {align_info['shift']:.4f}")
                
                # Compute theoretical bound
                bound = self.adaptive_cache.get_theoretical_bound(epoch)
                self.training_stats["convergence_bounds"].append(bound)
            else:
                # Fallback to static alignment
                loss_gm = self._grad_matching_loss_static(dense_grads, cur_grads)
                align_info = {"shift": 0.0, "refreshed": False}
            
            # Compare with static baseline if requested
            if use_static_baseline:
                loss_gm_static = self._grad_matching_loss_static(dense_grads, cur_grads)
                # Could log comparison here

            # Sparsity loss
            loss_sp = self._l1_sparsity_loss(edge_probs, weight_probs)

            # Total loss
            total = loss_cls + alpha * loss_gm + beta * loss_sp
            total.backward()

            optimizer.step()
            pruner_opt.step()

            # Projection step
            if (project_every is not None) and (epoch % project_every == 0) and (target_edge_sparsity is not None):
                with torch.no_grad():
                    hard_edge_mask = self._project_topk(
                        edge_gate_probs, target_edge_sparsity).bool()
                    p = hard_edge_mask.float().clamp(1e-6, 1-1e-6)
                    pruner.edge_log_alpha.data.copy_(
                        torch.log(p) - torch.log(1 - p))

            if (project_every is not None) and (epoch % project_every == 0) and (target_weight_sparsity is not None):
                with torch.no_grad():
                    all_w = torch.cat([w.view(-1) for w in weight_probs.values()])
                    k_keep = int(round((1 - target_weight_sparsity) * all_w.numel()))
                    if 0 < k_keep < all_w.numel():
                        thresh = torch.topk(all_w, k_keep, sorted=True).values[-1]
                        idx = 0
                        for real_name, w in weight_probs.items():
                            num = w.numel()
                            hard = (w.view(-1) >= thresh).float().view_as(w)
                            self._weight_parametrizations[real_name].set_gate(hard)
                            safe = pruner.safe_key(real_name)
                            p = hard.clamp(1e-6, 1-1e-6)
                            pruner.weight_log_alpha[safe].data.copy_(
                                torch.log(p) - torch.log(1-p))
                            idx += num

            # Evaluation
            val_acc_soft = self.evaluate_prune(self.data.val_mask, pruner=pruner, beta=tau)
            val_acc_hard = self.evaluate_prune(self.data.val_mask, pruner=pruner, use_hard=True, beta=tau)

            exp_s, hard_s, kept, total_e = self._edge_sparsity_metrics(edge_gate_probs)
            w_exp_s, w_hard_s, w_kept, w_total = self._weight_sparsity_metrics(weight_probs)

            # Logging
            if epoch % 10 == 0 or epoch == 1:
                shift_str = f", Shift: {align_info['shift']:.4f}" if use_adaptive else ""
                print(
                    f"Epoch {epoch}, Loss: {loss_cls.item():.4f}, "
                    f"GM: {loss_gm.item():.4f}{shift_str}, "
                    f"Val_hard: {val_acc_hard:.4f}, "
                    f"Edge: {hard_s:.3f}, Weight: {w_hard_s:.3f}"
                )

            if val_acc_hard > best_val:
                best_val = val_acc_hard
                torch.save(self.model.state_dict(),
                           f'./dense_models_nofeat/{self.dataset_name}_best_sparse.pth')

        # Final evaluation and results
        print(f"Best Val after sparse+AGA: {best_val:.4f}")
        
        # Log AGA statistics
        if use_adaptive:
            stats = self.adaptive_cache.get_statistics()
            print(f"\n=== Adaptive Gradient Alignment Statistics ===")
            print(f"Total cache refreshes: {stats['refresh_count']}")
            print(f"Average shift: {stats['avg_shift']:.4f}")
            if self.training_stats['convergence_bounds']:
                print(f"Final convergence bound: {self.training_stats['convergence_bounds'][-1]:.6f}")
        
        # Return results (similar to original)
        return self._finalize_and_save_results(
            pruner, alpha, beta, tau, target_edge_sparsity, target_weight_sparsity, best_val,
            retrain_after_pruning=retrain_after_pruning, retrain_epochs=retrain_epochs, retrain_lr=retrain_lr
        )

    def _finalize_and_save_results(
        self, 
        pruner, 
        alpha, 
        beta, 
        tau,
        target_edge_sparsity, 
        target_weight_sparsity,
        best_val,
        retrain_after_pruning: bool = False,
        retrain_epochs: int = 0,
        retrain_lr: float = 0.01
    ):
        """Finalize training and save results"""
        with torch.no_grad():
            # Get edge probabilities (sim * gates) and gate probabilities (gates only)
            final_edge_probs, final_weight_probs = pruner(
                self.data.x, self.data.edge_index, beta=tau)
            final_edge_gate_probs = pruner.edge_gate_probs(beta=tau)
            
            # Use gate probabilities for hard pruning (not edge_probs which can be negative)
            hard_edge_mask = (final_edge_gate_probs > 0.5)

            pruned_edge_index = self.data.edge_index[:, hard_edge_mask]
            # Use edge_probs as weights for kept edges (or use gate_probs, or 1.0)
            # Using edge_probs maintains the similarity weighting
            pruned_edge_weight = final_edge_probs[hard_edge_mask]

            weights_total = 0
            weights_kept_hard = 0
            for k, g in final_weight_probs.items():
                m = (g > 0.5)
                weights_total += m.numel()
                weights_kept_hard += int(m.sum().item())

            weight_sparsity_hard = 1.0 - (weights_kept_hard / weights_total) if weights_total > 0 else None

        # Optional: Retrain on hard-pruned graph (MUST be outside torch.no_grad() for gradients)
        if retrain_after_pruning:
            print(f"\n{'='*60}")
            print(f"RETRAINING on hard-pruned graph ({retrain_epochs} epochs)")
            print(f"{'='*60}")
            
            # Create new optimizer for retraining
            retrain_optimizer = optim.Adam(self.model.parameters(), lr=retrain_lr, weight_decay=5e-4)
            
            # Retrain on hard-pruned graph
            for retrain_epoch in range(1, retrain_epochs + 1):
                self.model.train()
                retrain_optimizer.zero_grad()
                
                # Forward pass on hard-pruned graph
                out = self.model(self.data.x, pruned_edge_index, edge_weight=pruned_edge_weight)
                loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
                
                loss.backward()
                retrain_optimizer.step()
                
                # Evaluate periodically
                if retrain_epoch % 10 == 0 or retrain_epoch == retrain_epochs:
                    self.model.eval()
                    with torch.no_grad():
                        val_acc_retrain = self.evaluate(self.data.val_mask, pruned_edge_index, pruned_edge_weight)
                        print(f"Retrain Epoch {retrain_epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc_retrain:.4f}")
            
            print(f"{'='*60}\n")
        
        # Final test evaluation
        with torch.no_grad():
            self.model.eval()
            logits = self.model(self.data.x, pruned_edge_index,
                                edge_weight=pruned_edge_weight)
            pred = logits[self.data.test_mask].max(dim=1)[1]
            test_acc_pruned = pred.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
            
            if retrain_after_pruning:
                print(f"Final Test Accuracy (HARD-PRUNED graph, after retraining): {test_acc_pruned:.4f}")
            else:
                print(f"Final Test Accuracy (HARD-PRUNED graph): {test_acc_pruned:.4f}")

            # Setup output directory
            run_dir = Path(getattr(self, "run_dir", ""))
            if not run_dir:
                tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                run_dir = Path("results") / f"{self.dataset_name}_run_{tag}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save pruned graph (without statistics metadata - calculate from file later)
            torch.save(
                {
                    "edge_index_pruned": pruned_edge_index.detach().cpu(),
                    "edge_weight_pruned": pruned_edge_weight.detach().cpu(),
                    "edge_mask_hard": hard_edge_mask.detach().cpu(),
                    "edge_mask_prob": final_edge_probs.detach().cpu(),
                },
                run_dir / "final_pruned_graph.pt",
            )

            torch.save(self.model.state_dict(), run_dir / "final_masked_model.pth")

            # Timing
            dense_avg_ms, dense_std_ms = self.measure_inference_time()
            soft_avg_ms, soft_std_ms = self.measure_inference_time(
                edge_index=self.data.edge_index,
                edge_weight=final_edge_probs,
            )
            hard_avg_ms, hard_std_ms = self.measure_inference_time(
                edge_index=pruned_edge_index,
                edge_weight=pruned_edge_weight,
            )

            # Calculate statistics from saved graph file (not from variables)
            # Load the saved graph to calculate statistics
            saved_graph = torch.load(run_dir / "final_pruned_graph.pt")
            pruned_edge_index_loaded = saved_graph["edge_index_pruned"]
            
            # Graph statistics - calculate from original and saved pruned graph
            nodes_before = int(self.data.num_nodes)
            nodes_after = nodes_before  # Nodes don't change, only edges are pruned
            edges_before = int(self.data.edge_index.size(1))
            edges_after = int(pruned_edge_index_loaded.size(1))  # From saved file
            edges_removed = edges_before - edges_after
            edge_spars_hard = 1.0 - edges_after / edges_before if edges_before > 0 else None
            edge_reduction_ratio = edges_after / edges_before if edges_before > 0 else None
            edge_reduction_percent = (1.0 - edge_reduction_ratio) * 100.0 if edge_reduction_ratio is not None else None

            # Weight statistics
            weight_sparsity_hard = 1.0 - (weights_kept_hard / weights_total) if weights_total > 0 else None
            weights_removed = weights_total - weights_kept_hard
            weight_reduction_ratio = weights_kept_hard / weights_total if weights_total > 0 else None
            weight_reduction_percent = (1.0 - weight_reduction_ratio) * 100.0 if weight_reduction_ratio is not None else None
            
            # Print statistics summary
            print(f"\n{'='*60}")
            print(f"PRUNING STATISTICS (calculated from saved graph)")
            print(f"{'='*60}")
            print(f"Nodes: {nodes_before} (unchanged)")
            print(f"Edges: {edges_before} → {edges_after} (removed {edges_removed}, {edge_reduction_percent:.2f}% reduction)")
            print(f"Weights: {weights_total} → {weights_kept_hard} (removed {weights_removed}, {weight_reduction_percent:.2f}% reduction)")
            print(f"{'='*60}\n")

            def _safe_div(a, b):
                return float(a) / float(b) if (b is not None and b > 0) else None

            speedup_soft = _safe_div(dense_avg_ms, soft_avg_ms)
            speedup_hard = _safe_div(dense_avg_ms, hard_avg_ms)

            # Build results dict with comprehensive statistics
            results_dict = {
                # Dataset and configuration
                "dataset": self.dataset_name,
                "device": str(self.device),
                "lambda_gm": alpha,
                "lambda_sp": beta,
                "tau": tau,
                "target_edge_sparsity": target_edge_sparsity,
                "target_weight_sparsity": target_weight_sparsity,
                
                # Performance metrics
                "Best Val after sparse+GM": float(best_val),
                "test_acc_pruned": float(test_acc_pruned),
                "retrained": retrain_after_pruning,
                "retrain_epochs": retrain_epochs if retrain_after_pruning else 0,
                
                # Graph statistics - BEFORE pruning
                "nodes_before": nodes_before,
                "edges_before": edges_before,
                
                # Graph statistics - AFTER hard pruning
                "nodes_after": nodes_after,
                "edges_after": edges_after,
                "edges_removed": edges_removed,
                "edge_sparsity_hard": edge_spars_hard,
                "edge_reduction_ratio": edge_reduction_ratio,
                "edge_reduction_percent": edge_reduction_percent,
                
                # Weight statistics - BEFORE pruning
                "weights_total": weights_total,
                
                # Weight statistics - AFTER hard pruning
                "weights_kept_hard": weights_kept_hard,
                "weights_removed": weights_removed,
                "weight_sparsity_hard": weight_sparsity_hard,
                "weight_reduction_ratio": weight_reduction_ratio,
                "weight_reduction_percent": weight_reduction_percent,
                
                # Timing metrics
                "dense_ms_avg": dense_avg_ms, 
                "dense_ms_std": dense_std_ms,
                "soft_ms_avg": soft_avg_ms, 
                "soft_ms_std": soft_std_ms,
                "hard_ms_avg": hard_avg_ms, 
                "hard_ms_std": hard_std_ms,
                "speedup_soft": speedup_soft,
                "speedup_hard": speedup_hard,
                
                # Legacy fields for backward compatibility
                "nodes": nodes_before,
                "edges_total": edges_before,
                "edges_kept_hard": edges_after,
            }

            # Add AGA statistics
            if self.adaptive_cache is not None:
                aga_stats = self.adaptive_cache.get_statistics()
                results_dict.update({
                    "aga_refresh_count": aga_stats["refresh_count"],
                    "aga_avg_shift": aga_stats["avg_shift"],
                    "aga_final_bound": aga_stats["latest_bound"],
                })

            # Save metrics
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(results_dict, f, indent=2)

            # Save training statistics
            with open(run_dir / "aga_training_stats.json", "w") as f:
                json.dump({
                    "shift_history": self.training_stats["shift_history"],
                    "refresh_epochs": self.training_stats["refresh_epochs"],
                    "convergence_bounds": self.training_stats["convergence_bounds"],
                }, f, indent=2)

            print(f"[INFO] Per-run files written to: {run_dir}")

            return results_dict

    # =========================================================================
    # ORIGINAL METHOD FOR COMPARISON (kept for ablation studies)
    # =========================================================================
    
    def train_sparse_with_grad_matching(
        self,
        dense_grads_path,
        alpha=0.2,
        beta=2e-4,
        tau=0.1,
        epochs=200,
        pruner_lr=1e-2,
        target_edge_sparsity=0.5,
        target_weight_sparsity=0.5,
        project_every=10,
        lc_lambda: float = 0.0,
        hetero: bool = False,
        retrain_after_pruning: bool = False,
        retrain_epochs: int = 50,
        retrain_lr: float = 0.01
    ):
        """Original static gradient matching method (for comparison/ablation)"""
        # This is the original implementation from your trainer.py
        # Keeping it here for ablation studies comparing static vs adaptive
        
        dense_grads = torch.load(dense_grads_path)
        num_edges = self.data.edge_index.size(1)
        node_feat_dim = self.data.x.size(1)

        pruner = DifferentiablePruner(
            num_edges=num_edges,
            model=self.model,
            node_feat_dim=node_feat_dim,
            device=self.device
        ).to(self.device)

        if target_edge_sparsity is None or target_edge_sparsity == 0:
            with torch.no_grad():
                pruner.edge_log_alpha.data.fill_(10.0)
        if target_weight_sparsity is None:
            for gate in self._weight_parametrizations.values():
                gate.set_gate(torch.ones_like(gate.gate))
            for safe, param in pruner.weight_log_alpha.items():
                param.data.fill_(10.0)

        joint_params = list(self.model.parameters()) + list(pruner.parameters())
        optimizer = optim.Adam(
            joint_params, lr=self.optimizer.param_groups[0]['lr'], weight_decay=0.0)
        pruner_opt = optim.Adam(pruner.parameters(), lr=pruner_lr)

        best_val = 0.0
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            pruner_opt.zero_grad()

            edge_probs, weight_probs = pruner(self.data.x, self.data.edge_index)
            edge_gate_probs = pruner.edge_gate_probs(beta=tau).detach()
            edge_probs = pruner.edge_probs(
                self.data.x, self.data.edge_index, beta=tau
            )

            for real_name, gate_module in self._weight_parametrizations.items():
                gate_tensor = weight_probs.get(real_name, None)
                if gate_tensor is not None:
                    gate_module.set_gate(gate_tensor.detach())

            out = self.model(self.data.x, self.data.edge_index, edge_weight=edge_probs)

            loss_cls = F.nll_loss(
                out[self.data.train_mask], self.data.y[self.data.train_mask])

            cur_grads = self._current_grads(loss_cls)
            loss_gm = self._grad_matching_loss_static(dense_grads, cur_grads)
            loss_sp = self._l1_sparsity_loss(edge_probs, weight_probs)

            total = loss_cls + alpha * loss_gm + beta * loss_sp
            total.backward()

            optimizer.step()
            pruner_opt.step()

            if (project_every is not None) and (epoch % project_every == 0) and (target_edge_sparsity is not None):
                with torch.no_grad():
                    hard_edge_mask = self._project_topk(
                        edge_gate_probs, target_edge_sparsity).bool()
                    p = hard_edge_mask.float().clamp(1e-6, 1-1e-6)
                    pruner.edge_log_alpha.data.copy_(
                        torch.log(p) - torch.log(1 - p))

            if (project_every is not None) and (epoch % project_every == 0) and (target_weight_sparsity is not None):
                with torch.no_grad():
                    all_w = torch.cat([w.view(-1) for w in weight_probs.values()])
                    k_keep = int(round((1 - target_weight_sparsity) * all_w.numel()))
                    if 0 < k_keep < all_w.numel():
                        thresh = torch.topk(all_w, k_keep, sorted=True).values[-1]
                        for real_name, w in weight_probs.items():
                            hard = (w.view(-1) >= thresh).float().view_as(w)
                            self._weight_parametrizations[real_name].set_gate(hard)
                            safe = pruner.safe_key(real_name)
                            p = hard.clamp(1e-6, 1-1e-6)
                            pruner.weight_log_alpha[safe].data.copy_(
                                torch.log(p) - torch.log(1-p))

            val_acc_soft = self.evaluate_prune(self.data.val_mask, pruner=pruner, beta=tau)
            val_acc_hard = self.evaluate_prune(self.data.val_mask, pruner=pruner, use_hard=True, beta=tau)

            exp_s, hard_s, kept, total_e = self._edge_sparsity_metrics(edge_gate_probs)
            w_exp_s, w_hard_s, w_kept, w_total = self._weight_sparsity_metrics(weight_probs)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss_cls.item():.4f}, "
                    f"Val_hard: {val_acc_hard:.4f}, "
                    f"Edge: {hard_s:.3f}, Weight: {w_hard_s:.3f}"
                )

            if val_acc_hard > best_val:
                best_val = val_acc_hard

        print(f"Best Val (static GM): {best_val:.4f}")
        
        # Compute final test accuracy on hard-pruned graph (for fair comparison)
        with torch.no_grad():
            # Get edge probabilities (sim * gates) and gate probabilities (gates only)
            final_edge_probs, final_weight_probs = pruner(
                self.data.x, self.data.edge_index, beta=tau)
            final_edge_gate_probs = pruner.edge_gate_probs(beta=tau)
            
            # Use gate probabilities for hard pruning (not edge_probs which can be negative)
            hard_edge_mask = (final_edge_gate_probs > 0.5)
            
            pruned_edge_index = self.data.edge_index[:, hard_edge_mask]
            # Use edge_probs as weights for kept edges
            pruned_edge_weight = final_edge_probs[hard_edge_mask]
            
            # Calculate weight statistics
            weights_total = 0
            weights_kept_hard = 0
            for k, g in final_weight_probs.items():
                m = (g > 0.5)
                weights_total += m.numel()
                weights_kept_hard += int(m.sum().item())
        
        # Optional: Retrain on hard-pruned graph
        if retrain_after_pruning:
            print(f"\n{'='*60}")
            print(f"RETRAINING on hard-pruned graph ({retrain_epochs} epochs)")
            print(f"{'='*60}")
            
            # Create new optimizer for retraining
            retrain_optimizer = optim.Adam(self.model.parameters(), lr=retrain_lr, weight_decay=5e-4)
            
            # Retrain on hard-pruned graph
            for retrain_epoch in range(1, retrain_epochs + 1):
                self.model.train()
                retrain_optimizer.zero_grad()
                
                # Forward pass on hard-pruned graph
                out = self.model(self.data.x, pruned_edge_index, edge_weight=pruned_edge_weight)
                loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
                
                loss.backward()
                retrain_optimizer.step()
                
                # Evaluate periodically
                if retrain_epoch % 10 == 0 or retrain_epoch == retrain_epochs:
                    self.model.eval()
                    with torch.no_grad():
                        val_acc_retrain = self.evaluate(self.data.val_mask, pruned_edge_index, pruned_edge_weight)
                        print(f"Retrain Epoch {retrain_epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc_retrain:.4f}")
            
            print(f"{'='*60}\n")
        
        # Final test evaluation
        with torch.no_grad():
            self.model.eval()
            logits = self.model(self.data.x, pruned_edge_index,
                                edge_weight=pruned_edge_weight)
            pred = logits[self.data.test_mask].max(dim=1)[1]
            test_acc_pruned = pred.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
            
            if retrain_after_pruning:
                print(f"Final Test Accuracy (HARD-PRUNED graph, after retraining): {test_acc_pruned:.4f}")
            else:
                print(f"Final Test Accuracy (HARD-PRUNED graph): {test_acc_pruned:.4f}")
            
            # Save pruned graph (without statistics metadata)
            # For static method, we don't have a run_dir, so calculate directly
            # But still calculate from the pruned_edge_index (which represents the saved graph)
            nodes_before = int(self.data.num_nodes)
            nodes_after = nodes_before
            edges_before = int(self.data.edge_index.size(1))
            edges_after = int(pruned_edge_index.size(1))  # This is what would be saved
            edges_removed = edges_before - edges_after
            edge_reduction_percent = (1.0 - edges_after / edges_before) * 100.0 if edges_before > 0 else 0.0
            weights_removed = weights_total - weights_kept_hard
            weight_reduction_percent = (1.0 - weights_kept_hard / weights_total) * 100.0 if weights_total > 0 else 0.0
            
            # Print statistics summary
            print(f"\n{'='*60}")
            print(f"PRUNING STATISTICS (Static)")
            print(f"{'='*60}")
            print(f"Nodes: {nodes_before} (unchanged)")
            print(f"Edges: {edges_before} → {edges_after} (removed {edges_removed}, {edge_reduction_percent:.2f}% reduction)")
            print(f"Weights: {weights_total} → {weights_kept_hard} (removed {weights_removed}, {weight_reduction_percent:.2f}% reduction)")
            print(f"{'='*60}\n")
        
        return {
            "best_val": best_val, 
            "test_acc_pruned": float(test_acc_pruned),
            "method": "static",
            # Add statistics for consistency
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "edges_before": edges_before,
            "edges_after": edges_after,
            "edges_removed": edges_removed,
            "edge_reduction_percent": edge_reduction_percent,
            "weights_total": weights_total,
            "weights_kept_hard": weights_kept_hard,
            "weights_removed": weights_removed,
            "weight_reduction_percent": weight_reduction_percent,
            "retrained": retrain_after_pruning,
            "retrain_epochs": retrain_epochs if retrain_after_pruning else 0,
        }
