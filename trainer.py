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


def _cosine(a, b, eps=1e-8):
    a = a.view(-1)
    b = b.view(-1).to(a.device)
    return 1.0 - (a @ b) / (a.norm() * b.norm() + eps)


class DenseModelTrainer:

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
            # keep only real Parameters with at least 2 dims (no bias, no LayerNorm weight1d)
            if not isinstance(p, torch.nn.Parameter):
                continue
            if (not p.requires_grad) or (p.ndim < 2) or real_name.endswith("bias"):
                continue

            # find the owning module and parameter field name
            mod_name, param_field = real_name.rsplit(".", 1)
            owning_module = module_dict[mod_name]

            # attach an elementwise gate (initialized to ones => no masking at start)
            gate = ElementwiseGate(torch.ones_like(p))
            parametrize.register_parametrization(
                owning_module, param_field, gate)
            canonical_name = f"{mod_name}.parametrizations.{param_field}.original"
            self._weight_parametrizations[canonical_name] = gate

    def train(self, epochs=200):
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
    def evaluate(self, mask):
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        pred = logits[mask].max(dim=1)[1]
        correct = pred.eq(self.data.y[mask]).sum().item()
        return correct / mask.sum().item()

    @torch.no_grad()
    def evaluate_prune(self, mask, pruner, use_hard=False, thresh=0.5):

        self.model.eval()

        if use_hard:
            edge_probs, _ = pruner(self.data.x, self.data.edge_index)
            hard_mask = (edge_probs > thresh)
            ei = self.data.edge_index[:, hard_mask]
            ew = edge_probs[hard_mask]
            logits = self.model(self.data.x, ei, edge_weight=ew)
        else:
            edge_probs, _ = pruner(self.data.x, self.data.edge_index)
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
        grads = torch.autograd.grad(loss, [p for p in self.model.parameters(
        ) if p.requires_grad], retain_graph=True, create_graph=False)
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
        """
        Mirror the **same** forward signature you already use in evaluate/test.
        If your model uses adj_t instead of edge_index, switch to that here.
        """
        x = self.data.x
        if edge_index is None:
            edge_index = self.data.edge_index  # <-- force COO path
        return self.model(x, edge_index, edge_weight=edge_weight)

    @torch.no_grad()
    def measure_inference_time(self, edge_index=None, edge_weight=None,
                               n_warmup: int = 10, n_runs: int = 200):
        """
        Forward-pass latency in ms (mean, std).
        Uses the model's exact forward signature used elsewhere in this codebase:
            self.model(self.data.x, edge_index, edge_weight=edge_weight)
        """
        self.model.eval()

        # Warm-up (avoid first-run overhead)
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
            times.append((t1 - t0) * 1000.0)  # ms

        mean_ms = sum(times) / len(times) if times else 0.0
        std_ms = _st.pstdev(times) if len(times) > 1 else 0.0
        return mean_ms, std_ms

    def _grad_matching_loss(self, dense_grads_dict, current_grads_dict):
        # robust GM: cosine per param, averaged
        per_param = []
        for name, g_cur in current_grads_dict.items():
            if (name in dense_grads_dict) and (dense_grads_dict[name] is not None):
                per_param.append(_cosine(g_cur, dense_grads_dict[name]))
        if len(per_param) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(per_param).mean()

    def _l1_sparsity_loss(self, edge_probs, weight_probs_dict):
        # simple, stable sparsity penalty
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
        """
        probs_1d: 1D tensor of edge gate probabilities in [0,1]
        Return: 1D boolean mask of which entries to keep (size = probs_1d.numel()).
        """
        flat = probs_1d.view(-1)
        if target_sparsity is None:
            return (flat > 0.5)

        k_keep = int(round((1 - target_sparsity) * flat.numel()))
        if k_keep <= 0:
            return torch.zeros_like(flat, dtype=torch.bool)
        if k_keep >= flat.numel():
            return torch.ones_like(flat, dtype=torch.bool)

        # find threshold
        thresh = torch.topk(flat, k_keep, sorted=True).values[-1]
        mask = (flat >= thresh)

        # tie-break if we got more than k_keep due to equals
        extra = int(mask.sum().item() - k_keep)
        if extra > 0:
            eq_idx = (flat == thresh).nonzero(as_tuple=False).view(-1)
            # deterministically drop the last 'extra' among the equals
            drop = eq_idx[-extra:]
            mask[drop] = False

        return mask

    def train_sparse_with_grad_matching(
        self,
        dense_grads_path,
        alpha=0.2,           # λ_gm (gradient matching weight)
        beta=2e-4,           # λ_sp (sparsity regularization weight)
        tau=0.1,             # τ (Concrete temperature) - NEW PARAMETER
        epochs=200,
        pruner_lr=1e-2,
        target_edge_sparsity=0.5,
        target_weight_sparsity=0.5,
        project_every=10,
        lc_lambda: float = 0.0,
        hetero: bool = False
    ):

        # 1) load teacher (dense) grads
        dense_grads = torch.load(dense_grads_path)
        # 2) create pruner
        num_edges = self.data.edge_index.size(1)
        node_feat_dim = self.data.x.size(1)

        pruner = DifferentiablePruner(
            num_edges=num_edges,
            model=self.model,
            node_feat_dim=node_feat_dim,
            device=self.device
        ).to(self.device)

        # ---- Optional: disable edge pruning (keep all edges ON) ----
        if target_edge_sparsity is None or target_edge_sparsity == 0:
            with torch.no_grad():
                pruner.edge_log_alpha.data.fill_(10.0)
        if target_weight_sparsity is None:
            # all-ones gates for the live parametrizations
            for gate in self._weight_parametrizations.values():
                gate.set_gate(torch.ones_like(gate.gate))

            # push +∞ (large positive) into pruner's logit params so concrete gate ~ 1.0
            for safe, param in pruner.weight_log_alpha.items():
                param.data.fill_(10.0)  # ~sigmoid(10)≈0.99995

                # 3) joint optimizer (model + pruner params)
                joint_params = list(self.model.parameters()) + \
                    list(pruner.parameters())

        joint_params = list(self.model.parameters()) + \
            list(pruner.parameters())

        optimizer = optim.Adam(
            joint_params, lr=self.optimizer.param_groups[0]['lr'], weight_decay=0.0)
        # optional second group if you want faster mask updates
        pruner_opt = optim.Adam(pruner.parameters(), lr=pruner_lr)

        best_val = 0.0
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            pruner_opt.zero_grad()

            # ---- pruner forward (edge & weight gates as probabilities in [0,1]) ----
            edge_probs, weight_probs = pruner(
                self.data.x, self.data.edge_index)

            edge_gate_probs = pruner.edge_gate_probs(beta=tau).detach()
            edge_probs = pruner.edge_probs(
                self.data.x, self.data.edge_index, beta=tau
            )

            # push the per-parameter Concrete gates into the parametrizations
            for real_name, gate_module in self._weight_parametrizations.items():
                gate_tensor = weight_probs.get(real_name, None)
                if gate_tensor is not None:
                    gate_module.set_gate(gate_tensor.detach())

            # ---- model forward with edge gating ----
            out = self.model(self.data.x, self.data.edge_index,
                             edge_weight=edge_probs)

            # ---- losses ----
            loss_cls = F.nll_loss(
                out[self.data.train_mask], self.data.y[self.data.train_mask])

            # current grads for GM w.r.t. task loss
            cur_grads = self._current_grads(loss_cls)
            loss_gm = self._grad_matching_loss(dense_grads, cur_grads)

            loss_sp = self._l1_sparsity_loss(edge_probs, weight_probs)

            total = loss_cls + alpha * loss_gm + beta * loss_sp
            total.backward()

            optimizer.step()
            pruner_opt.step()

            if (project_every is not None) and (epoch % project_every == 0) and (target_edge_sparsity is not None):
                with torch.no_grad():
                    # REPLACE any use of 'edge_probs' with 'edge_gate_probs'
                    hard_edge_mask = self._project_topk(
                        edge_gate_probs, target_edge_sparsity).bool()

                    # *** LOCK the projection into the pruner logits so it doesn't drift ***
                    p = hard_edge_mask.float().clamp(
                        1e-6, 1-1e-6)   # {0,1} -> (ε, 1-ε)
                    pruner.edge_log_alpha.data.copy_(
                        torch.log(p) - torch.log(1 - p))

            # ---- optional exact sparsity by projection weight
            if (project_every is not None) and (epoch % project_every == 0) and (target_weight_sparsity is not None):
                with torch.no_grad():
                    # Flatten all weight gates into one vector to do a global top-k keep
                    all_w = torch.cat([w.view(-1)
                                      for w in weight_probs.values()])
                    # reuse same target if you like
                    k_keep = int(
                        round((1 - target_weight_sparsity) * all_w.numel()))
                    if 0 < k_keep < all_w.numel():
                        thresh = torch.topk(
                            all_w, k_keep, sorted=True).values[-1]
                        # build hard masks per tensor and push back to pruner logits
                        idx = 0
                        for real_name, w in weight_probs.items():
                            num = w.numel()
                            hard = (w.view(-1) >= thresh).float().view_as(w)
                            # update live parametrization gate
                            self._weight_parametrizations[real_name].set_gate(
                                hard)
                            # also push into pruner logits so it “locks”
                            safe = pruner.safe_key(real_name)
                            p = hard.clamp(1e-6, 1-1e-6)
                            pruner.weight_log_alpha[safe].data.copy_(
                                torch.log(p) - torch.log(1-p))
                            idx += num

            # ---- eval & sparsity logs ----
            val_acc_soft = self.evaluate_prune(
                self.data.val_mask, pruner=pruner)
            val_acc_hard = self.evaluate_prune(
                self.data.val_mask, pruner=pruner, use_hard=True)

            # exp_s, hard_s, kept, total_e = self._edge_sparsity_metrics(
            #     edge_probs)
            exp_s, hard_s, kept, total_e = self._edge_sparsity_metrics(
                edge_gate_probs)
            w_exp_s, w_hard_s, w_kept, w_total = self._weight_sparsity_metrics(
                weight_probs)

            print(
                f"Epoch {epoch}, Loss: {loss_cls.item():.4f}, "
                f"sparsity_loss: {loss_sp.item():.4f}, Val Acc: {val_acc_soft:.4f}, Val Acc_hard: {val_acc_hard:.4f},"
                f"Edge Sparsity (exp/hard): {exp_s:.3f}/{hard_s:.3f} (kept {kept}/{total_e}), "
                f"Weight Sparsity (exp/hard): {w_exp_s:.3f}/{w_hard_s:.3f} (kept {w_kept}/{w_total})"
            )

            if val_acc_hard > best_val:
                best_val = val_acc_hard
                torch.save(self.model.state_dict(),
                           f'./dense_models_nofeat/{self.dataset_name}_best_sparse.pth')

        print(f"Best Val after sparse+GM: {best_val:.4f}")
        with torch.no_grad():

            final_edge_probs, final_weight_probs = pruner(
                self.data.x, self.data.edge_index)
            hard_edge_mask = (final_edge_probs > 0.5)

            pruned_edge_index = self.data.edge_index[:, hard_edge_mask]
            pruned_edge_weight = final_edge_probs[hard_edge_mask]

            weights_total = 0
            weights_kept_hard = 0
            for k, g in final_weight_probs.items():  # g is a tensor of gate probabilities
                m = (g > 0.5)
                weights_total += m.numel()
                weights_kept_hard += int(m.sum().item())

            weight_sparsity_hard = 1.0 - \
                (weights_kept_hard / weights_total) if weights_total > 0 else None

            # ---------- Inference timing (dense vs pruned) ----------

            self.model.eval()

            logits = self.model(self.data.x, pruned_edge_index,
                                edge_weight=pruned_edge_weight)
            pred = logits[self.data.test_mask].max(dim=1)[1]
            test_acc_pruned = pred.eq(self.data.y[self.data.test_mask]).sum(
            ).item() / self.data.test_mask.sum().item()
            print(
                f"Final Test Accuracy (HARD-PRUNED graph): {test_acc_pruned:.4f}")

            # ------------------ SAVE ARTIFACTS INTO PER-RUN FOLDER ------------------
            # Use run_dir set by main.py; fall back to a timestamped default if missing
            run_dir = Path(getattr(self, "run_dir", ""))
            if not run_dir:
                import datetime
                tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                run_dir = Path("results") / f"{self.dataset_name}_run_{tag}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # (1) Save graph + gates + model state into the run_dir
            torch.save(
                {
                    "edge_index_pruned": pruned_edge_index.detach().cpu(),
                    "edge_weight_pruned": pruned_edge_weight.detach().cpu(),
                    "edge_mask_hard": hard_edge_mask.detach().cpu(),
                    "edge_mask_prob": final_edge_probs.detach().cpu(),
                },
                run_dir / "final_pruned_graph.pt",
            )

            torch.save(
                {
                    "edge_mask_hard": hard_edge_mask.detach().cpu(),
                    "edge_mask_prob": final_edge_probs.detach().cpu(),
                    "edge_index": self.data.edge_index.detach().cpu(),
                },
                run_dir / "final_edge_mask.pt",
            )

            torch.save(
                {
                    "weight_gates": {k: v.detach().cpu() for k, v in final_weight_probs.items()}
                },
                run_dir / "final_weight_gates.pt",
            )

            torch.save(self.model.state_dict(),
                       run_dir / "final_masked_model.pth")

            dense_avg_ms, dense_std_ms = self.measure_inference_time()

            soft_avg_ms, soft_std_ms = self.measure_inference_time(
                edge_index=self.data.edge_index,
                edge_weight=final_edge_probs,
            )

            hard_avg_ms, hard_std_ms = self.measure_inference_time(
                edge_index=pruned_edge_index,
                edge_weight=pruned_edge_weight,
            )

            total_edges = int(self.data.edge_index.size(1))
            kept_edges = int(pruned_edge_index.size(1))
            edge_spars_hard = 1.0 - kept_edges / total_edges if total_edges > 0 else None

            def _safe_div(a, b):
                return float(a) / float(b) if (b is not None and b > 0) else None

            speedup_soft = _safe_div(dense_avg_ms, soft_avg_ms)
            speedup_hard = _safe_div(dense_avg_ms, hard_avg_ms)

            # Build the per-run metrics dict (keep your existing keys; add pruned accuracy)
            results_dict = {
                "dataset": self.dataset_name,
                "device": str(self.device),

                # Hyperparameters being tuned
                "lambda_gm": alpha,
                "lambda_sp": beta,
                "tau": tau,

                # set by main.py before each run; we include them for provenance
                "target_edge_sparsity": target_edge_sparsity,
                "target_weight_sparsity": target_weight_sparsity,
                "Best Val after sparse+GM": float(best_val),

                # NEW: explicitly store hard-pruned test accuracy from this block
                "test_acc_pruned": float(test_acc_pruned),

                # graph / sparsity
                "nodes": int(self.data.num_nodes),
                "edges_total": total_edges,
                "edges_kept_hard": kept_edges,
                "edge_sparsity_hard": edge_spars_hard,
                "weights_total": weights_total,
                "weights_kept_hard": weights_kept_hard,
                "weight_sparsity_hard": weight_sparsity_hard,

                # timings
                "dense_ms_avg": dense_avg_ms, "dense_ms_std": dense_std_ms,
                "soft_ms_avg":  soft_avg_ms,  "soft_ms_std":  soft_std_ms,
                "hard_ms_avg":  hard_avg_ms,  "hard_ms_std":  hard_std_ms,
                "speedup_soft": speedup_soft,
                "speedup_hard": speedup_hard,
            }

            # Write separate files in this run folder
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(results_dict, f, indent=2)

            timing_only = {k: results_dict[k] for k in [
                "dense_ms_avg", "dense_ms_std", "soft_ms_avg", "soft_ms_std",
                "hard_ms_avg", "hard_ms_std", "speedup_soft", "speedup_hard",
            ]}
            with open(run_dir / "timing.json", "w") as f:
                json.dump(timing_only, f, indent=2)

            print(f"[INFO] Per-run files written to: {run_dir}")

            # Return dict so main.py can aggregate all runs
            return results_dict
