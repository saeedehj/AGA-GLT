import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F
from typing import Optional


class ElementwiseGate(nn.Module):
    """
    Parametrization: returns weight * gate (gate is elementwise, same shape).
    The gate tensor is updated from the trainer every step.
    """

    def __init__(self, init_gate):
        super().__init__()
        self.register_buffer("gate", init_gate.clone())

    def set_gate(self, new_gate):
        self.gate.copy_(new_gate)

    def forward(self, X):
        return X * self.gate


class DifferentiablePruner(nn.Module):
    def __init__(self, num_edges, model, node_feat_dim, device):
        super().__init__()
        self.device = device
        self.edge_log_alpha = nn.Parameter(
            torch.zeros(num_edges, device=self.device))

        self.weight_log_alpha = nn.ParameterDict()
        self._safe_to_real_name = {}
        self._real_to_safe_name = {}

        for real_name, p in model.named_parameters():
            if "parametrizations" not in real_name:
                continue
            if not real_name.endswith(".original"):
                continue
            if (p is None) or (not isinstance(p, torch.nn.Parameter)):
                continue
            if (not p.requires_grad) or (p.ndim < 2):
                continue
            if real_name.endswith(".bias.original"):
                continue

            safe_name = real_name.replace(".", "___")
            self.weight_log_alpha[safe_name] = nn.Parameter(
                torch.zeros_like(p, device=self.device)
            )
            self._safe_to_real_name[safe_name] = real_name
            self._real_to_safe_name[real_name] = safe_name

        self.similarity_mlp = nn.Sequential(
            nn.Linear(2 * node_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

    def concrete_gate(self, log_alpha, beta=0.1):
        u = torch.rand_like(log_alpha, device=self.device)
        return torch.sigmoid((log_alpha + torch.log(u) - torch.log(1 - u)) / beta)

    def edge_probs(self, x, edge_index, beta=0.1):
        i, j = edge_index[0], edge_index[1]
        feat = torch.cat([x[i], x[j]], dim=-1)
        sim = self.similarity_mlp(feat).squeeze(-1)

        # use concrete gate instead of plain sigmoid
        gates = self.concrete_gate(self.edge_log_alpha, beta=beta)
        return sim * gates

    # def edge_probs(self, x, edge_index, beta=0.1, extras=None, mode="homo"):
    #     i, j = edge_index
    #     xi, xj = x[i], x[j]
    #     # pairwise algebra
    #     feats = [torch.abs(xi-xj), xi*xj, xi+xj]
    #     # cheap similarities
    #     cos = F.cosine_similarity(xi, xj, dim=-1, eps=1e-8).unsqueeze(-1)
    #     dot = (xi*xj).sum(-1, keepdim=True)
    #     feats += [cos, dot]
    #     # structural extras (precomputed): deg, jaccard, lap_pos, ppr_ij
    #     if extras:
    #         for nf in extras.get("node_side", []):   # tensors [N,d]
    #             feats += [nf[i], nf[j], torch.abs(nf[i]-nf[j])]
    #         for ef in extras.get("edge_side", []):   # tensors [E,d]
    #             feats += [ef]
    #     edge_x = torch.cat(feats, dim=-1)            # [E, D_edge]
    #     if not hasattr(self, "_sim_init") or not self._sim_init:
    #         in_dim = edge_x.size(-1)
    #         hidden = getattr(self, "sim_hidden", 64)  # or a constructor arg
    #         self.similarity_mlp = nn.Sequential(
    #             nn.Linear(in_dim, hidden, bias=True),
    #             nn.ReLU(inplace=True),
    #             nn.Linear(hidden, 1, bias=True),
    #         ).to(edge_x.device)
    #         self._sim_init = True
    #     logit = self.similarity_mlp(edge_x).squeeze(-1)
    #     p_feat = torch.sigmoid(logit)
    #     if mode == "hetero":
    #         p_feat = 1.0 - p_feat
    #     p_gate = self.concrete_gate(self.edge_log_alpha, beta=beta)
    #     return p_feat * p_gate

    # def edge_probs(self, x, edge_index, beta=0.1, mode="homo"):
    #     i, j = edge_index[0], edge_index[1]
    #     # feat = torch.cat([torch.abs(x[i]-x[j]), x[i]*x[j]], dim=-1)
    #     feat = torch.cat([x[i], x[j]], dim=-1)

    #     logit = self.similarity_mlp(feat).squeeze(-1)
    #     sim_prob = torch.sigmoid(logit)
    #     if mode == "hetero":
    #         sim_prob = 1.0 - sim_prob
    #     gates = self.concrete_gate(self.edge_log_alpha, beta=beta)
    #     return sim_prob * gates

    # def edge_probs(self, x, edge_index, beta=0.1, mode="homo"):
    #     i, j = edge_index[0], edge_index[1]
    #     cos = F.cosine_similarity(x[i], x[j], dim=-1)     # in [-1, 1]
    #     sim01 = (cos + 1) * 0.5                           # to [0,1]
    #     # hetero_prob = 1.0 - sim01                         # higher for dissimilar pairs
    #     hetero_prob = sim01
    #     gates = self.concrete_gate(self.edge_log_alpha, beta=beta)
    #     return hetero_prob * gates

    # def edge_probs(self, x, edge_index, beta=0.1, mode="homo"):
    #     i, j = edge_index[0], edge_index[1]
    #     feat = torch.cat([x[i], x[j]], dim=-1)
    #     logit = self.similarity_mlp(feat).squeeze(-1)

    #     # hetero_prob = torch.sigmoid(logit)   # reverse of sigmoid(logit)
    #     gates = self.concrete_gate(self.edge_log_alpha, beta=beta)
    #     return - logit * gates

    # def edge_probs(self, x, edge_index, beta=0.1):
    #     gates = self.concrete_gate(self.edge_log_alpha, beta=beta)
    #     return gates

    # for w =0

    def edge_gate_probs(self, beta: float = 0.1):
        """
        Return *pure* edge gate probabilities in [0,1], aligned with edge_index order.
        Use this for projection & sparsity metrics (not sim*gate).
        """
        return self.concrete_gate(self.edge_log_alpha, beta=beta)

    def forward(self, x, edge_index, beta=0.1):
        edge_gates = self.edge_probs(x, edge_index, beta=beta)

        weight_gates = {
            self._safe_to_real_name[safe]: self.concrete_gate(param, beta=beta)
            for safe, param in self.weight_log_alpha.items()
        }
        return edge_gates, weight_gates

    def safe_key(self, real_name: str) -> str:
        return self._real_to_safe_name.get(real_name, real_name)
