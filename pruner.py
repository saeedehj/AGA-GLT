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
