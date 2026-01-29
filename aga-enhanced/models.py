from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)

        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=x.size(0)
        )

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm) + self.bias

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class DenseGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(DenseGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class DenseGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super(DenseGAT, self).__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            edge_dim=1,
            dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            edge_dim=1,
            dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: Node features
        edge_index: Graph connectivity
        edge_weight: pruning mask (edge_probs), shape [num_edges]
                     If None, defaults to 1.0 for all edges.
        """
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), 1, device=x.device)
        else:
            edge_weight = edge_weight.view(-1, 1)

        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_weight)

        return F.log_softmax(x, dim=1)


class GINConvMasked(MessagePassing):
    """GIN-style conv with optional scalar edge weights (for pruning)."""

    def __init__(self, in_channels, out_channels, train_eps: bool = False):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('eps', torch.zeros(1))

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + (1.0 + self.eps) * x
        return self.mlp(out)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class DenseGIN(nn.Module):
    """
    TEDDY-style GIN for node classification.
    - n_layers counts GIN blocks BEFORE the final output block.
    - Final block maps hidden -> out_channels (no BN/Dropout after it).
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 n_layers: int = 2, dropout: float = 0.5, train_eps: bool = False):
        super().__init__()
        assert n_layers >= 1, "n_layers must be >= 1"
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # first layer
        self.convs.append(GINConvMasked(
            in_channels, hidden_channels, train_eps=train_eps))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # hidden layers (if any)
        for _ in range(n_layers - 1):
            self.convs.append(GINConvMasked(
                hidden_channels, hidden_channels, train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # final classification layer
        self.out_conv = GINConvMasked(
            hidden_channels, out_channels, train_eps=train_eps)

    def forward(self, x, edge_index, edge_weight=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_conv(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)
