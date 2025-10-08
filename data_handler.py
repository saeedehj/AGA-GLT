import os
import shutil
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_torch_sparse_tensor
from torch_geometric.datasets import Reddit
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import to_undirected
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage


class DataHandler:
    def __init__(self, dataset_name, device):
        self.dataset_name = dataset_name
        self.data = None
        self.device = device

    @staticmethod
    def _index_to_mask(idx, size):
        mask = torch.zeros(size, dtype=torch.bool)
        mask[idx] = True
        return mask

    def load_ogbn_arxiv(self, root):
        # ---- Collect available PyG classes safely (handles older/newer PyG) ----
        from torch.serialization import add_safe_globals, safe_globals
        avail = []

        # Core Data
        try:
            from torch_geometric.data import Data
            avail.append(Data)
        except Exception:
            pass

        # Storage classes (some versions don't expose all of these)
        try:
            from torch_geometric.data import storage as _stor_mod   # module
        except Exception:
            _stor_mod = None

        def _maybe(name):
            if _stor_mod is None:
                return None
            return getattr(_stor_mod, name, None)

        for n in ["BaseStorage", "GlobalStorage", "NodeStorage", "EdgeStorage", "AttrView", "TensorAttr"]:
            cls = _maybe(n)
            if cls is not None:
                avail.append(cls)

        # DataEdgeAttr moved around across versions
        dea = None
        try:
            from torch_geometric.data.data import DataEdgeAttr as _DEA
            dea = _DEA
        except Exception:
            try:
                from torch_geometric.data import DataEdgeAttr as _DEA
                dea = _DEA
            except Exception:
                pass
        if dea is not None:
            avail.append(dea)

        # Register whatever we actually found
        if avail:
            add_safe_globals(avail)

        # ---- Temporarily restore legacy unpickling (PyTorch 2.6+) ----
        import torch
        import torch.serialization as _ser
        _old_torch_load = torch.load
        _old_ser_load = _ser.load

        def _load_legacy(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _old_ser_load(*args, **kwargs)

        torch.load = _load_legacy
        _ser.load = _load_legacy

        # ---- Build dataset, retry after cache wipe if needed ----
        try:
            try:
                with safe_globals(avail):
                    dataset = PygNodePropPredDataset(
                        name='ogbn-arxiv', root=root)
            except Exception as e:
                msg = str(e)
                if "Weights only load failed" in msg or "Unsupported global" in msg:
                    import shutil
                    import os
                    proc_dir = os.path.join(
                        root, "ogbn-arxiv", "processed")  # NOTE: hyphen
                    shutil.rmtree(proc_dir, ignore_errors=True)
                    with safe_globals(avail):
                        dataset = PygNodePropPredDataset(
                            name='ogbn-arxiv', root=root)
                else:
                    raise
        finally:
            # always restore
            torch.load = _old_torch_load
            _ser.load = _old_ser_load

        # ---- Build Data object & masks ----
        data = dataset[0]
        if data.y.dim() == 2 and data.y.size(1) == 1:
            data.y = data.y.view(-1)

        split_idx = dataset.get_idx_split()
        n_nodes = data.x.size(0)
        data.train_mask = self._index_to_mask(split_idx["train"], n_nodes)
        data.val_mask = self._index_to_mask(split_idx["valid"], n_nodes)
        data.test_mask = self._index_to_mask(split_idx["test"],  n_nodes)

        # OGB arxiv is directed; many GCNs expect undirected
        data.edge_index = to_undirected(data.edge_index, num_nodes=n_nodes)

        data.num_features = data.x.size(-1)
        data.num_classes = int(data.y.max().item() + 1)

        data = data.to(self.device)
        data.adj_t = to_torch_sparse_tensor(data.edge_index)

        self.data = data
        return self.data

    def load_reddit(self, root):
        dataset = Reddit(root=root)
        data = dataset[0]

        # Ensure masks exist in bool
        data.train_mask = data.train_mask.bool()
        data.val_mask = data.val_mask.bool()
        data.test_mask = data.test_mask.bool()

        data.num_features = data.x.size(-1)
        # Reddit y is [N] with classes 0..40
        data.num_classes = int(data.y.max().item() + 1)
        data = data.to(self.device)
        data.adj_t = to_torch_sparse_tensor(data.edge_index)
        self.data = data

        return self.data

    def load_dataset(self):
        root = f'./dataset/{self.dataset_name}'

        if self.dataset_name in ["Cora", "Citeseer", "Pubmed"]:
            dataset = Planetoid(
                root=f'./dataset/{self.dataset_name}', name=self.dataset_name)
            self.data = dataset[0].to(self.device)
            self.data.adj_t = to_torch_sparse_tensor(self.data.edge_index)

            return self.data

        if self.dataset_name in ["arxiv", "ogbn-arxiv", "ogb-arxiv", "arXiv"]:
            return self.load_ogbn_arxiv(root=root)
        if self.dataset_name in ["reddit"]:
            return self.load_reddit(root=root)
        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def explicit_inductive_split(self, train_ratio=0.4, val_ratio=0.4):
        num_nodes = self.data.num_nodes
        perm = torch.randperm(num_nodes)

        train_idx = perm[:int(train_ratio * num_nodes)]
        val_idx = perm[int(train_ratio * num_nodes)
                           :int((train_ratio + val_ratio) * num_nodes)]
        test_idx = perm[int((train_ratio + val_ratio) * num_nodes):]

        self.data.train_mask = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device)
        self.data.val_mask = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device)
        self.data.test_mask = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device)

        self.data.train_mask[train_idx] = True
        self.data.val_mask[val_idx] = True
        self.data.test_mask[test_idx] = True

        print(f"Explicitly performed inductive split (Train: {self.data.train_mask.sum().item()}, "
              f"Val: {self.data.val_mask.sum().item()}, Test: {self.data.test_mask.sum().item()})")
        return self.data

    def load_and_save(self):
        self.load_dataset()
        # self.explicit_inductive_split()
        os.makedirs('./dataset', exist_ok=True)
        torch.save(
            self.data, f'./dataset/{self.dataset_name.lower()}_data.pt')
        print(f"{self.dataset_name} dataset explicitly saved with inductive splits at "
              f"'./dataset/{self.dataset_name.lower()}_data.pt'")
        return self.data

    def load_saved(self):
        load_path = f'./dataset/{self.dataset_name.lower()}_data.pt'
        self.data = torch.load(load_path).to(self.device)
        print(f"{self.dataset_name} dataset explicitly loaded from {load_path}")
        return self.data
