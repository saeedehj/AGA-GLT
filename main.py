import torch
from data_handler import DataHandler
from trainer import DenseModelTrainer
from models import DenseGCN, DenseGAT, DenseGIN
from pathlib import Path
import json
import csv
import datetime


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Explicitly using device: {device}')

    datasets = ['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv']

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name} dataset:")

        data_handler = DataHandler(dataset_name, device)
        data = data_handler.load_and_save()

        # Train and save dense model
        model = DenseGCN(in_channels=data.num_features, hidden_channels=512,
                         out_channels=data.y.max().item() + 1)

        # model = DenseGIN(
        #     in_channels=data.num_features,
        #     hidden_channels=512,
        #     out_channels=data.y.max().item() + 1,
        #     n_layers=2,
        #     dropout=0.5,
        #     train_eps=True
        # ).to(device)

        # model = DenseGAT(
        #     in_channels=data.num_features,
        #     hidden_channels=64,
        #     out_channels=data.y.max().item() + 1,
        #     heads=8
        # )

        trainer = DenseModelTrainer(model, data, dataset_name, device=device)
        trainer.train(epochs=200)
        dense_avg_ms, dense_std_ms = trainer.measure_inference_time()
        print(
            f"[Dense] {dataset_name}: {dense_avg_ms:.2f} ± {dense_std_ms:.2f} ms")

        trainer.save_model()
        trainer.compute_and_save_gradients()

        grad_path = f'./dense_gradients_GCN/{dataset_name.lower()}_dense_grads.pth'

        # ---------- SWEEP SPEC ----------
        weight_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        edge_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        sweep_root = Path("results") / f"{dataset_name}_sweep_{time_tag}"
        sweep_root.mkdir(parents=True, exist_ok=True)

        all_rows = []

        for e_s in edge_list:
            for w_s in weight_list:
                run_name = f"edge{e_s}_wt{w_s}"
                run_dir = sweep_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[SWEEP] {run_name}")

                # Pass per-run folder to trainer so it writes metrics/timing there
                setattr(trainer, "run_dir", str(run_dir))

                res = trainer.train_sparse_with_grad_matching(
                    dense_grads_path=grad_path,
                    alpha=0.2, beta=2e-4,
                    epochs=200,
                    pruner_lr=1e-2,
                    target_edge_sparsity=e_s,      # 0.9 to keep 10% edges
                    target_weight_sparsity=w_s,    # 0.9 to keep 10% weights
                    project_every=1,
                    lc_lambda=2e-3,                # try 1e-3 to 5e-3; 0 disables
                    hetero=False
                )
                # Optional mirrored single-file summary for convenience
                with open(run_dir / "summary.json", "w") as f:
                    json.dump(res, f, indent=2)

                all_rows.append(res)

        # ---------- AGGREGATED OUTPUTS ----------
        agg_jsonl = sweep_root / "ALL_RUNS.jsonl"
        with open(agg_jsonl, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[SWEEP] Aggregated JSONL -> {agg_jsonl}")

        csv_cols = [
            "dataset", "device",
            "target_edge_sparsity", "target_weight_sparsity",
            "nodes", "edges_total", "edges_kept_hard", "edge_sparsity_hard", "weights_total", "weights_kept_hard", "weight_sparsity_hard",
            "Best Val after sparse+GM",
            "test_acc_pruned",
            "dense_ms_avg", "dense_ms_std", "soft_ms_avg", "soft_ms_std",
            "hard_ms_avg", "hard_ms_std", "speedup_soft", "speedup_hard",
        ]
        agg_csv = sweep_root / "ALL_RUNS.csv"
        with open(agg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, None) for k in csv_cols})
        print(f"[SWEEP] Aggregated CSV   -> {agg_csv}")


if __name__ == "__main__":
    main()
