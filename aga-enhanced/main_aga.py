"""
Example Main File for FG-GLT with Adaptive Gradient Alignment

This demonstrates how to use the enhanced trainer with:
1. Adaptive gradient caching
2. Distribution shift detection
3. Automatic cache refresh
4. Selective parameter alignment

For ICML 2026 submission.

Author: Saeedeh (RMIT University)
"""

import torch
from pathlib import Path
import json
import datetime
import argparse

# Import your existing modules
from data_handler import DataHandler
from models import DenseGCN, DenseGAT, DenseGIN

# Import enhanced trainer with AGA
from enhanced_trainer import EnhancedDenseModelTrainer
from adaptive_gradient_alignment import AdaptiveAlignmentConfig, AlignmentMetric


def parse_args():
    parser = argparse.ArgumentParser(description='FG-GLT with Adaptive Gradient Alignment')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv'])
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'GIN'])
    
    # Sparsity targets
    parser.add_argument('--edge_sparsity', type=float, default=0.5)
    parser.add_argument('--weight_sparsity', type=float, default=0.5)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=512)
    
    # Gradient alignment
    parser.add_argument('--lambda_gm', type=float, default=0.2,
                        help='Gradient matching weight')
    parser.add_argument('--lambda_sp', type=float, default=2e-4,
                        help='Sparsity regularization weight')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Concrete temperature')
    
    # Adaptive Gradient Alignment (NEW)
    parser.add_argument('--use_adaptive', action='store_true', default=True,
                        help='Enable Adaptive Gradient Alignment')
    parser.add_argument('--kl_threshold', type=float, default=0.3,
                        help='Cosine distance threshold for cache refresh (0.3 = 0.7 similarity)')
    parser.add_argument('--top_k_ratio', type=float, default=0.5,
                        help='Fraction of parameters to align')
    parser.add_argument('--min_refresh_interval', type=int, default=20,
                        help='Minimum epochs between cache refreshes')
    parser.add_argument('--max_cache_age', type=int, default=50,
                        help='Maximum cache age before forced refresh')
    parser.add_argument('--use_low_rank', action='store_true', default=False,
                        help='Enable low-rank gradient compression')
    parser.add_argument('--low_rank_ratio', type=float, default=0.2,
                        help='Fraction of singular values to keep')
    
    # Retraining after pruning
    parser.add_argument('--retrain_after_pruning', action='store_true', default=False,
                        help='Retrain model on hard-pruned graph after pruning')
    parser.add_argument('--retrain_epochs', type=int, default=50,
                        help='Number of epochs for retraining')
    parser.add_argument('--retrain_lr', type=float, default=0.01,
                        help='Learning rate for retraining')
    
    # Ablation
    parser.add_argument('--compare_static', action='store_true', default=False,
                        help='Run comparison with static gradient matching')
    
    return parser.parse_args()


def create_model(model_name: str, num_features: int, num_classes: int, hidden: int = 512):
    """Create GNN model based on name"""
    if model_name == 'GCN':
        return DenseGCN(
            in_channels=num_features,
            hidden_channels=hidden,
            out_channels=num_classes
        )
    elif model_name == 'GAT':
        return DenseGAT(
            in_channels=num_features,
            hidden_channels=64,
            out_channels=num_classes,
            heads=8
        )
    elif model_name == 'GIN':
        return DenseGIN(
            in_channels=num_features,
            hidden_channels=hidden,
            out_channels=num_classes,
            n_layers=2,
            dropout=0.5,
            train_eps=True
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(args):
    """Run a single experiment with specified configuration"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    print(f"Edge Sparsity: {args.edge_sparsity}, Weight Sparsity: {args.weight_sparsity}")
    print(f"Adaptive Gradient Alignment: {'Enabled' if args.use_adaptive else 'Disabled'}")
    print(f"{'='*60}\n")
    
    data_handler = DataHandler(args.dataset, device)
    data = data_handler.load_and_save()
    
    # Create model
    model = create_model(
        args.model,
        num_features=data.num_features,
        num_classes=data.y.max().item() + 1,
        hidden=args.hidden
    ).to(device)
    
    # Create trainer
    trainer = EnhancedDenseModelTrainer(
        model, data, args.dataset, device=device, lr=args.lr
    )
    
    # Phase 1: Dense pretraining
    print("\n--- Phase 1: Dense Pretraining ---")
    trainer.train(epochs=200)
    trainer.save_model()
    trainer.compute_and_save_gradients()
    
    # Setup output directory
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("results") / f"{args.dataset}_{args.model}_AGA_{time_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer.run_dir = str(run_dir)
    
    # Dense gradient path
    grad_path = f'./dense_gradients_GCN/{args.dataset.lower()}_dense_grads.pth'
    
    # Phase 2: Sparse training with Adaptive Gradient Alignment
    print("\n--- Phase 2: Sparse Training with Adaptive Gradient Alignment ---")
    
    if args.use_adaptive:
        # Configure Adaptive Gradient Alignment
        aga_config = AdaptiveAlignmentConfig(
            kl_threshold=args.kl_threshold,
            top_k_ratio=args.top_k_ratio,
            layer_adaptive=True,
            use_low_rank=args.use_low_rank,
            low_rank_ratio=args.low_rank_ratio,
            min_refresh_interval=args.min_refresh_interval,
            max_cache_age=args.max_cache_age,
            alignment_metric=AlignmentMetric.COSINE,
            alignment_weight_decay=0.99
        )
        
        results = trainer.train_sparse_with_adaptive_grad_matching(
            dense_grads_path=grad_path,
            aga_config=aga_config,
            alpha=args.lambda_gm,
            beta=args.lambda_sp,
            tau=args.tau,
            epochs=args.epochs,
            target_edge_sparsity=args.edge_sparsity,
            target_weight_sparsity=args.weight_sparsity,
            project_every=10,
            use_adaptive=True,
            retrain_after_pruning=args.retrain_after_pruning,
            retrain_epochs=args.retrain_epochs,
            retrain_lr=args.retrain_lr
        )
    else:
        # Fallback to static gradient matching
        results = trainer.train_sparse_with_grad_matching(
            dense_grads_path=grad_path,
            alpha=args.lambda_gm,
            beta=args.lambda_sp,
            tau=args.tau,
            epochs=args.epochs,
            target_edge_sparsity=args.edge_sparsity,
            target_weight_sparsity=args.weight_sparsity,
            project_every=10,
            retrain_after_pruning=args.retrain_after_pruning,
            retrain_epochs=args.retrain_epochs,
            retrain_lr=args.retrain_lr
        )
    
    # Save experiment config
    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment completed. Results saved to: {run_dir}")
    print(f"{'='*60}")
    
    return results


def run_ablation_study(args):
    """
    Run ablation study comparing:
    1. Static gradient matching (original)
    2. Adaptive gradient matching (proposed)
    3. Different AGA configurations
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data once
    data_handler = DataHandler(args.dataset, device)
    data = data_handler.load_and_save()
    
    results_all = []
    dense_val_acc = None
    dense_test_acc = None
    
    # Configuration 1: Static (baseline)
    print("\n" + "="*60)
    print("ABLATION: Static Gradient Matching (Baseline)")
    print("="*60)
    
    model = create_model('GCN', data.num_features, data.y.max().item() + 1)
    trainer = EnhancedDenseModelTrainer(model, data, args.dataset, device=device)
    trainer.train(epochs=200)
    
    # Evaluate dense model after training (before pruning)
    dense_val_acc = trainer.evaluate(data.val_mask)
    dense_test_acc = trainer.evaluate(data.test_mask)
    print(f"Dense Model (before pruning) - Val Acc: {dense_val_acc:.4f}, Test Acc: {dense_test_acc:.4f}")
    
    trainer.compute_and_save_gradients()
    grad_path = f'./dense_gradients_GCN/{args.dataset.lower()}_dense_grads.pth'
    
    res_static = trainer.train_sparse_with_grad_matching(
        dense_grads_path=grad_path,
        alpha=args.lambda_gm,
        beta=args.lambda_sp,
        epochs=args.epochs,
        target_edge_sparsity=args.edge_sparsity,
        target_weight_sparsity=args.weight_sparsity,
        retrain_after_pruning=args.retrain_after_pruning,
        retrain_epochs=args.retrain_epochs,
        retrain_lr=args.retrain_lr
    )
    res_static["method"] = "static"
    res_static["dense_val_acc"] = float(dense_val_acc)
    res_static["dense_test_acc"] = float(dense_test_acc)
    results_all.append(res_static)
    
    # Configuration 2: Adaptive with default settings
    print("\n" + "="*60)
    print("ABLATION: Adaptive Gradient Alignment (Default)")
    print("="*60)
    
    model = create_model('GCN', data.num_features, data.y.max().item() + 1)
    trainer = EnhancedDenseModelTrainer(model, data, args.dataset, device=device)
    trainer.train(epochs=200)
    
    # Evaluate dense model after training (before pruning)
    dense_val_acc_adaptive = trainer.evaluate(data.val_mask)
    dense_test_acc_adaptive = trainer.evaluate(data.test_mask)
    print(f"Dense Model (before pruning) - Val Acc: {dense_val_acc_adaptive:.4f}, Test Acc: {dense_test_acc_adaptive:.4f}")
    
    aga_config = AdaptiveAlignmentConfig(
        kl_threshold=0.1,
        top_k_ratio=0.5,
        layer_adaptive=True
    )
    
    res_adaptive = trainer.train_sparse_with_adaptive_grad_matching(
        dense_grads_path=grad_path,
        aga_config=aga_config,
        alpha=args.lambda_gm,
        beta=args.lambda_sp,
        epochs=args.epochs,
        target_edge_sparsity=args.edge_sparsity,
        target_weight_sparsity=args.weight_sparsity,
        use_adaptive=True,
        retrain_after_pruning=args.retrain_after_pruning,
        retrain_epochs=args.retrain_epochs,
        retrain_lr=args.retrain_lr
    )
    res_adaptive["method"] = "adaptive_default"
    res_adaptive["dense_val_acc"] = float(dense_val_acc_adaptive)
    res_adaptive["dense_test_acc"] = float(dense_test_acc_adaptive)
    results_all.append(res_adaptive)
    
    # Configuration 3: Adaptive with aggressive refresh
    print("\n" + "="*60)
    print("ABLATION: Adaptive with Aggressive Refresh")
    print("="*60)
    
    model = create_model('GCN', data.num_features, data.y.max().item() + 1)
    trainer = EnhancedDenseModelTrainer(model, data, args.dataset, device=device)
    trainer.train(epochs=200)
    
    # Evaluate dense model after training (before pruning)
    dense_val_acc_aggressive = trainer.evaluate(data.val_mask)
    dense_test_acc_aggressive = trainer.evaluate(data.test_mask)
    print(f"Dense Model (before pruning) - Val Acc: {dense_val_acc_aggressive:.4f}, Test Acc: {dense_test_acc_aggressive:.4f}")
    
    aga_config_aggressive = AdaptiveAlignmentConfig(
        kl_threshold=0.05,  # Lower threshold = more frequent refresh
        top_k_ratio=0.7,
        min_refresh_interval=3,
        layer_adaptive=True
    )
    
    res_aggressive = trainer.train_sparse_with_adaptive_grad_matching(
        dense_grads_path=grad_path,
        aga_config=aga_config_aggressive,
        alpha=args.lambda_gm,
        beta=args.lambda_sp,
        epochs=args.epochs,
        target_edge_sparsity=args.edge_sparsity,
        target_weight_sparsity=args.weight_sparsity,
        use_adaptive=True,
        retrain_after_pruning=args.retrain_after_pruning,
        retrain_epochs=args.retrain_epochs,
        retrain_lr=args.retrain_lr
    )
    res_aggressive["method"] = "adaptive_aggressive"
    res_aggressive["dense_val_acc"] = float(dense_val_acc_aggressive)
    res_aggressive["dense_test_acc"] = float(dense_test_acc_aggressive)
    results_all.append(res_aggressive)
    
    # Configuration 4: Adaptive with low-rank compression
    print("\n" + "="*60)
    print("ABLATION: Adaptive with Low-Rank Compression")
    print("="*60)
    
    model = create_model('GCN', data.num_features, data.y.max().item() + 1)
    trainer = EnhancedDenseModelTrainer(model, data, args.dataset, device=device)
    trainer.train(epochs=200)
    
    # Evaluate dense model after training (before pruning)
    dense_val_acc_lowrank = trainer.evaluate(data.val_mask)
    dense_test_acc_lowrank = trainer.evaluate(data.test_mask)
    print(f"Dense Model (before pruning) - Val Acc: {dense_val_acc_lowrank:.4f}, Test Acc: {dense_test_acc_lowrank:.4f}")
    
    aga_config_lowrank = AdaptiveAlignmentConfig(
        kl_threshold=0.1,
        top_k_ratio=0.5,
        use_low_rank=True,
        low_rank_ratio=0.2,
        layer_adaptive=True
    )
    
    res_lowrank = trainer.train_sparse_with_adaptive_grad_matching(
        dense_grads_path=grad_path,
        aga_config=aga_config_lowrank,
        alpha=args.lambda_gm,
        beta=args.lambda_sp,
        epochs=args.epochs,
        target_edge_sparsity=args.edge_sparsity,
        target_weight_sparsity=args.weight_sparsity,
        use_adaptive=True,
        retrain_after_pruning=args.retrain_after_pruning,
        retrain_epochs=args.retrain_epochs,
        retrain_lr=args.retrain_lr
    )
    res_lowrank["method"] = "adaptive_lowrank"
    res_lowrank["dense_val_acc"] = float(dense_val_acc_lowrank)
    res_lowrank["dense_test_acc"] = float(dense_test_acc_lowrank)
    results_all.append(res_lowrank)
    
    # Print summary
    print("\n" + "="*90)
    print("ABLATION STUDY SUMMARY")
    print("="*90)
    print(f"{'Method':<30} {'Dense Val':<12} {'Dense Test':<12} {'Pruned Val':<12} {'Pruned Test':<12} {'Speedup':<10}")
    print("-"*90)
    
    for r in results_all:
        method = r.get("method", "unknown")
        dense_val = r.get("dense_val_acc", 0)
        dense_test = r.get("dense_test_acc", 0)
        val_acc = r.get("best_val", r.get("Best Val after sparse+GM", 0))
        test_acc = r.get("test_acc_pruned", 0)
        speedup = r.get("speedup_hard", 1.0)
        
        print(f"{method:<30} {dense_val:.4f}      {dense_test:.4f}      {val_acc:.4f}      {test_acc:.4f}      {speedup:.2f}x")
    
    # Save ablation results
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ablation_dir = Path("results") / f"ablation_{args.dataset}_{time_tag}"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    with open(ablation_dir / "ablation_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    
    print(f"\nAblation results saved to: {ablation_dir}")
    
    return results_all


def run_sparsity_sweep(args):
    """
    Run sweep over different sparsity levels with AGA.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_handler = DataHandler(args.dataset, device)
    data = data_handler.load_and_save()
    
    # Sparsity configurations
    edge_sparsities = [0.3, 0.5, 0.7]
    weight_sparsities = [0.3, 0.5, 0.7]
    
    results_all = []
    
    for e_s in edge_sparsities:
        for w_s in weight_sparsities:
            print(f"\n{'='*60}")
            print(f"Sweep: Edge Sparsity={e_s}, Weight Sparsity={w_s}")
            print(f"{'='*60}")
            
            model = create_model('GCN', data.num_features, data.y.max().item() + 1)
            trainer = EnhancedDenseModelTrainer(model, data, args.dataset, device=device)
            trainer.train(epochs=200)
            trainer.compute_and_save_gradients()
            
            grad_path = f'./dense_gradients_GCN/{args.dataset.lower()}_dense_grads.pth'
            
            aga_config = AdaptiveAlignmentConfig(
                kl_threshold=0.1,
                top_k_ratio=0.5,
                layer_adaptive=True
            )
            
            res = trainer.train_sparse_with_adaptive_grad_matching(
                dense_grads_path=grad_path,
                aga_config=aga_config,
                alpha=args.lambda_gm,
                beta=args.lambda_sp,
                epochs=args.epochs,
                target_edge_sparsity=e_s,
                target_weight_sparsity=w_s,
                use_adaptive=True,
                retrain_after_pruning=args.retrain_after_pruning,
                retrain_epochs=args.retrain_epochs,
                retrain_lr=args.retrain_lr
            )
            
            res["edge_sparsity_target"] = e_s
            res["weight_sparsity_target"] = w_s
            results_all.append(res)
    
    # Save sweep results
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = Path("results") / f"sweep_{args.dataset}_{time_tag}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    with open(sweep_dir / "sweep_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    
    print(f"\nSweep results saved to: {sweep_dir}")
    
    return results_all


def main():
    args = parse_args()
    
    if args.compare_static:
        # Run ablation study
        run_ablation_study(args)
    else:
        # Run single experiment
        run_single_experiment(args)


if __name__ == "__main__":
    main()
