"""
Script to run a custom experiment command multiple times and aggregate statistics.

Usage:
    python run_custom_experiment.py

Or modify the cmd variable below to match your desired command.
"""

import sys
from pathlib import Path

# Add parent directory to path to import run_experiments functions
sys.path.insert(0, str(Path(__file__).parent))

from run_experiments import (
    run_custom_experiment_multiple_times,
    print_ablation_summary
)
import json
import datetime


def main():
    # Modify this command to match your experiment
    cmd = [
        'python', 'main_aga.py',
        '--dataset', 'Cora',
        '--model', 'GCN',
        '--edge_sparsity', '0.7',
        '--weight_sparsity', '0.1',
        '--compare_static',
        '--epochs', '200'
    ]
    
    # Create results directory
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path("results") / f"custom_experiment_edge0.7_weight0.1_{time_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print("CUSTOM EXPERIMENT: Running 5 times")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'#'*80}\n")
    
    # Run experiment 5 times
    summary, all_results = run_custom_experiment_multiple_times(
        cmd, 
        n_runs=5, 
        dataset="Cora"
    )
    
    if summary is not None:
        # Save results
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(results_dir / "all_runs.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Also save a readable text summary
        with open(results_dir / "summary_table.txt", "w") as f:
            f.write("="*90 + "\n")
            f.write("ABLATION STUDY SUMMARY (5 runs - Mean ± Std)\n")
            f.write("="*90 + "\n")
            f.write(f"{'Method':<30} {'Dense Val':<15} {'Dense Test':<15} {'Pruned Val':<15} {'Pruned Test':<15} {'Speedup':<12}\n")
            f.write("-"*90 + "\n")
            
            for method, metrics in summary.items():
                dense_val = f"{metrics['dense_val']['mean']:.4f}±{metrics['dense_val']['std']:.4f}"
                dense_test = f"{metrics['dense_test']['mean']:.4f}±{metrics['dense_test']['std']:.4f}"
                pruned_val = f"{metrics['pruned_val']['mean']:.4f}±{metrics['pruned_val']['std']:.4f}"
                pruned_test = f"{metrics['pruned_test']['mean']:.4f}±{metrics['pruned_test']['std']:.4f}"
                speedup = f"{metrics['speedup']['mean']:.2f}±{metrics['speedup']['std']:.2f}x"
                
                f.write(f"{method:<30} {dense_val:<15} {dense_test:<15} {pruned_val:<15} {pruned_test:<15} {speedup:<12}\n")
        
        print(f"\n{'#'*80}")
        print(f"Results saved to: {results_dir}")
        print(f"  - summary.json: Aggregated statistics")
        print(f"  - all_runs.json: All individual run results")
        print(f"  - summary_table.txt: Formatted summary table")
        print(f"{'#'*80}\n")
    else:
        print("\n✗ ERROR: Failed to generate summary!")


if __name__ == "__main__":
    main()
