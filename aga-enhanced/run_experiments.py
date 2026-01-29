"""
Script to run comprehensive experiments:
1. Run ablation study (compare_static) 5 times with statistics (edge=0.1, weight=0.1)
2. Edge sparsity sweep: edge from 0.1 to 0.8 (step 0.1), weight fixed at 0.1
   Each configuration runs 5 times: 8 configurations × 5 runs = 40 runs
3. Weight sparsity sweep: weight from 0.1 to 0.8 (step 0.1), edge fixed at 0.1
   Each configuration runs 5 times: 8 configurations × 5 runs = 40 runs
"""

import subprocess
import json
import numpy as np
import datetime
from pathlib import Path
from collections import defaultdict
import statistics
import time
import glob


def run_command(cmd_list, run_id=None):
    """Run a command and return success status"""
    if run_id is not None:
        print(f"\n{'='*80}")
        print(f"Run {run_id + 1}/5")
        print(f"{'='*80}")
    
    print(f"Running: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    return result.stdout


def get_existing_ablation_dirs(dataset="Cora"):
    """Get list of existing ablation result directories"""
    pattern = f"results/ablation_{dataset}_*"
    dirs = [Path(d) for d in glob.glob(pattern) if Path(d).is_dir()]
    return set(dirs)


def find_new_ablation_results(existing_dirs, dataset="Cora", max_wait=30):
    """Find newly created ablation results JSON file"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        pattern = f"results/ablation_{dataset}_*/ablation_results.json"
        files = glob.glob(pattern)
        
        # Find files in directories that weren't in the existing set
        for file_path in files:
            file_dir = Path(file_path).parent
            if file_dir not in existing_dirs:
                # Verify file is complete (not being written)
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)  # Try to parse to ensure it's complete
                    return file_path
                except (json.JSONDecodeError, IOError):
                    continue  # File still being written
        
        time.sleep(1)  # Wait 1 second before checking again
    
    # Fallback: return most recent if no new one found
    if files:
        latest = max(files, key=lambda x: Path(x).stat().st_mtime)
        return latest
    
    return None


def load_ablation_results(json_path):
    """Load ablation results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def aggregate_ablation_results(all_runs):
    """Aggregate results from multiple runs"""
    # Group by method
    method_results = defaultdict(list)
    
    for run_results in all_runs:
        for result in run_results:
            method = result.get("method", "unknown")
            method_results[method].append(result)
    
    # Compute statistics
    summary = {}
    for method, runs in method_results.items():
        if len(runs) == 0:
            continue
        
        dense_vals = [r.get("dense_val_acc", 0) for r in runs]
        dense_tests = [r.get("dense_test_acc", 0) for r in runs]
        pruned_vals = [r.get("best_val", r.get("Best Val after sparse+GM", 0)) for r in runs]
        pruned_tests = [r.get("test_acc_pruned", 0) for r in runs]
        speedups = [r.get("speedup_hard", 1.0) for r in runs]
        
        summary[method] = {
            'dense_val': {
                'mean': np.mean(dense_vals),
                'std': np.std(dense_vals)
            },
            'dense_test': {
                'mean': np.mean(dense_tests),
                'std': np.std(dense_tests)
            },
            'pruned_val': {
                'mean': np.mean(pruned_vals),
                'std': np.std(pruned_vals)
            },
            'pruned_test': {
                'mean': np.mean(pruned_tests),
                'std': np.std(pruned_tests)
            },
            'speedup': {
                'mean': np.mean(speedups),
                'std': np.std(speedups)
            }
        }
    
    return summary


def run_ablation_study_multiple_times(base_cmd, n_runs=5, dataset="Cora"):
    """Run ablation study n times and compute statistics"""
    print(f"\n{'#'*80}")
    print(f"Running Ablation Study {n_runs} times")
    print(f"{'#'*80}\n")
    
    all_results = []
    
    # Get existing directories before starting
    existing_dirs = get_existing_ablation_dirs(dataset)
    
    for run_id in range(n_runs):
        print(f"\n{'='*80}")
        print(f"Run {run_id + 1}/{n_runs}")
        print(f"{'='*80}\n")
        
        # Get directories before this run
        dirs_before = get_existing_ablation_dirs(dataset)
        
        # Run the command
        output = run_command(base_cmd, run_id=run_id)
        
        if output is None:
            print(f"Warning: Run {run_id + 1} command failed")
            continue
        
        # Find the newly created results file
        results_file = find_new_ablation_results(dirs_before, dataset, max_wait=60)
        
        if results_file:
            try:
                results = load_ablation_results(results_file)
                all_results.append(results)
                print(f"✓ Loaded results from: {results_file}")
            except Exception as e:
                print(f"✗ Error loading results from {results_file}: {e}")
        else:
            print(f"✗ Warning: Could not find results file for run {run_id + 1}")
            print("   This run will be skipped in the final statistics")
    
    if len(all_results) == 0:
        print("\n✗ ERROR: No results were successfully loaded!")
        return None, []
    
    # Aggregate results
    summary = aggregate_ablation_results(all_results)
    
    print(f"\n✓ Successfully processed {len(all_results)}/{n_runs} runs")
    
    return summary, all_results


def print_ablation_summary(summary):
    """Print formatted ablation study summary"""
    print(f"\n{'='*90}")
    print("ABLATION STUDY SUMMARY (5 runs - Mean ± Std)")
    print(f"{'='*90}")
    print(f"{'Method':<30} {'Dense Val':<15} {'Dense Test':<15} {'Pruned Val':<15} {'Pruned Test':<15} {'Speedup':<12}")
    print("-"*90)
    
    for method, metrics in summary.items():
        dense_val = f"{metrics['dense_val']['mean']:.4f}±{metrics['dense_val']['std']:.4f}"
        dense_test = f"{metrics['dense_test']['mean']:.4f}±{metrics['dense_test']['std']:.4f}"
        pruned_val = f"{metrics['pruned_val']['mean']:.4f}±{metrics['pruned_val']['std']:.4f}"
        pruned_test = f"{metrics['pruned_test']['mean']:.4f}±{metrics['pruned_test']['std']:.4f}"
        speedup = f"{metrics['speedup']['mean']:.2f}±{metrics['speedup']['std']:.2f}x"
        
        print(f"{method:<30} {dense_val:<15} {dense_test:<15} {pruned_val:<15} {pruned_test:<15} {speedup:<12}")


def get_existing_experiment_dirs(dataset="Cora", model="GCN"):
    """Get list of existing experiment result directories"""
    pattern = f"results/{dataset}_{model}_AGA_*"
    dirs = [Path(d) for d in glob.glob(pattern) if Path(d).is_dir()]
    return set(dirs)


def find_new_experiment_results(existing_dirs, dataset="Cora", model="GCN", max_wait=30):
    """Find newly created experiment results JSON file"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        pattern = f"results/{dataset}_{model}_AGA_*/metrics.json"
        files = glob.glob(pattern)
        
        # Find files in directories that weren't in the existing set
        for file_path in files:
            file_dir = Path(file_path).parent
            if file_dir not in existing_dirs:
                # Verify file is complete (not being written)
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)  # Try to parse to ensure it's complete
                    return file_path
                except (json.JSONDecodeError, IOError):
                    continue  # File still being written
        
        time.sleep(1)  # Wait 1 second before checking again
    
    # Fallback: return most recent if no new one found
    if files:
        latest = max(files, key=lambda x: Path(x).stat().st_mtime)
        return latest
    
    return None


def load_experiment_results(json_path):
    """Load experiment results from metrics.json"""
    with open(json_path, 'r') as f:
        return json.load(f)


def run_sparsity_sweep(base_cmd, sparsity_type, fixed_value, sweep_range, n_runs=5):
    """
    Run sparsity sweep
    
    Args:
        base_cmd: Base command list
        sparsity_type: 'edge' or 'weight'
        fixed_value: Fixed sparsity value (0.1)
        sweep_range: List of values to sweep [0.1, 0.2, ..., 0.8]
        n_runs: Number of runs per configuration (default 5)
    """
    print(f"\n{'#'*80}")
    print(f"Running {sparsity_type.upper()} Sparsity Sweep")
    print(f"Fixed {('weight' if sparsity_type == 'edge' else 'edge')}_sparsity = {fixed_value}")
    print(f"Sweeping {sparsity_type}_sparsity: {sweep_range}")
    print(f"Runs per configuration: {n_runs}")
    print(f"{'#'*80}\n")
    
    # Print the sequence clearly
    print("Sequence:")
    for i, sparsity_val in enumerate(sweep_range):
        if sparsity_type == 'edge':
            print(f"  {i+1}. edge={sparsity_val}, weight={fixed_value} ({n_runs} runs)")
        else:
            print(f"  {i+1}. edge={fixed_value}, weight={sparsity_val} ({n_runs} runs)")
    print()
    
    all_results = []
    
    for idx, sparsity_val in enumerate(sweep_range):
        cmd = base_cmd.copy()
        
        if sparsity_type == 'edge':
            cmd.extend(['--edge_sparsity', str(sparsity_val)])
            cmd.extend(['--weight_sparsity', str(fixed_value)])
            config_desc = f"edge={sparsity_val}, weight={fixed_value}"
        else:
            cmd.extend(['--edge_sparsity', str(fixed_value)])
            cmd.extend(['--weight_sparsity', str(sparsity_val)])
            config_desc = f"edge={fixed_value}, weight={sparsity_val}"
        
        config_results = []
        
        # Get existing directories before this configuration
        dirs_before_config = get_existing_experiment_dirs()
        
        for run_id in range(n_runs):
            print(f"\n{'='*60}")
            print(f"Configuration {idx+1}/{len(sweep_range)}: {config_desc} (Run {run_id + 1}/{n_runs})")
            print(f"{'='*60}")
            
            # Get directories before this run
            dirs_before_run = get_existing_experiment_dirs()
            
            output = run_command(cmd)
            
            if output is None:
                print(f"Warning: Command failed for {sparsity_type}={sparsity_val}, run {run_id + 1}")
                continue
            
            # Find the newly created results file
            results_file = find_new_experiment_results(dirs_before_run, max_wait=60)
            
            if results_file:
                try:
                    results = load_experiment_results(results_file)
                    results['sparsity_type'] = sparsity_type
                    results['sparsity_value'] = sparsity_val
                    results['fixed_value'] = fixed_value
                    results['run_id'] = run_id
                    config_results.append(results)
                    print(f"✓ Loaded results from: {results_file}")
                except Exception as e:
                    print(f"✗ Error loading results: {e}")
            else:
                print(f"✗ Warning: Could not find results file for {sparsity_type}={sparsity_val}, run {run_id + 1}")
        
        all_results.append({
            'sparsity_type': sparsity_type,
            'sparsity_value': sparsity_val,
            'fixed_value': fixed_value,
            'runs': config_results
        })
    
    return all_results


def print_sweep_summary(sweep_results, metric_key='test_acc_pruned'):
    """Print formatted sweep summary"""
    print(f"\n{'='*80}")
    print(f"{sweep_results[0]['sparsity_type'].upper()} SPARSITY SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Sparsity':<12} {'Mean':<12} {'Std':<12}")
    print("-"*80)
    
    for config in sweep_results:
        sparsity = config['sparsity_value']
        values = [r.get(metric_key, 0) for r in config['runs']]
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{sparsity:<12.1f} {mean_val:<12.4f} {std_val:<12.4f}")


def run_custom_experiment_multiple_times(cmd_list, n_runs=5, dataset="Cora"):
    """
    Run a custom command multiple times and aggregate statistics
    
    Args:
        cmd_list: List of command arguments (e.g., ['python', 'main_aga.py', '--dataset', 'Cora', ...])
        n_runs: Number of runs (default 5)
        dataset: Dataset name for finding result files (default "Cora")
    
    Returns:
        summary: Aggregated statistics dictionary
        all_results: List of all individual run results
    """
    print(f"\n{'#'*80}")
    print(f"Running Custom Experiment {n_runs} times")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"{'#'*80}\n")
    
    all_results = []
    
    # Get existing directories before starting
    existing_dirs = get_existing_ablation_dirs(dataset)
    
    for run_id in range(n_runs):
        print(f"\n{'='*80}")
        print(f"Run {run_id + 1}/{n_runs}")
        print(f"{'='*80}\n")
        
        # Get directories before this run
        dirs_before = get_existing_ablation_dirs(dataset)
        
        # Run the command
        output = run_command(cmd_list, run_id=run_id)
        
        if output is None:
            print(f"Warning: Run {run_id + 1} command failed")
            continue
        
        # Find the newly created results file
        results_file = find_new_ablation_results(dirs_before, dataset, max_wait=60)
        
        if results_file:
            try:
                results = load_ablation_results(results_file)
                all_results.append(results)
                print(f"✓ Loaded results from: {results_file}")
            except Exception as e:
                print(f"✗ Error loading results from {results_file}: {e}")
        else:
            print(f"✗ Warning: Could not find results file for run {run_id + 1}")
            print("   This run will be skipped in the final statistics")
    
    if len(all_results) == 0:
        print("\n✗ ERROR: No results were successfully loaded!")
        return None, []
    
    # Aggregate results
    summary = aggregate_ablation_results(all_results)
    
    print(f"\n✓ Successfully processed {len(all_results)}/{n_runs} runs")
    
    # Print summary table
    print_ablation_summary(summary)
    
    return summary, all_results


def main():
    # Base command
    base_cmd = [
        'python', 'main_aga.py',
        '--dataset', 'Cora',
        '--model', 'GCN'
    ]
    
    # Create results directory
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path("results") / f"comprehensive_experiments_{time_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run ablation study 5 times
    # Using edge=0.1, weight=0.1 to match the starting point of sweeps
    ablation_cmd = base_cmd.copy()
    ablation_cmd.extend(['--edge_sparsity', '0.1'])
    ablation_cmd.extend(['--weight_sparsity', '0.1'])
    ablation_cmd.append('--compare_static')
    
    ablation_summary, ablation_all_runs = run_ablation_study_multiple_times(
        ablation_cmd, n_runs=5, dataset="Cora"
    )
    print_ablation_summary(ablation_summary)
    
    # Save ablation results
    with open(results_dir / "ablation_study_5runs_summary.json", "w") as f:
        json.dump(ablation_summary, f, indent=2)
    with open(results_dir / "ablation_study_5runs_all.json", "w") as f:
        json.dump(ablation_all_runs, f, indent=2)
    
    # 2. Edge sparsity sweep: edge from 0.1 to 0.8 (step 0.1), weight fixed at 0.1
    # Sequence: edge=0.1,weight=0.1 (5 runs) → edge=0.2,weight=0.1 (5 runs) → ... → edge=0.8,weight=0.1 (5 runs)
    print(f"\n{'#'*80}")
    print("EDGE SPARSITY SWEEP")
    print("Sequence: edge=0.1,weight=0.1 (5 runs) → edge=0.2,weight=0.1 (5 runs) → ... → edge=0.8,weight=0.1 (5 runs)")
    print(f"{'#'*80}\n")
    
    edge_sweep_cmd = base_cmd.copy()
    edge_sweep_cmd.append('--compare_static')
    edge_sweep_range = [round(0.1 + i * 0.1, 1) for i in range(8)]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"Edge sparsity values: {edge_sweep_range}")
    print(f"Fixed weight_sparsity: 0.1")
    print(f"Runs per configuration: 5")
    print(f"Using --compare_static flag\n")
    
    edge_sweep_results = run_sparsity_sweep(
        edge_sweep_cmd, 
        'edge', 
        fixed_value=0.1, 
        sweep_range=edge_sweep_range,
        n_runs=5
    )
    
    print_sweep_summary(edge_sweep_results)
    
    with open(results_dir / "edge_sparsity_sweep.json", "w") as f:
        json.dump(edge_sweep_results, f, indent=2)
    
    # 3. Weight sparsity sweep: weight from 0.1 to 0.8 (step 0.1), edge fixed at 0.1
    # Sequence: edge=0.1,weight=0.1 (5 runs) → edge=0.1,weight=0.2 (5 runs) → ... → edge=0.1,weight=0.8 (5 runs)
    print(f"\n{'#'*80}")
    print("WEIGHT SPARSITY SWEEP")
    print("Sequence: edge=0.1,weight=0.1 (5 runs) → edge=0.1,weight=0.2 (5 runs) → ... → edge=0.1,weight=0.8 (5 runs)")
    print(f"{'#'*80}\n")
    
    weight_sweep_cmd = base_cmd.copy()
    weight_sweep_cmd.append('--compare_static')
    weight_sweep_range = [round(0.1 + i * 0.1, 1) for i in range(8)]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"Fixed edge_sparsity: 0.1")
    print(f"Weight sparsity values: {weight_sweep_range}")
    print(f"Runs per configuration: 5")
    print(f"Using --compare_static flag\n")
    
    weight_sweep_results = run_sparsity_sweep(
        weight_sweep_cmd,
        'weight',
        fixed_value=0.1,
        sweep_range=weight_sweep_range,
        n_runs=5
    )
    
    print_sweep_summary(weight_sweep_results)
    
    with open(results_dir / "weight_sparsity_sweep.json", "w") as f:
        json.dump(weight_sweep_results, f, indent=2)
    
    print(f"\n{'#'*80}")
    print("All experiments completed!")
    print(f"Results saved to: {results_dir}")
    print(f"{'#'*80}\n")
    
    # Print summary
    print("\nSUMMARY:")
    print(f"1. Ablation study: 5 runs completed")
    print(f"2. Edge sparsity sweep: {len(edge_sweep_range)} configurations × 5 runs = {len(edge_sweep_range) * 5} total runs")
    print(f"3. Weight sparsity sweep: {len(weight_sweep_range)} configurations × 5 runs = {len(weight_sweep_range) * 5} total runs")


def run_custom_experiment_example():
    """
    Example: Run a specific experiment command 5 times and get statistics
    This function demonstrates how to use run_custom_experiment_multiple_times
    """
    # Example command matching user's request
    cmd = [
        'python', 'main_aga.py',
        '--dataset', 'Cora',
        '--model', 'GCN',
        '--edge_sparsity', '0.5',
        '--weight_sparsity', '0.1',
        '--compare_static',
        '--epochs', '200'
    ]
    
    # Create results directory
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path("results") / f"custom_experiment_{time_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        print(f"\n{'#'*80}")
        print(f"Results saved to: {results_dir}")
        print(f"{'#'*80}\n")
    
    return summary, all_results


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run custom experiment
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        run_custom_experiment_example()
    else:
        main()
