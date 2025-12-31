"""
Batch Inference: Systematic evaluation of model interpolation performance

This script:
1. Discovers all available preprocessed graphs (training + intermediate ARs)
2. Runs inference on intermediate AR cases
3. Computes error metrics (RMSE, MAE, R²) vs ground truth
4. Calculates distance from training AR space
5. Analyzes correlation between distance and error
6. Generates comprehensive visualizations

Usage:
    python batch_infer.py --intermediate-only
    python batch_infer.py --all-cases
    python batch_infer.py --re 1000 --plot
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from Models.model import WSSPredictor
from config import config
from dataset import WSSDataset, get_dataloaders
from evaluate import denormalize_wss


def euclidean_distance(point1, point2):
    """Compute Euclidean distance between two 2D points (AR representations)."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def average_distance_from_training(ar_point, training_ars):
    """
    Compute average Euclidean distance from a point to training AR set.
    
    AR is represented as (Lx, Ly) coordinates.  
    """
    distances = [euclidean_distance(ar_point, train_ar) for train_ar in training_ars]
    return np.mean(distances)


def parse_ar_string(ar_str):
    """Convert AR string like '1.5x1' to (Lx, Ly) tuple."""
    parts = ar_str.lower().replace('x', ' ').split()
    lx, ly = float(parts[0]), float(parts[1])
    return (lx, ly)


def normalize_graph(graph, norm_stats):
    """Apply dataset normalization to a graph (mimics WSSDataset.__getitem__())."""
    # Load normalization stats as tensors
    target_mean = torch.tensor(norm_stats['target_mean'])
    target_std = torch.tensor(norm_stats['target_std'])
    
    # Normalize targets using log1p transform (same as dataset.py)
    sign = torch.sign(graph.y)
    log_mag = torch.log1p(torch.abs(graph.y))
    signed_log = sign * log_mag
    graph.y = (signed_log - target_mean) / target_std
    
    return graph


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = WSSPredictor(
        node_feature_dim=config.node_feature_dim,
        flow_param_dim=config.flow_param_dim,
        hidden_dim=config.hidden_dim,
        context_dim=config.context_dim,
        output_dim=config.target_dim,
        num_geom_layers=config.num_geom_layers,
        num_task_layers=config.num_task_layers,
        task_hidden_dim=config.task_hidden_dim,
        dropout=config.dropout_rate
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def run_inference_single(model, graph_data, device):
    """Run model inference on a single graph from dataset."""
    # Add batch attribute
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    
    # Ensure flow params are tensors (dataset may have scalars)
    if not isinstance(graph_data.re, torch.Tensor):
        graph_data.re = torch.tensor([graph_data.re], dtype=torch.float)
    if not isinstance(graph_data.lx, torch.Tensor):
        graph_data.lx = torch.tensor([graph_data.lx], dtype=torch.float)
    if not isinstance(graph_data.ly, torch.Tensor):
        graph_data.ly = torch.tensor([graph_data.ly], dtype=torch.float)
    
    # Move to device
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        pred_normalized = model(graph_data).cpu()
    
    # Get ground truth (normalized from dataset)
    true_normalized = graph_data.y.cpu()
    
    return pred_normalized, true_normalized


def compute_metrics(pred_normalized, true_normalized, norm_stats):
    """Compute error metrics in physical units."""
    # Denormalize predictions and ground truth (both are torch tensors)
    pred_physical = denormalize_wss(pred_normalized, norm_stats)
    true_physical = denormalize_wss(true_normalized, norm_stats)
    
    # Compute errors
    abs_error = np.abs(pred_physical - true_physical)
    squared_error = (pred_physical - true_physical) ** 2
    
    # Per-component metrics
    mae_x = np.mean(abs_error[:, 0])
    mae_y = np.mean(abs_error[:, 1])
    rmse_x = np.sqrt(np.mean(squared_error[:, 0]))
    rmse_y = np.sqrt(np.mean(squared_error[:, 1]))
    
    # Magnitude metrics
    pred_mag = np.sqrt(pred_physical[:, 0]**2 + pred_physical[:, 1]**2)
    true_mag = np.sqrt(true_physical[:, 0]**2 + true_physical[:, 1]**2)
    
    mae_mag = np.mean(np.abs(pred_mag - true_mag))
    rmse_mag = np.sqrt(np.mean((pred_mag - true_mag)**2))
    
    return {
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_mag': mae_mag,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_mag': rmse_mag,
    }


def identify_training_ars(train_loader):
    """Extract unique aspect ratios from training set."""
    ars = set()
    for graph in train_loader.dataset:
        ars.add(graph.ar)
    
    # Convert to (Lx, Ly) tuples
    training_ar_points = [parse_ar_string(ar) for ar in ars]
    return list(ars), training_ar_points


def compute_baseline_rmse(df_training):
    """Compute baseline RMSE for each Reynolds number from training ARs."""
    baselines = {}
    for re in df_training['re'].unique():
        re_data = df_training[df_training['re'] == re]
        baselines[re] = re_data['rmse_mag'].mean()
    return baselines


def batch_evaluate(
    model,
    all_graphs,
    training_ar_strings,
    training_ar_points,
    norm_stats,
    device,
    intermediate_only=False
):
    """Run evaluation on all graphs and compute error growth rate."""
    results = []
    
    print(f"\nEvaluating {len(all_graphs)} cases...")
    print(f"Training ARs: {sorted(training_ar_strings)}")
    
    for graph in tqdm(all_graphs, desc="Running inference"):
        ar_str = graph.ar
        re = int(graph.re.item() if isinstance(graph.re, torch.Tensor) else graph.re)
        
        # Skip training ARs if intermediate_only
        if intermediate_only and ar_str in training_ar_strings:
            continue
        
        # Run inference (graph already has normalized data from dataset)
        pred_normalized, true_normalized = run_inference_single(model, graph, device)
        
        # Compute metrics in physical units
        metrics = compute_metrics(pred_normalized, true_normalized, norm_stats)
        
        # Compute distance from training ARs
        ar_point = parse_ar_string(ar_str)
        avg_dist = average_distance_from_training(ar_point, training_ar_points)
        
        # Store results
        result = {
            'ar': ar_str,
            're': re,
            'lx': ar_point[0],
            'ly': ar_point[1],
            'distance_from_training': avg_dist,
            'is_training_ar': ar_str in training_ar_strings,
            **metrics
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Compute baselines from training ARs and error growth rate
    df_training = df[df['is_training_ar']].copy()
    if len(df_training) > 0:
        print("\nComputing baseline RMSE from training ARs...")
        baselines = compute_baseline_rmse(df_training)
        
        # Compute error growth rate
        df['error_growth_rate'] = df.apply(
            lambda row: row['rmse_mag'] / baselines.get(row['re'], row['rmse_mag']) 
            if row['re'] in baselines else 1.0,
            axis=1
        )
    else:
        print("\nWarning: No training AR data available. Error growth rate cannot be computed.")
        df['error_growth_rate'] = np.nan
    
    return df


def plot_heatmap(df, output_dir, re_values=None):
    """Plot heatmap showing error pattern across Re and AR."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter by Re if specified
    if re_values is not None:
        df = df[df['re'].isin(re_values)]
    
    # Filter to intermediate ARs only
    df_interp = df[~df['is_training_ar']].copy()
    
    # Check if we have error growth rate data
    has_growth_rate = not df_interp['error_growth_rate'].isna().all()
    
    if has_growth_rate:
        metrics = ['rmse_mag', 'mae_mag', 'error_growth_rate']
        titles = ['RMSE (Magnitude)', 'MAE (Magnitude)', 'Error Growth Rate']
        cmaps = ['YlOrRd', 'YlOrRd', 'RdYlGn_r']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        metrics = ['rmse_mag', 'mae_mag']
        titles = ['RMSE (Magnitude)', 'MAE (Magnitude)']
        cmaps = ['YlOrRd', 'YlOrRd']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Error Heatmap: Reynolds Number vs Aspect Ratio', 
                 fontsize=14, fontweight='bold')
    
    for idx, (metric, title, cmap) in enumerate(zip(metrics, titles, cmaps)):
        ax = axes[idx]
        
        # Create pivot table
        pivot = df_interp.pivot_table(
            values=metric,
            index='re',
            columns='ar',
            aggfunc='mean'
        )
        
        # Sort columns by AR value (convert '1.1x1' to 1.1 for sorting)
        ar_order = sorted(pivot.columns, key=lambda x: float(x.split('x')[0]))
        pivot = pivot[ar_order]
        
        # Plot heatmap
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticklabels(pivot.index)
        
        # Labels
        ax.set_xlabel('Aspect Ratio', fontsize=11)
        ax.set_ylabel('Reynolds Number', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        if metric == 'error_growth_rate':
            cbar.set_label('Growth Rate (×baseline)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Error (Pa)', rotation=270, labelpad=20)
        
        # Add text annotations for values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    # Format value
                    if metric == 'error_growth_rate':
                        text = f'{val:.2f}×'
                    else:
                        text = f'{val:.1e}'
                    
                    # Choose text color based on background
                    color = 'white' if val > pivot.values.mean() else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                           color=color, fontsize=7)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'error_heatmap.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Heatmap saved: {plot_path}")
    plt.close()


def plot_error_growth_correlation(df, output_dir, selected_re=None):
    """Plot distance vs error growth rate correlation."""
    output_dir = Path(output_dir)
    
    df_interp = df[~df['is_training_ar']].copy()
    
    # Check if we have growth rate data
    if df_interp['error_growth_rate'].isna().all():
        print("⚠ Skipping error growth plot: no baseline data available")
        return
    
    if selected_re is None:
        unique_res = sorted(df_interp['re'].unique())
        n_plots = min(6, len(unique_res))
        indices = np.linspace(0, len(unique_res)-1, n_plots, dtype=int)
        selected_re = [unique_res[i] for i in indices]
    
    n_cols = 3
    n_rows = int(np.ceil(len(selected_re) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Distance vs Error Growth Rate', 
                 fontsize=14, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, re in enumerate(selected_re):
        ax = axes_flat[idx]
        re_data = df_interp[df_interp['re'] == re].sort_values('distance_from_training')
        
        if len(re_data) == 0:
            ax.axis('off')
            continue
        
        ax.scatter(re_data['distance_from_training'], re_data['error_growth_rate'],
                  s=100, alpha=0.7, edgecolors='black', linewidth=1.5, c='orange')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=2, 
                  label='Baseline (training)')
        
        if len(re_data) > 2:
            z = np.polyfit(re_data['distance_from_training'], re_data['error_growth_rate'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(re_data['distance_from_training'].min(),
                                re_data['distance_from_training'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Linear fit')
            
            corr = np.corrcoef(re_data['distance_from_training'], re_data['error_growth_rate'])[0, 1]
            
            ax.text(0.05, 0.95, f'r = {corr:.3f}',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Distance from Training ARs', fontsize=10)
        ax.set_ylabel('Error Growth Rate (×baseline)', fontsize=10)
        ax.set_title(f'Re = {int(re)}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        for _, row in re_data.iterrows():
            ax.annotate(row['ar'], 
                       (row['distance_from_training'], row['error_growth_rate']),
                       textcoords="offset points", xytext=(0,5), 
                       ha='center', fontsize=7, alpha=0.7)
    
    for idx in range(len(selected_re), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'correlation_distance_vs_growth_rate.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Distance vs Growth Rate plots saved: {plot_path}")
    plt.close()


def plot_correlation_small_multiples(df, output_dir, selected_re=None):
    """Plot small multiples showing correlation for selected Re values."""
    output_dir = Path(output_dir)
    
    # Filter to intermediate ARs only
    df_interp = df[~df['is_training_ar']].copy()
    
    # Select Reynolds numbers to plot
    if selected_re is None:
        unique_res = sorted(df_interp['re'].unique())
        # Select 5-6 evenly spaced Re values
        n_plots = min(6, len(unique_res))
        indices = np.linspace(0, len(unique_res)-1, n_plots, dtype=int)
        selected_re = [unique_res[i] for i in indices]
    
    # Create subplots
    n_cols = 3
    n_rows = int(np.ceil(len(selected_re) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Distance vs RMSE for Selected Reynolds Numbers', 
                 fontsize=14, fontweight='bold')
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, re in enumerate(selected_re):
        ax = axes_flat[idx]
        re_data = df_interp[df_interp['re'] == re].sort_values('distance_from_training')
        
        if len(re_data) == 0:
            ax.axis('off')
            continue
        
        # Scatter plot
        ax.scatter(re_data['distance_from_training'], re_data['rmse_mag'],
                  s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Fit linear regression
        if len(re_data) > 2:
            z = np.polyfit(re_data['distance_from_training'], re_data['rmse_mag'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(re_data['distance_from_training'].min(),
                                re_data['distance_from_training'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Linear fit')
            
            # Compute correlation
            corr = np.corrcoef(re_data['distance_from_training'], re_data['rmse_mag'])[0, 1]
            
            # Add correlation text
            ax.text(0.05, 0.95, f'r = {corr:.3f}',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add equation
            slope, intercept = z
            eq_text = f'y = {slope:.2e}x + {intercept:.2e}'
            ax.text(0.05, 0.85, eq_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        ax.set_xlabel('Distance from Training ARs', fontsize=10)
        ax.set_ylabel('RMSE (Pa)', fontsize=10)
        ax.set_title(f'Re = {int(re)}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add AR labels to points
        for _, row in re_data.iterrows():
            ax.annotate(row['ar'], 
                       (row['distance_from_training'], row['rmse_mag']),
                       textcoords="offset points", xytext=(0,5), 
                       ha='center', fontsize=7, alpha=0.7)
    
    # Hide unused subplots
    for idx in range(len(selected_re), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'correlation_small_multiples.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Correlation plots saved: {plot_path}")
    plt.close()


def generate_formatted_report(df, output_dir, selected_re=None):
    """Generate formatted report showing metrics for each AR by Re."""
    output_dir = Path(output_dir)
    
    df_interp = df[~df['is_training_ar']].copy()
    
    if selected_re is None:
        selected_re = sorted(df_interp['re'].unique())
    
    has_growth_rate = not df_interp['error_growth_rate'].isna().all()
    
    print("\n" + "="*80)
    print("DETAILED METRICS BY REYNOLDS NUMBER")
    print("="*80)
    print("")
    
    md_lines = []
    md_lines.append("# Detailed Error Metrics Report")
    md_lines.append("")
    md_lines.append("## Results by Reynolds Number")
    md_lines.append("")
    
    for re in selected_re:
        re_data = df_interp[df_interp['re'] == re].sort_values('lx')
        
        if len(re_data) == 0:
            continue
        
        if len(re_data) > 2:
            corr = np.corrcoef(re_data['distance_from_training'], re_data['rmse_mag'])[0, 1]
            header = f"## Re = {int(re)} (correlation: r={corr:.3f})"
        else:
            header = f"## Re = {int(re)}"
        
        print(header)
        print("")
        md_lines.append(header)
        md_lines.append("")
        
        for _, row in re_data.iterrows():
            ar = row['ar']
            mae = row['mae_mag']
            rmse = row['rmse_mag']
            
            if has_growth_rate and not np.isnan(row['error_growth_rate']):
                growth = row['error_growth_rate']
                line = f"- **AR {ar}**: MAE = **{mae:.2e} Pa**, RMSE = **{rmse:.2e} Pa**, Growth = **{growth:.2f}× baseline**"
            else:
                line = f"- **AR {ar}**: MAE = **{mae:.2e} Pa**, RMSE = **{rmse:.2e} Pa**"
            
            print(line)
            md_lines.append(line)
        
        print("")
        print("---")
        print("")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Save markdown report
    md_path = output_dir / 'detailed_metrics_report.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"✓ Detailed report saved: {md_path}")
    
    # Save summary CSV
    cols = ['re', 'ar', 'lx', 'ly', 'distance_from_training',
            'mae_mag', 'rmse_mag', 'mae_x', 'mae_y', 'rmse_x', 'rmse_y']
    if has_growth_rate:
        cols.append('error_growth_rate')
    
    summary_df = df_interp[cols].copy()
    summary_df = summary_df.sort_values(['re', 'lx'])
    
    summary_path = output_dir / 'metrics_summary.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.6e')
    print(f"✓ Summary CSV saved: {summary_path}")


def print_summary(df, training_ar_strings):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total cases evaluated: {len(df)}")
    print(f"  Reynolds numbers: {sorted(df['re'].unique())}")
    print(f"  Aspect ratios: {sorted(df['ar'].unique())}")
    print(f"  Training ARs: {sorted(training_ar_strings)}")
    
    # Separate training vs intermediate
    train_df = df[df['is_training_ar']]
    interp_df = df[~df['is_training_ar']]
    
    print(f"\n  Training AR cases: {len(train_df)}")
    print(f"  Intermediate AR cases: {len(interp_df)}")
    
    # Performance comparison
    if len(train_df) > 0 and len(interp_df) > 0:
        print("\nPerformance Comparison:")
        print(f"  Training ARs - RMSE: {train_df['rmse_mag'].mean():.6e} ± {train_df['rmse_mag'].std():.6e} Pa")
        print(f"  Intermediate ARs - RMSE: {interp_df['rmse_mag'].mean():.6e} ± {interp_df['rmse_mag'].std():.6e} Pa")
        print(f"  Training ARs - MAE: {train_df['mae_mag'].mean():.6e} ± {train_df['mae_mag'].std():.6e} Pa")
        print(f"  Intermediate ARs - MAE: {interp_df['mae_mag'].mean():.6e} ± {interp_df['mae_mag'].std():.6e} Pa")
        
        if not interp_df['error_growth_rate'].isna().all():
            print(f"  Intermediate ARs - Error Growth: {interp_df['error_growth_rate'].mean():.3f}× ± {interp_df['error_growth_rate'].std():.3f}×")
    
    # Per-Re analysis
    print("\nPer Reynolds Number Analysis:")
    for re in sorted(df['re'].unique()):
        re_df = df[df['re'] == re]
        re_interp = re_df[~re_df['is_training_ar']]
        
        if len(re_interp) > 0:
            corr = np.corrcoef(re_interp['distance_from_training'], re_interp['rmse_mag'])[0, 1]
            print(f"\n  Re={re}:")
            print(f"    RMSE (mag): {re_df['rmse_mag'].mean():.6e} Pa")
            print(f"    MAE (mag): {re_df['mae_mag'].mean():.6e} Pa")
            
            if not re_interp['error_growth_rate'].isna().all():
                print(f"    Error growth: {re_interp['error_growth_rate'].mean():.3f}× baseline")
            
            print(f"    Correlation (distance vs RMSE): {corr:.4f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Batch Inference on Model Interpolation')
    parser.add_argument('--intermediate-only', action='store_true',
                       help='Evaluate only intermediate ARs (exclude training ARs)')
    parser.add_argument('--all-cases', action='store_true',
                       help='Evaluate all available cases')
    parser.add_argument('--re', type=int, nargs='+',
                       help='Filter by specific Reynolds numbers')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate visualizations (default: True)')
    parser.add_argument('--model', type=str, default='Models/best_model.pt',
                       help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='results/batch_evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("BATCH INFERENCE - INTERPOLATION ANALYSIS")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, device)
    
    # Load datasets with proper normalization
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)
    training_ar_strings, training_ar_points = identify_training_ars(train_loader)
    
    # Load normalization stats for denormalization
    norm_stats_path = config.processed_data_dir / 'normalization_stats.json'
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    # Combine all graphs from train/val/test datasets  
    print("\nCombining all available graphs...")
    all_graphs = []
    all_graphs.extend(list(train_loader.dataset))
    all_graphs.extend(list(val_loader.dataset))
    all_graphs.extend(list(test_loader.dataset))
    
    # Load intermediate AR graphs (1.1x1 through 1.9x1) and normalize them
    print("Loading intermediate AR graphs...")
    intermediate_ars = ['1.1x1', '1.2x1', '1.3x1', '1.4x1', '1.5x1', '1.6x1', '1.7x1', '1.8x1', '1.9x1']
    for ar in intermediate_ars:
        ar_dir = config.processed_data_dir / ar
        if ar_dir.exists():
            for pt_file in ar_dir.glob('Re*.pt'):
                graph = torch.load(pt_file, weights_only=False)
                # Apply same normalization as dataset does
                graph = normalize_graph(graph, norm_stats)
                all_graphs.append(graph)
    
    print(f"Found {len(all_graphs)} preprocessed cases")
    
    # Run evaluation
    results_df = batch_evaluate(
        model=model,
        all_graphs=all_graphs,
        training_ar_strings=training_ar_strings,
        training_ar_points=training_ar_points,
        norm_stats=norm_stats,
        device=device,
        intermediate_only=args.intermediate_only
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'evaluation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")
    
    # Print summary
    print_summary(results_df, training_ar_strings)
    
    # Generate formatted detailed report
    print("\nGenerating detailed report...")
    report_re = [100, 500, 1000, 1500, 2000, 2100] if args.re is None else args.re
    generate_formatted_report(results_df, output_dir, selected_re=report_re)
    
    # Generate plots
    if args.plot:
        print("\nGenerating visualizations...")
        plot_heatmap(results_df, output_dir, re_values=args.re)
        
        # Select representative Re values for correlation plots
        selected_re = [100, 500, 1000, 1500, 2000, 2100] if args.re is None else args.re
        plot_correlation_small_multiples(results_df, output_dir, selected_re=selected_re)
        plot_error_growth_correlation(results_df, output_dir, selected_re=selected_re)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
