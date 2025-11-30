"""
Evaluation script for trained WSS prediction model.
Computes comprehensive metrics and generates visualization plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import config
from dataset import get_dataloaders
from Models.model import WSSPredictor


def load_best_model(checkpoint_path='Models/best_model.pt'):
    """Load the best trained model from checkpoint."""
    # Force CPU since we may not have CUDA available
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    
    return model


def denormalize_wss(wss_normalized, stats):
    """
    Convert normalized WSS back to physical units.
    Uses the same method as dataset.py:
    - Sign-preserving log1p normalization with mean/std scaling
    """
    # Convert stats to tensors
    target_mean = torch.tensor(stats['target_mean'], dtype=torch.float32)
    target_std = torch.tensor(stats['target_std'], dtype=torch.float32)
    
    # Input can be numpy or torch tensor
    if isinstance(wss_normalized, np.ndarray):
        wss_normalized = torch.from_numpy(wss_normalized).float()
    
    # Reverse normalization: denormalized_log = normalized * std + mean
    sign = torch.sign(wss_normalized)
    log_mag = wss_normalized.abs() * target_std + target_mean
    
    # Reverse log1p transform: mag = exp(log) - 1
    mag = torch.expm1(log_mag)
    
    # Restore sign and convert to numpy
    wss_physical = (sign * mag).numpy()
    
    return wss_physical


def evaluate_model(model, test_loader, stats):
    """
    Comprehensive evaluation on test set.
    Returns metrics dictionary and all predictions/ground truth.
    """
    model.eval()
    
    all_preds_norm = []
    all_targets_norm = []
    all_preds_phys = []
    all_targets_phys = []
    all_re = []
    all_aspect = []
    test_loss = 0.0
    
    device = torch.device('cpu')  # Use CPU for evaluation
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            pred = model(batch)
            
            # Compute loss in normalized space
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            test_loss += loss.item()
            
            # Denormalize predictions and targets
            pred_phys = denormalize_wss(pred.cpu().numpy(), stats)
            target_phys = denormalize_wss(batch.y.cpu().numpy(), stats)
            
            # Store results
            all_preds_norm.append(pred.cpu().numpy())
            all_targets_norm.append(batch.y.cpu().numpy())
            all_preds_phys.append(pred_phys)
            all_targets_phys.append(target_phys)
            
            # Store metadata (Re and aspect ratio from flow_params)
            # batch.re, batch.lx, batch.ly are batched tensors
            re_vals = batch.re.cpu().numpy()
            lx_vals = batch.lx.cpu().numpy()
            ly_vals = batch.ly.cpu().numpy()
            aspect_vals = lx_vals / ly_vals
            
            # Expand to per-node (each graph in batch has different number of nodes)
            # We need to replicate each graph's Re/aspect for all its nodes
            batch_idx = batch.batch.cpu().numpy()  # [N_total] array of graph indices
            for graph_idx in range(len(re_vals)):
                node_mask = (batch_idx == graph_idx)
                num_nodes_this_graph = node_mask.sum()
                all_re.extend([re_vals[graph_idx]] * num_nodes_this_graph)
                all_aspect.extend([aspect_vals[graph_idx]] * num_nodes_this_graph)
    
    # Concatenate all results
    preds_norm = np.vstack(all_preds_norm)  # [N_total, 2]
    targets_norm = np.vstack(all_targets_norm)
    preds_phys = np.vstack(all_preds_phys)
    targets_phys = np.vstack(all_targets_phys)
    
    # Compute metrics
    test_loss = test_loss / len(test_loader)
    
    # Physical units metrics (for each component and magnitude)
    mae_x = mean_absolute_error(targets_phys[:, 0], preds_phys[:, 0])
    mae_y = mean_absolute_error(targets_phys[:, 1], preds_phys[:, 1])
    
    rmse_x = np.sqrt(mean_squared_error(targets_phys[:, 0], preds_phys[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(targets_phys[:, 1], preds_phys[:, 1]))
    
    r2_x = r2_score(targets_phys[:, 0], preds_phys[:, 0])
    r2_y = r2_score(targets_phys[:, 1], preds_phys[:, 1])
    
    # Magnitude metrics
    target_mag = np.linalg.norm(targets_phys, axis=1)
    pred_mag = np.linalg.norm(preds_phys, axis=1)
    
    mae_mag = mean_absolute_error(target_mag, pred_mag)
    rmse_mag = np.sqrt(mean_squared_error(target_mag, pred_mag))
    r2_mag = r2_score(target_mag, pred_mag)
    
    # Relative error (percentage)
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-14
    rel_error_x = np.abs(targets_phys[:, 0] - preds_phys[:, 0]) / (np.abs(targets_phys[:, 0]) + epsilon) * 100
    rel_error_y = np.abs(targets_phys[:, 1] - preds_phys[:, 1]) / (np.abs(targets_phys[:, 1]) + epsilon) * 100
    rel_error_mag = np.abs(target_mag - pred_mag) / (target_mag + epsilon) * 100
    
    # Filter out extreme outliers for relative error stats (when true value is near zero)
    rel_error_x_filtered = rel_error_x[np.abs(targets_phys[:, 0]) > 1e-12]
    rel_error_y_filtered = rel_error_y[np.abs(targets_phys[:, 1]) > 1e-12]
    rel_error_mag_filtered = rel_error_mag[target_mag > 1e-12]
    
    metrics = {
        'test_loss_normalized': test_loss,
        'mae_x_Pa': mae_x,
        'mae_y_Pa': mae_y,
        'mae_magnitude_Pa': mae_mag,
        'rmse_x_Pa': rmse_x,
        'rmse_y_Pa': rmse_y,
        'rmse_magnitude_Pa': rmse_mag,
        'r2_x': r2_x,
        'r2_y': r2_y,
        'r2_magnitude': r2_mag,
        'median_rel_error_x_percent': float(np.median(rel_error_x_filtered)),
        'median_rel_error_y_percent': float(np.median(rel_error_y_filtered)),
        'median_rel_error_magnitude_percent': float(np.median(rel_error_mag_filtered)),
        'mean_rel_error_x_percent': float(np.mean(rel_error_x_filtered)),
        'mean_rel_error_y_percent': float(np.mean(rel_error_y_filtered)),
        'mean_rel_error_magnitude_percent': float(np.mean(rel_error_mag_filtered)),
    }
    
    results = {
        'preds_norm': preds_norm,
        'targets_norm': targets_norm,
        'preds_phys': preds_phys,
        'targets_phys': targets_phys,
        'pred_mag': pred_mag,
        'target_mag': target_mag,
        'rel_error_x': rel_error_x,
        'rel_error_y': rel_error_y,
        'rel_error_mag': rel_error_mag,
        're_values': np.array(all_re),
        'aspect_ratios': np.array(all_aspect),
    }
    
    return metrics, results


def print_metrics(metrics):
    """Print formatted metrics summary."""
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nNormalized Loss (MSE in log-space):")
    print(f"  Test Loss: {metrics['test_loss_normalized']:.6f}")
    
    print(f"\nPhysical Units - Mean Absolute Error (MAE):")
    print(f"  WSS_x:      {metrics['mae_x_Pa']:.6e} Pa")
    print(f"  WSS_y:      {metrics['mae_y_Pa']:.6e} Pa")
    print(f"  Magnitude:  {metrics['mae_magnitude_Pa']:.6e} Pa")
    
    print(f"\nPhysical Units - Root Mean Squared Error (RMSE):")
    print(f"  WSS_x:      {metrics['rmse_x_Pa']:.6e} Pa")
    print(f"  WSS_y:      {metrics['rmse_y_Pa']:.6e} Pa")
    print(f"  Magnitude:  {metrics['rmse_magnitude_Pa']:.6e} Pa")
    
    print(f"\nR² Score (Coefficient of Determination):")
    print(f"  WSS_x:      {metrics['r2_x']:.6f}")
    print(f"  WSS_y:      {metrics['r2_y']:.6f}")
    print(f"  Magnitude:  {metrics['r2_magnitude']:.6f}")
    
    print(f"\nRelative Error (filtered for |true| > 1e-12 Pa):")
    print(f"  WSS_x median:      {metrics['median_rel_error_x_percent']:.2f}%")
    print(f"  WSS_y median:      {metrics['median_rel_error_y_percent']:.2f}%")
    print(f"  Magnitude median:  {metrics['median_rel_error_magnitude_percent']:.2f}%")
    print(f"  WSS_x mean:        {metrics['mean_rel_error_x_percent']:.2f}%")
    print(f"  WSS_y mean:        {metrics['mean_rel_error_y_percent']:.2f}%")
    print(f"  Magnitude mean:    {metrics['mean_rel_error_magnitude_percent']:.2f}%")
    
    print("\n" + "="*60)


def plot_predictions_vs_truth(results, output_dir='results'):
    """Generate scatter plots comparing predictions to ground truth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Component x
    ax = axes[0]
    ax.scatter(results['targets_phys'][:, 0], results['preds_phys'][:, 0], 
               alpha=0.3, s=10, c='blue', edgecolors='none')
    
    # Perfect prediction line
    min_val = min(results['targets_phys'][:, 0].min(), results['preds_phys'][:, 0].min())
    max_val = max(results['targets_phys'][:, 0].max(), results['preds_phys'][:, 0].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True WSS_x (Pa)', fontsize=12)
    ax.set_ylabel('Predicted WSS_x (Pa)', fontsize=12)
    ax.set_title('WSS X-Component', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Component y
    ax = axes[1]
    ax.scatter(results['targets_phys'][:, 1], results['preds_phys'][:, 1], 
               alpha=0.3, s=10, c='green', edgecolors='none')
    
    min_val = min(results['targets_phys'][:, 1].min(), results['preds_phys'][:, 1].min())
    max_val = max(results['targets_phys'][:, 1].max(), results['preds_phys'][:, 1].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True WSS_y (Pa)', fontsize=12)
    ax.set_ylabel('Predicted WSS_y (Pa)', fontsize=12)
    ax.set_title('WSS Y-Component', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Magnitude
    ax = axes[2]
    ax.scatter(results['target_mag'], results['pred_mag'], 
               alpha=0.3, s=10, c='purple', edgecolors='none')
    
    min_val = min(results['target_mag'].min(), results['pred_mag'].min())
    max_val = max(results['target_mag'].max(), results['pred_mag'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True WSS Magnitude (Pa)', fontsize=12)
    ax.set_ylabel('Predicted WSS Magnitude (Pa)', fontsize=12)
    ax.set_title('WSS Magnitude', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_truth.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'predictions_vs_truth.png'}")
    plt.close()


def plot_error_distribution(results, output_dir='results'):
    """Generate histograms of error distributions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Absolute errors in physical units
    abs_error_x = np.abs(results['targets_phys'][:, 0] - results['preds_phys'][:, 0])
    abs_error_y = np.abs(results['targets_phys'][:, 1] - results['preds_phys'][:, 1])
    abs_error_mag = np.abs(results['target_mag'] - results['pred_mag'])
    
    # Top row: Absolute errors
    ax = axes[0, 0]
    ax.hist(abs_error_x, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Error (Pa)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WSS_x Absolute Error', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(abs_error_y, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Error (Pa)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WSS_y Absolute Error', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.hist(abs_error_mag, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Error (Pa)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WSS Magnitude Absolute Error', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Bottom row: Relative errors (filtered)
    rel_error_x_filtered = results['rel_error_x'][np.abs(results['targets_phys'][:, 0]) > 1e-12]
    rel_error_y_filtered = results['rel_error_y'][np.abs(results['targets_phys'][:, 1]) > 1e-12]
    rel_error_mag_filtered = results['rel_error_mag'][results['target_mag'] > 1e-12]
    
    ax = axes[1, 0]
    ax.hist(rel_error_x_filtered, bins=50, color='blue', alpha=0.7, edgecolor='black', range=(0, 100))
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WSS_x Relative Error (|true| > 1e-12)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(rel_error_y_filtered, bins=50, color='green', alpha=0.7, edgecolor='black', range=(0, 100))
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WSS_y Relative Error (|true| > 1e-12)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.hist(rel_error_mag_filtered, bins=50, color='purple', alpha=0.7, edgecolor='black', range=(0, 100))
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WSS Magnitude Relative Error (|true| > 1e-12)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'error_distribution.png'}")
    plt.close()


def analyze_by_re_and_aspect(results, output_dir='results'):
    """Analyze error breakdown by Reynolds number and aspect ratio."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Compute errors per point
    abs_error_mag = np.abs(results['target_mag'] - results['pred_mag'])
    rel_error_mag = results['rel_error_mag']
    
    # Filter relative errors
    valid_mask = results['target_mag'] > 1e-12
    rel_error_mag_filtered = rel_error_mag[valid_mask]
    re_filtered = results['re_values'][valid_mask]
    aspect_filtered = results['aspect_ratios'][valid_mask]
    
    # 1. Absolute error vs Reynolds number
    ax = axes[0, 0]
    scatter = ax.scatter(results['re_values'], abs_error_mag, c=results['aspect_ratios'], 
                        cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('Absolute Error in WSS Magnitude (Pa)', fontsize=12)
    ax.set_title('Error vs Reynolds Number', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Aspect Ratio (Lx/Ly)', fontsize=10)
    
    # 2. Relative error vs Reynolds number
    ax = axes[0, 1]
    scatter = ax.scatter(re_filtered, rel_error_mag_filtered, c=aspect_filtered, 
                        cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('Relative Error in WSS Magnitude (%)', fontsize=12)
    ax.set_title('Relative Error vs Reynolds Number', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Aspect Ratio (Lx/Ly)', fontsize=10)
    
    # 3. Error by aspect ratio (box plot)
    ax = axes[1, 0]
    unique_aspects = np.unique(results['aspect_ratios'])
    aspect_labels = [f'{ar:.2f}' for ar in unique_aspects]
    
    error_by_aspect = []
    for ar in unique_aspects:
        mask = results['aspect_ratios'] == ar
        errors = abs_error_mag[mask]
        error_by_aspect.append(errors)
    
    bp = ax.boxplot(error_by_aspect, labels=aspect_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Aspect Ratio (Lx/Ly)', fontsize=12)
    ax.set_ylabel('Absolute Error in WSS Magnitude (Pa)', fontsize=12)
    ax.set_title('Error Distribution by Aspect Ratio', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Mean error by Reynolds number bins
    ax = axes[1, 1]
    re_bins = np.linspace(results['re_values'].min(), results['re_values'].max(), 20)
    re_centers = (re_bins[:-1] + re_bins[1:]) / 2
    
    mean_errors = []
    std_errors = []
    for i in range(len(re_bins) - 1):
        mask = (results['re_values'] >= re_bins[i]) & (results['re_values'] < re_bins[i+1])
        if mask.sum() > 0:
            mean_errors.append(abs_error_mag[mask].mean())
            std_errors.append(abs_error_mag[mask].std())
        else:
            mean_errors.append(np.nan)
            std_errors.append(np.nan)
    
    mean_errors = np.array(mean_errors)
    std_errors = np.array(std_errors)
    
    ax.plot(re_centers, mean_errors, 'o-', color='blue', linewidth=2, markersize=6, label='Mean Error')
    ax.fill_between(re_centers, mean_errors - std_errors, mean_errors + std_errors, 
                     alpha=0.3, color='blue', label='± 1 Std Dev')
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (Pa)', fontsize=12)
    ax.set_title('Mean Error vs Reynolds Number', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_re_and_aspect.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'error_by_re_and_aspect.png'}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("EVALUATING TRAINED WSS PREDICTION MODEL")
    print("="*60)
    
    # Load normalization stats
    stats_path = Path(config.processed_data_dir) / 'normalization_stats.json'
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print(f"\nLoaded normalization stats from: {stats_path}")
    
    # Load test data
    print("\nLoading test data...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=config.batch_size)
    print(f"Test set: {len(test_loader.dataset)} graphs, {len(test_loader)} batches")
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_best_model('Models/best_model.pt')
    
    # Evaluate
    print("\nRunning evaluation on test set...")
    metrics, results = evaluate_model(model, test_loader, stats)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics to JSON
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Generate visualizations
    print("\nGenerating visualization plots...")
    plot_predictions_vs_truth(results, output_dir)
    plot_error_distribution(results, output_dir)
    analyze_by_re_and_aspect(results, output_dir)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - test_metrics.json")
    print("  - predictions_vs_truth.png")
    print("  - error_distribution.png")
    print("  - error_by_re_and_aspect.png")


if __name__ == '__main__':
    main()
