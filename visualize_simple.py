"""
Simple spatial visualization of WSS predictions on cavity boundaries.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import json

from config import config
from dataset import WSSDataset
from evaluate import load_best_model, denormalize_wss
from torch_geometric.loader import DataLoader


def safe_float(val):
    """Convert any value to float safely."""
    if isinstance(val, (int, float)):
        return float(val)
    elif hasattr(val, 'item'):
        return float(val.item())
    else:
        return float(val)


def plot_single_case(data, pred_phys, true_phys, case_idx, output_dir):
    """Plot ground truth, prediction, and error for one test case."""
    
    pos = data.pos.cpu().numpy()
    lx = safe_float(data.lx)
    ly = safe_float(data.ly)
    re = safe_float(data.re)
    ar = lx / ly  # Compute numeric aspect ratio
    
    # Get actual bounds from node positions
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    
    # Compute magnitudes and error
    true_mag = np.linalg.norm(true_phys, axis=1)
    pred_mag = np.linalg.norm(pred_phys, axis=1)
    error_mag = np.abs(pred_mag - true_mag)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common limits for truth and pred
    vmin = min(true_mag.min(), pred_mag.min())
    vmax = max(true_mag.max(), pred_mag.max())
    
    # Plot 1: Ground Truth
    ax = axes[0]
    rect = Rectangle((x_min, y_min), lx, ly, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    sc1 = ax.scatter(pos[:, 0], pos[:, 1], c=true_mag, s=50, cmap='viridis', 
                    vmin=vmin, vmax=vmax, edgecolors='black', linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
    plt.colorbar(sc1, ax=ax, label='WSS (Pa)')
    
    # Plot 2: Prediction
    ax = axes[1]
    rect = Rectangle((x_min, y_min), lx, ly, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    sc2 = ax.scatter(pos[:, 0], pos[:, 1], c=pred_mag, s=50, cmap='viridis',
                    vmin=vmin, vmax=vmax, edgecolors='black', linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title('Prediction', fontsize=12, fontweight='bold')
    plt.colorbar(sc2, ax=ax, label='WSS (Pa)')
    
    # Plot 3: Error
    ax = axes[2]
    rect = Rectangle((x_min, y_min), lx, ly, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    sc3 = ax.scatter(pos[:, 0], pos[:, 1], c=error_mag, s=50, cmap='Reds',
                    edgecolors='black', linewidth=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title('Absolute Error', fontsize=12, fontweight='bold')
    plt.colorbar(sc3, ax=ax, label='Error (Pa)')
    
    # Overall title
    mae = error_mag.mean()
    fig.suptitle(f'Case {case_idx+1}: Re={re:.0f}, AR={ar:.2f}, MAE={mae:.2e} Pa',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / f'case_{case_idx+1:02d}_Re{re:.0f}_AR{ar:.2f}.png'.replace('.', 'p')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_summary_grid(all_data, all_preds, all_true, output_dir):
    """Create grid showing all test cases with error coloring."""
    
    n_cases = len(all_data)
    n_cols = 8
    n_rows = int(np.ceil(n_cases / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5*n_rows))
    axes = axes.flatten()
    
    for idx, (data, pred, true) in enumerate(zip(all_data, all_preds, all_true)):
        ax = axes[idx]
        
        pos = data.pos.cpu().numpy()
        lx = safe_float(data.lx)
        ly = safe_float(data.ly)
        re = safe_float(data.re)
        ar = lx / ly  # Compute numeric aspect ratio
        
        error_mag = np.linalg.norm(pred - true, axis=1)
        
        # Draw cavity
        rect = Rectangle((0, 0), lx, ly, fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Color by error
        ax.scatter(pos[:, 0], pos[:, 1], c=error_mag, s=8, cmap='Reds',
                  vmin=0, vmax=1e-7)
        
        ax.set_xlim(-0.05*lx, 1.05*lx)
        ax.set_ylim(-0.05*ly, 1.05*ly)
        ax.set_aspect('equal')
        ax.set_title(f'Re{re:.0f} AR={ar:.1f}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_cases, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('All Test Cases - Error Magnitude', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'summary_grid.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("="*60)
    print("SPATIAL VISUALIZATION")
    print("="*60)
    
    # Setup
    output_dir = Path('results/spatial_viz')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load stats
    stats_path = Path(config.processed_data_dir) / 'normalization_stats.json'
    with open(stats_path) as f:
        stats = json.load(f)
    
    # Load data
    print("\nLoading test data...")
    test_dataset = WSSDataset(split='test', normalize=True, seed=42)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    print("Loading model...")
    model = load_best_model('Models/best_model.pt')
    model.eval()
    
    # Generate predictions
    print("Generating predictions...")
    all_data = []
    all_preds_phys = []
    all_true_phys = []
    
    device = torch.device('cpu')
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            # Get unnormalized data
            data_unnorm = torch.load(test_dataset.files[idx], weights_only=False)
            
            # Predict and denormalize
            pred_norm = model(batch)
            pred_phys = denormalize_wss(pred_norm.cpu().numpy(), stats)
            true_phys = data_unnorm.y.cpu().numpy()
            
            all_data.append(data_unnorm)
            all_preds_phys.append(pred_phys)
            all_true_phys.append(true_phys)
    
    print(f"Got {len(all_data)} test cases")
    
    # Generate individual plots
    print("\nGenerating individual plots...")
    for idx, (data, pred, true) in enumerate(zip(all_data, all_preds_phys, all_true_phys)):
        plot_single_case(data, pred, true, idx, output_dir)
        print(f"  [{idx+1}/{len(all_data)}] Done")
    
    # Generate summary
    print("\nGenerating summary grid...")
    plot_summary_grid(all_data, all_preds_phys, all_true_phys, output_dir)
    
    print("\n" + "="*60)
    print("DONE!")
    print(f"Results in: {output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()
