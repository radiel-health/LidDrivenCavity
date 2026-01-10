# Baseline 4-Wall Model Results

This directory contains archived results from the original model trained on all 4 walls (top/bottom/left/right).

## Contents

- `best_model.pt` - Best model checkpoint from training
- `training_history.json` - Training and validation loss history
- `comprehensive_validation_results/` - Per-case validation metrics
- `comprehensive_inference_results/` - Inference predictions on test cases

## Model Configuration

- **Walls trained on**: 4 walls (top, bottom, left, right)
- **filter_top_wall**: False
- **Architecture**: Two-stream GNN with FiLM modulation
  - Flow Encoder: MLP([Re, Lx, Ly] → context_dim=64)
  - Geometry Encoder: 3 GCN layers (hidden_dim=64)
  - Task Head: 2 GCN layers + BNN MLP (hidden_dim=128)
- **Training**: MSE loss on log-normalized WSS
- **Batch size**: 8
- **Epochs**: 200 (with early stopping)

## Purpose

These results serve as a baseline for comparison with the new 3-wall model (filter_top_wall=True).

The hypothesis is that removing the top (moving lid) wall from training will improve prediction quality on the three stationary walls (bottom/left/right), as the top wall consistently showed higher errors and is not representative of the final application domain.

## Date Archived

January 6, 2026

## Next Steps

1. Enable `filter_top_wall=True` in config.py
2. Retrain model on 3-wall data
3. Compare validation and inference metrics against this baseline
4. Compute improvement percentage on stationary walls
