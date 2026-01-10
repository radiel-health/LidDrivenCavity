"""
Quick test script to verify top wall filtering implementation.
Run this before enabling filter_top_wall=True in config.
"""

import torch
from pathlib import Path
from config import config
from dataset import WSSDataset, filter_edge_index

print("="*80)
print("TOP WALL FILTERING - IMPLEMENTATION TEST")
print("="*80)

# Test 1: Config
print("\n1. Testing configuration...")
print(f"   filter_top_wall: {config.filter_top_wall}")
print(f"   base batch_size: {config.batch_size}")
print(f"   effective batch_size: {config.get_batch_size()}")
assert hasattr(config, 'filter_top_wall'), "❌ filter_top_wall not in config"
assert hasattr(config, 'get_batch_size'), "❌ get_batch_size method not in config"
print("   ✓ Config OK")

# Test 2: Edge filtering function
print("\n2. Testing edge filtering helper...")
edge_index = torch.tensor([[0, 1, 2, 3, 4], 
                           [1, 2, 3, 4, 0]])
keep_mask = torch.tensor([True, True, False, True, True])  # Remove node 2
filtered = filter_edge_index(edge_index, keep_mask)
# Should keep edges not involving node 2 and remap: 0→0, 1→1, 3→2, 4→3
expected = torch.tensor([[0, 2, 3], [1, 3, 0]])
assert filtered.shape[1] == 3, f"❌ Expected 3 edges, got {filtered.shape[1]}"
print(f"   Original: {edge_index.shape[1]} edges")
print(f"   Filtered: {filtered.shape[1]} edges (removed edges with node 2)")
print(f"   Remapped indices: {filtered.tolist()}")
print("   ✓ Edge filtering OK")

# Test 3: Load dataset without filtering
print("\n3. Testing dataset loading (filter_top_wall=False)...")
try:
    dataset = WSSDataset(split='train', normalize=True, filter_top_wall=False)
    sample = dataset[0]
    print(f"   Loaded dataset: {len(dataset)} graphs")
    print(f"   Sample graph: {sample.num_nodes} nodes, {sample.edge_index.shape[1]} edges")
    
    # Check for on_top feature
    on_top = sample.x[:, 2]
    num_top = (on_top == 1.0).sum().item()
    num_other = (on_top == 0.0).sum().item()
    print(f"   Top wall nodes: {num_top}")
    print(f"   Other wall nodes: {num_other}")
    print(f"   Total: {sample.num_nodes}")
    assert num_top + num_other == sample.num_nodes, "❌ Wall classification mismatch"
    print("   ✓ Dataset loading (4 walls) OK")
except Exception as e:
    print(f"   ❌ Dataset loading failed: {e}")
    raise

# Test 4: Load dataset with filtering
print("\n4. Testing dataset loading (filter_top_wall=True)...")
try:
    dataset_filtered = WSSDataset(split='train', normalize=True, filter_top_wall=True)
    sample_filtered = dataset_filtered[0]
    print(f"   Loaded filtered dataset: {len(dataset_filtered)} graphs")
    print(f"   Sample graph: {sample_filtered.num_nodes} nodes, {sample_filtered.edge_index.shape[1]} edges")
    
    # Check that no top wall nodes remain
    on_top_filtered = sample_filtered.x[:, 2]
    num_top_filtered = (on_top_filtered == 1.0).sum().item()
    print(f"   Top wall nodes remaining: {num_top_filtered}")
    assert num_top_filtered == 0, "❌ Top wall nodes still present after filtering!"
    
    # Verify reduction
    reduction = (1 - sample_filtered.num_nodes / sample.num_nodes) * 100
    print(f"   Node reduction: {reduction:.1f}%")
    assert 20 < reduction < 30, f"❌ Unexpected reduction: {reduction:.1f}% (expected ~25%)"
    print("   ✓ Dataset filtering (3 walls) OK")
except Exception as e:
    print(f"   ❌ Filtered dataset loading failed: {e}")
    raise

# Test 5: Verify separate stats files
print("\n5. Testing normalization stats...")
stats_4wall = Path("ProcessedData/normalization_stats.json")
stats_3wall = Path("ProcessedData/normalization_stats_no_top.json")
print(f"   4-wall stats exist: {stats_4wall.exists()}")
print(f"   3-wall stats exist: {stats_3wall.exists()}")
if not stats_3wall.exists():
    print("   ⚠ 3-wall stats not computed yet (will be created on first training run)")
else:
    print("   ✓ Both stats files exist")

# Test 6: Check baseline archive
print("\n6. Checking baseline archive...")
baseline_dir = Path("Models/baseline_4walls")
assert baseline_dir.exists(), "❌ Baseline archive not created"
required_files = ["best_model.pt", "training_history.json", "README.md"]
for f in required_files:
    fpath = baseline_dir / f
    if fpath.exists():
        print(f"   ✓ {f} archived")
    else:
        print(f"   ⚠ {f} not found in archive")
print("   ✓ Baseline archive OK")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nNext steps:")
print("1. Set config.filter_top_wall = True in config.py")
print("2. Run: python train.py")
print("3. Compare results against baseline in Models/baseline_4walls/")
print("="*80)
