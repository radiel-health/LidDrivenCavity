# Purpose: Convert your boundary CSVs into PyTorch Geometric graph objects (run this ONCE)
# What it needs to do:

# Read each CSV file (parse space-delimited format)
# Extract coordinates, WSS, pressure from the columns
# Create normalized node features (x/Lx, y/Ly, boundary flags, etc.)
# Build edge connectivity (which boundary nodes connect to which)
# Package everything into a PyG Data object with:

# x: node features
# edge_index: connectivity
# y: WSS target values
# flow_params: [Re, Lx, Ly]
# pos: coordinates (for visualization later)


# Save each graph as a .pt file

# Output: One .pt file per CSV (so ~195 graph files in data/processed/)
# Why: You do this preprocessing once, then training loads graphs quickly


# why --> 

"""
Preprocessing: Convert boundary CSV files to PyTorch Geometric graphs

This script:
1. Loads boundary data from ANSYS CSV files (moving + stationary walls)
2. Computes normalized, physics-informed features (including arc length)
3. Constructs graph connectivity (boundary mesh edges)
4. Saves PyG Data objects for training

Run once:
    python Preprocessing/pre_process.py

Output: ProcessedData/{aspect_ratio}/Re{re}.pt files
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import config

def load_boundary_csv(csv_path):
    """
    Load boundary data from ANSYS CSV file (space-delimited)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        dict with:
            - coords: [n_nodes, 2] numpy array
            - wss_mag: [n_nodes] numpy array (magnitude)
            - wss_x: [n_nodes] numpy array (x-component)
            - wss_y: [n_nodes] numpy array (y-component)
            - pressure: [n_nodes] numpy array
            - wall_type: str ('moving' or 'stationary')
    """
    # Load space-delimited CSV
    df = pd.read_csv(csv_path, delim_whitespace=True)
    
    # Extract coordinates (using actual column names from your data)
    x = df['x-coordinate'].values
    y = df['y-coordinate'].values
    coords = np.column_stack([x, y])
    
    # Extract WSS components
    wss_mag = df['wall-shear'].values
    wss_x = df['x-wall-shear'].values
    wss_y = df['y-wall-shear'].values
    
    # Extract pressure
    pressure = df['pressure'].values
    
    # Determine wall type from filename
    wall_type = 'moving' if 'moving_wall' in str(csv_path) else 'stationary'
    
    return {
        'coords': coords,
        'wss_mag': wss_mag,
        'wss_x': wss_x,
        'wss_y': wss_y,
        'pressure': pressure,
        'wall_type': wall_type,
    }

def load_case_data(ar, re):
    """
    Load both moving and stationary wall data for one case
    
    Args:
        ar: aspect ratio ("1x1", "1x2", "2x1")
        re: Reynolds number
        
    Returns:
        Combined dict with all boundary nodes, or None if files missing
    """
    folder_name = config.aspect_ratio_folders[ar]
    re_dir = f"Re{re}"
    base_path = config.data_root / folder_name / re_dir
    
    # Construct file paths
    moving_file = base_path / f"moving_wall_full_Re{re}.csv"
    
    # Try both stationary file naming conventions
    stat_file = base_path / f"stat_walls_full_Re{re}.csv"
    if not stat_file.exists():
        stat_file = base_path / f"stationary_walls_full_Re{re}.csv"
    
    # Check if files exist
    if not moving_file.exists() or not stat_file.exists():
        return None
    
    try:
        # Load both files
        moving_data = load_boundary_csv(moving_file)
        stat_data = load_boundary_csv(stat_file)
        
        # Concatenate all boundary nodes
        coords = np.vstack([moving_data['coords'], stat_data['coords']])
        wss_mag = np.concatenate([moving_data['wss_mag'], stat_data['wss_mag']])
        wss_x = np.concatenate([moving_data['wss_x'], stat_data['wss_x']])
        wss_y = np.concatenate([moving_data['wss_y'], stat_data['wss_y']])
        pressure = np.concatenate([moving_data['pressure'], stat_data['pressure']])
        
        # Create wall type labels (1=moving, 0=stationary)
        n_moving = len(moving_data['coords'])
        n_stat = len(stat_data['coords'])
        wall_type_labels = np.concatenate([
            np.ones(n_moving), # array of ones
            np.zeros(n_stat)  # array of zeros
        ])

        # above, we can decipher the moving walls to the stationary walls, because of the 'coords' array that seperates the walls that don't move to the ones that do...
        
        return {
            'coords': coords,
            'wss_mag': wss_mag,
            'wss_x': wss_x,
            'wss_y': wss_y,
            'pressure': pressure,
            'wall_type': wall_type_labels,
        }
    
    except Exception as e:
        print(f"    Error loading {ar}/Re{re}: {str(e)}")
        return None
    
def classify_walls(coords, Lx, Ly):
    """
    Determine which wall each node is on using actual coordinate bounds.
    
    Note: Lx, Ly are the PHYSICAL domain dimensions (e.g., 1m × 1m),
    but the mesh coordinates might not start at (0,0). We use actual bounds.
    
    Args:
        coords: [n_nodes, 2] array
        Lx, Ly: domain dimensions (used for tolerance calculation only)
        
    Returns:
        4 binary masks: on_bottom, on_right, on_top, on_left
        Each is [n_nodes] boolean array
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Use actual coordinate bounds (mesh might not start at origin)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Tolerance for floating point comparison
    tol = config.wall_tolerance * min(x_max - x_min, y_max - y_min)
    
    # Classify each wall based on actual bounds
    on_bottom = np.abs(y - y_min) < tol
    on_top = np.abs(y - y_max) < tol
    on_left = np.abs(x - x_min) < tol
    on_right = np.abs(x - x_max) < tol
    
    return on_bottom, on_right, on_top, on_left

def sort_boundary_ccw(coords, Lx, Ly, wall_classifications=None):
    """
    Sort boundary nodes counter-clockwise starting from bottom-left corner
    
    Strategy:
        1. Classify nodes into 4 walls
        2. Sort each wall independently
        3. Concatenate in CCW order (bottom → right → top → left)
        4. Skip corner duplicates
    
    Args:
        coords: [n_nodes, 2] array
        Lx, Ly: domain dimensions
        wall_classifications: Optional tuple (on_bottom, on_right, on_top, on_left)
                            If provided, skips wall classification step
        
    Returns:
        sorted_indices: [n_nodes] array of indices in CCW order
    """
    # Reuse wall classifications if provided, otherwise compute them
    if wall_classifications is not None:
        on_bottom, on_right, on_top, on_left = wall_classifications
    else:
        on_bottom, on_right, on_top, on_left = classify_walls(coords, Lx, Ly)
    
    # Get indices for each wall
    bottom_idx = np.where(on_bottom)[0]
    right_idx = np.where(on_right)[0]
    top_idx = np.where(on_top)[0]
    left_idx = np.where(on_left)[0]
    
    # Sort each wall:
    # Bottom: left to right (increasing x)
    bottom_sorted = bottom_idx[np.argsort(coords[bottom_idx, 0])]
    
    # Right: bottom to top (increasing y)
    right_sorted = right_idx[np.argsort(coords[right_idx, 1])]
    
    # Top: right to left (decreasing x)
    top_sorted = top_idx[np.argsort(-coords[top_idx, 0])]
    
    # Left: top to bottom (decreasing y)
    left_sorted = left_idx[np.argsort(-coords[left_idx, 1])]
    
    # Check if corners are duplicated (appear in multiple walls)
    n_walls_per_node = (on_bottom.astype(int) + on_right.astype(int) + 
                        on_top.astype(int) + on_left.astype(int))
    has_corner_duplicates = np.any(n_walls_per_node > 1)
    
    if has_corner_duplicates:
        # Corners appear in multiple wall lists - skip them when concatenating
        sorted_indices = np.concatenate([
            bottom_sorted,           # All bottom nodes (includes both bottom corners)
            right_sorted[1:],        # Right nodes (skip bottom-right corner)
            top_sorted[1:],          # Top nodes (skip top-right corner)
            left_sorted[1:-1],       # Left nodes (skip both corners)
        ])
    else:
        # No duplicates - just concatenate all walls
        sorted_indices = np.concatenate([
            bottom_sorted,
            right_sorted,
            top_sorted,
            left_sorted,
        ])
    
    return sorted_indices


def compute_arc_length(coords, Lx, Ly, wall_classifications=None):
    """
    Compute normalized arc length (0 to 1) around boundary perimeter
    
    This is a key surface coordinate that helps the model understand
    position along the boundary path.
    
    Args:
        coords: [n_nodes, 2] array
        Lx, Ly: domain dimensions
        
    Returns:
        s: [n_nodes] array of normalized arc lengths (0 to 1)
    """
    n_nodes = len(coords)
    
    # 1. Sort nodes CCW
    sorted_idx = sort_boundary_ccw(coords, Lx, Ly, wall_classifications)
    sorted_coords = coords[sorted_idx]
    
    # 2. Compute distance between consecutive nodes (with wraparound)
    # Next coord is shifted by 1, then wrap last back to first
    next_coords = np.roll(sorted_coords, -1, axis=0)
    distances = np.linalg.norm(sorted_coords - next_coords, axis=1)
    
    # 3. Cumulative arc length
    cumulative = np.cumsum(distances)
    arc_lengths = np.zeros(n_nodes)
    arc_lengths[0] = 0.0  # Start at 0
    arc_lengths[1:] = cumulative[:-1]  # Shift by one
    
    # 4. Normalize by total perimeter
    total_perimeter = cumulative[-1]
    s_normalized = arc_lengths / total_perimeter
    
    # 5. Unsort: map back to original node order
    s_per_node = np.zeros(n_nodes)
    s_per_node[sorted_idx] = s_normalized
    
    return s_per_node

def compute_corner_distance(coords, Lx, Ly):
    """
    Near corners we have wacky behavior (stress concentrations).
    But, we don't actually care about sharp 90 degree corners, the human body has no 
         sharp corners...
    Compute normalized distance to nearest corner
    
    Corners are regions of stress concentration in lid-driven cavity,
    so distance to corner is an important feature.
    
    Args:
        coords: [n_nodes, 2] array
        Lx, Ly: domain dimensions
        
    Returns:
        corner_dist: [n_nodes] array of normalized distances
    """
    # Four corners
    corners = np.array([
        [0, 0],      # bottom-left
        [Lx, 0],     # bottom-right
        [Lx, Ly],    # top-right
        [0, Ly],     # top-left
    ])
    
    # Distance from each node to each corner
    # Broadcasting: coords[n, 2] vs corners[4, 2]
    # distances[n, 4] = distance from node n to corner c
    distances = np.linalg.norm(
        coords[:, np.newaxis, :] - corners[np.newaxis, :, :],
        axis=2
    )
    
    # Minimum distance to any corner
    min_corner_dist = np.min(distances, axis=1)
    
    # Normalize by diagonal length
    diagonal = np.sqrt(Lx**2 + Ly**2)
    normalized = min_corner_dist / diagonal
    
    return normalized


def create_features(coords, wall_type_labels, Lx, Ly):
    """
    Create all node features for boundary mesh
    
    Features (10 total):
        0: x_normalized (x / Lx)
        1: y_normalized (y / Ly)
        2: on_top (binary)
        3: on_bottom (binary)
        4: on_left (binary)
        5: on_right (binary)
        6: arc_length (0 to 1, surface coordinate)
        7: distance_to_nearest_corner (normalized)
        8: wall_type (1=moving/top, 0=stationary)
        9: (reserved for local_mesh_spacing if needed)
    
    Args:
        coords: [n_nodes, 2] array
        wall_type_labels: [n_nodes] array (1=moving, 0=stationary)
        Lx, Ly: domain dimensions
        
    Returns:
        features: [n_nodes, 10] torch tensor
    """
    x = coords[:, 0]
    y = coords[:, 1]
    n_nodes = len(coords)
    
    # Get actual coordinate bounds (mesh might not start at origin)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # 1. Normalized Cartesian coordinates (scale-invariant, mapped to [0,1])
    # Use actual bounds for normalization, not Lx/Ly
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    
    # 2. Wall classification (which wall am I on?)
    # Compute once and reuse for arc length calculation
    wall_classifications = classify_walls(coords, Lx, Ly)
    on_bottom, on_right, on_top, on_left = wall_classifications
    
    # 3. Arc length (position along boundary path)
    # Pass wall classifications to avoid recomputation
    s = compute_arc_length(coords, Lx, Ly, wall_classifications)
    
    # 4. Corner proximity (WSS singularities at corners)
    corner_dist = compute_corner_distance(coords, Lx, Ly)
    
    # 5. Wall type (moving vs stationary)
    wall_type = wall_type_labels
    
    # 6. Placeholder for future feature (local mesh spacing)
    # For now, set to zero
    placeholder = np.zeros(n_nodes)
    
    # Stack all features [n_nodes, 10]
    features = np.stack([
        x_norm,                  # 0
        y_norm,                  # 1
        on_top.astype(float),    # 2
        on_bottom.astype(float), # 3
        on_left.astype(float),   # 4
        on_right.astype(float),  # 5
        s,                       # 6 - arc length
        corner_dist,             # 7
        wall_type,               # 8
        placeholder,             # 9 - reserved
    ], axis=1)
    
    # Convert to torch tensor
    return torch.tensor(features, dtype=torch.float32)

def create_edges(coords, Lx, Ly, wall_classifications=None):
    """
    Create edge connectivity for boundary mesh (ring graph)
    
    Connects consecutive nodes along boundary perimeter
    
    Args:
        coords: [n_nodes, 2] array
        Lx, Ly: domain dimensions
        wall_classifications: Optional tuple (on_bottom, on_right, on_top, on_left)
                            If provided, passed to sort_boundary_ccw for efficiency
        
    Returns:
        edge_index: [2, n_edges] torch tensor (undirected)
    """
    n_nodes = len(coords)
    
    # Sort nodes CCW to get proper connectivity
    sorted_idx = sort_boundary_ccw(coords, Lx, Ly, wall_classifications)
    
    # Connect consecutive nodes in sorted order
    edges = []
    for i in range(n_nodes):
        curr = sorted_idx[i]
        next = sorted_idx[(i + 1) % n_nodes]  # Wrap around at end
        edges.append([curr, next])
    
    edges = np.array(edges, dtype=np.int64)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # Make undirected (add reverse edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return edge_index
def create_graph(coords, wss_components, pressure, wall_type_labels, flow_params, metadata):
    """
    Package everything into PyG Data object
    
    Args:
        coords: [n_nodes, 2] numpy array
        wss_components: dict with 'mag', 'x', 'y' numpy arrays [n_nodes]
        pressure: [n_nodes] numpy array
        wall_type_labels: [n_nodes] numpy array (1=moving, 0=stationary)
        flow_params: [3] list [Re, Lx, Ly]
        metadata: dict with 're', 'ar', 'Lx', 'Ly'
        
    Returns:
        data: PyG Data object
    """
    Lx = metadata['Lx']
    Ly = metadata['Ly']
    
    # Compute wall classifications once
    wall_classifications = classify_walls(coords, Lx, Ly)
    
    # Create features (will use wall_classifications internally via create_features)
    features = create_features(coords, wall_type_labels, Lx, Ly)
    
    # Create edges (pass wall_classifications to avoid recomputation)
    edge_index = create_edges(coords, Lx, Ly, wall_classifications)
    
    # Target: WSS components [n_nodes, 2] (x and y components)
    # Model will predict components, magnitude can be derived
    if config.predict_components_only:
        y_target = np.column_stack([wss_components['x'], wss_components['y']])
    else:
        # Include magnitude as well [n_nodes, 3]
        y_target = np.column_stack([
            wss_components['mag'],
            wss_components['x'],
            wss_components['y']
        ])
    
    # Create Data object
    data = Data(
        # Node features [n_nodes, 10]
        x=features,
        
        # Edge connectivity [2, n_edges]
        edge_index=edge_index,
        
        # Target: WSS components [n_nodes, 2 or 3]
        y=torch.tensor(y_target, dtype=torch.float32),
        
        # Flow parameters [3]  [Re, Lx, Ly]
        flow_params=torch.tensor(flow_params, dtype=torch.float32),
        
        # Node positions (for visualization) [n_nodes, 2]
        pos=torch.tensor(coords, dtype=torch.float32),
        
        # Metadata (for reference)
        re=metadata['re'],
        ar=metadata['ar'],
        lx=metadata['Lx'],
        ly=metadata['Ly'],
        
        # Store number of nodes for batching
        num_nodes=len(coords),
    )
    
    return data

def process_one_case(ar, re):
    """
    Process one case (moving + stationary walls) → one PyG graph
    
    Args:
        ar: aspect ratio string ("1x1", "2x1", "1x2")
        re: Reynolds number (int)
        
    Returns:
        PyG Data object, or None if files don't exist or should be excluded
    """
    # Check if should exclude (data quality issues)
    if config.should_exclude_case(ar, re):
        return None
    
    # Load data (combines moving and stationary walls)
    data = load_case_data(ar, re)
    
    if data is None:
        return None
    
    try:
        # Extract data
        coords = data['coords']
        wss_components = {
            'mag': data['wss_mag'],
            'x': data['wss_x'],
            'y': data['wss_y'],
        }
        pressure = data['pressure']
        wall_type_labels = data['wall_type']
        
        # Get domain size
        Lx = config.domain_sizes[ar]['Lx']
        Ly = config.domain_sizes[ar]['Ly']
        
        # Create graph
        flow_params = [float(re), Lx, Ly]
        metadata = {
            're': re,
            'ar': ar,
            'Lx': Lx,
            'Ly': Ly,
        }
        
        graph = create_graph(
            coords, 
            wss_components, 
            pressure, 
            wall_type_labels,
            flow_params, 
            metadata
        )
        
        return graph
        
    except Exception as e:
        print(f"    ❌ Error creating graph for {ar}/Re{re}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Process all CSV files and save PyG graphs
    """
    print("=" * 80)
    print("PREPROCESSING: CSV → PyG Graphs")
    print("=" * 80)
    print()
    
    # Create output directories
    print("Creating output directories...")
    config.create_directories()
    print()
    
    # Print configuration summary
    print(f"Data source: {config.data_root}")
    print(f"Output: {config.processed_data_dir}")
    print(f"Node features: {config.node_feature_dim}")
    print(f"Target: {config.target_dim}D WSS {'components (x,y)' if config.predict_components_only else '(mag,x,y)'}")
    print()
    
    # Statistics
    total_cases = 0
    processed = 0
    skipped = 0
    failed = 0
    failed_files = []
    skipped_files = []
    
    # Process all aspect ratios
    for ar in config.aspect_ratios:
        print(f"\n{'='*80}")
        print(f"Processing aspect ratio: {ar}")
        print(f"{'='*80}")
        
        Lx = config.domain_sizes[ar]['Lx']
        Ly = config.domain_sizes[ar]['Ly']
        print(f"Domain size: {Lx} × {Ly} m")
        
        # Check for exclusions
        if ar in config.exclude_cases:
            print(f"⚠️  Will skip: Re {config.exclude_cases[ar]} (data quality)")
        
        ar_processed = 0
        ar_skipped = 0
        ar_failed = 0
        
        # Process all Re values for this aspect ratio
        with tqdm(config.re_values, desc=f"{ar}", ncols=100) as pbar:
            for re in pbar:
                total_cases += 1
                pbar.set_description(f"{ar}/Re{re:4d}")
                
                # Check if should skip
                if config.should_exclude_case(ar, re):
                    skipped += 1
                    ar_skipped += 1
                    skipped_files.append(f"{ar}/Re{re}")
                    continue
                
                # Process
                graph = process_one_case(ar, re)
                
                if graph is not None:
                    # Save
                    output_path = config.processed_data_dir / ar / f"Re{re}.pt"
                    torch.save(graph, output_path)
                    processed += 1
                    ar_processed += 1
                    
                    # Update progress bar with stats
                    pbar.set_postfix({
                        'nodes': graph.num_nodes,
                        'edges': graph.edge_index.shape[1]
                    })
                else:
                    failed += 1
                    ar_failed += 1
                    failed_files.append(f"{ar}/Re{re}")
        
        # Aspect ratio summary
        print(f"\n{ar} Summary:")
        print(f"  ✅ Processed: {ar_processed}")
        if ar_skipped > 0:
            print(f"  ⚠️  Skipped: {ar_skipped}")
        if ar_failed > 0:
            print(f"  ❌ Failed: {ar_failed}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total cases attempted: {total_cases}")
    print(f"✅ Successfully processed: {processed}")
    if skipped > 0:
        print(f"⚠️  Skipped (excluded): {skipped}")
        if len(skipped_files) <= 10:
            for f in skipped_files:
                print(f"    - {f}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
        print("\nFailed files:")
        for f in failed_files[:10]:  # Show first 10
            print(f"    - {f}")
        if len(failed_files) > 10:
            print(f"    ... and {len(failed_files) - 10} more")
    
    print(f"\n📁 Processed graphs saved to: {config.processed_data_dir}")
    print(f"📊 Expected usable cases: {config.total_cases_usable}")
    print(f"📊 Actual processed: {processed}")
    
    if processed == config.total_cases_usable:
        print("\n✅ SUCCESS: All expected cases processed!")
    else:
        print(f"\n⚠️  WARNING: Processed {processed}, expected {config.total_cases_usable}")
    
    print("=" * 80)

def test_single_case():
    """
    Test preprocessing on a single case for debugging
    """
    print("=" * 80)
    print("TESTING: Single Case Processing")
    print("=" * 80)
    
    # Test case: 1x1, Re=200 (should have clean data)
    test_ar = "1x1"
    test_re = 200
    
    print(f"\nTest case: {test_ar}, Re={test_re}")
    print(f"Domain: {config.domain_sizes[test_ar]}")
    
    # Process
    print("\nProcessing...")
    graph = process_one_case(test_ar, test_re)
    
    if graph is None:
        print("❌ Failed to process test case!")
        return False
    
    # Inspect graph
    print("\n✅ Graph created successfully!")
    print(f"\nGraph properties:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Target shape: {graph.y.shape}")
    print(f"  Flow params: {graph.flow_params}")
    print(f"  Metadata: Re={graph.re}, AR={graph.ar}, Lx={graph.lx}, Ly={graph.ly}")
    
    # Check feature ranges
    print(f"\nFeature ranges:")
    for i, name in enumerate(['x_norm', 'y_norm', 'on_top', 'on_bottom', 
                              'on_left', 'on_right', 'arc_length', 
                              'corner_dist', 'wall_type', 'placeholder']):
        feat_min = graph.x[:, i].min().item()
        feat_max = graph.x[:, i].max().item()
        print(f"  {i}. {name:15s}: [{feat_min:8.4f}, {feat_max:8.4f}]")
    
    # Check target ranges
    print(f"\nTarget (WSS) ranges:")
    print(f"  x-component: [{graph.y[:, 0].min():.6e}, {graph.y[:, 0].max():.6e}]")
    print(f"  y-component: [{graph.y[:, 1].min():.6e}, {graph.y[:, 1].max():.6e}]")
    
    # Check for NaNs
    has_nan_x = torch.isnan(graph.x).any()
    has_nan_y = torch.isnan(graph.y).any()
    print(f"\nData quality:")
    print(f"  NaNs in features: {'❌ YES' if has_nan_x else '✅ NO'}")
    print(f"  NaNs in targets: {'❌ YES' if has_nan_y else '✅ NO'}")
    
    # Save test graph
    test_path = config.processed_data_dir / f"test_{test_ar}_Re{test_re}.pt"
    torch.save(graph, test_path)
    print(f"\n📁 Test graph saved to: {test_path}")
    
    print("\n" + "=" * 80)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CFD data to PyG graphs')
    parser.add_argument('--test', action='store_true', help='Run test on single case')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    if args.test:
        # Test mode: process single case
        success = test_single_case()
        sys.exit(0 if success else 1)
    else:
        # Full processing
        main()