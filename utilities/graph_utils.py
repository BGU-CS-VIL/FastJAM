# Packages
import torch
from torch_geometric.data import Data
from collections import defaultdict
import itertools
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _normalize_keypoints_RoMa(coords, image_size):
    """
    Normalize keypoints to [-1, 1] using a homography that centers at (0,0)
    
    Args:
        coords: list or array of [x, y] or shape (N, 2)
        image_size: tuple (H, W)

    Returns:
        normalized_coords: list of [x_norm, y_norm]
    """
    H, W = image_size
    coords = np.array(coords)
    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])  # (N, 3)

    T = torch.tensor([
        [2.0 / W, 0.0, -1.0],
        [0.0, 2.0 / H, -1.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    coords_h = torch.from_numpy(coords_h).float()  # (N, 3)
    norm_coords_h = (T @ coords_h.T).T  # (N, 3)
    norm_coords = norm_coords_h[:, :2] / norm_coords_h[:, 2:3]
    return norm_coords.tolist()

def build_graph_from_matches_RoMa(matched_keypoints_by_pair, image_size, device='cpu'):
    node_map = {}  # (x, y, image_idx) -> node_id
    image_kpts = defaultdict(list)           # image_idx -> list of node_ids
    image_kpts_coords = defaultdict(list)    # image_idx -> list of [x, y]
    point_index_map = defaultdict(dict)      # image_idx -> {(x,y): local_index}
    node_coords = []
    next_node_id = 0
    source = []
    target = []

    for (i, j), (mkpts0, mkpts1) in matched_keypoints_by_pair.items():
        match_pairs = []
        for pt0, pt1 in zip(mkpts0, mkpts1):
            key0 = (float(pt0[0]), float(pt0[1]), i)
            key1 = (float(pt1[0]), float(pt1[1]), j)

            # Add key0
            if key0 not in node_map:
                node_map[key0] = next_node_id
                local_idx_i = len(image_kpts_coords[i])
                point_index_map[i][(pt0[0], pt0[1])] = local_idx_i
                image_kpts[i].append(next_node_id)
                image_kpts_coords[i].append([pt0[0], pt0[1]])
                node_coords.append([pt0[0], pt0[1]])
                next_node_id += 1
            else:
                local_idx_i = point_index_map[i][(pt0[0], pt0[1])]

            # Add key1
            if key1 not in node_map:
                node_map[key1] = next_node_id
                local_idx_j = len(image_kpts_coords[j])
                point_index_map[j][(pt1[0], pt1[1])] = local_idx_j
                image_kpts[j].append(next_node_id)
                image_kpts_coords[j].append([pt1[0], pt1[1]])
                node_coords.append([pt1[0], pt1[1]])
                next_node_id += 1
            else:
                local_idx_j = point_index_map[j][(pt1[0], pt1[1])]

            # Add match
            match_pairs.append((local_idx_i, local_idx_j))

            # Graph edges
            source.append(node_map[key0])
            target.append(node_map[key1])
            source.append(node_map[key1])
            target.append(node_map[key0])

    for node_ids in image_kpts.values():
        for u, v in itertools.permutations(node_ids, 2):
            source.append(u)
            target.append(v)

    batch = torch.zeros(len(node_coords), dtype=torch.long)
    for (x, y, img_idx), node_id in node_map.items():
        batch[node_id] = img_idx

    norm_coords = _normalize_keypoints_RoMa(node_coords, image_size)
    x = torch.tensor(norm_coords, dtype=torch.float).to(device)
    edge_index = torch.tensor([source, target], dtype=torch.long).to(device)
    batch = batch.to(device)

    graph = Data(x=x, edge_index=edge_index, batch=batch)
    return graph, image_kpts_coords

def build_graph_from_fused_keypoints(fused_keypoints, fused_batches, orig_to_fused_map, original_data):
    device = fused_keypoints.device

    u = original_data.edge_index[0]
    v = original_data.edge_index[1]

    # GPU-optimized mapping using tensor operations
    # Create a mapping tensor for fast lookup
    # Ensure mapping tensor is large enough for all possible node indices
    max_orig_node_from_map = max(orig_to_fused_map.keys()) if orig_to_fused_map else 0
    max_orig_node_from_edges = max(torch.max(u).item(), torch.max(v).item()) if u.size(0) > 0 else 0
    max_orig_node = max(max_orig_node_from_map, max_orig_node_from_edges)
    mapping_tensor = torch.full((max_orig_node + 1,), -1, dtype=torch.long, device=device)
    
    for orig_idx, fused_idx in orig_to_fused_map.items():
        mapping_tensor[orig_idx] = fused_idx
    
    # Vectorized mapping
    mapped_u = mapping_tensor[u]
    mapped_v = mapping_tensor[v]
    
    # Filter valid edges
    mask = (mapped_u >= 0) & (mapped_v >= 0) & (mapped_u != mapped_v)
    edge_u = mapped_u[mask]
    edge_v = mapped_v[mask]

    # Remove duplicate edges using a low-memory CPU-based hash unique
    if edge_u.size(0) > 0:
        # Move to CPU to avoid large GPU allocations during deduplication
        num_nodes = int(fused_keypoints.shape[0])
        keys = (edge_u.detach().to('cpu', non_blocking=True).long() * num_nodes 
                + edge_v.detach().to('cpu', non_blocking=True).long())
        # Unique on CPU (sorted for deterministic order), then reconstruct on device
        unique_keys = torch.unique(keys, sorted=True)
        edge_u_unique = (unique_keys // num_nodes).to(device, non_blocking=True)
        edge_v_unique = (unique_keys %  num_nodes).to(device, non_blocking=True)
        edge_index_new = torch.stack([edge_u_unique, edge_v_unique], dim=0)
        # Cleanup CPU temporaries
        del keys, unique_keys
    else:
        edge_index_new = torch.empty((2, 0), dtype=torch.long, device=device)

    return Data(x=fused_keypoints, edge_index=edge_index_new, batch=fused_batches)