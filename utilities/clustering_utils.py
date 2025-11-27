import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from typing import Dict, Tuple
from pdc_dp_means import DPMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dpmeans_clustering(keypoints: torch.Tensor, 
                          batch: torch.Tensor, 
                          delta: float = 10.0,
                          n_init: int = 10,
                          device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    DP-Means clustering for keypoints using the PDC-DP-Means implementation.
    
    Args:
        keypoints: (N, 2) tensor of normalized keypoints
        batch: (N,) tensor indicating which image each keypoint belongs to
        delta: delta parameter for DP-Means (replaces lambda from the paper)
        n_init: number of initializations for DP-Means
        device: device to use
        
    Returns:
        Tuple of (fused_keypoints, fused_batches, orig_to_fused_map)
    """
    # Convert to CPU for DP-Means (it's a CPU-based algorithm)
    keypoints_cpu = keypoints.cpu().numpy()
    batch_cpu = batch.cpu().numpy()
    
    all_means = []
    all_batches = []
    orig_to_fused_map = {}
    fused_node_id = 0
    
    # Process each image separately
    unique_batches = np.unique(batch_cpu)
    
    for b in unique_batches:
        # Get keypoints for this image
        mask = (batch_cpu == b)
        keypoints_in_image = keypoints_cpu[mask]
        original_indices = np.where(mask)[0]
        
        if len(keypoints_in_image) == 0:
            continue
            
        # Apply DP-Means clustering
        dpmeans = DPMeans(n_clusters=1, n_init=n_init, delta=delta)
        cluster_labels = dpmeans.fit_predict(keypoints_in_image)
        
        # Get cluster centers
        cluster_centers = dpmeans.cluster_centers_
        
        # Create mapping from original indices to fused indices
        # Group keypoints by cluster
        cluster_groups = {}
        for i, (orig_idx, cluster_id) in enumerate(zip(original_indices, cluster_labels)):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(orig_idx)
        
        # Create fused nodes for each cluster
        for cluster_id, orig_indices_in_cluster in cluster_groups.items():
            cluster_center = cluster_centers[cluster_id]
            
            # Create new fused node for this cluster
            all_means.append(torch.tensor(cluster_center, device=device).unsqueeze(0))
            all_batches.append(torch.tensor([int(b)], device=device))
            
            # Map all original indices in this cluster to the same fused node
            for orig_idx in orig_indices_in_cluster:
                orig_to_fused_map[int(orig_idx)] = fused_node_id
            
            fused_node_id += 1
    
    if all_means:
        fused_keypoints = torch.cat(all_means, dim=0)
        fused_batches = torch.cat(all_batches, dim=0)
    else:
        fused_keypoints = torch.empty((0, 2), device=device)
        fused_batches = torch.empty((0,), dtype=torch.long, device=device)
    
    print(f"DP-Means clustering: {len(keypoints)} -> {len(fused_keypoints)} keypoints")
    
    return fused_keypoints, fused_batches, orig_to_fused_map