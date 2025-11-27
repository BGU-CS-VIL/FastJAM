# Packages
import torch
import os
import json
import numpy as np

# Global homography tensor cache for memory optimization
class HomographyTensorCache:
    """Global tensor cache for homography operations to reduce memory allocation overhead."""
    def __init__(self, device):
        self.device = device
        self.ones_1d_cache = {}

    
    def get_ones_1d(self, batch_size):
        """Get or create ones tensor of shape (batch_size, 1)."""
        if batch_size not in self.ones_1d_cache:
            self.ones_1d_cache[batch_size] = torch.ones(batch_size, 1, device=self.device)
        return self.ones_1d_cache[batch_size]
    
    def clear_cache(self):
        """Clear all cached tensors to free memory."""
        self.ones_1d_cache.clear()

# Global homography tensor cache instance
_homography_cache = None

def get_homography_cache(device):
    """Get global homography tensor cache instance."""
    global _homography_cache
    if _homography_cache is None or _homography_cache.device != device:
        _homography_cache = HomographyTensorCache(device)
    return _homography_cache

def get_homography_matrix(params, matrix_exp = False):
    """
    Given a tensor of shape (batch, 8) (each row: 8 parameters),
    construct the corresponding batch of 3x3 homography matrices.
    
    OPTIMIZED VERSION: Uses tensor caching to reduce memory allocation overhead.
    
    We set H = [p0, p1, p2; p3, p4, p5; p6, p7, 1]
    """
    batch_size = params.shape[0]
    
    # Get tensor cache
    cache = get_homography_cache(params.device)
    
    if matrix_exp:
    # Concatenate parameters in the order: p0, p1, p2, p3, p4, p5, p6, p7, -(p0 + p5)
        H = torch.cat([params[:, 0:2], params[:, 2:3],
                    params[:, 3:5], params[:, 5:6],
                    params[:, 6:8], -(params[:, 0:1] + params[:, 5:6])], dim=1)  # shape: (batch, 9)
        H = H.view(-1, 3, 3)
        H = torch.matrix_exp(H)
    else:
        # Use cached ones tensor instead of creating new one
        ones = cache.get_ones_1d(batch_size)
        # Concatenate parameters in the order: p0, p1, p2, p3, p4, p5, p6, p7, 1
        H = torch.cat([params[:, 0:2], params[:, 2:3],
                    params[:, 3:5], params[:, 5:6],
                    params[:, 6:8], ones], dim=1)  # shape: (batch, 9)
        H = H.view(-1, 3, 3)
        
    return H

def apply_homography(points, H):
    """
    Apply homography H (3x3) to points (N x 2).
    Points are converted to homogeneous coordinates and normalized.
    Returns warped points of shape (N, 2).
    """
    N = points.shape[0]
    ones = torch.ones(N, 1, device=points.device)
    pts_h = torch.cat([points, ones], dim=1)  # (N, 3)
    pts_trans = (H @ pts_h.T).T  # (N, 3)
    pts_trans = pts_trans / pts_trans[:, 2:3]
    return pts_trans[:, :2]

def warp_kps(kps_matrix, H_n, matrix_exp):
    """
    Applies all pairwise homographies H_j⁻¹ @ H_i to kps_matrix[i, j].

    Args:
        kps_matrix: Tensor of shape (B, B, N, 2)
        H_n
        matrix_exp: Apply matrix exponential to H before use

    Returns:
        warped_points: Tensor of shape (B, B, N, 2), where
        warped_points[j, i] = kps_matrix[i, j] transformed by H_j⁻¹ @ H_i
    """
    B = kps_matrix.shape[0]
    H_inv = torch.linalg.inv(H_n).to(H_n.device)

    # Compute H_combined[j, i] = H_j⁻¹ @ H_i
    H_combined = torch.matmul(H_inv[:, None], H_n[None])  # shape (B, B, 3, 3)

    # Since _apply_batch_batch_homography assumes H[i, j] applies to kps_matrix[i, j],
    # we need to **transpose** the kps_matrix and pass H_combined
    kps_transposed = kps_matrix.transpose(0, 1)  # shape (B, B, N, 2), now kps_matrix[j, i]
    warped_transposed = _apply_batch_batch_homography(kps_transposed, H_combined)

    return warped_transposed.transpose(0, 1)  # back to [j, i]

def _apply_batch_batch_homography(kps_matrix, H):
    """
    Warps pairwise keypoints using corresponding homographies.

    Args:
        kps_matrix: Tensor of shape (B, B, N, 2), keypoints from i to j
        H: Tensor of shape (B, B, 3, 3), H_combined[j, i] = H_j⁻¹ @ H_i
        matrix_exp: Whether to apply matrix exponential to H

    Returns:
        warped_points: Tensor of shape (B, B, N, 2)
    """
    B, _, N, _ = kps_matrix.shape

    # Add homogeneous coordinate
    ones = torch.ones((B, B, N, 1), device=kps_matrix.device)
    points_hom = torch.cat([kps_matrix, ones], dim=-1)  # (B, B, N, 3)

    # Reshape for batch multiplication
    points_hom_flat = points_hom.reshape(B * B, N, 3)
    H_flat = H.reshape(B * B, 3, 3)

    points_transformed = torch.matmul(H_flat, points_hom_flat.transpose(1, 2))  # (B*B, 3, N)
    points_transformed = points_transformed.transpose(1, 2).reshape(B, B, N, 3)
    # Normalize to Euclidean
    warped_points = points_transformed[..., :2] / (points_transformed[..., 2:3])

    return warped_points

def export_homographies_to_json(thetas, best_reflections, config, image_paths, pck_scores=None):
    """
    Export homographies/thetas to JSON file.
    
    Args:
        thetas: Tensor of homography matrices (B, 3, 3)
        best_reflections: Tensor of reflection flags (B,)
        config: Configuration dictionary
        image_paths: List of image file paths
        pck_scores: Dictionary of PCK scores (optional)
    """
    # Get results_dir from config
    results_dir = config['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Create save path
    save_path = f"{results_dir}/canonical_homographies.json"
    
    # Convert tensors to numpy arrays and then to lists for JSON serialization
    thetas_np = thetas.detach().cpu().numpy()
    best_reflections_np = best_reflections.detach().cpu().numpy()
    
    # Convert PCK scores to JSON-serializable format
    pck_scores_serializable = {}
    if pck_scores is not None:
        for key, value in pck_scores.items():
            # Convert numpy types to Python native types
            if hasattr(value, 'item'):  # numpy scalar
                pck_scores_serializable[key] = value.item()
            elif hasattr(value, 'tolist'):  # numpy array
                pck_scores_serializable[key] = value.tolist()
            else:
                pck_scores_serializable[key] = float(value) if value is not None else None
    
    # Create the data structure
    export_data = {
        "ref": config['ref'],
        "num_images": len(image_paths),
        "image_paths": image_paths,
        "homographies": thetas_np.tolist(),
        "reflections": best_reflections_np.tolist(),
        "pck_scores": pck_scores_serializable,
        "config": {
            "stn_n": config['stn_n'],
            "hidden_channels": config['hidden_channels'],
            "num_layers": config['num_layers'],
            "start_with_id_matrix": config['start_with_id_matrix'],
            "matrix_exp": config['matrix_exp']
        }
    }
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    if pck_scores:
        print(f"PCK scores available in json file: {save_path}")
    else:
        print("No PCK scores available")