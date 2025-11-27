# Packages
import torch
from torch import nn

# Project
from utilities.homography_utils import warp_kps

def _geman_mcclure_loss(a, sigma=0.01):
    """
    Compute Geman-McClure robust loss.
    
    Args:
        a (Tensor): difference tensor (e.g., prediction - target), shape (..., D)
        sigma (float): regularization constant (controls outlier suppression)

    Returns:
        Tensor: same shape as `a`, containing per-element loss
    """
    a_squared = a ** 2
    return a_squared / (a_squared + sigma ** 2)

def congealing_loss_pair_normalized(kps_matrix, H_n, mask, matrix_exp, sigma):
    """
    Computes normalized congealing loss between warped keypoints of image i to j.

    Args:
        kps_matrix: Tensor (B, B, N, 2) – keypoints from image i matched to j
        H_n: Homography after recurrent STN n times
        mask: Tensor (B, B, N) – binary mask of valid keypoints
        matrix_exp: flag if Lie alg. or not

    Returns:
        total_loss: scalar – mean pairwise loss over all valid non-diagonal pairs
        pair_loss: Tensor (B, B) – loss per pair (i, j)
    """
    B, _, N, _ = kps_matrix.shape

    # Step 1: Warp keypoints from image i to j using H_j⁻¹ @ H_i
    warped_kps = warp_kps(kps_matrix, H_n, matrix_exp)  # shape (B, B, N, 2)
    # Step 2: Compare warped keypoints to reference keypoints from image j
    kps_target = kps_matrix.transpose(0, 1)  # kps[j, i], still shape (B, B, N, 2)
    diff = warped_kps - kps_target
    diff_norm = diff.norm(dim=-1) # (B, B, N)
    diff_norm_masked = diff_norm * mask
    pairwise_loss_sum  = _geman_mcclure_loss(diff_norm_masked, sigma).sum(dim=-1)  # (B, B)
    # pairwise_loss_sum = (diff_norm_masked ** 2).sum(dim=-1)  # (B, B)
    if torch.isnan(pairwise_loss_sum).any() or torch.isinf(pairwise_loss_sum).any():
            raise ValueError("pairwise_loss_sum invalid.")

    # Step 4: Normalize per-pair
    valid_counts = mask.sum(dim=-1)  # (B, B)
    pair_loss = torch.zeros_like(valid_counts, dtype=torch.float32)
    valid = valid_counts > 0
    pair_loss[valid] = pairwise_loss_sum[valid] / valid_counts[valid]

    # Zero diagonal (self comparisons)
    pair_loss.fill_diagonal_(0.0)

    # Step 5: Average over valid non-diagonal pairs
    total_loss = pair_loss.sum() / (B * (B - 1))

    return total_loss, pair_loss

@torch.no_grad()
def optimize_reflections_coord_descent_matrix(kps_reflected_matrix, H_n, mask, matrix_exp, sigma, max_iter=10):
    """
    Optimize reflection choice for each image using coordinate descent.

    Args:
        kps_reflected_matrix: (B, 2, B, N, 2)
        H_n
        mask: (B, B, N)
        matrix_exp flag if lie alg

    Returns:
        best_reflections: (B,) tensor of 0 (original) or 1 (reflected)
    """
    B = kps_reflected_matrix.shape[0]
    device = kps_reflected_matrix.device
    reflections = torch.zeros(B, dtype=torch.long, device=device)

    for _ in range(max_iter):
        updated = False
        for i in range(B):
            current_kps_matrix = kps_reflected_matrix[torch.arange(B), reflections]  # (B, B, N, 2)

            # Try flipping just image i
            flipped = reflections.clone()
            flipped[i] = 1 - flipped[i]
            flipped_kps_matrix = kps_reflected_matrix[torch.arange(B), flipped]

            loss_orig, _ = congealing_loss_pair_normalized(current_kps_matrix, H_n, mask, matrix_exp, sigma)
            loss_flip, _ = congealing_loss_pair_normalized(flipped_kps_matrix, H_n, mask, matrix_exp, sigma)

            if loss_flip < loss_orig:
                reflections[i] = 1 - reflections[i]
                updated = True

        if not updated:
            break  # converged

    return reflections