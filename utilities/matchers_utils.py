import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.cm as cm
from torchvision import transforms
import time
from pathlib import Path

ROMA_ROOT = Path(__file__).resolve().parent / "third_party" / "RoMa"
if str(ROMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ROMA_ROOT))

from romatch import roma_outdoor
from romatch.utils.batch_matcher import create_cached_matcher_from_roma, BatchImageMatcher

def apply_nms(kpts, mconf, radius=10):
    """
    Apply non-maximum suppression on keypoints based on confidence and distance.

    Args:
        kpts (np.ndarray): (N, 2) keypoints
        mconf (np.ndarray): (N,) confidence scores
        radius (float): suppression radius in pixels

    Returns:
        np.ndarray: indices of selected keypoints
    """
    # Sort by descending confidence
    sorted_indices = np.argsort(-mconf)
    kpts = kpts[sorted_indices]
    mconf = mconf[sorted_indices]
    selected_indices = []
    suppressed = np.zeros(len(kpts), dtype=bool)

    for i in range(len(kpts)):
        if suppressed[i]:
            continue
        selected_indices.append(sorted_indices[i])  # use original index
        dists = np.linalg.norm(kpts - kpts[i], axis=1)
        suppressed[dists < radius] = True
        suppressed[i] = False  # keep this one

    return np.array(selected_indices)

def get_mask_path(image_path):
    """
    Construct the mask path from an image path using os.path.
    Example image path: ./data/spair_sets/spair_bus/test/images/img_000.png
    Corresponding mask path: ./data/spair_sets/spair_bus/test/masks/img_000_mask.png
    """
    filename = os.path.splitext(os.path.basename(image_path))[0]  # 'img_000'
    split_dir = os.path.dirname(os.path.dirname(image_path))      # .../spair_bus/test
    mask_path = os.path.join(split_dir, "masks", f"{filename}_mask.png")
    return mask_path

def to_masked_path(original_path):
    """
    Convert a SPair image path to its masked image path.
    Input example:  ./data/spair_sets/spair_bus/test/images/img_000.png
    Output example: ./data/spair_sets/spair_bus/test/masks/img_000.jpg
    """
    filename = os.path.splitext(os.path.basename(original_path))[0] + ".jpg"  # 'img_000.jpg'
    split_dir = os.path.dirname(os.path.dirname(original_path))               # .../spair_bus/test
    masked_path = os.path.join(split_dir, "masks", filename)
    return masked_path

def RoMa_extract_matches_cached(roma_model, image_paths, nms_radius_prec: float, max_keypoints: int, batch_size: int, device: str = "cuda"):
    """
    Cached version with batch processing: uses BatchImageMatcher for efficient multi-image matching with caching.
    Processes pairs in batches to leverage both caching and batch processing benefits.
    
    Returns:
        dict[(i, j)] -> (kptsA_xy[N,2], kptsB_xy[N,2])
    """
    matched_keypoints_by_pair = {}
    image_pairs = list(itertools.combinations(range(len(image_paths)), 2))
    
    # === Resize target from model ===
    H, W = roma_model.get_output_resolution()
    target_size = (W, H)  # torchvision uses (W,H)

    # === Preprocess transform ===
    preprocess = transforms.Compose([
        transforms.Resize(target_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])

    # === Cache: tensors for images (for match), RGB arrays for plotting, masks ===
    tensor_cache = {}
    rgb_cache = {}
    mask_cache = {}

    def get_tensor(idx: int) -> torch.Tensor:
        if idx not in tensor_cache:
            im = Image.open(image_paths[idx]).convert('RGB')
            t = preprocess(im).to(device, non_blocking=True)
            tensor_cache[idx] = t
            # cache RGB for plotting
            rgb_cache[idx] = np.array(im.resize(target_size))
        return tensor_cache[idx]

    def get_mask(idx: int) -> np.ndarray:
        if idx in mask_cache:
            return mask_cache[idx]
        mpath = get_mask_path(image_paths[idx])
        if os.path.exists(mpath):
            mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask = None
        mask_cache[idx] = mask
        return mask

    # === Create cached matcher ===
    cached_matcher = create_cached_matcher_from_roma(roma_model, device=device)
    
    # === Process pairs in batches ===
    print(f"Matching {len(image_pairs)} pairs with caching and batch processing (batch_size={batch_size})...")
    for start in tqdm(range(0, len(image_pairs), batch_size), desc="Cached batch matching"):
        batch_pairs = image_pairs[start:start + batch_size]

        # Build batched tensors
        ims_A = torch.stack([get_tensor(i) for (i, j) in batch_pairs], dim=0)
        ims_B = torch.stack([get_tensor(j) for (i, j) in batch_pairs], dim=0)

        # Create batch for cached matcher
        batch = {
            "im_A": ims_A,
            "im_B": ims_B
        }
        
        # Get corresps from cached matcher (batched)
        with torch.inference_mode():
            corresps = cached_matcher.forward(batch, batched=True)
        
        # Process each pair in the batch
        B = len(batch_pairs)
        for b in range(B):
            i, j = batch_pairs[b]
            
            # Extract corresps for this specific pair
            pair_corresps = {}
            for scale, corresp in corresps.items():
                if isinstance(corresp, dict):
                    # Handle dict format
                    pair_corresp = {}
                    for key, value in corresp.items():
                        if isinstance(value, torch.Tensor) and value.dim() > 2:
                            pair_corresp[key] = value[b:b+1]  # Keep batch dimension
                        else:
                            pair_corresp[key] = value
                    pair_corresps[scale] = pair_corresp
                else:
                    # Handle tensor format
                    if isinstance(corresp, torch.Tensor) and corresp.dim() > 2:
                        pair_corresps[scale] = corresp[b:b+1]  # Keep batch dimension
                    else:
                        pair_corresps[scale] = corresp
            
            # Process corresps to get warp and certainty
            warp, certainty = cached_matcher.process_corresps_to_warp_certainty(pair_corresps, H, W, device)
            
            # Sample matches for this specific pair
            matches, conf = roma_model.sample(warp, certainty)
            
            # Convert to pixel coordinates
            kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H, W, H, W)
            kptsA = kptsA.detach().cpu().numpy()
            kptsB = kptsB.detach().cpu().numpy()
            mconf = conf.detach().cpu().numpy()

            # === Mask filtering ===
            maskA = get_mask(i)
            maskB = get_mask(j)
            if maskA is not None and maskB is not None and kptsA.size > 0:
                xA = np.clip(kptsA[:, 0].astype(int), 0, W - 1)
                yA = np.clip(kptsA[:, 1].astype(int), 0, H - 1)
                xB = np.clip(kptsB[:, 0].astype(int), 0, W - 1)
                yB = np.clip(kptsB[:, 1].astype(int), 0, H - 1)
                validA = (maskA[yA, xA] == 255)
                validB = (maskB[yB, xB] == 255)
                valid = validA & validB
                if valid.any():
                    kptsA = kptsA[valid]
                    kptsB = kptsB[valid]
                    mconf = mconf[valid]
                else:
                    # no valid points left
                    kptsA = kptsA[:0]
                    kptsB = kptsB[:0]
                    mconf = mconf[:0]

            # === NMS in pixel space ===
            if kptsA.size > 0:
                nms_radius = int(max(H, W) * nms_radius_prec)
                keep_idx = apply_nms(kptsA, mconf, radius=nms_radius)
                kptsA = kptsA[keep_idx]
                kptsB = kptsB[keep_idx]
                mconf = mconf[keep_idx]

                # === Top-K by confidence ===
                if kptsA.shape[0] > max_keypoints:
                    order = np.argsort(-mconf)[:max_keypoints]
                    kptsA = kptsA[order]
                    kptsB = kptsB[order]
                    mconf = mconf[order]

            matched_keypoints_by_pair[(i, j)] = (kptsA, kptsB)

    # Print cache statistics
    cache_stats = cached_matcher.get_cache_stats()
    dino_stats = cache_stats.get('dino_cache', cache_stats)  # Handle both old and new formats
    
    print(f"Cache statistics: {dino_stats['hits']} hits, {dino_stats['misses']} misses, "
          f"hit rate: {dino_stats['hit_rate']:.2%}")

    # --- Aggressive cleanup to free RoMa tensors and caches ---
    try:
        if hasattr(cached_matcher, 'clear_cache'):
            cached_matcher.clear_cache()
        if hasattr(cached_matcher, 'cleanup_encoders'):
            cached_matcher.cleanup_encoders()
    except Exception:
        pass

    tensor_cache.clear()
    rgb_cache.clear()
    mask_cache.clear()
    try:
        del cached_matcher
    except Exception:
        pass

    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return matched_keypoints_by_pair

# Global tensor pool for memory optimization
class TensorPool:
    """Global tensor pool to reuse large tensors and reduce memory allocation overhead."""
    def __init__(self, device):
        self.device = device
        self.kps_matrix_pool = {}
        self.mask_pool = {}
        self.ones_pool = {}
        self.T_pool = {}
    
    def get_kps_matrix(self, B, N):
        """Get or create kps_matrix tensor of shape (B, B, N, 2)."""
        key = (B, N)
        if key not in self.kps_matrix_pool:
            # Pre-allocate with -1 values
            tensor = torch.zeros((B, B, N, 2), dtype=torch.float32, device=self.device)
            tensor.fill_(-1.0)  # More efficient than tensor - 1
            self.kps_matrix_pool[key] = tensor
        return self.kps_matrix_pool[key]
    
    def get_mask(self, B, N):
        """Get or create mask tensor of shape (B, B, N)."""
        key = (B, N)
        if key not in self.mask_pool:
            self.mask_pool[key] = torch.zeros((B, B, N), dtype=torch.bool, device=self.device)
        return self.mask_pool[key]
    
    def get_ones(self, n):
        """Get or create ones tensor of shape (n, 1)."""
        if n not in self.ones_pool:
            self.ones_pool[n] = torch.ones((n, 1), device=self.device)
        return self.ones_pool[n]
    
    def get_T_matrix(self, W, H):
        """Get or create normalization matrix T."""
        key = (W, H)
        if key not in self.T_pool:
            self.T_pool[key] = torch.tensor([
                [2.0 / W, 0.0, -1.0],
                [0.0, 2.0 / H, -1.0],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=self.device)
        return self.T_pool[key]
    
    def clear_cache(self):
        """Clear all cached tensors to free memory."""
        self.kps_matrix_pool.clear()
        self.mask_pool.clear()
        self.ones_pool.clear()
        self.T_pool.clear()

# Global tensor pool instance
_tensor_pool = None

def get_tensor_pool(device):
    """Get global tensor pool instance."""
    global _tensor_pool
    if _tensor_pool is None or _tensor_pool.device != device:
        _tensor_pool = TensorPool(device)
    return _tensor_pool

def create_mask_and_kps_matrix(matched_keypoints_by_pair, B, device, image_size=None):
    """
    Creates normalized keypoint matrix and mask from matched keypoints.
    Normalization maps (x, y) from image space to [-1, 1] with center at (0, 0).
    
    OPTIMIZED VERSION: Uses tensor pooling to reduce memory allocation overhead.

    Args:
        matched_keypoints_by_pair: dict with keys (i, j) and values (kps_i, kps_j) as arrays or tensors
        B: int – number of images
        device: torch.device or str
        image_size: (H, W) – needed for normalization

    Returns:
        kps_matrix: (B, B, N, 2) tensor – normalized keypoints from i to j
        mask: (B, B, N) tensor – True for valid keypoints
    """
    assert image_size is not None, "image_size=(H, W) must be provided for normalization"
    H, W = image_size

    # Determine max number of keypoints (N)
    max_kps = 0
    for (i, j), (kps_i, _) in matched_keypoints_by_pair.items():
        max_kps = max(max_kps, kps_i.shape[0])
    N = max_kps

    # Get tensor pool
    pool = get_tensor_pool(device)
    
    # Get pre-allocated tensors from pool
    kps_matrix = pool.get_kps_matrix(B, N)
    mask = pool.get_mask(B, N)
    
    # Reset tensors to initial state (more efficient than creating new ones)
    kps_matrix.fill_(-1.0)  # Reset to -1
    mask.zero_()  # Reset to False

    # Get cached normalization matrix
    T = pool.get_T_matrix(W, H)

    # Populate using optimized operations
    for (i, j), (kps_i, kps_j) in matched_keypoints_by_pair.items():
        # Convert to tensors if needed (avoid unnecessary conversions)
        if not isinstance(kps_i, torch.Tensor):
            kps_i = torch.tensor(kps_i, dtype=torch.float32, device=device)
        elif kps_i.device != device:
            kps_i = kps_i.to(device)
            
        if not isinstance(kps_j, torch.Tensor):
            kps_j = torch.tensor(kps_j, dtype=torch.float32, device=device)
        elif kps_j.device != device:
            kps_j = kps_j.to(device)

        n = kps_i.shape[0]

        # Get cached ones tensors
        ones_i = pool.get_ones(n)
        ones_j = pool.get_ones(n)
        
        # Convert to homogeneous and normalize using T
        kps_i_h = torch.cat([kps_i, ones_i], dim=1)  # (N, 3)
        kps_i_norm = (T @ kps_i_h.T).T[:, :2]

        kps_j_h = torch.cat([kps_j, ones_j], dim=1)
        kps_j_norm = (T @ kps_j_h.T).T[:, :2]

        # Use in-place operations where possible
        kps_matrix[i, j, :n] = kps_i_norm
        kps_matrix[j, i, :n] = kps_j_norm
        mask[i, j, :n] = True
        mask[j, i, :n] = True

    return kps_matrix, mask