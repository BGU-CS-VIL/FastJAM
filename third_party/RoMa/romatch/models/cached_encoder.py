import torch
import torch.nn as nn
import hashlib
from typing import Dict, Optional, Tuple, Any
from .encoders import CNNandDinov2


class CachedCNNandDinov2(nn.Module):
    """
    A cached version of CNNandDinov2 that stores DINO features for images
    to avoid recomputing them when the same image appears in multiple pairs.
    
    This is particularly useful when processing N images with all (N, 2) pairs,
    where each image appears in N-1 pairs.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize the base encoder
        self.base_encoder = CNNandDinov2(*args, **kwargs)
        
        # Copy attributes from base encoder
        self.cnn = self.base_encoder.cnn
        self.amp = self.base_encoder.amp
        self.amp_dtype = self.base_encoder.amp_dtype
        self.dinov2_vitl14 = self.base_encoder.dinov2_vitl14
        
        # Cache for storing DINO features
        self._dino_cache: Dict[str, torch.Tensor] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Cache configuration
        self.max_cache_size = 1000  # Maximum number of cached features
        self.cache_enabled = True
        
        # Hash cache to avoid recomputing hashes for the same images
        self._hash_cache: Dict[str, str] = {}
        self._hash_cache_stats = {"hits": 0, "misses": 0}
        
    def _compute_image_hash(self, x: torch.Tensor) -> str:
        """Compute a hash for the input tensor to use as cache key."""
        # Convert to CPU and create a hash of the tensor data
        x_cpu = x.cpu()
        # Use a combination of shape and data hash for uniqueness
        shape_str = str(x_cpu.shape)
        data_hash = hashlib.md5(x_cpu.numpy().tobytes()).hexdigest()
        return f"{shape_str}_{data_hash}"
    
    def _compute_image_hash_batch_safe(self, x_batch: torch.Tensor) -> list:
        """
        SAFE batch hash computation that maintains 100% accuracy.
        
        Strategy: Batch CPU transfers but use full image data for each hash.
        This reduces the number of CPU transfers from N to 1, while maintaining
        perfect cache accuracy by using the original full-image hashing method.
        """
        B = x_batch.shape[0]
        hashes = []
        
        # Convert entire batch to CPU in one transfer
        x_batch_cpu = x_batch.cpu()
        
        # Process each image in the batch using the original method
        for i in range(B):
            img_tensor = x_batch_cpu[i:i+1]  # Get individual image from CPU batch
            shape_str = str(img_tensor.shape)
            data_hash = hashlib.md5(img_tensor.numpy().tobytes()).hexdigest()
            hash_key = f"{shape_str}_{data_hash}"
            hashes.append(hash_key)
        
        return hashes
    
    def _compute_image_hash_smart(self, x: torch.Tensor) -> str:
        """
        SMART hash computation with two-level caching.
        
        Strategy:
        1. First check if we've seen this exact tensor before (hash cache)
        2. If not, compute hash using original method but cache the result
        3. This reduces CPU transfers for repeated images while maintaining accuracy
        """
        # Create a quick tensor fingerprint for hash cache lookup
        # This is much faster than full CPU transfer
        tensor_fingerprint = f"{x.shape}_{x.device}_{x.dtype}_{x.data_ptr()}"
        
        # Check hash cache first
        if tensor_fingerprint in self._hash_cache:
            self._hash_cache_stats["hits"] += 1
            return self._hash_cache[tensor_fingerprint]
        
        # Cache miss - compute hash using original method
        self._hash_cache_stats["misses"] += 1
        hash_result = self._compute_image_hash(x)
        
        # Store in hash cache
        self._hash_cache[tensor_fingerprint] = hash_result
        
        return hash_result
    
    def _get_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DINO features for a batch of images."""
        B, C, H, W = x.shape
        
        with torch.no_grad():
            if self.dinov2_vitl14[0].device != x.device:
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
            
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
            features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0, 2, 1).reshape(B, 1024, H//14, W//14)
            del dinov2_features_16
            
        return features_16
    
    def _cache_dino_features(self, x: torch.Tensor, features: torch.Tensor) -> None:
        """Cache DINO features for individual images in the batch."""
        if not self.cache_enabled:
            return
            
        # Process each image in the batch individually
        for i in range(x.shape[0]):
            img_tensor = x[i:i+1]  # Single image
            img_features = features[i:i+1]  # Corresponding features
            
            # Create cache key
            cache_key = self._compute_image_hash(img_tensor)
            
            # Store in cache if not already present
            if cache_key not in self._dino_cache:
                # Check cache size limit
                if len(self._dino_cache) >= self.max_cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._dino_cache))
                    del self._dino_cache[oldest_key]
                
                # Store features on CPU to save GPU memory
                self._dino_cache[cache_key] = img_features.cpu()
    
    def _get_cached_dino_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get DINO features, using cache when possible.
        Returns (features, cache_hit_mask) where cache_hit_mask indicates which images were cached.
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Initialize output tensor
        all_features = torch.zeros(B, 1024, H//14, W//14, device=device, dtype=self.amp_dtype)
        cache_hit_mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Check cache for each image
        images_to_compute = []
        compute_indices = []
        
        for i in range(B):
            img_tensor = x[i:i+1]
            cache_key = self._compute_image_hash(img_tensor)
            
            if cache_key in self._dino_cache:
                # Cache hit
                cached_features = self._dino_cache[cache_key].to(device).to(self.amp_dtype)
                all_features[i] = cached_features[0]
                cache_hit_mask[i] = True
                self._cache_stats["hits"] += 1
            else:
                # Cache miss - need to compute
                images_to_compute.append(img_tensor)
                compute_indices.append(i)
                self._cache_stats["misses"] += 1
        
        # Compute features for images not in cache
        if images_to_compute:
            batch_to_compute = torch.cat(images_to_compute, dim=0)
            computed_features = self._get_dino_features(batch_to_compute)
            
            # Store computed features in cache
            self._cache_dino_features(batch_to_compute, computed_features)
            
            # Fill in the computed features
            for idx, orig_idx in enumerate(compute_indices):
                all_features[orig_idx] = computed_features[idx]
        
        return all_features, cache_hit_mask
    
    def forward(self, x: torch.Tensor, upsample: bool = False) -> Dict[int, torch.Tensor]:
        """
        Forward pass with DINO feature caching.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            upsample: Whether to upsample (affects DINO computation)
            
        Returns:
            Feature pyramid dictionary
        """
        B, C, H, W = x.shape
        
        # Check if encoders have been cleaned up
        if self.cnn is None:
            raise RuntimeError("CNN encoder has been cleaned up. Cannot perform forward pass.")
        
        # Get CNN features (always computed)
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            # Get DINO features with caching
            if self.cache_enabled and self.dinov2_vitl14 is not None:
                dino_features, cache_hits = self._get_cached_dino_features(x)
                feature_pyramid[16] = dino_features
            elif self.dinov2_vitl14 is not None:
                # Fallback to original behavior
                dino_features = self._get_dino_features(x)
                feature_pyramid[16] = dino_features
                cache_hits = torch.zeros(B, dtype=torch.bool, device=x.device)
            else:
                # DINO encoder has been cleaned up, skip DINO features
                print("Warning: DINO encoder has been cleaned up, skipping DINO features")
                # Create a placeholder tensor for scale 16
                feature_pyramid[16] = torch.zeros(B, 1024, H//14, W//14, device=x.device, dtype=self.amp_dtype)
        
        return feature_pyramid
    
    def train(self, mode: bool = True):
        """Set training mode for the base encoder."""
        if self.base_encoder is not None:
            return self.base_encoder.train(mode)
        return self
    
    def clear_cache(self):
        """Clear both DINO feature cache and hash cache."""
        self._dino_cache.clear()
        self._hash_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        self._hash_cache_stats = {"hits": 0, "misses": 0}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # DINO feature cache stats
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        # Hash cache stats
        total_hash_requests = self._hash_cache_stats["hits"] + self._hash_cache_stats["misses"]
        hash_hit_rate = self._hash_cache_stats["hits"] / total_hash_requests if total_hash_requests > 0 else 0.0
        
        return {
            "dino_cache": {
                "hits": self._cache_stats["hits"],
                "misses": self._cache_stats["misses"], 
                "hit_rate": hit_rate,
                "cache_size": len(self._dino_cache)
            },
            "hash_cache": {
                "hits": self._hash_cache_stats["hits"],
                "misses": self._hash_cache_stats["misses"],
                "hit_rate": hash_hit_rate,
                "cache_size": len(self._hash_cache)
            },
            "total_cpu_transfers_saved": self._hash_cache_stats["hits"],
            "max_cache_size": self.max_cache_size
        }
    
    def set_cache_config(self, enabled: bool = True, max_size: int = 1000):
        """Configure cache settings."""
        self.cache_enabled = enabled
        self.max_cache_size = max_size
        
        # Clear cache if disabled
        if not enabled:
            self.clear_cache()
    
    def precompute_features(self, images: torch.Tensor) -> None:
        """
        Precompute and cache DINO features for a batch of images.
        Useful for warming up the cache before processing pairs.
        
        Args:
            images: Tensor of shape (N, C, H, W) containing N images
        """
        if not self.cache_enabled:
            return
            
        with torch.no_grad():
            # Process each image individually to build cache
            for i in range(images.shape[0]):
                img = images[i:i+1]
                cache_key = self._compute_image_hash(img)
                
                if cache_key not in self._dino_cache:
                    features = self._get_dino_features(img)
                    self._cache_dino_features(img, features)
    
    def cleanup_encoders(self):
        """
        Delete DINO and VGG encoders to free up memory after feature extraction.
        This should be called after all features have been extracted and cached.
        """
        # Delete DINO encoder
        if hasattr(self, 'dinov2_vitl14') and self.dinov2_vitl14 is not None:
            del self.dinov2_vitl14
            self.dinov2_vitl14 = None
        
        # Delete VGG encoder
        if hasattr(self, 'cnn') and self.cnn is not None:
            del self.cnn
            self.cnn = None
        
        # Delete base encoder
        if hasattr(self, 'base_encoder') and self.base_encoder is not None:
            del self.base_encoder
            self.base_encoder = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class CachedRegressionMatcher(nn.Module):
    """
    A cached version of RegressionMatcher that uses CachedCNNandDinov2
    for efficient processing of multiple image pairs.
    """
    
    def __init__(self, encoder: CachedCNNandDinov2, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Copy other attributes from the original RegressionMatcher
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def extract_backbone_features(self, batch, batched=True, upsample=False):
        """Extract features using the cached encoder."""
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        
        if batched:
            X = torch.cat((x_q, x_s), dim=0)
            feature_pyramid = self.encoder(X, upsample=upsample)
        else:
            feature_pyramid = (
                self.encoder(x_q, upsample=upsample),
                self.encoder(x_s, upsample=upsample),
            )
        return feature_pyramid
    
    def get_output_resolution(self):
        """Get the output resolution for the matcher."""
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def process_corresps_to_warp_certainty(self, corresps, H, W, device):
        """Process corresps dictionary to return warp and certainty tensors like the original match function."""
        import torch.nn.functional as F
        
        # Get the finest scale (typically scale 1)
        finest_scale = 1
        
        # Apply the exact same logic as RoMa's match function
        im_A_to_im_B = corresps[finest_scale]["flow"]
        certainty = corresps[finest_scale]["certainty"]
        
        # Apply certainty attenuation if enabled (same as RoMa)
        if hasattr(self, 'attenuate_cert') and self.attenuate_cert:
            low_res_certainty = F.interpolate(
                corresps[16]["certainty"],
                size=(H, W),
                align_corners=False,
                mode="bilinear",
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = (
                factor * low_res_certainty * (low_res_certainty < cert_clamp)
            )
            certainty = certainty - low_res_certainty
        
        # Interpolation if needed (same as RoMa)
        if finest_scale != 1:
            im_A_to_im_B = F.interpolate(
                im_A_to_im_B, size=(H, W), align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
                certainty, size=(H, W), align_corners=False, mode="bilinear"
            )
        
        # Permute im_A_to_im_B to coordinate format (same as RoMa)
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        
        # Create im_A meshgrid (same as RoMa)
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device),
                torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device),
            ),
            indexing="ij",
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(1, 2, H, W)
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        
        # Apply sigmoid to convert logits to probabilities (same as RoMa)
        certainty = certainty.sigmoid()
        
        # Filter invalid flows and clamp (same as RoMa)
        if (im_A_to_im_B.abs() > 1).any():
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        
        # Create warp tensor (same as RoMa)
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        
        return warp, certainty
    
    def forward(self, batch, batched=True, upsample=False, scale_factor=1):
        """Forward pass using cached features."""
        feature_pyramid = self.extract_backbone_features(
            batch, batched=batched, upsample=upsample
        )
        
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
            
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor,
        )
        
        return corresps
    
    def forward_symmetric(self, batch, batched=True, upsample=False, scale_factor=1):
        """Forward pass with symmetric matching using cached features."""
        feature_pyramid = self.extract_backbone_features(
            batch, batched=batched, upsample=upsample
        )
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim=0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor,
        )
        return corresps
    
    def get_cache_stats(self):
        """Get cache statistics from the encoder."""
        return self.encoder.get_cache_stats()
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.encoder.clear_cache()
    
    def precompute_image_features(self, images: torch.Tensor):
        """Precompute features for a batch of images."""
        self.encoder.precompute_features(images)
    
    def cleanup_encoders(self):
        """Clean up encoders to free memory after feature extraction."""
        if hasattr(self, 'encoder') and self.encoder is not None:
            self.encoder.cleanup_encoders()
