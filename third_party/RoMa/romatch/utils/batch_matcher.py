import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import numpy as np
from itertools import combinations
import time
from ..models.cached_encoder import CachedCNNandDinov2, CachedRegressionMatcher
from ..utils.utils import get_tuple_transform_ops


class BatchImageMatcher:
    """
    A utility class for efficiently processing N images with all (N, 2) pairs
    using cached DINO features to avoid redundant computations.
    
    This is particularly useful when you have N images and want to match
    all possible pairs, where each image appears in N-1 pairs.
    """
    
    def __init__(self, 
                 matcher: CachedRegressionMatcher,
                 device: Optional[torch.device] = None,
                 cache_size: int = 1000,
                 precompute_features: bool = True):
        """
        Initialize the batch matcher.
        
        Args:
            matcher: A CachedRegressionMatcher instance
            device: Device to use for computation
            cache_size: Maximum number of cached features
            precompute_features: Whether to precompute features for all images
        """
        self.matcher = matcher
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_size = cache_size
        self.precompute_features = precompute_features
        
        # Configure cache
        self.matcher.encoder.set_cache_config(enabled=True, max_size=cache_size)
        
        # Store for processed images
        self._processed_images: Dict[str, torch.Tensor] = {}
        self._image_hashes: List[str] = []
        
    def _load_and_preprocess_image(self, image_input: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Load and preprocess a single image."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, torch.Tensor):
            return image_input.to(self.device)
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
        
        # Get target resolution
        h, w = self.matcher.get_output_resolution()
        
        # Resize image
        image = image.resize((w, h))
        
        # Apply transforms
        transform = get_tuple_transform_ops(resize=None, normalize=True)
        image_tensor, _ = transform((image, image))  # Use same transform for both
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def add_images(self, images: List[Union[str, Image.Image, torch.Tensor]]) -> List[str]:
        """
        Add images to the batch processor.
        
        Args:
            images: List of images (paths, PIL Images, or tensors)
            
        Returns:
            List of image identifiers
        """
        image_ids = []
        
        for i, image_input in enumerate(images):
            # Load and preprocess image
            image_tensor = self._load_and_preprocess_image(image_input)
            
            # Create unique identifier
            image_id = f"image_{i}_{hash(str(image_input))}"
            image_ids.append(image_id)
            
            # Store processed image
            self._processed_images[image_id] = image_tensor
            self._image_hashes.append(image_id)
        
        # Precompute features if enabled
        if self.precompute_features and len(images) > 0:
            self._precompute_all_features()
        
        return image_ids
    
    def _precompute_all_features(self):
        """Precompute DINO features for all added images."""
        if not self._processed_images:
            return
        
        print("Precomputing DINO features for all images...")
        start_time = time.time()
        
        # Stack all images into a batch
        image_tensors = list(self._processed_images.values())
        batch_tensor = torch.stack(image_tensors, dim=0)
        
        # Precompute features
        self.matcher.precompute_image_features(batch_tensor)
        
        elapsed = time.time() - start_time
        print(f"Precomputed features for {len(image_tensors)} images in {elapsed:.2f}s")
        
        # Print cache stats
        stats = self.matcher.get_cache_stats()
        print(f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, "
              f"hit rate: {stats['hit_rate']:.2%}")
        
        # Clean up encoders after feature extraction to free memory
        print("Cleaning up encoders to free memory...")
        self.matcher.cleanup_encoders()
    
    def match_all_pairs(self, 
                       return_correspondences: bool = True,
                       return_certainty: bool = True,
                       symmetric: bool = True) -> List[Dict]:
        """
        Match all possible pairs of images.
        
        Args:
            return_correspondences: Whether to return correspondence maps
            return_certainty: Whether to return certainty maps
            symmetric: Whether to use symmetric matching
            
        Returns:
            List of dictionaries containing match results for each pair
        """
        if len(self._processed_images) < 2:
            raise ValueError("Need at least 2 images to create pairs")
        
        results = []
        total_pairs = len(list(combinations(range(len(self._processed_images)), 2)))
        
        print(f"Processing {total_pairs} image pairs...")
        start_time = time.time()
        
        # Process all pairs
        for i, (idx1, idx2) in enumerate(combinations(range(len(self._image_hashes)), 2)):
            image_id1 = self._image_hashes[idx1]
            image_id2 = self._image_hashes[idx2]
            
            # print(f"Processing pair {i+1}/{total_pairs}: {image_id1} <-> {image_id2}")
            
            # Get image tensors
            im_A = self._processed_images[image_id1]
            im_B = self._processed_images[image_id2]
            
            # Create batch
            batch = {
                "im_A": im_A.unsqueeze(0),  # Add batch dimension
                "im_B": im_B.unsqueeze(0)
            }
            
            # Match the pair
            with torch.no_grad():
                if symmetric:
                    corresps = self.matcher.forward_symmetric(batch, batched=True)
                else:
                    corresps = self.matcher.forward(batch, batched=True)
            
            # Extract results
            result = {
                "image_id1": image_id1,
                "image_id2": image_id2,
                "pair_index": i,
                "correspondences": corresps if return_correspondences else None,
                "certainty": corresps[1]["certainty"] if return_certainty else None,
                "flow": corresps[1]["flow"] if return_correspondences else None
            }
            
            results.append(result)
        
        elapsed = time.time() - start_time
        print(f"Completed {total_pairs} pairs in {elapsed:.2f}s ({elapsed/total_pairs:.3f}s per pair)")
        
        # Print final cache stats
        stats = self.matcher.get_cache_stats()
        print(f"Final cache stats: {stats['hits']} hits, {stats['misses']} misses, "
              f"hit rate: {stats['hit_rate']:.2%}")
        
        return results
    
    def match_specific_pairs(self, 
                           pairs: List[Tuple[int, int]],
                           return_correspondences: bool = True,
                           return_certainty: bool = True,
                           symmetric: bool = True) -> List[Dict]:
        """
        Match specific pairs of images by their indices.
        
        Args:
            pairs: List of (index1, index2) tuples specifying which images to match
            return_correspondences: Whether to return correspondence maps
            return_certainty: Whether to return certainty maps
            symmetric: Whether to use symmetric matching
            
        Returns:
            List of dictionaries containing match results for each specified pair
        """
        if not pairs:
            return []
        
        results = []
        
        print(f"Processing {len(pairs)} specific image pairs...")
        start_time = time.time()
        
        for i, (idx1, idx2) in enumerate(pairs):
            if idx1 >= len(self._image_hashes) or idx2 >= len(self._image_hashes):
                raise ValueError(f"Invalid indices: {idx1}, {idx2}. "
                               f"Valid range: 0-{len(self._image_hashes)-1}")
            
            image_id1 = self._image_hashes[idx1]
            image_id2 = self._image_hashes[idx2]
            
            # print(f"Processing pair {i+1}/{len(pairs)}: {image_id1} <-> {image_id2}")
            
            # Get image tensors
            im_A = self._processed_images[image_id1]
            im_B = self._processed_images[image_id2]
            
            # Create batch
            batch = {
                "im_A": im_A.unsqueeze(0),
                "im_B": im_B.unsqueeze(0)
            }
            
            # Match the pair
            with torch.no_grad():
                if symmetric:
                    corresps = self.matcher.forward_symmetric(batch, batched=True)
                else:
                    corresps = self.matcher.forward(batch, batched=True)
            
            # Extract results
            result = {
                "image_id1": image_id1,
                "image_id2": image_id2,
                "pair_index": i,
                "image_indices": (idx1, idx2),
                "correspondences": corresps if return_correspondences else None,
                "certainty": corresps[1]["certainty"] if return_certainty else None,
                "flow": corresps[1]["flow"] if return_correspondences else None
            }
            
            results.append(result)
        
        elapsed = time.time() - start_time
        print(f"Completed {len(pairs)} pairs in {elapsed:.2f}s ({elapsed/len(pairs):.3f}s per pair)")
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get current cache statistics."""
        return self.matcher.get_cache_stats()
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.matcher.clear_cache()
    
    def clear_images(self):
        """Clear all stored images."""
        self._processed_images.clear()
        self._image_hashes.clear()
    
    def get_image_count(self) -> int:
        """Get the number of stored images."""
        return len(self._processed_images)
    
    def get_image_ids(self) -> List[str]:
        """Get list of image identifiers."""
        return self._image_hashes.copy()


def create_cached_matcher_from_roma(roma_matcher, device: Optional[torch.device] = None) -> CachedRegressionMatcher:
    """
    Create a CachedRegressionMatcher from an existing RoMa matcher.
    
    Args:
        roma_matcher: An existing RoMa RegressionMatcher
        device: Device to use
        
    Returns:
        CachedRegressionMatcher with the same configuration
    """
    # Create cached encoder
    cached_encoder = CachedCNNandDinov2(
        cnn_kwargs=roma_matcher.encoder.cnn_kwargs if hasattr(roma_matcher.encoder, 'cnn_kwargs') else {},
        amp=roma_matcher.encoder.amp,
        dinov2_weights=None,  # Will be loaded from existing model
        amp_dtype=roma_matcher.encoder.amp_dtype
    )
    
    # Copy weights from original encoder to the base encoder
    cached_encoder.base_encoder.load_state_dict(roma_matcher.encoder.state_dict())
    
    # Create cached matcher
    cached_matcher = CachedRegressionMatcher(
        encoder=cached_encoder,
        decoder=roma_matcher.decoder,
        h=roma_matcher.h_resized,
        w=roma_matcher.w_resized,
        sample_mode=getattr(roma_matcher, 'sample_mode', 'threshold_balanced'),
        upsample_preds=getattr(roma_matcher, 'upsample_preds', False),
        symmetric=getattr(roma_matcher, 'symmetric', False),
        sample_thresh=getattr(roma_matcher, 'sample_thresh', 0.05),
        name=getattr(roma_matcher, 'name', None),
        attenuate_cert=getattr(roma_matcher, 'attenuate_cert', None),
        upsample_res=getattr(roma_matcher, 'upsample_res', None)
    )
    
    if device:
        cached_matcher = cached_matcher.to(device)
    
    return cached_matcher
