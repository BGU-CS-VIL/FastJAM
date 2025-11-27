"""
Simple example showing how to use cached RoMa for efficient multi-image matching.
"""

import torch
from PIL import Image
from romatch import roma_outdoor
from romatch.utils.batch_matcher import create_cached_matcher_from_roma, BatchImageMatcher


def simple_cached_example():
    """Simple example of using cached RoMa matching."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a standard RoMa matcher
    print("Creating RoMa matcher...")
    roma_matcher = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    
    # Convert to cached version
    print("Converting to cached version...")
    cached_matcher = create_cached_matcher_from_roma(roma_matcher, device=device)
    
    # Create batch processor
    batch_processor = BatchImageMatcher(
        matcher=cached_matcher,
        device=device,
        cache_size=1000,  # Cache up to 1000 images
        precompute_features=True  # Precompute DINO features for all images
    )
    
    # Add your images (replace with your actual image paths)
    image_paths = [
        "assets/toronto_A.jpg",
        "assets/toronto_B.jpg", 
        "assets/sacre_coeur_A.jpg",
        "assets/sacre_coeur_B.jpg"
    ]
    
    # Filter to existing images
    existing_images = [path for path in image_paths if path]  # Add your actual paths here
    
    if len(existing_images) < 2:
        print("Please provide at least 2 image paths")
        return
    
    print(f"Adding {len(existing_images)} images...")
    image_ids = batch_processor.add_images(existing_images)
    
    # Match all possible pairs
    print("Matching all pairs...")
    results = batch_processor.match_all_pairs()
    
    print(f"Completed matching {len(results)} pairs")
    
    # Print cache statistics
    stats = batch_processor.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
    print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
    
    # Example: Match specific pairs
    if len(existing_images) >= 3:
        print("\nMatching specific pairs...")
        specific_pairs = [(0, 1), (0, 2)]  # Match first image with second and third
        specific_results = batch_processor.match_specific_pairs(specific_pairs)
        print(f"Completed {len(specific_results)} specific pairs")


if __name__ == "__main__":
    simple_cached_example()
