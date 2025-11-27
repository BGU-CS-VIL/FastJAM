"""
Demo script showing how to use the cached RoMa matcher for efficient processing
of N images with all (N, 2) pairs.

This demonstrates the performance benefits of caching DINO features when
each image appears in multiple pairs.
"""

import os
import time
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from romatch import roma_outdoor
from romatch.models.cached_encoder import CachedCNNandDinov2, CachedRegressionMatcher
from romatch.utils.batch_matcher import BatchImageMatcher, create_cached_matcher_from_roma


def benchmark_comparison(image_paths: List[str], 
                        device: torch.device,
                        num_runs: int = 1) -> Dict[str, float]:
    """
    Compare performance between standard RoMa and cached RoMa for multiple image pairs.
    
    Args:
        image_paths: List of paths to images
        device: Device to use for computation
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing results
    """
    print(f"Benchmarking with {len(image_paths)} images ({len(list(combinations(range(len(image_paths)), 2)))} pairs)")
    
    # Load images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # Standard RoMa matcher
    print("\n=== Standard RoMa (No Caching) ===")
    standard_times = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Create standard matcher
        standard_matcher = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
        
        start_time = time.time()
        
        # Process all pairs with standard matcher
        pair_count = 0
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                with torch.no_grad():
                    warp, certainty = standard_matcher.match(images[i], images[j], device=device)
                pair_count += 1
        
        elapsed = time.time() - start_time
        standard_times.append(elapsed)
        print(f"  Processed {pair_count} pairs in {elapsed:.2f}s")
    
    # Cached RoMa matcher
    print("\n=== Cached RoMa (With Caching) ===")
    cached_times = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Create cached matcher
        standard_matcher = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
        cached_matcher = create_cached_matcher_from_roma(standard_matcher, device=device)
        
        # Create batch matcher
        batch_matcher = BatchImageMatcher(
            matcher=cached_matcher,
            device=device,
            cache_size=1000,
            precompute_features=True
        )
        
        start_time = time.time()
        
        # Add all images
        image_ids = batch_matcher.add_images(images)
        
        # Match all pairs
        results = batch_matcher.match_all_pairs(return_correspondences=False, return_certainty=False)
        
        elapsed = time.time() - start_time
        cached_times.append(elapsed)
        
        # Print cache stats
        stats = batch_matcher.get_cache_stats()
        print(f"  Processed {len(results)} pairs in {elapsed:.2f}s")
        print(f"  Cache hit rate: {stats['hit_rate']:.2%}")
        
        # Clear for next run
        batch_matcher.clear_cache()
        batch_matcher.clear_images()
    
    # Calculate averages
    avg_standard = np.mean(standard_times)
    avg_cached = np.mean(cached_times)
    speedup = avg_standard / avg_cached
    
    return {
        "standard_time": avg_standard,
        "cached_time": avg_cached,
        "speedup": speedup,
        "standard_std": np.std(standard_times),
        "cached_std": np.std(cached_times)
    }


def demo_cached_matching():
    """Demonstrate cached matching with sample images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using device: {device}")
    
    # Get available images
    image_paths = [
        "assets/toronto_A.jpg",
        "assets/toronto_B.jpg", 
        "assets/sacre_coeur_A.jpg",
        "assets/sacre_coeur_B.jpg"
    ]
    
    # Filter to existing images
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if len(existing_paths) < 2:
        print("Not enough images found. Please ensure you have at least 2 images in the assets folder.")
        return
    
    print(f"Found {len(existing_paths)} images: {existing_paths}")
    
    # Create cached matcher
    print("\n=== Creating Cached Matcher ===")
    standard_matcher = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    cached_matcher = create_cached_matcher_from_roma(standard_matcher, device=device)
    
    # Create batch matcher
    batch_matcher = BatchImageMatcher(
        matcher=cached_matcher,
        device=device,
        cache_size=1000,
        precompute_features=True
    )
    
    # Add images
    print("\n=== Adding Images ===")
    image_ids = batch_matcher.add_images(existing_paths)
    print(f"Added {len(image_ids)} images: {image_ids}")
    
    # Match all pairs
    print("\n=== Matching All Pairs ===")
    results = batch_matcher.match_all_pairs(return_correspondences=True, return_certainty=True)
    
    print(f"\nResults for {len(results)} pairs:")
    for i, result in enumerate(results):
        print(f"  Pair {i+1}: {result['image_id1']} <-> {result['image_id2']}")
        if result['certainty'] is not None:
            avg_certainty = result['certainty'].mean().item()
            print(f"    Average certainty: {avg_certainty:.3f}")
    
    # Print final cache stats
    stats = batch_matcher.get_cache_stats()
    print(f"\nFinal Cache Statistics:")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    # Demonstrate specific pair matching
    print("\n=== Matching Specific Pairs ===")
    if len(existing_paths) >= 3:
        specific_pairs = [(0, 1), (0, 2)]  # Match first image with second and third
        specific_results = batch_matcher.match_specific_pairs(specific_pairs)
        
        print(f"Results for {len(specific_results)} specific pairs:")
        for i, result in enumerate(specific_results):
            print(f"  Pair {i+1}: {result['image_id1']} <-> {result['image_id2']} "
                  f"(indices: {result['image_indices']})")


def demo_memory_efficiency():
    """Demonstrate memory efficiency of cached approach."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"\n=== Memory Efficiency Demo ===")
    print(f"Using device: {device}")
    
    # Create test images
    num_images = 5
    test_images = []
    for i in range(num_images):
        # Create a random test image
        img_array = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        test_images.append(img)
    
    print(f"Created {num_images} test images")
    
    # Standard approach
    print("\n--- Standard Approach ---")
    standard_matcher = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    
    start_time = time.time()
    pair_count = 0
    
    for i in range(num_images):
        for j in range(i + 1, num_images):
            with torch.no_grad():
                warp, certainty = standard_matcher.match(test_images[i], test_images[j], device=device)
            pair_count += 1
    
    standard_time = time.time() - start_time
    print(f"Standard approach: {pair_count} pairs in {standard_time:.2f}s")
    
    # Cached approach
    print("\n--- Cached Approach ---")
    cached_matcher = create_cached_matcher_from_roma(standard_matcher, device=device)
    batch_matcher = BatchImageMatcher(
        matcher=cached_matcher,
        device=device,
        cache_size=1000,
        precompute_features=True
    )
    
    start_time = time.time()
    
    # Add images
    image_ids = batch_matcher.add_images(test_images)
    
    # Match all pairs
    results = batch_matcher.match_all_pairs(return_correspondences=False, return_certainty=False)
    
    cached_time = time.time() - start_time
    print(f"Cached approach: {len(results)} pairs in {cached_time:.2f}s")
    
    # Print performance comparison
    speedup = standard_time / cached_time
    print(f"\nPerformance Comparison:")
    print(f"  Standard time: {standard_time:.2f}s")
    print(f"  Cached time: {cached_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Print cache efficiency
    stats = batch_matcher.get_cache_stats()
    print(f"\nCache Efficiency:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Total requests: {stats['hits'] + stats['misses']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")


def visualize_matching_results(results: List[Dict], image_paths: List[str], save_dir: str = "demo/cached_results"):
    """Visualize the matching results."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== Visualizing Results ===")
    print(f"Saving visualizations to {save_dir}")
    
    for i, result in enumerate(results):
        if result['flow'] is not None and result['certainty'] is not None:
            # Create a simple visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Load original images
            img1 = Image.open(image_paths[0])  # Simplified for demo
            img2 = Image.open(image_paths[1])  # Simplified for demo
            
            # Display images
            axes[0].imshow(img1)
            axes[0].set_title(f"Image 1: {result['image_id1']}")
            axes[0].axis('off')
            
            axes[1].imshow(img2)
            axes[1].set_title(f"Image 2: {result['image_id2']}")
            axes[1].axis('off')
            
            # Display certainty map
            certainty_map = result['certainty'][0, 0].cpu().numpy()
            im = axes[2].imshow(certainty_map, cmap='viridis')
            axes[2].set_title('Certainty Map')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/pair_{i+1}_matching.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Saved {len(results)} visualization files")


if __name__ == "__main__":
    import argparse
    from itertools import combinations
    
    parser = argparse.ArgumentParser(description="Demo cached RoMa matching")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--memory_demo", action="store_true", help="Run memory efficiency demo")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    if args.benchmark:
        # Run benchmark comparison
        image_paths = [
            "assets/toronto_A.jpg",
            "assets/toronto_B.jpg", 
            "assets/sacre_coeur_A.jpg",
            "assets/sacre_coeur_B.jpg"
        ]
        existing_paths = [path for path in image_paths if os.path.exists(path)]
        
        if len(existing_paths) >= 2:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            
            results = benchmark_comparison(existing_paths, device, args.num_runs)
            
            print(f"\n=== Benchmark Results ===")
            print(f"Standard RoMa: {results['standard_time']:.2f}s ± {results['standard_std']:.2f}s")
            print(f"Cached RoMa: {results['cached_time']:.2f}s ± {results['cached_std']:.2f}s")
            print(f"Speedup: {results['speedup']:.2f}x")
        else:
            print("Not enough images found for benchmark")
    
    elif args.memory_demo:
        demo_memory_efficiency()
    
    else:
        # Run basic demo
        demo_cached_matching()
        
        if args.visualize:
            # This would require the actual image paths from the demo
            pass
