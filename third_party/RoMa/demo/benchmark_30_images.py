"""
Benchmark script to test runtime of 30 images with and without caching.
This script generates synthetic images and measures performance for both approaches.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from itertools import combinations

# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from romatch import roma_outdoor
from romatch.utils.batch_matcher import create_cached_matcher_from_roma, BatchImageMatcher


def generate_synthetic_images(num_images: int = 30, 
                            image_size: Tuple[int, int] = (448, 448),
                            device: torch.device = None) -> List[Image.Image]:
    """
    Generate synthetic test images for benchmarking.
    
    Args:
        num_images: Number of images to generate
        image_size: Size of each image (width, height)
        device: Device to use for computation
        
    Returns:
        List of PIL Images
    """
    print(f"Generating {num_images} synthetic images of size {image_size}...")
    
    images = []
    for i in range(num_images):
        # Create a synthetic image with some variation
        # Use different patterns to ensure images are unique
        pattern_type = i % 4
        
        if pattern_type == 0:
            # Random noise pattern
            img_array = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
        elif pattern_type == 1:
            # Gradient pattern
            x = np.linspace(0, 1, image_size[0])
            y = np.linspace(0, 1, image_size[1])
            X, Y = np.meshgrid(x, y)
            gradient = (X + Y) * 255
            img_array = np.stack([gradient, gradient * 0.8, gradient * 0.6], axis=2).astype(np.uint8)
        elif pattern_type == 2:
            # Checkerboard pattern
            checker_size = 20
            checker = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
            for row in range(0, image_size[1], checker_size):
                for col in range(0, image_size[0], checker_size):
                    if (row // checker_size + col // checker_size) % 2 == 0:
                        checker[row:row+checker_size, col:col+checker_size] = 255
            img_array = checker
        else:
            # Circular pattern
            center_x, center_y = image_size[0] // 2, image_size[1] // 2
            y, x = np.ogrid[:image_size[1], :image_size[0]]
            mask = (x - center_x)**2 + (y - center_y)**2 < (min(image_size) // 3)**2
            img_array = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
            img_array[mask] = 255
        
        # Add some noise to make each image unique
        noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        images.append(img)
    
    print(f"Generated {len(images)} synthetic images")
    return images


def benchmark_standard_roma(images: List[Image.Image], 
                           device: torch.device,
                           num_runs: int = 1) -> Dict[str, float]:
    """
    Benchmark standard RoMa without caching.
    
    Args:
        images: List of PIL Images
        device: Device to use
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with timing results
    """
    print("\n=== Standard RoMa (No Caching) ===")
    
    times = []
    total_pairs = len(list(combinations(range(len(images)), 2)))
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Create fresh matcher for each run
        matcher = roma_outdoor(device=device, coarse_res=560, upsample_res=(560, 560))
        
        start_time = time.time()
        pair_count = 0
        
        # Process all pairs
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                with torch.no_grad():
                    try:
                        warp, certainty = matcher.match(images[i], images[j], device=device)
                        pair_count += 1
                    except Exception as e:
                        print(f"Error matching pair ({i}, {j}): {e}")
                        continue
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Processed {pair_count} pairs in {elapsed:.2f}s")
        
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return {
        'times': times,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_pairs': total_pairs
    }


def benchmark_cached_roma(images: List[Image.Image], 
                         device: torch.device,
                         num_runs: int = 1,
                         cache_size: int = 1000,
                         precompute: bool = True) -> Dict[str, float]:
    """
    Benchmark cached RoMa with caching.
    
    Args:
        images: List of PIL Images
        device: Device to use
        num_runs: Number of benchmark runs
        cache_size: Maximum cache size
        precompute: Whether to precompute features
        
    Returns:
        Dictionary with timing results
    """
    print("\n=== Cached RoMa (With Caching) ===")
    
    times = []
    cache_stats_list = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Create fresh matcher for each run
        standard_matcher = roma_outdoor(device=device, coarse_res=560, upsample_res=(560, 560))
        cached_matcher = create_cached_matcher_from_roma(standard_matcher, device=device)
        
        # Create batch processor
        batch_processor = BatchImageMatcher(
            matcher=cached_matcher,
            device=device,
            cache_size=cache_size,
            precompute_features=precompute
        )
        
        start_time = time.time()
        
        # Add images
        image_ids = batch_processor.add_images(images)
        
        # Match all pairs
        results = batch_processor.match_all_pairs(
            return_correspondences=False, 
            return_certainty=False
        )
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Get cache stats
        stats = batch_processor.get_cache_stats()
        cache_stats_list.append(stats)
        
        print(f"  Processed {len(results)} pairs in {elapsed:.2f}s")
        print(f"  Cache hit rate: {stats['hit_rate']:.2%}")
        
        # Clear for next run
        batch_processor.clear_cache()
        batch_processor.clear_images()
        
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate average cache stats
    avg_hit_rate = np.mean([stats['hit_rate'] for stats in cache_stats_list])
    avg_hits = np.mean([stats['hits'] for stats in cache_stats_list])
    avg_misses = np.mean([stats['misses'] for stats in cache_stats_list])
    
    return {
        'times': times,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'avg_hit_rate': avg_hit_rate,
        'avg_hits': avg_hits,
        'avg_misses': avg_misses,
        'cache_stats': cache_stats_list
    }


def run_comprehensive_benchmark(num_images: int = 30,
                              num_runs: int = 3,
                              image_size: Tuple[int, int] = (448, 448),
                              cache_size: int = 1000,
                              precompute: bool = True):
    """
    Run comprehensive benchmark comparing standard vs cached RoMa.
    
    Args:
        num_images: Number of images to test
        num_runs: Number of benchmark runs
        image_size: Size of test images
        cache_size: Cache size for cached version
        precompute: Whether to precompute features
    """
    print(f"=== Comprehensive Benchmark: {num_images} Images ===")
    print(f"Number of pairs: {num_images * (num_images - 1) // 2}")
    print(f"Benchmark runs: {num_runs}")
    print(f"Image size: {image_size}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using device: {device}")
    
    # Generate test images
    images = generate_synthetic_images(num_images, image_size, device)
    
    # Benchmark standard RoMa
    print(f"\n{'='*50}")
    standard_results = benchmark_standard_roma(images, device, num_runs)
    
    # Benchmark cached RoMa
    print(f"\n{'='*50}")
    cached_results = benchmark_cached_roma(images, device, num_runs, cache_size, precompute)
    
    # Calculate performance metrics
    speedup = standard_results['mean_time'] / cached_results['mean_time']
    time_saved = standard_results['mean_time'] - cached_results['mean_time']
    efficiency = (time_saved / standard_results['mean_time']) * 100
    
    # Print results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Images: {num_images}")
    print(f"Pairs: {standard_results['total_pairs']}")
    print(f"Runs: {num_runs}")
    print(f"Image size: {image_size}")
    print()
    
    print("PERFORMANCE COMPARISON:")
    print(f"  Standard RoMa: {standard_results['mean_time']:.2f}s ± {standard_results['std_time']:.2f}s")
    print(f"  Cached RoMa:  {cached_results['mean_time']:.2f}s ± {cached_results['std_time']:.2f}s")
    print()
    
    print("SPEEDUP ANALYSIS:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.2f}s")
    print(f"  Efficiency gain: {efficiency:.1f}%")
    print()
    
    print("CACHE PERFORMANCE:")
    print(f"  Average hit rate: {cached_results['avg_hit_rate']:.2%}")
    print(f"  Average hits: {cached_results['avg_hits']:.0f}")
    print(f"  Average misses: {cached_results['avg_misses']:.0f}")
    print()
    
    # Calculate theoretical maximum speedup
    # Each image appears in (N-1) pairs, so we should compute DINO features N times instead of N*(N-1)/2 times
    theoretical_speedup = (num_images * (num_images - 1) / 2) / num_images
    print(f"THEORETICAL ANALYSIS:")
    print(f"  Theoretical max speedup: {theoretical_speedup:.2f}x")
    print(f"  Achieved speedup: {speedup:.2f}x")
    print(f"  Efficiency vs theoretical: {(speedup / theoretical_speedup) * 100:.1f}%")
    
    return {
        'standard_results': standard_results,
        'cached_results': cached_results,
        'speedup': speedup,
        'time_saved': time_saved,
        'efficiency': efficiency
    }


def create_performance_plot(results: Dict, save_path: str = "benchmark_results.png"):
    """Create a performance comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Timing comparison
    methods = ['Standard RoMa', 'Cached RoMa']
    times = [results['standard_results']['mean_time'], results['cached_results']['mean_time']]
    errors = [results['standard_results']['std_time'], results['cached_results']['std_time']]
    
    bars = ax1.bar(methods, times, yerr=errors, capsize=5, color=['red', 'green'], alpha=0.7)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Runtime Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time, error in zip(bars, times, errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + error + 1,
                f'{time:.1f}s', ha='center', va='bottom')
    
    # Speedup analysis
    speedup = results['speedup']
    efficiency = results['efficiency']
    
    ax2.bar(['Speedup', 'Efficiency (%)'], [speedup, efficiency], 
            color=['blue', 'orange'], alpha=0.7)
    ax2.set_ylabel('Value')
    ax2.set_title('Performance Metrics')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    ax2.text(0, speedup + 0.1, f'{speedup:.2f}x', ha='center', va='bottom')
    ax2.text(1, efficiency + 1, f'{efficiency:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Performance plot saved to {save_path}")


def test_different_cache_sizes(num_images: int = 30, 
                             cache_sizes: List[int] = [100, 500, 1000, 2000]):
    """Test performance with different cache sizes."""
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT CACHE SIZES")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    # Generate test images
    images = generate_synthetic_images(num_images, (448, 448), device)
    
    results = []
    
    for cache_size in cache_sizes:
        print(f"\nTesting cache size: {cache_size}")
        
        # Benchmark cached version
        cached_results = benchmark_cached_roma(images, device, num_runs=1, cache_size=cache_size)
        
        results.append({
            'cache_size': cache_size,
            'time': cached_results['mean_time'],
            'hit_rate': cached_results['avg_hit_rate'],
            'hits': cached_results['avg_hits'],
            'misses': cached_results['avg_misses']
        })
        
        print(f"  Time: {cached_results['mean_time']:.2f}s")
        print(f"  Hit rate: {cached_results['avg_hit_rate']:.2%}")
    
    # Find optimal cache size
    best_result = min(results, key=lambda x: x['time'])
    print(f"\nOptimal cache size: {best_result['cache_size']}")
    print(f"Best time: {best_result['time']:.2f}s")
    print(f"Best hit rate: {best_result['hit_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark RoMa caching with 30 images")
    parser.add_argument("--num_images", type=int, default=30, help="Number of images to test")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--cache_size", type=int, default=1000, help="Cache size for cached version")
    parser.add_argument("--no_precompute", action="store_true", help="Disable feature precomputation")
    parser.add_argument("--test_cache_sizes", action="store_true", help="Test different cache sizes")
    parser.add_argument("--save_plot", type=str, default="benchmark_30_images.png", help="Save performance plot")
    
    args = parser.parse_args()
    
    # Run main benchmark
    results = run_comprehensive_benchmark(
        num_images=args.num_images,
        num_runs=args.num_runs,
        cache_size=args.cache_size,
        precompute=not args.no_precompute
    )
    
    # Create performance plot
    create_performance_plot(results, args.save_plot)
    
    # Test different cache sizes if requested
    if args.test_cache_sizes:
        test_different_cache_sizes(args.num_images)
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {args.save_plot}")
    print(f"Speedup achieved: {results['speedup']:.2f}x")
    print(f"Time saved: {results['time_saved']:.2f}s")
