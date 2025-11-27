import os
import gc
import sys
import time
from datetime import datetime
import torch
import cv2
import numpy as np
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

ROMA_ROOT = Path(__file__).resolve().parent / "third_party" / "RoMa"
if str(ROMA_ROOT) not in sys.path:
    sys.path.insert(0, str(ROMA_ROOT))

from romatch import roma_outdoor

from train_args import get_argparser
from utilities.homography_utils import export_homographies_to_json
from utilities.models import SAGEHomographyNet
from utilities.plot_utils import (
    plot_warped_grid_images_canonical_single,
    plot_warped_grid_images_ref_single,
)
from utilities.train_utils import train_gnn_reflections_matrix
from utilities.graph_utils import (
    build_graph_from_matches_RoMa,
    build_graph_from_fused_keypoints,
)
from utilities.matchers_utils import RoMa_extract_matches_cached, to_masked_path
from utilities.clustering_utils import dpmeans_clustering

from transformers.reflection_transformer import ReflectionTransformer
from transformers.sequence_transformer import SequenceTransformer
from transformers.homography_transformer import HomographyTransformer
from eval.pck import get_pck, get_pairs

def seed_everything(seed):
    if seed is not None and seed >= 0:
        print(f"Setting seed to {seed}")
        # Python's built-in random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for deterministic behavior
        
        torch.backends.cudnn.deterministic = True   # needed for reproducibility
        torch.backends.cudnn.benchmark = False  
    
        # OpenCV
        cv2.setRNGSeed(seed)
    else:
        print("No seed set, results will not be reproducible")
    
def init_config(parser):
    config = vars(parser.parse_args())

    if config['run_name'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_folder_name = Path(config["data_folder"]).name
        config['run_name'] = f"{timestamp}_{data_folder_name}"

    return config

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    parser = get_argparser()
    config = init_config(parser)
    config['device'] = device
    
    # Create results directory based on run_name
    results_dir = Path(__file__).resolve().parent / "results" / config['run_name']
    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir = str(results_dir)
    config['results_dir'] = results_dir

    if config['seed'] != -1:
        seed_everything(config['seed'])
    # RoMa matcher
    roma_model = roma_outdoor(device=device, coarse_res=config['roma_coarse_res'], upsample_res=config['roma_upsample_res'])
    start = time.time()
    matched_keypoints_by_pair = {}
    image_size = roma_model.get_output_resolution()

    image_dir = f"{config['data_folder']}/images"
    image_paths = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir))]
    image_paths_masked = [to_masked_path(p) for p in image_paths]
    matched_keypoints_by_pair = RoMa_extract_matches_cached(roma_model, image_paths, config['nms_radius_prec'], config['max_keypoints'], config['roma_batch_size'], device=device)
    del roma_model
    gc.collect()
    torch.cuda.empty_cache()
    end = time.time()
    print(f"RoMa matches: {end - start:.3f} seconds")
    # Graph
    graph_data, image_kpts_coords = build_graph_from_matches_RoMa(
            matched_keypoints_by_pair, image_size, device=device
    )
    start = time.time()
    
    # GPU-optimized clustering
    fused_keypoints, fused_batches, orig_to_fused_map = dpmeans_clustering(
        graph_data.x, graph_data.batch,
        delta=config['dpmeans_delta'], n_init=config['dpmeans_n_init'], device=device)
    end = time.time()
    print(f"DP-Means clustering + fusion: {end - start:.3f} seconds")
    
    fused_graph = build_graph_from_fused_keypoints(
        fused_keypoints=fused_keypoints,
        fused_batches=fused_batches,
        orig_to_fused_map=orig_to_fused_map,
        original_data=graph_data
    )
    
    # For GPU clustering, use the original matched keypoints for training
    # The fused graph will be used for the model forward pass
    fused_matched_keypoints_by_pair = matched_keypoints_by_pair

    # GPU RAM clean-up
    if 'roma_model' in locals():
        del roma_model
    if 'fused_keypoints' in locals():
        del fused_keypoints
    if 'fused_batches' in locals():
        del fused_batches
    if 'graph_data' in locals():
        del graph_data
    del image_kpts_coords
    if 'groups' in locals():
        del groups
    del matched_keypoints_by_pair
    gc.collect()
    torch.cuda.empty_cache()

    model = SAGEHomographyNet(config['start_with_id_matrix'], config['matrix_exp'],
                          hidden_channels=config['hidden_channels'], num_layers=config['num_layers'])
    # train
    start = time.time()
    loss, best_reflections = train_gnn_reflections_matrix(model, fused_graph, fused_matched_keypoints_by_pair,
                                                                     image_paths_masked, image_size,
                                                                     reflection_epoch_check = config['reflection_epoch_check'],
                                                                     num_epochs = config['num_epochs'], lr=config['lr'],
                                                                     weight_decay = config['weight_decay'],
                                                                     matrix_exp=config['matrix_exp'], sigma=config['sigma'],
                                                                     patience_sched=config['patience_sched'], factor_sched=config['factor_sched'],
                                                                     stn_n=config['stn_n'])
    end = time.time()
    print(f"Training: {end - start:.3f} seconds")
    # pck
    result_dict = {}
    with torch.no_grad():
        thetas = model.forward_n(fused_graph, config['stn_n'])

        # Load and stack input images
        transform = transforms.Compose([transforms.ToTensor()])
        images_tensor = torch.stack([
            transform(Image.open(path).convert("RGB")) for path in image_paths
        ]).to(device)

        # Compose transformations per image
        transformers = [
            SequenceTransformer(
                [
                    ReflectionTransformer(image_size, best_reflection.unsqueeze(0)),
                    HomographyTransformer(image_size, (torch.linalg.inv(theta)).unsqueeze(0), lie_algebra=False)
                ],
                combine_transformations=True
            )
            for theta, best_reflection in zip(thetas, best_reflections)
        ]

        # Evaluate PCK (only if pck folder exists)
        pck_folder = os.path.join(config['data_folder'], 'pck')
        if os.path.exists(pck_folder):
            pck_scores, _ = get_pck(config['alpha_list'], images_tensor, config['data_folder'],
                                    transformers, max_pairs_to_return=1, mirror=True)
            
            # Store results
            for alpha in config['alpha_list']:
                result_dict[f"pck/pck@{alpha}"] = pck_scores[alpha]['score']
        else:
            print(f"Warning: PCK folder not found at {pck_folder}, skipping PCK evaluation")
            # Set default PCK scores
            for alpha in config['alpha_list']:
                result_dict[f"pck/pck@{alpha}"] = -1.0

    # Export homographies/thetas to JSON
    export_homographies_to_json(thetas, best_reflections, config, image_paths, result_dict)
    
    # Visualize Results
    if config['visualize_results']:
        plot_warped_grid_images_canonical_single(image_paths_masked, model, fused_graph, image_size,
                                                config['stn_n'], best_reflections,
                                                dpi=150,
                                                save_path=f"{config['results_dir']}/canonical")
        plot_warped_grid_images_ref_single(image_paths_masked, model, fused_graph, image_size,
                                    stn_n=config['stn_n'], ref=config['ref'],
                                    best_reflections=best_reflections,
                                    save_path=f"{config['results_dir']}/to_{config['ref']}_ith",
                                    dpi=150)

    print("=" * 160)

if __name__ == "__main__":
    main()