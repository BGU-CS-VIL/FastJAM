# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM

import argparse
import subprocess
import sys
from pathlib import Path
from prepare_data import download_cub, download_cub_metadata, load_acsm_data_and_process
from prepare_data import load_acsm_data_and_process_class
from prepare_data import visualize_images


def run_grounded_sam(script_name: str, description: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "third_party" / "Grounded-Segment-Anything" / script_name
    if not script_path.exists():
        print(f"[Grounded-SAM] Skipping {description}: script not found at {script_path}")
        return

    print(f"[Grounded-SAM] Running {description} via {script_path} ...")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[Grounded-SAM] WARNING: {description} failed (exit code {exc.returncode}). See logs above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data of benchmarks SPair-71K/CUB_200_2011.')
    parser.add_argument("--out_format", type=str, choices=['png', 'jpg'], default='png', help="format to store images")
    parser.add_argument("--size", type=int, default=560, help="resolution of images for the dataset")
    parser.add_argument("--cub_acsm_class", required=True, choices=[str(i) for i in range(1, 201)], type=str,
                        help='If specified, constructs the CUB dataset by specific class. This will use the same pre-processing '
                        'as the CUB validation split from GANgealing (and originally, ACSM paper) but within a specific class (without shuffling).')
    parser.add_argument("--custom_set_size", type=int, default=None,  
                        help='If specified, the number of images in the output dataset will be this number. '
                                'If not specified, the number of images will be the default. ')
    args = parser.parse_args()

    acsm_class_val = int(args.cub_acsm_class)
    assert acsm_class_val >= 1 and acsm_class_val <= 200, 'CUB class must be between 1 and 200'

    Path('data').mkdir(parents=True, exist_ok=True)
    method = 'cub_crop'
    out = f'data/cub_classes/cub_class_{acsm_class_val:03d}'
    path = download_cub('data')
    
    load_acsm_data_and_process_class(path, method=method, size=args.size, out_path=out, image_format=args.out_format, 
                                     custom_set_size=args.custom_set_size, acsm_class_id_val=acsm_class_val)
    run_grounded_sam("grounded_sam_cub_classes.py", "CUB class mask generation")