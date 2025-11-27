# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM

import argparse
import subprocess
import sys
from pathlib import Path
from prepare_data import download_spair, download_cub_metadata, load_acsm_data_and_process
from prepare_data import load_spair_data_and_process


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

    
    args = parser.parse_args()
    Path('data').mkdir(parents=True, exist_ok=True)
    path = download_spair('data')
    load_spair_data_and_process(path, method='pad', size=args.size, spair_sets_path='data/spair_sets', image_format=args.out_format)
    run_grounded_sam("grounded_sam_spair_split.py", "SPair-71k mask generation")