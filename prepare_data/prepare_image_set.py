# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM

import argparse
import subprocess
import sys
from pathlib import Path
from prepare_data import download_spair, download_cub_metadata, load_acsm_data_and_process
from prepare_data import load_image_folder_and_process


def run_grounded_sam(out_path: str, object_class: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "third_party" / "Grounded-Segment-Anything" / "grounded_sam_custom_image_set.py"
    if not script_path.exists():
        print(f"[Grounded-SAM] Skipping custom mask generation: script not found at {script_path}")
        return

    print(f"[Grounded-SAM] Running custom mask generation for class '{object_class}' via {script_path} ...")
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root",
        str(Path(out_path).resolve()),
        "--object-class",
        object_class,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[Grounded-SAM] WARNING: custom mask generation failed (exit code {exc.returncode}). See logs above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_format", type=str, choices=['png', 'jpg'], default='png', help="format to store images")
    parser.add_argument("--size", type=int, default=560, help="resolution of images for the dataset")
    parser.add_argument("--path", required=True, type=str, default=None, help="path to image set")
    parser.add_argument("--out", required=True, type=str, default=None, help="path to output folder, for example: data/cool_cats")
    parser.add_argument("--object_class", required=True, type=str, help="object class name used for grounding prompts (e.g., dog, airplane)")

    args = parser.parse_args()
    Path('data').mkdir(parents=True, exist_ok=True)
    load_image_folder_and_process(args.path, method='pad', size=args.size, out_path=args.out, image_format=args.out_format)
    run_grounded_sam(args.out, args.object_class)