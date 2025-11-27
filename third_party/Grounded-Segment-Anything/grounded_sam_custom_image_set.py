import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv  # noqa: F401 (kept for parity with other scripts)
import torch
import torchvision
from tqdm import tqdm

from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import SamPredictor, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Grounded-SAM mask generation for a custom image set.")
    parser.add_argument("--data-root", required=True, type=str, help="Path to processed dataset root (contains images/).")
    parser.add_argument("--object-class", required=True, type=str, help="Object class name used for grounding prompts.")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.8, help="NMS IoU threshold.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()

    images_dir = data_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Expected images directory at {images_dir}")

    masks_dir = data_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    grounding_config = script_dir / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    grounding_ckpt = script_dir / "groundingdino_swint_ogc.pth"
    sam_ckpt = script_dir / "sam_vit_h_4b8939.pth"

    if not grounding_config.exists() or not grounding_ckpt.exists() or not sam_ckpt.exists():
        raise FileNotFoundError("Missing Grounded-SAM checkpoints/config. Please download them before running this script.")

    grounding_model = Model(
        model_config_path=str(grounding_config),
        model_checkpoint_path=str(grounding_ckpt),
    )

    sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt))
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    def segment(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        masks = []
        for box in boxes:
            mask_candidates, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
            masks.append(mask_candidates[np.argmax(scores)])
        return np.array(masks)

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not image_paths:
        print(f"[Grounded-SAM] No images found under {images_dir}, exiting.")
        return

    start_time = time.time()
    classes = [args.object_class]
    print(f"[Grounded-SAM] Generating masks for {len(image_paths)} images in {data_root} using class prompt '{args.object_class}'.")

    for image_path in tqdm(image_paths):
        image = cv2.imread(str(image_path))
        detections = grounding_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        if detections.xyxy.size == 0:
            combined_mask = np.zeros(image.shape[:2], dtype=bool)
        else:
            keep = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                args.nms_threshold,
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[keep]
            detections.confidence = detections.confidence[keep]

            masks = segment(
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                boxes=detections.xyxy,
            )
            combined_mask = np.any(masks, axis=0) if masks.size > 0 else np.zeros(image.shape[:2], dtype=bool)

        annotated = image.copy()
        annotated[~combined_mask] = 0

        stem = image_path.stem
        cv2.imwrite(str(masks_dir / f"{stem}.jpg"), annotated)
        cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), (combined_mask.astype(np.uint8) * 255))
        with open(masks_dir / f"{stem}.json", "w") as fh:
            json.dump(combined_mask.astype(int).tolist(), fh)

    print(f"[Grounded-SAM] Finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

