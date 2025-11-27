import cv2
import numpy as np
import supervision as sv
import os
import time
import json
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm

from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = Path(__file__).resolve().parent
# Use current working directory as the repo root (should be FastJAM when running from terminal)
REPO_ROOT = Path.cwd()

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = SCRIPT_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = SCRIPT_DIR / "groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = SCRIPT_DIR / "sam_vit_h_4b8939.pth"

# Load models
grounding_dino_model = Model(
    model_config_path=str(GROUNDING_DINO_CONFIG_PATH),
    model_checkpoint_path=str(GROUNDING_DINO_CHECKPOINT_PATH)
)

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=str(SAM_CHECKPOINT_PATH))
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Hyperparameters
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# Run script
base_root = (REPO_ROOT / "data" / "cub_subsets").resolve()
start_time = time.time()

if not base_root.exists():
    raise FileNotFoundError(f"CUB subsets data not found at {base_root}")

for class_folder in sorted(base_root.iterdir()):
    if not class_folder.is_dir():
        continue

    split_path = class_folder / "images"
    if not split_path.exists():
        continue

    CLASSES = ['bird']
    result_folder = class_folder / "masks"
    result_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in split_path.iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    print(f"Processing {class_folder.name} with {len(image_paths)} images...")

    for image_path in tqdm(image_paths):
        image = cv2.imread(str(image_path))
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # Apply NMS
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        combined_mask = np.any(detections.mask, axis=0) if detections.mask is not None and len(detections.mask) > 0 else np.zeros(image.shape[:2], dtype=bool)
        annotated_image = image.copy()
        annotated_image[~combined_mask] = 0

        filename = image_path.stem
        cv2.imwrite(str(result_folder / f"{filename}.jpg"), annotated_image)
        cv2.imwrite(str(result_folder / f"{filename}_mask.png"), (combined_mask.astype(np.uint8) * 255))
        with open(result_folder / f"{filename}.json", 'w') as f:
            json.dump(combined_mask.astype(int).tolist(), f)

print(f"✅ Finished in {time.time() - start_time:.2f} seconds")