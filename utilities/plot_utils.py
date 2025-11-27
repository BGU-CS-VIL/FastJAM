# Packages
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# Project
from .homography_utils import get_homography_matrix, apply_homography

from transformers.reflection_transformer import ReflectionTransformer
from transformers.sequence_transformer import SequenceTransformer
from transformers.homography_transformer import HomographyTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_warped_grid_images_canonical_single(image_paths, model, graph_data, image_size,
                                             stn_n, best_reflections, dpi=150, save_path=None):
    import os
    import matplotlib.pyplot as plt

    device = graph_data.x.device
    B = len(image_paths)
    W, H = image_size

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # === Step 1: Compute transformations ===
    model.eval()
    with torch.no_grad():
        thetas = model.forward_n(graph_data, stn_n)

    avg_theta = thetas.mean(dim=0)  # shape: (3, 3)

    # === Step 2: Load images as tensors ===
    transform = transforms.Compose([transforms.ToTensor()])
    images_tensor = torch.stack([
        transform(Image.open(path).convert("RGB")) for path in image_paths
    ]).to(device)

    # === Step 3: Warp and plot each image ===
    for i in range(B):
        img_tensor = images_tensor[i].unsqueeze(0)  # shape: (1, 3, H, W)

        transformer = SequenceTransformer(
            [
                ReflectionTransformer(image_size, best_reflections[i].unsqueeze(0)),
                HomographyTransformer(image_size, torch.linalg.inv(thetas[i]).unsqueeze(0), lie_algebra=False),
                HomographyTransformer(image_size, avg_theta.unsqueeze(0), lie_algebra=False),
            ],
            combine_transformations=True
        )

        warped_img = transformer(img_tensor)
        warped_np = (warped_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        plt.imshow(warped_np)
        plt.axis("off")

        if save_path is not None:
            fig.savefig(os.path.join(save_path, f"img_{i}_to_canonical.png"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

def plot_warped_grid_images_ref_single(image_paths, model, graph_data, image_size, stn_n, ref,
                                       save_path=None, best_reflections=None, dpi=150):
    import os
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image

    device = graph_data.x.device
    B = len(image_paths)
    W, H = image_size

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Step 1: Compute transformations
    model.eval()
    with torch.no_grad():
        thetas = model.forward_n(graph_data, stn_n)

    # Step 2: Load images as tensors
    transform = transforms.Compose([transforms.ToTensor()])
    images_tensor = torch.stack([
        transform(Image.open(path).convert("RGB")) for path in image_paths
    ]).to(device)

    # Step 3: Warp and plot each image
    for i in range(B):
        img_tensor = images_tensor[i].unsqueeze(0)  # shape: (1, 3, H, W)

        transformer = SequenceTransformer(
            [
                ReflectionTransformer(image_size, best_reflections[i].unsqueeze(0)),
                HomographyTransformer(image_size, torch.linalg.inv(thetas[i]).unsqueeze(0), lie_algebra=False),
                HomographyTransformer(image_size, thetas[ref].unsqueeze(0), lie_algebra=False),
                ReflectionTransformer(image_size, best_reflections[ref].unsqueeze(0)),
            ],
            combine_transformations=True
        )

        warped_img = transformer(img_tensor)
        warped_np = (warped_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        plt.imshow(warped_np)
        plt.axis("off")

        if save_path is not None:
            fig.savefig(os.path.join(save_path, f"img_{i}_to_{ref}.png"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()