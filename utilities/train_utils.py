# Packages
import time
import os
from datetime import datetime
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Project
from .models import SAGEHomographyNet
from .matchers_utils import create_mask_and_kps_matrix
from .loss_functions import congealing_loss_pair_normalized, optimize_reflections_coord_descent_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _reflect_keypoints_matrix(kps_matrix):
    """
    Reflect keypoints across the y-axis (vertical axis) in normalized space.
    Since keypoints are centered at (0,0), reflection is done by negating x.

    Args:
        kps_matrix: (B, B, N, 2), keypoints normalized

    Returns:
        reflected_kps: (B, 2, B, N, 2), where reflected_kps[b, r] is kps_matrix with
                       image b reflected if r == 1 (x → -x), unchanged if r == 0.
    """
    reflected = kps_matrix.clone()
    reflected[:, :, :, 0] = -reflected[:, :, :, 0]  # Reflect x across y-axis
    return torch.stack([kps_matrix, reflected], dim=1)  # (B, 2, B, N, 2)

def train_gnn_reflections_matrix(model_gcn: SAGEHomographyNet, graph_data, matched_keypoints_by_pair,
                                 image_paths, image_size,
                                 reflection_epoch_check=100,
                                 num_epochs=1000, lr=2e-4,
                                 weight_decay=0.0, visualize=False,
                                 matrix_exp=False, sigma=0.01, patience_sched=10, factor_sched=0.5, stn_n=1):

    optimizer = optim.AdamW(model_gcn.parameters(), lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=factor_sched,
        patience=patience_sched
    )
    loss_history_gcn = []

    graph_data = graph_data.clone().detach().to(device)
    B = int(graph_data.batch.max().item()) + 1

    kps_matrix, mask = create_mask_and_kps_matrix(matched_keypoints_by_pair, B, device, image_size)
    kps_reflected_matrix = _reflect_keypoints_matrix(kps_matrix)
    model_gcn.to(device)

    if visualize:
        for part in image_paths[0].split('/')[::-1]:
            if "spair_" in part or "cub_" in part:
                dataset_folder = part
                break
        else:
            dataset_folder = "unknown_dataset"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    best_reflections = torch.zeros(B, dtype=torch.long, device=device)
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        model_gcn.train()
        optimizer.zero_grad()

        H_n = model_gcn.forward_n(graph_data, n=stn_n)

        if epoch % reflection_epoch_check == 0:
            best_reflections = optimize_reflections_coord_descent_matrix(kps_reflected_matrix, H_n, mask, matrix_exp, sigma=sigma)

        current_kps_matrix = kps_reflected_matrix[torch.arange(B), best_reflections]
        loss, pair_loss = congealing_loss_pair_normalized(current_kps_matrix, H_n, mask, matrix_exp, sigma)
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"}, refresh=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_gcn.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        loss_history_gcn.append(loss.item())

    plt.figure(figsize=(6, 4))
    plt.plot(loss_history_gcn)
    plt.title("Graph Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.close()

    return loss_history_gcn, best_reflections