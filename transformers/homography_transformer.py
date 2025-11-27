# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM

import torch
from typing import Optional, Tuple

from transformers.transformer import Transformer

class HomographyTransformer(Transformer):
    def __init__(self, img_size, theta, lie_algebra=True):
        super().__init__(img_size, theta)
        self.lie_algebra = lie_algebra
        assert theta.dim() == 3, f"Expected theta to have 3 dimensions, but got {theta.dim()} dimensions (shape: {theta.shape})"
        assert theta.shape[-2:] == (3, 3), f"Expected theta to have shape (3, 3), but got {theta.shape}"
        self.theta_inv = -self.theta if lie_algebra else torch.inverse(self.theta)
        self.mirror = torch.det(theta) < 0  # [N] type: bool


    def try_combine(self, other_transformer):
        if isinstance(other_transformer, HomographyTransformer):
            theta1 = torch.matrix_exp(self.theta) if self.lie_algebra else self.theta                                         # [N, 3, 3] 
            theta2 = torch.matrix_exp(other_transformer.theta) if other_transformer.lie_algebra else other_transformer.theta  # [N, 3, 3]
            return HomographyTransformer(self.img_size, torch.bmm(theta1, theta2), lie_algebra=False)

        return None


    def make_grid(self, initial_grid: Optional[torch.Tensor] = None, inverse=False) -> torch.Tensor:
        theta = self.theta_inv if inverse else self.theta

        if self.lie_algebra:
            theta = torch.matrix_exp(theta)   # [N, 3, 3]

        if initial_grid is None:
            N, H, W = theta.shape[0], self.img_size[0], self.img_size[1]
            initial_grid = self._make_initial_grid([N, 1, H, W], device=theta.device)  # [N, H, W, 2]
        else:
            assert initial_grid.shape[0] == theta.shape[0], f"Expected initial_grid to have {theta.shape[0]} rows, but got {initial_grid.shape[0]} rows"
            assert initial_grid.shape[3] == 2, f"Expected initial_grid to have shape [N, H, W, 2], but got {initial_grid.shape}"
            assert initial_grid.shape[1] == self.img_size[0] and initial_grid.shape[2] == self.img_size[1], f"Expected initial_grid to have shape [N, {self.img_size[0]}, {self.img_size[1]}, 2], but got {initial_grid.shape}"
            N, H, W, _ = initial_grid.shape

        initial_grid = initial_grid.permute(0, 3, 1, 2) # [N, 2, H, W]
        initial_grid = initial_grid.reshape(theta.shape[0], 2, -1) # [N, 2, H*W]
        initial_grid = torch.cat([initial_grid, torch.ones_like(initial_grid[:, :1, :])], dim=1) # [N, 3, H*W]

        grid_t = torch.bmm(theta, initial_grid) # ([N, 3, 3] @ [N, 3, H*W]) = [N, 3, H*W]
        grid_t = grid_t[:, :2, :] / (grid_t[:, 2, :].unsqueeze(1) + 1e-8)  # [N, 2, H*W]
        grid_t = grid_t.permute(0, 2, 1).reshape(N, H, W, 2)               # [N, H, W, 2]
        return grid_t
        
        
    def warp_points(self, x: torch.Tensor, inverse=False, normalize=True, **kwargs) -> torch.Tensor:
        """map points to new space by applying transformation.
        Points are in (x, y) format.
        Note that in case of varying number of points, N should be 1.
        
        Args:
            x (torch.tensor): expected shape (N, P, 2) for points
            inverse (bool, optional): forward or backward transformation. Defaults to False.
            normalize (bool, optional): if points are in range [0, W]x[0, H], they will be normalized and denormalized after transformation.

        Returns:
            torch.tensor: expected shape (N, P, 2) for warped points
        """
        assert x.dim() == 3, f"Expected x to have 3 dimensions, but got {x.dim()} dimensions (shape: {x.shape})"

        inverse = not inverse  # Hack because learned theta for grid warping is inverse transformation

        theta = self.theta_inv if inverse else self.theta # [N, 3, 3]
            
        if self.lie_algebra:
            theta = torch.matrix_exp(theta)     # [N, 3, 3]

        if normalize:   # from [0, W]x[0, H] to [-1, 1]x[-1, 1]
            x = self._normalize_points(x, inverse=False)

        pointsA_hom = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=2)   # [N, P, 3]
        points_transformed = torch.bmm(pointsA_hom, theta.transpose(1, 2))  # [N, P, 3] @ [N, 3, 3] = [N, P, 3]
        points_transformed = points_transformed[:, :, :2] / (points_transformed[:, :, 2:] + 1e-8)   # [N, P, 2]

        if normalize:   # from [-1, 1]x[-1, 1] to [0, W]x[0, H]
            points_transformed = self._normalize_points(points_transformed, inverse=True)
        return points_transformed


    def select_single_reflection_transformer(self, config, batch_size, reflection_num):
        self.theta_inv = self.theta_inv.reshape(batch_size, reflection_num, *self.theta_inv.shape[1:])[torch.arange(config.shape[0]), config]
        super().select_single_reflection_transformer(config, batch_size, reflection_num)