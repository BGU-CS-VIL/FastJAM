# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM

import torch
import torch.nn.functional as F
from typing import Optional, List
from copy import deepcopy

class Transformer:
    def __init__(self, img_size, theta, *args, **kwargs):
        super().__init__()
        self.img_size = img_size    # (H, W)
        self.theta = theta          # Defined by each transformer

    def _make_initial_grid(self, shape: List[int], device) -> torch.Tensor:
        """Create initial grid for transformation of range [-1, 1]x[-1, 1]
        
        Args:
            shape (List[int]): shape of the grid (shape: (N, _, H, W))
            
        Returns:
            torch.Tensor: initial grid (shape: (N, H, W, 2))
        """
        N, _, H, W = shape
        y, x = torch.meshgrid([torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device)], indexing='ij')
        grid = torch.stack([x, y], dim=-1)          # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1) # (N, H, W, 2)
        return grid
        

    def make_grid(self, initial_grid: Optional[torch.Tensor] = None, inverse=False) -> torch.Tensor:
        """Create grid for transformation
        
        Args:
            initial_grid (torch.Tensor, optional): grid to apply transformation on (shape: (N, H, W, 2)). 
            Defaults to None (meshgrid will be created).
            inverse (bool, optional): forward or backward transformation. Defaults to False.
            
        Returns:
            torch.Tensor: grid for transformation (shape: (N, H, W, 2))
        """
        raise NotImplementedError


    def __call__(self, x: torch.Tensor, inverse=False) -> torch.Tensor:
        """Apply transformation on image / feature-map

        Args:
            x (torch.Tensor): expected shape (N, C, H, W) for image / feature-map
            inverse (bool, optional): forward or backward transformation. Defaults to False.

        Returns:
            torch.Tensor: expected shape (N, C, H, W) for transformed image / feature-map
        """
        grid = self.make_grid(inverse=inverse)      # TODO: moving it to __init__ will make things much faster (but double check ther eis no abuse of the transformer object (with different batch sizes / resolutions), maybe update grid in set_image_size)
        assert grid.shape[0] == x.shape[0], f"Expected grid to have same batch size as x, but got grid {grid.shape} and x {x.shape}"
        
        if x.dim() == 5:    # Case of double reflections x is (N=B*Ref, Ref, C, H, W) and grid is (N=B*Ref, H, W, 2)
            num_reflections = x.shape[1]
            grid = grid.unsqueeze(1).repeat(1, num_reflections, 1, 1, 1).reshape(-1, *grid.shape[-3:])  # (N=B*Ref*Ref, H, W, 2)
            x = x.reshape(-1, *x.shape[-3:])                                                            # (N=B*Ref*Ref, C, H, W)
            return self.warp_image(x, grid).reshape(-1, num_reflections, *x.shape[-3:])                 # (N=B*Ref, Ref, C, H, W)
        else:               # No double reflections, x is (N, C, H, W) and grid is (N, H, W, 2) where N can be B or B*Ref 
            return self.warp_image(x, grid)


    def set_image_size(self, img_size):
        assert len(img_size) == 2, f"Expected img_size to have length 2, but got {len(img_size)}"
        self.img_size = img_size


    def warp_image(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected x to have 4 dimensions, but got {x.dim()}"
        assert x.shape[0] == grid.shape[0], f"Expected grid to have same batch size as x, but got {grid.shape[0]} and {x.shape[0]}"
        assert grid.shape[-3:-1] == x.shape[-2:], f"Expected grid to have shape (N, {x.shape[-2]}, {x.shape[-1]}, 2), but got {grid.shape}"
        grid_smapled = F.grid_sample(x, grid, align_corners=False)
        return grid_smapled
        
    
    def _normalize_points(self, x: torch.Tensor, inverse=False) -> torch.Tensor:
        """Normalize or denormalize points to/from range [-1, 1]x[-1, 1] from/to [0, W]x[0, H]

        Args:
            x (torch.Tensor): expected shape (N, P, 2) for points. points in (x, y) format.
            inverse (bool, optional): normalize or denormalize. Defaults to False (normalize).

        Returns:
            torch.Tensor: expected shape (N, P, 2) for normalized or denormalized points
        """
        H, W = self.img_size
        if inverse:
            return (x + 1) * torch.tensor([W, H], dtype=x.dtype, device=x.device) / 2
        else:
            return (x * 2 / torch.tensor([W, H], dtype=x.dtype, device=x.device)) - 1


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
        raise NotImplementedError
    

    def try_combine(self, other_transformer):
        """Try to combine with other transformer.
        If successful, return new combined transformer instance.
        Otherwise, return None.
        """
        return None
    
    
    def select_single_reflection_transformer(self, config, batch_size, reflection_num):
        self.theta = self.theta.reshape(batch_size, reflection_num, *self.theta.shape[1:])[torch.arange(config.shape[0]), config]

        
        