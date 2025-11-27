# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM

import torch
import logging
from typing import Dict, List, Tuple, Optional, Union

from transformers.transformer import Transformer
from transformers.homography_transformer import HomographyTransformer

class SequenceTransformer(Transformer):
    def __init__(self, transformers : List[Transformer], combine_transformations=True):
        assert len(transformers) > 0, "Expected at least one transformer"
        assert all([transformer.img_size == transformers[0].img_size for transformer in transformers]), "All transformers should have the same img_size"
        assert all([transformer.theta.shape[0] == transformers[0].theta.shape[0] for transformer in transformers]), "All transformers should have the same batch_size"
        assert all([transformer.theta.device == transformers[0].theta.device for transformer in transformers]), "All transformers should have the same device"

        batch_size = transformers[0].theta.shape[0]

        # Flatten SequenceTransformers given in transformers
        flattened_transformers = []
        for transformer in transformers:
            if isinstance(transformer, SequenceTransformer):
                flattened_transformers += transformer.transformers
            else:
                flattened_transformers.append(transformer)
        transformers = flattened_transformers

        # Reduce the number of transformations (and thus the number of grids / interpolation) by combining (combinable) transformations
        if combine_transformations:
            transformers = self._combine_transformations(transformers)

        # Check if transformation is going to mirror the image by checking if homography is mirrored
        self.mirror = torch.zeros(batch_size, dtype=torch.bool, device=transformers[0].theta.device)
        for transformer in transformers:
            if isinstance(transformer, HomographyTransformer):
                self.mirror = self.mirror ^ transformer.mirror  # XOR

        self.transformers = transformers
        super().__init__(img_size=transformers[0].img_size, 
                         theta=torch.zeros(batch_size, device=transformers[0].theta.device)) # Bit of abuse of the super class, but it should work fine (only using batch_size and device)



    @staticmethod
    def _combine_transformations(transformers: List[Transformer]) -> List[Transformer]:
        res_transformers = []

        for i, transformer in enumerate(transformers):
            if i == 0:
                res_transformers.append(transformer)
            else:
                combine_res = res_transformers[-1].try_combine(transformer)
                if combine_res is None:
                    res_transformers.append(transformer)
                else:
                    res_transformers[-1] = combine_res

        return res_transformers


    def try_combine(self, other_transformer):
        if isinstance(other_transformer, SequenceTransformer):
            return SequenceTransformer(self.transformers + other_transformer.transformers)
        elif isinstance(other_transformer, Transformer):
            return SequenceTransformer(self.transformers + [other_transformer])
        return None


    def set_image_size(self, img_size):
        for tf in self.transformers:
            tf.set_image_size(img_size)
        return super().set_image_size(img_size)


    def make_grid(self, initial_grid: Optional[torch.Tensor]=None, inverse=False, return_all_grids=False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """make single grid for transformation (forward or backward)

        Args:
            - initial_grid (torch.Tensor, optional): grid to apply transformation on (shape: (N, H, W, 2)) with (x,y) format.
            Defaults to None (meshgrid will be created).
            - inverse (bool, optional): forward or backward transformation. Defaults to False.
            - return_all_grids (bool, optional): grid generating strategy. Defaults to True (return all grids).
            True (default): return all grids (list of grids) for each transformation on the identity grid.
            False: return the result of warping the initial_grid with all transformations in sequence.

        Returns:
            torch.Tensor: grid after transformations (shape: (N, H, W, 2)) or list of grids
        """

        if return_all_grids:
            assert initial_grid is None, "Expected initial_grid to be None when return_all_grids=True"
            return self._make_grids(inverse=inverse)
        else:
            return self._make_transformed_grid(initial_grid=initial_grid, inverse=inverse)



    def _make_transformed_grid(self, initial_grid: Optional[torch.Tensor]=None, inverse=False) -> torch.Tensor:
        transformers = self.transformers[::-1] if not inverse else self.transformers    # when using transforms on grid, order is reversed
        N, H, W = self.theta.shape[0], self.img_size[0], self.img_size[1]

        if initial_grid is None:
            initial_grid = self._make_initial_grid([N, 1, H, W], device=self.theta.device)  # [N, H, W, 2]
        else:
            assert initial_grid.shape[0] == N, f"Expected initial_grid to have {N} rows, but got {initial_grid.shape[0]} rows"
            assert initial_grid.shape[3] == 2, f"Expected initial_grid to have shape [N, H, W, 2], but got {initial_grid.shape}"
            assert initial_grid.shape[1] == self.img_size[0] and initial_grid.shape[2] == self.img_size[1], f"Expected initial_grid to have shape [N, {self.img_size[0]}, {self.img_size[1]}, 2], but got {initial_grid.shape}"

        grid = initial_grid
        for transformer in transformers:
            grid = transformer.make_grid(initial_grid=grid, inverse=inverse)
        return grid
    
    def _make_grids(self, inverse=False) -> List[torch.Tensor]:
        transformers = self.transformers[::-1] if inverse else self.transformers
        return [transformer.make_grid(inverse=inverse) for transformer in transformers]
    
        

    def warp_image(self, x: torch.Tensor, grids: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """map image to new space by applying transformation grid(s). Interpolation is done after every warp (when there are multiple grids).
        It is recommended to use combine_transformations=True optimization when creating the SequenceTransformer to reduce the number of grids.
        
        Args:
            x (torch.tensor): expected shape (N, C, H, W) for image / feature-map
            grids (torch.tensor | List[torch.Tensor]): grid or list of grids with expected shape (N, H, W, 2) each
        
        Returns:
            torch.tensor: expected shape (N, C, H, W) for warped image
        """
        if not isinstance(grids, list):
            grids = [grids]

        for grid in grids:
            x = super().warp_image(x, grid)
        return x



    def warp_points(self, x: torch.Tensor, inverse=False, normalize=True, **kwargs) -> torch.Tensor:
        """map points to new space by applying transformation.
        Points are in (x, y) format.
        Note that in case of varying number of points, N should be 1.
        
        Args:
            x (torch.tensor): expected shape (N, P, 2) for points
            inverse (bool, optional): forward or backward transformation. Defaults to False.
            normalize (bool, optional): if points are in range [0, H]x[0, W], they will be normalized and denormalized after transformation.

        Returns:
            torch.tensor: expected shape (N, P, 2) for warped points
        """
        if normalize:
            x = self._normalize_points(x, inverse=False)

        transformers = self.transformers[::-1] if inverse else self.transformers
        for transformer in transformers:
            x = transformer.warp_points(x, inverse=inverse, normalize=False)

        if normalize:
            x = self._normalize_points(x, inverse=True)

        return x
    
    
    def select_single_reflection_transformer(self, config, batch_size, reflection_num):
        for transformer in self.transformers:
            transformer.select_single_reflection_transformer(config, batch_size, reflection_num)
        super().select_single_reflection_transformer(config, batch_size, reflection_num)