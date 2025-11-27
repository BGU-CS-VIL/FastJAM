
# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM
# "SpaceJAM" code taken and modified from "Neural Congealing": https://github.com/dolev104/neural_congealing

from typing import List

import torch
import os
from torchvision.utils import make_grid

from transformers.transformer import Transformer

NUMBER_OF_CUB_SUBSETS = 14  # number of CUB subsets used for numeric evaluation (see paper for details)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

def images2grid(images, **grid_kwargs):
    # images should be (N, C, H, W)
    grid = make_grid(images, **grid_kwargs)
    out = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return out


def load_pairs_kps(path, dataset_type):
    keypoints_path = f'{path}/keypoints.pt'
    orig_pairs_path = f'{path}/pairs.pt'
    pairs_path = f'{path}/pairs_indices_in_set.pt'
    thresh_path = f'{path}/pck_thresholds.pt'
    inverse_ops_path = f'{path}/inverse_coordinates.pt'
    permutation_path = f'{path}/permutation.pt'

    if dataset_type in ("spair", "willow"):
        orig_pairs = torch.load(orig_pairs_path, weights_only=True)
        fixed_pairs = torch.load(pairs_path, weights_only=True)
        thresholds = torch.load(thresh_path, weights_only=True)
        inverse_ops = torch.load(inverse_ops_path, weights_only=True)
    elif dataset_type == "cub":
        orig_pairs, fixed_pairs, thresholds, inverse_ops = None, None, None, None
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not supported")
    keypoints = torch.load(keypoints_path, weights_only=True)
    mirror_permutation = torch.load(permutation_path, weights_only=True) if os.path.exists(permutation_path) else list(range(keypoints.shape[1]))

    return orig_pairs, fixed_pairs, keypoints, thresholds, inverse_ops, mirror_permutation


def get_pairs(data_folder: str):
    if 'cub' in data_folder:
        dataset_type = 'cub'
    elif 'spair' in data_folder:
        dataset_type = 'spair'
    elif 'willow' in data_folder:
        dataset_type = 'willow'
    else:
        dataset_type = 'unknown'

    orig_pairs, pairs, kps, thresholds, inverse_ops, mirror_permutation = load_pairs_kps(f"{data_folder}/pck", dataset_type)
    number_of_images = kps.shape[0]

    if dataset_type == 'cub':
        pairs = torch.combinations(torch.arange(number_of_images), r=2)
        orig_pairs = pairs

    return pairs, orig_pairs, kps, thresholds, inverse_ops, mirror_permutation

# TODO: this can be optimized by batching the points and warping them all at once.
def extract_and_transform_pairs(images, transformers, indices, pairs, orig_pairs, kps, mirror_permutation, mirror=False):
    """Function to extract pairs from the dataset and transform them using the provided transformer.
    Args:
        - images (torch.Tensor): tensor of all images in the dataset. [N, 3, H, W] \n
        - transformers (List[Transformer]): List of singleton transformers to use for the transformation. \n
        - pairs (torch.Tensor): tensor of image pairs to extract.     [Pairs, 2] \n
        - orig_pairs (torch.Tensor): mapping from pairs to the indices in the kps tensor. [Pairs, 2] \n
        - kps (torch.Tensor): tensor of keypoints (x,y,vis)           [N, Points, 3] \n
        - indices (List[int]): list of indices in the pairs tensor to extract. \n
        - mirror_permutation (List[int]): permutation tensor to apply to the keypoints to get the mirrored version. List of ints (len = len(Points))

    Returns:
        List[dict]: List with element for each pair, containing the following keys: \n
            idxA, idxB (int): indices of the images in the pair. \n 
            imgA, imgA_warped, imgB_warped, imgB (all (3, H, W))  \n
            gt_pointsA, gt_pointsB, gt_pointsA_warped, gt_pointsB_warped, est_pointsA, est_pointsB (all (N, 2)) \n
    """
    outputs = []

    device = images.device
    assert len(transformers) == len(images), f"Currently not supporting batching transformers in pck evaluation. Got {len(transformers)} transformers and {len(images)} images"
    assert len(mirror_permutation) == kps.shape[-2], f"mirror_permutation should be of length {kps.shape[-2]}, but got {len(mirror_permutation)}"

    old_image_size = transformers[0].img_size
    new_image_size = images.shape[-2:]
    for tf in transformers:
        tf.set_image_size(new_image_size)

    for i in indices:
        im_idxA, im_idxB = pairs[i].tolist()
        imgA, imgB = images[im_idxA].to(device), images[im_idxB].to(device)
        transformerA, transformerB = transformers[im_idxA], transformers[im_idxB]
        
        pair_idxA, pair_idxB = orig_pairs[i].tolist()
        gt_kpsA, gt_kpsB = kps[pair_idxA].to(device), kps[pair_idxB].to(device)

        # Permute the keypoints if needed
        if mirror:
            gt_kpsA = gt_kpsA[mirror_permutation] if transformerA.mirror[0] else gt_kpsA
            gt_kpsB = gt_kpsB[mirror_permutation] if transformerB.mirror[0] else gt_kpsB 

        if gt_kpsA.size(-1) == 3: 
            visible_kps = gt_kpsA[..., 2] * gt_kpsB[..., 2]                             # [AllKP]
            gt_kpsA, gt_kpsB = gt_kpsA[:, :2], gt_kpsB[:, :2]                           # [AllKP, 2]
            gt_kpsA, gt_kpsB = gt_kpsA[visible_kps == 1], gt_kpsB[visible_kps == 1]     # [KP, 2]
        
        assert transformerA.img_size == imgA.shape[-2:], f"transformerA.img_size = {transformerA.img_size}, imgA.shape = {imgA.shape}"
        
        # Transfer the images
        imgA_warped = transformerA(imgA.unsqueeze(0)).squeeze(0)
        imgB_warped = transformerB(imgB.unsqueeze(0)).squeeze(0)
        est_imgA = transformerB(imgA_warped.unsqueeze(0), inverse=True).squeeze(0)
        est_imgB = transformerA(imgB_warped.unsqueeze(0), inverse=True).squeeze(0)
    
        # Transfer the points     
        gt_kpsA_warped = transformerA.warp_points(gt_kpsA.unsqueeze(0)).squeeze(0)
        gt_kpsB_warped = transformerB.warp_points(gt_kpsB.unsqueeze(0)).squeeze(0)
        est_kpsA = transformerA.warp_points(gt_kpsB_warped.unsqueeze(0), inverse=True).squeeze(0)
        est_kpsB = transformerB.warp_points(gt_kpsA_warped.unsqueeze(0), inverse=True).squeeze(0)


        outputs.append(dotdict({
            'idxA': im_idxA, # int
            'idxB': im_idxB, # int
            'pair_idxA': pair_idxA, # int
            'pair_idxB': pair_idxB, # int
            'imgA': imgA, # (3, H, W)
            'imgB': imgB, # (3, H, W)
            'imgA_warped': imgA_warped, # (3, H, W)
            'imgB_warped': imgB_warped, # (3, H, W)
            'est_imgA': est_imgA, # (3, H, W)
            'est_imgB': est_imgB, # (3, H, W)
            'gt_pointsA': gt_kpsA, # (KP, 2)
            'gt_pointsB': gt_kpsB, # (KP, 2)
            'gt_pointsA_warped': gt_kpsA_warped, # (KP, 2)
            'gt_pointsB_warped': gt_kpsB_warped, # (KP, 2)
            'est_pointsA_warped': gt_kpsB_warped, # (KP, 2) 
            'est_pointsB_warped': gt_kpsA_warped, # (KP, 2)
            'est_pointsA': est_kpsA, # (KP, 2)
            'est_pointsB': est_kpsB, # (KP, 2)
        }))


    for tf in transformers:
        tf.set_image_size(old_image_size)
        
    return outputs


def calculate_dist_perc(outputs, thresholds=None, inverse_ops=None):
    dist_perc_per_pair = []     # list for each pair, containing a 2 binary list of correct points

    for output in outputs:
        gt_kpsA, gt_kpsB = output.gt_pointsA, output.gt_pointsB
        est_kpsA, est_kpsB = output.est_pointsA, output.est_pointsB
        pair_idxA, pair_idxB = output.pair_idxA, output.pair_idxB
        
        if thresholds is not None and inverse_ops is not None:  # SPair specify thresholds (the size of the object in the image)
            thresholdA, thresholdB = thresholds[pair_idxA], thresholds[pair_idxB]
            scaleA, scaleB = inverse_ops[pair_idxA, 2], inverse_ops[pair_idxB, 2]
            imgA_thresh, imgB_thresh = thresholdA * scaleA, thresholdB * scaleB
        else:
            imgA_thresh, imgB_thresh = max(output.imgA.shape[-2:]), max(output.imgB.shape[-2:])
            
        dist_perc_per_pair.append(dotdict({
            'dist_perc_A': torch.norm(gt_kpsA - est_kpsA, dim=-1) / imgA_thresh,
            'dist_perc_B': torch.norm(gt_kpsB - est_kpsB, dim=-1) / imgB_thresh,
        }))

    return dist_perc_per_pair


# permutation_index = None

def get_pck(alpha_list: List[float], images, data_folder, transformers: List[Transformer], max_pairs_to_return=10, mirror=False):
    """Return the pck score (as dict) and the outputs (as list of dotdicts). \n
    The number of pairs is O(N^2) where N is the number of images in the dataset, so we limit the number of pairs to return. \n
    The list of outputs will contain dict for the first max_pairs_to_return pairs and None for the rest. 
    """

    pairs, orig_pairs, kps, thresholds, inverse_ops, mirror_permutation = get_pairs(data_folder)

    if 'cub' in data_folder:
        transfer_bidirectional = True
        assert thresholds is None
        assert inverse_ops is None
    elif 'spair' in data_folder or 'willow' in data_folder:
        transfer_bidirectional = False
        assert thresholds is not None
        assert inverse_ops is not None
    else:
        raise NotImplementedError(f"dataset type {data_folder} not supported")
    

    # for static type checking
    assert type(pairs) == torch.Tensor        # [num_pairs, 2]
    assert type(orig_pairs) == torch.Tensor   # [num_pairs, 2]
    assert type(kps) == torch.Tensor          # [num_orig_pairs, num_keypoints_in_image, 3] 
    if 'spair' in data_folder:
        assert type(thresholds) == torch.Tensor   # [num_pairs*2]
        assert type(inverse_ops) == torch.Tensor  # [num_pairs*2, 3]
    
    num_pairs = len(pairs)

    batch_size = max_pairs_to_return    
    num_batches = num_pairs // batch_size + (num_pairs % batch_size > 0)
    first_batch_outputs = None

    num_of_thresholds_to_check = len(alpha_list) 
    total_correct = [0] * num_of_thresholds_to_check
    total_key_points_seen = [0] * num_of_thresholds_to_check
    total_errors = [0] * num_of_thresholds_to_check

    for batch_num in range(num_batches):
        indices = range(batch_num * batch_size, min((batch_num + 1) * batch_size, num_pairs))
        outputs = extract_and_transform_pairs(images, transformers, indices, pairs, orig_pairs, kps, mirror_permutation, mirror)

        # calculate dist perc for each pair
        dist_perc_per_pair_dict = calculate_dist_perc(outputs, thresholds, inverse_ops)
        
        # calculate correct points for each alpha
        for i, alpha in enumerate(alpha_list):
            if transfer_bidirectional:
                total_correct[i] += sum((pair.dist_perc_A < alpha).sum() + (pair.dist_perc_B < alpha).sum() for pair in dist_perc_per_pair_dict)
                total_key_points_seen[i] += sum(pair.dist_perc_A.shape[0] + pair.dist_perc_B.shape[0] for pair in dist_perc_per_pair_dict)
                total_errors[i] += sum(pair.dist_perc_A.sum() + pair.dist_perc_B.sum() for pair in dist_perc_per_pair_dict)
            else:
                total_correct[i] += sum((pair.dist_perc_B < alpha).sum() for pair in dist_perc_per_pair_dict)
                total_key_points_seen[i] += sum(pair.dist_perc_B.shape[0] for pair in dist_perc_per_pair_dict)
                total_errors[i] += sum(pair.dist_perc_B.sum() for pair in dist_perc_per_pair_dict)
                
        # if it's the first batch, return the outputs
        if batch_num == 0:
            first_batch_outputs = (outputs, dist_perc_per_pair_dict)
            
    # prepare the final pck score as a dict for each alpha
    pck_score_res = {
        alpha: {
            "correct": total_correct[i],
            "total": total_key_points_seen[i],
            "score": total_correct[i] / total_key_points_seen[i] if total_key_points_seen[i] != 0 else -1,
        } 
        for i, alpha in enumerate(alpha_list)
    }
        
    assert first_batch_outputs is not None, f"first_batch_outputs should not be None when num_batches > 0, num_batches = {num_batches}"
    return pck_score_res, first_batch_outputs

