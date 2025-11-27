# FastJAM: a Fast Joint Alignment Model for Images
Omri Hirsch*, Ron Shapira Weber*, Shira Ifergane, and Oren Freifeld  

[![arXiv](https://img.shields.io/badge/arXiv-2510.22842-b31b1b.svg?style=flat)](https://arxiv.org/abs/2510.22842)  
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://bgu-cs-vil.github.io/FastJAM/)  

![FastJAM teaser](website_FastJAM_teaser_video.gif)

## Requirements
To set up the environment for this project, you need to install the required dependencies listed in `environment.yml`. This file specifies the necessary packages and channels to ensure that your environment is properly configured.

 1. Install Conda: If you don't have Conda installed, you can get it by installing Miniconda or Anaconda. Download Miniconda or Download Anaconda.
 2. Create the Environment: To create a Conda environment with the dependencies specified in `environment.yml`, use the following command:
```bash
conda env create -f environment.yml
```
 3. Activate the Environment: Once the environment is created, activate it using:
```bash
conda activate fastjam
```

## Download data 
You can download and preprocess the following datasets according to the paper: 
```bash
python prepare_data/prepare_spair.py

python prepare_data/prepare_cub_class.py --cub_acsm_class <class_num>

python prepare_data/prepare_cub_subsets.py
```
Each prepare script automatically launches the matching Grounded-SAM helper under `third_party/Grounded-Segment-Anything/` to create segmentation masks (e.g., `grounded_sam_spair_split.py`). Ensure the GroundedDINO and SAM checkpoints referenced in those scripts are downloaded before running the prepare commands.

In order to run FastJAM on custom images set, use:
```bash
python prepare_data/prepare_image_set.py --path <image-dir> --out <out-dir> --object_class <name>
```
 Ensure `<image-dir>` only contains `.png` or `.jpg` files.

The custom flow automatically runs `third_party/Grounded-Segment-Anything/grounded_sam_custom_image_set.py` to create masks using the provided object class prompt (e.g., `dog`, `airplane`).

### How to run
To train the entire model on one of the dataset, simply run:
```bash
python train.py --data_folder <processed-images-dir>
```
**Note:** The `<processed-images-dir>` should be the same as `<out-dir>` from the `prepare_image_set.py` command above (or the output directory from the other prepare scripts).

The `data_folder` must contain the following subfolders:
- `images/` - directory containing the input images
- `masks/` - directory containing the corresponding segmentation masks

For CUB and SPair datasets, the folder will also contain additional subfolders such as `PCK/` and other dataset-specific directories.

For example, the following paths are valid `data_folder` values:  
```./data/spair_sets/spair_aeroplane/test```  
```./data/cub_subsets/cub_subset_0```  
```./data/cub_classes/cub_class_001/test```  
For more details, run:
```bash
python train.py --help
```

 The outputs of `train.py` include:
- Canonical visualizations warped into the reference space $(\mathcal{C})$.
- A JSON file containing homographies and auxiliary metadata.
- Visualizations warped into the ref image space via $\theta_i \cdot \theta_{\text{ref}}^{-1}$.

## Citation
If you use this work, please cite:
```
@inproceedings{Hirsch:NeurIPS:2025:FastJAM,
      title={{FastJAM}: a Fast Joint Alignment Model for Images},
      author={Hirsch, Omri and Weber, Ron Shapira and Ifergane, Shira and Freifeld, Oren},
      year={2025},
      journal={NeurIPS},
}
```

## Third-party code

This repository includes code from the following projects:

- [RoMa: Robust Dense Feature Matching](https://github.com/Parskatt/RoMa) (MIT License)
  - Located under `third_party/RoMa/`.
  - Original authors: Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, Michael Felsberg.  
  - If you use RoMa in academic work, please cite:

    Edstedt et al., "RoMa: Robust Dense Feature Matching", CVPR 2024.  

- [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) (Apache-2.0)
  - Located under `third_party/grounded_sam/`.
  - If you use Grounded SAM, please cite:

    Ren et al., "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks", 2024.
