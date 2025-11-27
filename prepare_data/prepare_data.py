# Code taken and modified from "SpaceJAM": https://github.com/BGU-CS-VIL/SpaceJAM
# "SpaceJAM" code taken and modified from "GANgealing": https://github.com/wpeebles/gangealing

import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.io import loadmat
from torchvision.datasets.utils import (download_and_extract_archive,
                                        download_file_from_google_drive,
                                        extract_archive, download_url)
from tqdm import tqdm

import sys
sys.path.append('.')
from utilities.CUB_data_utils import acsm_crop, perturb_bbox, square_bbox

# When an image is mirrored, any key points with left/right distinction need to be swapped.
# These are the permutations of key point indices that accomplishes this:
CUB_PERMUTATION = [0, 1, 2, 3, 4, 5, 10, 11, 12, 9, 6, 7, 8, 13, 14]
SPAIR_PERMUTATIONS = {
    'aeroplane': [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 22, 23, 24],
    'bicycle': [0, 1, 3, 2, 4, 5, 7, 6, 8, 10, 9, 11],
    'bird': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 13, 12, 15, 14, 16],
    'boat': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 13],
    'bottle': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8],
    'bus': [1, 0, 3, 2, 4, 6, 5, 7, 18, 19, 20, 21, 22, 23, 15, 14, 17, 16, 8, 9, 10, 11, 12, 13, 25, 24, 27, 26],
    'car': [1, 0, 3, 2, 4, 5, 7, 6, 8, 9, 20, 21, 22, 23, 24, 25, 17, 16, 19, 18, 10, 11, 12, 13, 14, 15, 27, 26, 29, 28],
    'cat': [1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13, 14],
    'chair': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12],
    'cow': [1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13, 14, 16, 15, 18, 17, 20, 19],
    'dog': [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 12, 11, 13, 14, 15],
    'horse': [1, 0, 3, 2, 5, 4, 7, 6, 8, 9, 11, 10, 13, 12, 14, 15, 17, 16, 19, 18],
    'motorbike': [1, 0, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'person': [1, 0, 3, 2, 4, 5, 6, 7, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18],
    'pottedplant': [2, 1, 0, 3, 5, 4, 8, 7, 6],
    'sheep': [1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13, 14, 16, 15, 18, 17, 20, 19],
    'train': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
    'tvmonitor': [2, 1, 0, 7, 6, 5, 4, 3, 10, 9, 8, 15, 14, 13, 12, 11]
}



NUMBER_OF_CUB_SUBSETS = 14  # number of CUB subsets used for numeric evaluation (see paper for details)


def download_cub_metadata(to_path):
    # Downloads some metadata so we can use image pre-processing consistent with ACSM for CUB
    # and get the lists of subset images for constructing the same 14 subsets from the dataset as shown in the paper
    cub_metadata_folder = f'{to_path}/cub_metadata'
    if not os.path.isdir(cub_metadata_folder):
        print('Downloading metadata used to form ACSM\'s CUB validation set, together with CUB subsets filenames')
        cub_metadata_file_id = "1Upa-5mjMqHZGTHuDEk7mZCCMUs7To232"
        download_file_from_google_drive(cub_metadata_file_id, to_path)
        zip_path = f'{cub_metadata_folder}.zip'
        shutil.move(f'{to_path}/{cub_metadata_file_id}', zip_path)
        extract_archive(zip_path, remove_finished=True)
    else:
        print(f'Found pre-existing CUB metadata folder at {cub_metadata_folder}')
    return cub_metadata_folder

def download_pf_willow(to_path):
    # Downloads the PF-Willow dataset
    willow_dir = f'{to_path}/willow'
    if not os.path.isdir(willow_dir):
        print(f'Downloading PF-Willow dataset to {to_path}')
        willow_url = "http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip"
        download_and_extract_archive(willow_url, willow_dir, remove_finished=True)

        # Remove __MACOSX folder if present using os
        macosx_path = os.path.join(willow_dir, '__MACOSX')
        if os.path.isdir(macosx_path):
            for root, dirs, files in os.walk(macosx_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(macosx_path)

        # Download the CSV file containing the test pairs
        csv_url = "http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_pf.csv"
        print(f'Downloading test pairs CSV to {willow_dir}')
        download_url(csv_url, willow_dir, filename='test_pairs_pf.csv')
    else:
        print('Found pre-existing PF-Willow directory')
    return willow_dir

def download_cub(to_path):
    # Downloads the CUB-200-2011 dataset
    cub_dir = f'{to_path}/CUB_200_2011'
    if not os.path.isdir(cub_dir):
        print(f'Downloading CUB_200_2011 to {to_path}')
        cub_url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
        download_and_extract_archive(cub_url, cub_dir, remove_finished=True)
    else:
        print('Found pre-existing CUB directory')
    return f'{cub_dir}/CUB_200_2011'


def download_spair(to_path):
    # Downloads and extracts the SPair-71K dataset
    spair_dir = f'{to_path}/SPair-71k'
    if not os.path.isdir(spair_dir):
        print(f'Downloading SPair-71k to {to_path}')
        spair_url = 'http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz'
        download_and_extract_archive(spair_url, to_path, remove_finished=True)
    else:
        print('Found pre-existing SPair-71K directory')
    return spair_dir


def border_pad(img, target_res, resize=True, to_pil=True):
    original_width, original_height = img.size
    if original_height <= original_width:
        if resize:
            img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        half_height = (target_res - height) / 2
        int_half_height = int(half_height)
        lh = int_half_height
        rh = int_half_height + (half_height > int_half_height)
        img = np.pad(img, pad_width=[(lh, rh), (0, 0), (0, 0)])
    else:
        if resize:
            img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        half_width = (target_res - width) / 2
        int_half_width = int(half_width)
        lw = int_half_width
        rw = int_half_width + (half_width > int_half_width)
        img = np.pad(img, pad_width=[(0, 0), (lw, rw), (0, 0)])
    if to_pil:
        img = Image.fromarray(img)
    return img


def pad(img, target_res, bgval=0):
    original_width, original_height = img.size
    if original_height <= original_width:
        img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        half_height = (target_res - height) / 2
        int_half_height = int(half_height)
        lh = int_half_height
        rh = int_half_height + (half_height > int_half_height)
        img = np.pad(img, pad_width=[(lh, rh), (0, 0), (0, 0)], constant_values=bgval)
    else:
        img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        half_width = (target_res - width) / 2
        int_half_width = int(half_width)
        lw = int_half_width
        rw = int_half_width + (half_width > int_half_width)
        img = np.pad(img, pad_width=[(0, 0), (lw, rw), (0, 0)], constant_values=bgval)
    
    img = Image.fromarray(img)
    return img 



def center_crop(img, target_res):
    # From official StyleGAN2 create_lsun method:
    img = np.asarray(img)
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    img = Image.fromarray(img, 'RGB')
    img = img.resize((target_res, target_res), Image.LANCZOS)
    return img


def cub_crop(img, target_res, bbox, border=True):
    # This function mimics ACSM's pre-processing used for the CUB dataset (up to image resampling and padding color)
    img = np.asarray(img)
    img = acsm_crop(img, bbox, 0, border=border)
    return Image.fromarray(img).resize((target_res, target_res), Image.LANCZOS)


def resize_and_convert(img, size, method, bbox=None):
    if method == 'border':
        img = border_pad(img, size)
    elif method == 'center':
        img = center_crop(img, size)
    elif method == 'pad':
        img = pad(img, size)
    elif method == 'none':
        pass
    elif method == 'stretch':
        img = img.resize((size, size), Image.LANCZOS)
    elif method == 'cub_crop':
        img = cub_crop(img, size, bbox, border=False)
    elif method == 'cub_crop_border':
        img = cub_crop(img, size, bbox, border=True)
    else:
        raise NotImplementedError

    return img


def process_images_and_save(out_path, files, method, size, bboxes, image_format):
    output_path_images = f'{out_path}/images'
    Path(output_path_images).mkdir(parents=True, exist_ok=True)
    num_files = len(files)
    print(f'Found {num_files} files for {out_path}')
    print(f'Example file being loaded: {files[0]}')
    files = [(i, file, bbox) for i, (file, bbox) in enumerate(zip(files, bboxes))]
    print("Saving images...")
    for img_file in tqdm(files):
        i, file, bbox = img_file
        img = Image.open(file).convert('RGB')
        out_img = resize_and_convert(img, size, method, bbox)
        # save image
        image_filename = f'{output_path_images}/img_{i:03d}.{image_format}'
        out_img.save(image_filename)
        if i == 0:
            print(f'Example file being saved: {image_filename}')


def load_image_folder_and_process(path, method, size, out_path, image_format):
    files = sorted(list(glob(f"{path}/*.jpeg")) + list(glob(f"{path}/*.jpg")) + list(glob(f"{path}/*.png")))
    bboxes = [None] * len(files)  # This means no bounding boxes are used
    process_images_and_save(out_path, files, method, size, bboxes, image_format)

def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3]  # zero-out any non-visible key points 
    return kps, offset_x, offset_y, scale


def preprocess_kps_resize(kps, img_width, img_height, size):
    # Once an image has been pre-processed via resizing (stretched),
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the resized image.
    kps = kps.clone()
    scale_x = size / img_width
    scale_y = size / img_height
    kps[:, [0, 1]] *= torch.tensor([scale_x, scale_y])
    kps *= kps[:, 2:3]  # zero-out any non-visible key points
    return kps, scale_x, scale_y, min(scale_x, scale_y)
    
    

def preprocess_kps_box_crop(kps, bbox, size):
    # Once an image has been pre-processed via a box crop,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the cropped image.
    kps = kps.clone()
    kps[:, 0] -= bbox[0] + 1
    kps[:, 1] -= bbox[1] + 1
    w = 1 + bbox[2] - bbox[0]
    h = 1 + bbox[3] - bbox[1]
    assert w == h
    kps[:, [0, 1]] *= size / float(w)
    return kps


def load_CUB_keypoints(path):
    names = ['img_index', 'kp_index', 'x', 'y', 'visible']
    landmarks = pd.read_table(path, header=None, names=names, delim_whitespace=True, engine='python')
    landmarks = landmarks.to_numpy().reshape((11788, 15, 5))[..., [2, 3, 4]]  # (num_images, num_kps, 3)
    landmarks = torch.from_numpy(landmarks).float()
    return landmarks


def extract_cub_subset_acsm_data(files, filenames, kps, b, out_path, size, custom_set_size=None, curr_images_list_file=None):
    if curr_images_list_file is not None:
        with open(curr_images_list_file, "r") as fd:
            subset_filenames = fd.read().splitlines()

        if custom_set_size is not None:
            subset_filenames = subset_filenames[:custom_set_size]
    else:
        subset_filenames = filenames
        if custom_set_size is not None:
            if custom_set_size > len(filenames):
                custom_set_size = len(filenames)
            subset_filenames = random.sample(filenames, custom_set_size)

    output_path_pck = f'{out_path}/pck'
    Path(output_path_pck).mkdir(parents=True, exist_ok=True)
    subset_files = []
    bboxes = []
    kps_out = []
    for file in subset_filenames:
        curr_file_index = filenames.index(file)  # find index of current file in filenames

        curr_file = files[curr_file_index]
        curr_kps = kps[curr_file_index]
        curr_b = b[curr_file_index]

        x1, y1, x2, y2 = curr_b[0, 0]
        bbox = np.array([x1[0, 0], y1[0, 0], x2[0, 0], y2[0, 0]]) - 1
        bbox = perturb_bbox(bbox, 0.05, 0)
        bbox = square_bbox(bbox)
        bboxes.append(bbox)
        kps_out.append(preprocess_kps_box_crop(curr_kps, bbox, size))
        subset_files.append(curr_file)
    bboxes = np.stack(bboxes)
    kps_out = torch.stack(kps_out)
    torch.save(kps_out, f'{output_path_pck}/keypoints.pt')
    # When an image is mirrored horizontally, the designation between key points with a left versus right distinction
    # needs to be swapped. This is the permutation of CUB key points which accomplishes this swap:
    torch.save(CUB_PERMUTATION, f'{output_path_pck}/permutation.pt')  
    assert bboxes.shape[0] == len(subset_filenames)
    return subset_files, bboxes


def load_acsm_data_and_process(path, metadata_path='data/cub_metadata', method='cub_crop', size=256, out_path=None, image_format='png', custom_set_size=None):
    from scipy.io import loadmat
    mat_path = f'{metadata_path}/acsm_val_cub_cleaned.mat'
    mat = loadmat(mat_path)
    files = [f'{path}/images/{file[0]}' for file in mat['images']['rel_path'][0]]
    # These are the indices retained by ACSM (others are filtered):
    indices = [i[0, 0] - 1 for i in mat['images']['id'][0]]
    kps = load_CUB_keypoints(f'{path}/parts/part_locs.txt')[indices]
    b = mat['images']['bbox'][0]

    filenames = [f'CUB_200_2011/images/{file[0]}' for file in mat['images']['rel_path'][0]]
    # upload subsets' original paths and process each subset's data
    for set_idx in tqdm(range(NUMBER_OF_CUB_SUBSETS)):
        curr_out_path = f'{out_path}/cub_subset_{set_idx}' + (f'_setsize_{custom_set_size}' if custom_set_size else '')
        curr_images_list_file = f'{metadata_path}/cub_subset_{set_idx}_original_paths.txt'
        subset_files, bboxes = extract_cub_subset_acsm_data(files, filenames, kps, b, curr_out_path, size, custom_set_size=custom_set_size, curr_images_list_file=curr_images_list_file)
        process_images_and_save(curr_out_path, subset_files, method, size, bboxes, image_format)
        visualize_images(curr_out_path, size)


def load_acsm_data_and_process_class(path, method='cub_crop', size=256, out_path=None, image_format='png', custom_set_size=None, acsm_class_id_val=1):
    # Create a dictionary to map integers to names
    classes_name_dict = {}
    with open(f'{path}/classes.txt','r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into ID and image name using space as a delimiter
            parts = line.strip().split(' ')
            if len(parts) == 2:
                class_id = int(parts[0])
                class_name = parts[1]
                classes_name_dict[class_id] = class_name
    class_name = '001.Black_footed_Albatross'
    if acsm_class_id_val in classes_name_dict:
        class_name = classes_name_dict[acsm_class_id_val]
    
    image_id_mapping = {}
    image_path_mapping = {}
    all_images_paths = []
    with open(f'{path}/images.txt','r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into ID and image name using space as a delimiter
            parts = line.strip().split(' ')
            if len(parts) == 2:
                image_id = int(parts[0])
                image_name = parts[1]
                image_id_mapping[image_name] = image_id
                image_path_mapping[image_id] = image_name
                all_images_paths.append(image_name)
    
    id_traintest_mapping = {}
    with open(f'{path}/train_test_split.txt','r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into ID and image name using space as a delimiter
            parts = line.strip().split(' ')
            if len(parts) == 2:
                image_id = int(parts[0])
                image_train_flag = int(parts[1])
                id_traintest_mapping[image_id] = bool(image_train_flag)
    bbox_mapping = {}
    with open(f'{path}/bounding_boxes.txt','r') as file:
        # Read each line in the file
        for line in file:
            # Split the line into ID and image name using space as a delimiter
            parts = line.strip().split(' ')
            if len(parts) >= 5:
                image_id = int(parts[0])
                box = [float(x) for x in parts[1:5]]
                bbox_mapping[image_id] = [box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1]
    
    
    for set_name in ['train', 'test']:
        indices = [image_id_mapping[image_path]-1 for image_path in all_images_paths if class_name in image_path and id_traintest_mapping[image_id_mapping[image_path]]==(set_name=='train')]
        filenames = [f'CUB_200_2011/images/{image_path_mapping[index+1]}' for index in indices]
        # These are the indices retained by ACSM (others are filtered):
        kps = load_CUB_keypoints(f'{path}/parts/part_locs.txt')[indices]
        
        # Define the data type for the structured array
        dtype_bbox = np.dtype([("x1", "O"), ("y1", "O"), ("x2", "O"), ("y2", "O")])
        structured_arrays = []
        for key, values in bbox_mapping.items():
            if key-1 in indices:
                structured_array = np.array([(np.array([np.array([values[0]], dtype=np.int32)]),
                                            np.array([np.array([values[1]], dtype=np.int32)]),
                                            np.array([np.array([values[2]], dtype=np.int32)]),
                                            np.array([np.array([values[3]], dtype=np.int32)]))], dtype=dtype_bbox)
                structured_arrays.append(np.array([structured_array]))

        b = np.vstack(np.array([structured_arrays]))  # Stack the structured arrays vertically

        # upload subsets' original paths and process each subset's data
        curr_out_path = f'{out_path}/{set_name}' + (f'_setsize_{custom_set_size}' if custom_set_size else '')
        files = [f'data/CUB_200_2011/{file}' for file in filenames]
        subset_files, bboxes = extract_cub_subset_acsm_data(files, filenames, kps, b, curr_out_path, size, custom_set_size=custom_set_size)
        process_images_and_save(curr_out_path, subset_files, method, size, bboxes, image_format)
        visualize_images(curr_out_path, size)

def load_pf_willow_pck_data_from_test_pairs(path, category, size, out_path):
    csv_path = os.path.join(path, 'test_pairs_pf.csv')
    output_path_pck = os.path.join(out_path, 'pck')
    Path(output_path_pck).mkdir(parents=True, exist_ok=True)

    def simplified(cat): return cat.split('(')[0].lower()
    def match_category(x): return simplified(x.split('/')[1]) == category

    df = pd.read_csv(csv_path)
    df = df[df['imageA'].apply(match_category) & df['imageB'].apply(match_category)]

    files_set = set(df['imageA']).union(set(df['imageB']))
    files_list = sorted(list(files_set))
    image_index_map = {f: i for i, f in enumerate(files_list)}

    keypoints = []
    inverse_coords = []
    thresholds = []
    missing_files = []

    print(f"[INFO] Processing {len(files_list)} unique images for category '{category}'")

    for file_rel in files_list:
        file_abs = os.path.join(path, file_rel)
        mat_path = file_abs.replace('.png', '.mat')

        if not os.path.isfile(mat_path):
            print(f"[WARN] Missing .mat file: {mat_path}")
            missing_files.append(file_rel)
            continue

        mat = loadmat(mat_path)
        if 'pts_coord' not in mat or mat['pts_coord'].shape != (2, 10):
            print(f"[WARN] Invalid keypoints in {mat_path}")
            missing_files.append(file_rel)
            continue

        pts = mat['pts_coord']
        kps = np.zeros((10, 3), dtype=np.float32)
        kps[:, 0] = pts[0]
        kps[:, 1] = pts[1]
        kps[:, 2] = 1

        img = Image.open(file_abs).convert('RGB')
        w, h = img.size
        kps_tensor = torch.from_numpy(kps)
        kps_tensor, _, _, scale = preprocess_kps_pad(kps_tensor, w, h, size)

        keypoints.append(kps_tensor)
        inverse_coords.append([0, 0, scale])
        thresholds.append(max(w, h))

    files_list = [f for f in files_list if f not in missing_files]
    kps_img = torch.stack(keypoints)
    torch.save(kps_img, f'{output_path_pck}/keypoints_per_image.pt')

    valid_pairs = []
    for _, row in df.iterrows():
        a = row['imageA']
        b = row['imageB']
        if a in image_index_map and b in image_index_map:
            valid_pairs.append([image_index_map[a], image_index_map[b]])

    pairs_tensor = torch.tensor(valid_pairs)
    torch.save(pairs_tensor, f'{output_path_pck}/pairs.pt')
    torch.save(pairs_tensor, f'{output_path_pck}/pairs_indices_in_set.pt')

    kps = torch.stack([kps_img[a] for a, b in valid_pairs] + [kps_img[b] for a, b in valid_pairs])
    torch.save(kps, f'{output_path_pck}/keypoints.pt')

    thresh_tensor = torch.tensor(thresholds, dtype=torch.float32)
    inv_tensor = torch.tensor(inverse_coords, dtype=torch.float32)

    thresholds_expanded = torch.stack([thresh_tensor[a] for a, b in valid_pairs] + [thresh_tensor[b] for a, b in valid_pairs])
    inverse_expanded = torch.stack([inv_tensor[a] for a, b in valid_pairs] + [inv_tensor[b] for a, b in valid_pairs])

    torch.save(thresholds_expanded, f'{output_path_pck}/pck_thresholds.pt')
    torch.save(inverse_expanded, f'{output_path_pck}/inverse_coordinates.pt')
    torch.save(list(range(10)), f'{output_path_pck}/permutation.pt')

    return [os.path.join(path, f) for f in files_list], [None] * len(files_list)


def load_spair_pck_data(path, category, split, size, out_path):
    output_path_pck = f'{out_path}/pck'
    Path(output_path_pck).mkdir(parents=True, exist_ok=True)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*{category}.json'))

    files_no_repetition = []  # images' names in set, with no repetitions
    pairs_set_indices = []  # original index in set with no repetitions
    thresholds = []
    inverse = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
        image_kps = json
    print(f'Number of SPair key points for {category} <= {num_kps}')
    kps = []            # [num_pairs, num_kps, 3]
    kps_per_image = []  # [num_imgs, num_kps, 3]
    blank_kps = torch.zeros(num_kps, 3)
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        
        # get image set with no repetitions
        source_img_new = source_fn not in files_no_repetition
        target_img_new = target_fn not in files_no_repetition
        
        if source_img_new:
            files_no_repetition.append(source_fn)
        if target_img_new:
            files_no_repetition.append(target_fn)
        pairs_set_indices.append(files_no_repetition.index(source_fn))
        pairs_set_indices.append(files_no_repetition.index(target_fn))

        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])
        # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
        # they are computed as well:
        thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]))

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        kp_ixs = torch.tensor([int(id) for id in data["kps_ids"]]).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        source_kps, _, _, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, _, _, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)

        kps.append(source_kps)
        kps.append(target_kps)
        inverse.append([0, 0, src_scale])
        inverse.append([0, 0, trg_scale])
    kps = torch.stack(kps)  # [num_pairs, num_kps, 3]

    
    for file in files_no_repetition:
        imname_json = file.split('/')[-1].replace('.jpg', '.json')
        im_annotations = f'{path}/ImageAnnotation/{category}/{imname_json}'
        with open(im_annotations) as f:
            im_kps = torch.zeros(num_kps, 3, dtype=torch.float)
            im_json = json.load(f)
            im_kps_dict = im_json['kps']
            w, h = im_json['image_width'], im_json['image_height']
            for i in range(num_kps):
                kp_val = im_kps_dict[str(i)]
                if kp_val is not None:
                    im_kps[i, :] = torch.tensor([kp_val[0], kp_val[1], 1])
                    
            im_kps, _, _, _ = preprocess_kps_pad(im_kps, w, h, size)
            kps_per_image.append(im_kps)
            
    kps_per_image = torch.stack(kps_per_image)  # [num_imgs, num_kps, 3]
    
    used_kps, = torch.where(kps_per_image[:, :, 2].any(dim=0))
    print(f'used keypoints: {used_kps}')
    kps = kps[:, used_kps, :]
    kps_per_image = kps_per_image[:, used_kps, :]
    
    print(f'Final number of used key points: {kps.size(1)}')
    num_imgs_in_pairs = len(thresholds)  # Total number of images (= 2 * number of pairs)
    torch.save(torch.arange(num_imgs_in_pairs).view(num_imgs_in_pairs // 2, 2), f'{output_path_pck}/pairs.pt')
    torch.save(torch.tensor(pairs_set_indices).view(num_imgs_in_pairs // 2, 2), f'{output_path_pck}/pairs_indices_in_set.pt')
    torch.save(torch.tensor(thresholds, dtype=torch.float), f'{output_path_pck}/pck_thresholds.pt')
    torch.save(torch.tensor(inverse), f'{output_path_pck}/inverse_coordinates.pt')
    torch.save(kps, f'{output_path_pck}/keypoints.pt') # [num_pairs, num_kps, 3]
    torch.save(kps_per_image, f'{output_path_pck}/keypoints_per_image.pt') # [num_imgs, num_kps, 3]

    assert category in SPAIR_PERMUTATIONS, f'No permutation for {category} found.'
    # When an image is mirrored horizontally, the designation between key points with a left versus right distinction
    # needs to be swapped. This is the permutation of SPair key points which accomplishes this swap:
    print('SPAIR_PERMUTATIONS[category]:', SPAIR_PERMUTATIONS[category])
    torch.save(SPAIR_PERMUTATIONS[category], f'{output_path_pck}/permutation.pt')

    return files_no_repetition, [None] * len(files_no_repetition)  # No bounding boxes are used


def load_pf_willow_data_and_process(path, method, size, willow_sets_path, image_format='png'):
    # Only the 4 PF-Willow categories
    categories = ['car', 'duck', 'motorbike', 'winebottle']

    for category in categories:
        print(f'Processing PF-Willow category {category}')

        out_path = f'{willow_sets_path}/{category}'
        Path(out_path).mkdir(parents=True, exist_ok=True)

        files_no_repetition, bboxes = load_pf_willow_pck_data_from_test_pairs(path, category, size, out_path)
        process_images_and_save(out_path, files_no_repetition, method, size, bboxes, image_format)
        visualize_images(out_path, size)

def load_spair_data_and_process(path, method, size, spair_sets_path, image_format='png'):
    for category in [
        'aeroplane', 
        'bicycle', 
        'bird', 
        'boat', 
        'bottle', 
        'bus', 
        'car', 
        'cat', 
        'chair', 
        'cow', 
        'dog', 
        'horse', 
        'motorbike', 
        'person', 
        'pottedplant', 
        'sheep', 
        'train', 
        'tvmonitor'
        ]:
        print(f'Processing SPair category {category}')
        
        Path(f'{spair_sets_path}/spair_{category}').mkdir(parents=True, exist_ok=True)
        for split in ['trn', 'val', 'test']:
            split_str = 'train' if split == 'trn' else split
            out_path = f'{spair_sets_path}/spair_{category}/{split_str}'
            Path(out_path).mkdir(parents=True, exist_ok=True)
            
            files_no_repetition, bboxes = load_spair_pck_data(path, category, split, size, out_path)
            process_images_and_save(out_path, files_no_repetition, method, size, bboxes, image_format)
            visualize_images(out_path, size)
    
    

def visualize_images(set_path, size):
    images_path = f'{set_path}/images'
    pck_path = f'{set_path}/pck'
    vis_path = f'{set_path}/vis_images'
    
    if not os.path.isdir(images_path):
        print(f'Images folder not found at {images_path}. Skipping visualization.')
        return
    if not os.path.isdir(pck_path):
        print(f'PCK folder not found at {pck_path}. Skipping visualization.')
        return

    Path(vis_path).mkdir(parents=True, exist_ok=True)
    images = sorted(list(glob(f"{images_path}/*.png")) + list(glob(f"{images_path}/*.jpg")))
    
    if os.path.isfile(f'{pck_path}/keypoints_per_image.pt'):     # A bit of a (ugly) hack to support both SPair and CUB that have different keypoint format
        keypoints = torch.load(f'{pck_path}/keypoints_per_image.pt', weights_only=True)
    else:
        keypoints = torch.load(f'{pck_path}/keypoints.pt', weights_only=True)  # (num_images, num_kps, 3)
    
    print('keypoints.shape:', keypoints.shape)
    num_of_vis_kps_per_image = []
    for i, image in enumerate(images):
        num_of_vis_kps_per_image.append((keypoints[i][:,2]>0).sum())
        img = Image.open(image).convert('RGB')
        img = np.asarray(img)
        img = Image.fromarray(img)
        img = img.resize((size*2,size*2), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        for j, kp in enumerate(keypoints[i]):
            if kp[2] > 0:
                kp *= 2
                draw.ellipse((kp[0] - 2, kp[1] - 2, kp[0] + 2, kp[1] + 2), fill='red')
                draw.text((kp[0] + 2, kp[1] + 2), str(int(j)), fill='white', stroke_fill='black', stroke_width=3)
        img.save(f'{vis_path}/img_{i:03d}.png')
            
    print('The top 5 images with the most visible keypoints are:')
    print("\n".join(map(lambda x: f"{vis_path}/img_{x:03d}.png", np.argsort(num_of_vis_kps_per_image)[-5:])))
