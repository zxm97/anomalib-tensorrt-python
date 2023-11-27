#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Defines a `load_data` function that returns a generator yielding
feed_dicts so that this script can be used as the argument for
the --data-loader-script command-line parameter.
"""
import os
import random
import numpy as np
import cv2

from typing import Any, cast
import albumentations as A
from omegaconf import DictConfig, OmegaConf

calib_num_images = 100

dataset_path = r"D:/surface_defect_datasets/mvtec_anomaly_detection/transistor/train/good"
# metadata_path = "data/metadata_transistor_dfkde.json"
metadata_path = "data/metadata_transistor_efficient_ad.json"

imgs = os.listdir(dataset_path)
random.shuffle(imgs)

def load_metadata(path):
    """Loads the meta data from the given path.

    Args:
        path (str | Path | dict | None, optional): Path to JSON file containing the metadata.
            If no path is provided, it returns an empty dict. Defaults to None.

    Returns:
        dict | DictConfig: Dictionary containing the metadata.
    """
    # metadata: dict[str, float | np.ndarray | Tensor] | DictConfig = {}
    print("Reading metadata from file {}...".format(path))
    metadata = DictConfig = {}
    if path is not None:
        config = OmegaConf.load(path)
        metadata = cast(DictConfig, config)

    print('metadata: ', metadata)
    return metadata

metadata = load_metadata(metadata_path)

def pre_process(image: np.ndarray) -> np.ndarray:
    """Pre process the input image by applying transformations.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: pre-processed image.
    """

    transform = A.from_dict(metadata["transform"])
    processed_image = transform(image=image)["image"]

    if len(processed_image.shape) == 3:
        processed_image = np.expand_dims(processed_image, axis=0)

    if processed_image.shape[-1] == 3:
        processed_image = processed_image.transpose(0, 3, 1, 2)

    return processed_image

def load_data():
    for img_ind, img_name in enumerate(imgs):
        if img_ind > (calib_num_images - 1):
            break
        # img = cv2.imdecode(np.fromfile(os.path.join(dataset_path, img_name), dtype=np.uint8), 1) # H, W, C
        img = cv2.imread(os.path.join(dataset_path, img_name))
        img = pre_process(img)
        print(img_ind+1, 'of', calib_num_images)
        yield {"input": img}  # Still totally real data

# if __name__ == "__main__":
#     loader = load_data()
#     print(next(loader))
#     print(next(loader))
#     print(next(loader))



