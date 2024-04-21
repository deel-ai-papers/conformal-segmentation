import os
import numpy as np
import torch
from collections import namedtuple
from dotenv import load_dotenv
from typing import get_args, Optional
import matplotlib.pyplot as plt

from mmengine.runner import Runner

from cose._typing import DatasetNames

_IMPLEMENTED_DATASET = tuple(datype for datype in get_args(DatasetNames))


ProjectPaths = namedtuple(
    "ProjectPaths",
    [
        "COSE_PATH",
        "RAW_DATA_PATH",
        "MODEL_CONFIG_PATH",
    ],
)


def load_project_paths(dataset_name: Optional[DatasetNames] = None) -> ProjectPaths:
    if dataset_name is None:
        load_dotenv()
        COSE_PATH = os.getenv("COSE_PATH")
        RAW_DATA_PATH = None
        MODEL_CONFIG_PATH = None

        config_paths = ProjectPaths(COSE_PATH, None, None)
        return config_paths

    if dataset_name not in _IMPLEMENTED_DATASET:
        raise ValueError(
            f"INVALID {dataset_name = }: must be in {_IMPLEMENTED_DATASET}"
        )

    load_dotenv()
    COSE_PATH = os.getenv("COSE_PATH")

    if dataset_name == "Cityscapes":
        COSE_DATASET = os.getenv("COSE_DATA_CITYSCAPES")
    elif dataset_name == "ADE20K":
        COSE_DATASET = os.getenv("COSE_DATA_ADE20K")
    elif dataset_name == "LoveDA":
        COSE_DATASET = os.getenv("COSE_DATA_LOVEDA")

    RAW_DATA_PATH = f"{COSE_DATASET}"
    MODEL_CONFIG_PATH = f"{COSE_PATH}/models/{dataset_name}"

    config_paths = ProjectPaths(
        COSE_PATH,
        RAW_DATA_PATH,
        MODEL_CONFIG_PATH,
    )

    return config_paths


def run_mmseg_training(data_root, ann_dir, mmengine_config_file, do_data_split=False):
    ### split train/val/calib set randomly

    if do_data_split:
        split_data(data_root_dir=data_root, annotations_dir=ann_dir)

    runner = Runner.from_cfg(mmengine_config_file)
    runner.train()


def plot_ground_truth_with_mask(image, ground_truth_mask):
    plt.imshow(image)
    plt.imshow(ground_truth_mask, alpha=0.5)
    plt.show()


def gt_from_code(
    code, path_to_labels_, torch_device_str: Optional[str]
) -> torch.Tensor:
    np_gt = np.loadtxt(f"{path_to_labels_}/{code}.regions.txt").astype(int)

    tens = torch.from_numpy(np_gt)

    if torch_device_str:
        try:
            tens = tens.to(torch.device(torch_device_str))
            return tens
        except:
            raise ValueError(f"CUDA Device: could not allocate to: {torch_device_str}")
    else:
        return tens


def load_splits_codes(path_to_splits, verbose=False):
    try:
        split_train = f"{path_to_splits}/train.txt"
        split_calib = f"{path_to_splits}/calib.txt"
        split_test = f"{path_to_splits}/val.txt"

        train_codes = open(split_train, "r").read().split("\n")
        train_codes = [s for s in train_codes if s]
        calib_codes = open(split_calib, "r").read().split("\n")
        calib_codes = [s for s in calib_codes if s]
        test_codes = open(split_test, "r").read().split("\n")
        test_codes = [s for s in test_codes if s]

        if verbose:
            print(calib_codes[-3:])
            print(train_codes[:3])
            print(test_codes[:3])

        return train_codes, calib_codes, test_codes
    except:
        raise ValueError(f"--- ERROR: could not load data splits")
