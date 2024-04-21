# import os
from typing import Tuple, Optional, Literal
from PIL import Image  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from mmengine.config import Config  # type: ignore
from mmseg.datasets import CityscapesDataset  # type: ignore
from mmseg.datasets import ADE20KDataset, LoveDADataset

from cose.utils import load_project_paths
from cose.models import get_available_models_for_dataset
from cose._typing import DatasetNames


_MODELS_CITYSCAPES = get_available_models_for_dataset("Cityscapes")
_MODELS_ADE20K = get_available_models_for_dataset("ADE20K")
_MODELS_LOVEDA = get_available_models_for_dataset("LoveDA")

# # TODO: use the following to get extra classes from dataset descriptors
# # Assumes the user runs checks and preprocessing after downloading repo, data and models
# COSE_PATH = load_project_paths().COSE_PATH
# PATH_TO_DATASET_DESCRIPTORS = f"{COSE_PATH}/datasets/dataset_descriptors.json"

# === WARNING:
# === __extra labels__ in ground truth ("void", missing data, missing label for a pixel)
# === must be excluded from calibration and the computation of metrics:
# === these are equivalent to "undefined": handle with care.
#
# === [self.extralabels] were IGNORED DURING TRAINING (by the model's designer).
# === They must be ignored during calibration, since the model CAN ONLY
# === predict an actual class (in 0..N_classes-1), and these "extra pixels"
# === will always be incorrectly predicted (they get no softmax value)
#
EXTRA_CLASSES_NUMBERS = {
    "Cityscapes": [255],
    "ADE20K": [255],
    "LoveDA": [255],
}


def get_available_datasets():
    return DATASETS


class CoseDatasetCityscapes(CityscapesDataset):
    ## This class is old: most of the methods are useless now.

    def __init__(
        self,
        model_config_name: str = "pspnet",  # TODO: just to be consistent with other classes, to be impl
        data_partition: str = "val",
        name: DatasetNames = "Cityscapes",
    ):
        self.extralabels = EXTRA_CLASSES_NUMBERS[name]
        self.name = name
        paths = load_project_paths(self.name)

        config_path = f"{paths.MODEL_CONFIG_PATH}"  # CONFIG_PATH

        try:
            model_config_path = (
                f"{config_path}/{_MODELS_CITYSCAPES[model_config_name]['config']}"
            )
        except Exception as e:
            raise e

        config = Config.fromfile(model_config_path)

        super().__init__(
            data_root=paths.RAW_DATA_PATH,
            data_prefix=config.val_dataloader.dataset["data_prefix"],
            pipeline=config.test_pipeline,
            seg_map_suffix="_gtFine_labelTrainIds.png",
        )
        self.task = data_partition

        ## setup colors-to-classes relations for visualization
        self.classes = self.METAINFO["classes"]
        self.n_classes = len(self.classes)
        self.palette = self.METAINFO["palette"]
        self.set_color_relations()

    def set_color_relations(self):
        self.color_relations = {
            i: lab for i, lab in enumerate(zip(self.classes, self.palette))
        }

    def get_path(self, idx: int) -> Tuple[str, str]:
        data_sample = self[idx]["data_samples"]
        return data_sample.img_path, data_sample.seg_map_path

    def read_input_image(self, idx: int):
        data_sample = self[idx]["data_samples"]
        img = Image.open(data_sample.img_path)
        return img

    def plot_input_image(self, idx: int):
        data_sample = self[idx]["data_samples"]
        img = Image.open(data_sample.img_path)
        fig, ax = plt.subplots()
        ax.imshow(np.asarray(img))

    def plot_ground_truth(self, idx: int):
        data_sample = self[idx]["data_samples"]
        gt_mask = Image.open(data_sample.seg_map_path)
        gt_mask = np.array(gt_mask)
        fig, ax = plt.subplots()
        ax.imshow(gt_mask)


class CoseDatasetADE20K(ADE20KDataset):

    def __init__(
        self,
        model_config_name: Literal["deeplabv3", "segformer"] = "segformer",
        data_partition: str = "val",
        name: DatasetNames = "ADE20K",
    ):
        self.extralabels = EXTRA_CLASSES_NUMBERS[name]
        self.name = name

        ## TEMP: for the moment, only inferences on validation data are supported
        if data_partition != "val":
            raise ValueError(f"Invalid data_partition: {data_partition}")

        paths = load_project_paths(self.name)
        config_path = f"{paths.MODEL_CONFIG_PATH}"  # CONFIG_PATH

        model_config_path = (
            f"{config_path}/{_MODELS_ADE20K[model_config_name]['config']}"
        )

        config = Config.fromfile(model_config_path)

        # WARNING: by default, ADE20K and LoveDA in mmseg have reduce_zero_label = True
        # THIS BREAKS THINGS: the 0-label (background) is set to 255, and the others
        # are decremented by one.
        # From the doc: https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/datasets.html
        #
        # > To ignore label 0 (such as ADE20K dataset), we can use reduce_zero_label (default to False)
        # > argument of BaseSegDataset and its subclasses. When reduce_zero_label is True, label 0 in
        # > segmentation annotations would be set as 255 (models of MMSegmentation would ignore label 255
        # >  in calculating loss) and indices of other labels will minus 1:
        #
        # > gt_semantic_seg[gt_semantic_seg == 0] = 255
        # > gt_semantic_seg = gt_semantic_seg - 1
        # > gt_semantic_seg[gt_semantic_seg == 254] = 255
        #
        # HACK: change config default before passing to pipeline, to keep label 0 as is.

        super().__init__(
            data_root=paths.RAW_DATA_PATH,
            data_prefix=config.val_dataloader.dataset["data_prefix"],
            pipeline=config.test_pipeline,
        )

        self.classes = self.METAINFO["classes"]
        self.n_classes = len(self.classes)
        self.palette = self.METAINFO["palette"]


class CoseDatasetLoveDA(LoveDADataset):

    def __init__(
        self,
        model_config_name: str = "pspnet",
        data_partition: str = "val",
        name: DatasetNames = "LoveDA",
    ):
        self.extralabels = EXTRA_CLASSES_NUMBERS[name]
        self.name = name

        ## TEMP: for the moment, only inferences on validation data are supported
        if data_partition != "val":
            raise ValueError(f"Invalid data_partition: {data_partition}")

        paths = load_project_paths(self.name)
        config_path = f"{paths.MODEL_CONFIG_PATH}"

        model_config_path = (
            f"{config_path}/{_MODELS_LOVEDA[model_config_name]['config']}"
        )

        config = Config.fromfile(model_config_path)

        super().__init__(
            data_root=paths.RAW_DATA_PATH,
            data_prefix=config.val_dataloader.dataset["data_prefix"],
            pipeline=config.test_pipeline,
        )

        self.classes = self.METAINFO["classes"]
        self.n_classes = len(self.classes)
        self.palette = self.METAINFO["palette"]


DATASETS = {
    "Cityscapes": CoseDatasetCityscapes,
    "ADE20K": CoseDatasetADE20K,
    "LoveDA": CoseDatasetLoveDA,
}
