import glob
import argparse
import torch
from typing import Literal, Optional
from mmengine.registry import init_default_scope  # type: ignore

from cose.models import CityscapesPSPNet, ADE20kSegformer, ADE20kDeepLabV3, LoveDAPSPNet  # type: ignore
from cose.datasets import CoseDatasetCityscapes, CoseDatasetADE20K, CoseDatasetLoveDA  # type: ignore
from cose.conformal import lac_multimask, aps_multimask, split_dataset_idxs  # type: ignore

import matplotlib as mpl
from typing import Literal, Union
from PIL.Image import Image as ImageType
from PIL import Image

from cose.conformal import (
    aps_multimask,
    lac_multimask,
    PredictionHandler,  # type: ignore
)


def setup_gpu(device_str):
    if torch.cuda.is_available():
        return torch.device(device_str)
    else:
        print(f" --- WARNING: CUDA is not available, torch device is CPU")
        return torch.device("cpu")


def parse_arguments(default_dataset="Cityscapes"):
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--gpu",
        type=str,
        help="Default: [cuda:0]. Specify [cuda:n] if you have multiple GPUs",
        default="cuda:0",
    )
    parser.add_argument(
        "--usecase",
        type=str,
        help="Default: [Cityscapes]. Specify [ADE20K] if you want to use ADE20K dataset",
        default=default_dataset,
    )
    parser.add_argument(
        "--share_url",
        action="store_true",
        help="Default: [False]. Creates a shareable url via gradio web services",
    )

    args = parser.parse_args()
    return args


def setup_inference_cityscapes(
    torch_device,
    n_calib: Optional[int] = None,
    random_seed: Optional[int] = None,
    pretrained_model="pspnet",
    # recover_calib_test_splits=False,
    recover_calib_test_splits=True,
    img_extension="png",
):
    dataset = CoseDatasetCityscapes(data_partition="val")

    if pretrained_model == "pspnet":
        model = CityscapesPSPNet(torch_device)
    else:
        raise ValueError(f"Invalid pretrained_model: {pretrained_model} unknown")

    if recover_calib_test_splits:
        if n_calib is None or random_seed is None:
            raise ValueError
        else:
            _n_calib = n_calib
            _, test_id = split_dataset_idxs(
                len_dataset=len(dataset), n_calib=_n_calib, random_seed=random_seed
            )

            val_dir_path = f"{dataset.data_prefix['img_path']}"

            png_files = glob.glob(val_dir_path + f"/*/*.{img_extension}")
            test_file_paths = [png_files[i] for i in test_id]
    else:
        val_dir_path = f"{dataset.data_prefix['img_path']}"
        png_files = glob.glob(val_dir_path + f"/*/*.{img_extension}")
        test_file_paths = png_files

    ## gradio does not take objects as input, hence we wrap model in dictionary
    model = {"model": model}
    return dataset, model, test_file_paths


def setup_inference_ADE20K(
    torch_device,
    n_calib: Optional[int] = None,
    random_seed: Optional[int] = None,
    pretrained_model: Literal["deeplabv3", "segformer"] = "segformer",
    recover_calib_test_splits=True,
    img_extension="jpg",
):
    ## load dataset
    dataset = CoseDatasetADE20K(
        model_config_name=pretrained_model, data_partition="val"
    )

    # Load pretrained model
    if pretrained_model == "deeplabv3":
        model = ADE20kDeepLabV3(torch_device)
    elif pretrained_model in ["segformer", "deeplabv3"]:
        model = ADE20kSegformer(torch_device)
    else:
        raise ValueError(f"Invalid pretrained_model: {pretrained_model}")

    if recover_calib_test_splits and n_calib is None:
        raise ValueError("n_calib must be specified")
    elif recover_calib_test_splits and n_calib is not None and random_seed is not None:
        _n_calib = n_calib
        _, test_id = split_dataset_idxs(
            len_dataset=len(dataset), n_calib=_n_calib, random_seed=random_seed
        )

        val_dir_path = f"{dataset.data_prefix['img_path']}"
        png_files = glob.glob(f"{val_dir_path}/*.{img_extension}")  #
        test_file_paths = [png_files[i] for i in test_id]
    else:
        val_dir_path = f"{dataset.data_prefix['img_path']}"
        png_files = glob.glob(f"{val_dir_path}/*.{img_extension}")  #
        test_file_paths = png_files

    ## gradio does not take objects as input, hence we wrap model in dictionary
    model = {"model": model}
    return dataset, model, test_file_paths


def setup_inference_LoveDA(
    torch_device,
    n_calib: Optional[int] = None,
    random_seed: Optional[int] = None,
    pretrained_model: Literal["pspnet"] = "pspnet",
    recover_calib_test_splits=True,
    img_extension="png",
):
    dataset = CoseDatasetLoveDA(
        model_config_name=pretrained_model, data_partition="val"
    )

    if pretrained_model == "pspnet":
        model = LoveDAPSPNet(torch_device)
    else:
        raise ValueError(f"Invalid pretrained_model: {pretrained_model}")

    if recover_calib_test_splits and n_calib is None:
        raise ValueError("n_calib must be specified")
    elif recover_calib_test_splits and n_calib is not None and random_seed is not None:
        _n_calib = n_calib
        _, test_id = split_dataset_idxs(
            len_dataset=len(dataset), n_calib=_n_calib, random_seed=random_seed
        )

        val_dir_path = f"{dataset.data_prefix['img_path']}"
        png_files = glob.glob(f"{val_dir_path}/*.{img_extension}")  #
        test_file_paths = [png_files[i] for i in test_id]
    else:
        val_dir_path = f"{dataset.data_prefix['img_path']}"
        png_files = glob.glob(f"{val_dir_path}/*.{img_extension}")  #
        test_file_paths = png_files

    ## gradio does not take objects as input, hence we wrap model in dictionary
    model = {"model": model}
    return dataset, model, test_file_paths


def setup_mmseg_inference(
    torch_device,
    use_case: Literal["Cityscapes", "ADE20K", "LoveDA"],
    random_seed: Optional[int] = None,
    n_calib: Optional[int] = None,
):
    ## setup mmseg working environment
    init_default_scope("mmseg")

    if n_calib is None:  # HACK: used for gradio app
        n_calib = 250
        # raise ValueError("n_calib must be specified")

    if use_case == "Cityscapes":
        dataset, model, test_file_paths = setup_inference_cityscapes(
            torch_device=torch_device, n_calib=n_calib, random_seed=random_seed
        )
    elif use_case == "ADE20K":
        dataset, model, test_file_paths = setup_inference_ADE20K(
            torch_device=torch_device, n_calib=n_calib, random_seed=random_seed
        )
    elif use_case == "LoveDA":
        dataset, model, test_file_paths = setup_inference_LoveDA(
            torch_device=torch_device, n_calib=n_calib, random_seed=random_seed
        )
    else:
        raise ValueError(f"Invalid use_case: {use_case}")

    return dataset, model, test_file_paths


def setup_mmseg_score():
    return init_default_scope("mmseg")


def heatmap_from_multimaks(cose_multimask: torch.Tensor):
    return cose_multimask.sum(dim=0)


def heatmap_from_softmax(
    mode: Literal["LAC", "APS"], softmax: torch.Tensor, threshold: float, n_labels: int
):
    if mode == "LAC":
        multimask = lac_multimask(
            threshold=threshold, predicted_softmax=softmax, n_labels=n_labels
        )
    elif mode == "APS":
        multimask = aps_multimask(
            threshold=threshold, predicted_softmax=softmax, n_labels=n_labels
        )
    else:
        raise ValueError(f"mode must be either 'lac' or 'aps', not {mode}")

    return heatmap_from_multimaks(multimask)


def segmask_from_softmax(softmax: torch.Tensor):
    return torch.argmax(softmax, dim=0)


def threshold_heatmap_from_input_img(
    input_img: Union[str, ImageType],
    mode: Literal["LAC", "APS"],
    threshold: float,
    usecase: Literal["Cityscapes", "ADE20K", "LoveDA"],
    normalize_by_total_number_of_classes: bool = False,
    random_seed=42,
):
    if threshold < 0 or threshold > 1:
        raise ValueError(f"threshold must be in [0, 1], not {threshold}")
    if mode not in ["LAC", "APS"]:
        raise ValueError(f"mode must be either 'LAC' or 'APS', not {mode}")

    args = parse_arguments()
    device = setup_gpu(args.gpu)
    dataset, model, _ = setup_mmseg_inference(
        torch_device=device, use_case=usecase, random_seed=random_seed
    )

    with torch.no_grad():
        predictor = PredictionHandler(model["model"].mmseg_model)
        softmax_prediction = predictor.predict(input_img)
        segmask = segmask_from_softmax(softmax_prediction)
        # print(segmask.unique())
        segmask = segmask.cpu().numpy()

        if mode == "LAC":
            multimask = lac_multimask(
                threshold=threshold,
                predicted_softmax=softmax_prediction,
                n_labels=dataset.n_classes,
            )
        elif mode == "APS":
            multimask = aps_multimask(
                threshold=threshold,
                predicted_softmax=softmax_prediction,
                n_labels=dataset.n_classes,
            )

        heatmap = heatmap_from_multimaks(multimask).cpu().numpy()

        if normalize_by_total_number_of_classes:
            heatmap = heatmap / dataset.n_classes
        else:
            max_labs_per_pxl = multimask.sum(dim=0).max().cpu().numpy()
            heatmap = heatmap / max_labs_per_pxl

        cmap = mpl.colormaps["tab20"]
        segmask = Image.fromarray(cmap(segmask / segmask.max(), bytes=True))  # * 255)
        cmap = mpl.colormaps["turbo"]
        heatmap = Image.fromarray(cmap(heatmap, bytes=True))

        return segmask, heatmap
