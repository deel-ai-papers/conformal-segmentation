from typing import Sequence, Union
import torch

from mmseg.apis.inference import _preprare_data  # type: ignore
from mmseg.apis import init_model  # type: ignore

from cose.utils import load_project_paths


__DATASETS = ["Cityscapes", "ADE20K", "LoveDA"]


def get_available_models_for_dataset(
    dataset_name: str,
):

    if dataset_name == "Cityscapes":
        return {
            "pspnet": {
                "config": "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py",
                "checkpoint": "pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth",
            }
        }
    elif dataset_name == "ADE20K":
        return {
            "deeplabv3": {
                "config": "deeplabv3_r50-d8_4xb4-80k_ade20k-512x512.py",
                "checkpoint": "deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth",
            },
            "segformer": {
                "config": "segformer_mit-b5_8xb2-160k_ade20k-512x512.py",
                "checkpoint": "segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth",
            },
        }
    elif dataset_name == "LoveDA":
        return {
            "pspnet": {
                "config": "pspnet_r50-d8_4xb4-80k_loveda-512x512.py",
                "checkpoint": "pspnet_r50-d8_512x512_80k_loveda_20211104_155728-88610f9f.pth",
            }
        }
    else:
        raise ValueError(f"Invalid {dataset_name = }: must be in {__DATASETS}")


class CoseModel:
    def __init__(self, device):
        self.device = device

    def __call__(self, imgs):  # -> Sequence[Image]:
        return self.predict(imgs)

    def init_model(self, model_cfg_path, checkpoint_path, device):
        self.mmseg_model = init_model(model_cfg_path, checkpoint_path, device=device)

    def prepare_data(self, imgs):
        """Source: mmseg.apis.inference._preprare_data"""
        data, is_batch = _preprare_data(imgs, self.mmseg_model)

        return data, is_batch

    @torch.no_grad()
    def predict_softmax(self, imgs: Union[Sequence[str], str]):
        if not isinstance(imgs, str):
            # TODO (maybe): loop inputs, do many one-batch predictions.
            raise NotImplementedError(
                f"input [img] is not implemented. You must pass str"
            )

        data, is_batch = self.prepare_data(imgs)
        pred = self.mmseg_model.test_step(data)
        pred = pred[0]  ## batch size is ALWAYS 1, for mmsegmentation library
        pred_softmaxes = torch.nn.functional.softmax(pred.seg_logits.data, dim=0)

        torch.cuda.empty_cache()
        return pred_softmaxes

    @torch.no_grad()
    def predict(self, imgs: Union[Sequence[str], str]):
        data, is_batch = self.prepare_data(imgs)
        output = self.mmseg_model.test_step(data)
        return output

    @torch.no_grad()
    def _test_step(self, data):
        output = self.mmseg_model.test_step(data)
        return output


class CityscapesPSPNet(CoseModel):
    def __init__(self, device):
        super().__init__(device)

        _paths = load_project_paths(dataset_name="Cityscapes")
        model_cfg_path = _paths.MODEL_CONFIG_PATH

        mod_py = get_available_models_for_dataset("Cityscapes")["pspnet"]["config"]
        mod_ckpt = get_available_models_for_dataset("Cityscapes")["pspnet"][
            "checkpoint"
        ]

        model_name_py = f"{model_cfg_path}/{mod_py}"
        checkpoint_path = f"{model_cfg_path}/{mod_ckpt}"

        # self.init_model returns self.mmseg_model
        self.init_model(model_name_py, checkpoint_path, device=device)


class ADE20kSegformer(CoseModel):
    def __init__(self, device):
        super().__init__(device)

        dat_nam = "ADE20K"
        _paths = load_project_paths(dataset_name=dat_nam)
        model_cfg_path = _paths.MODEL_CONFIG_PATH

        mod_py = get_available_models_for_dataset(dat_nam)["segformer"]["config"]
        mod_ckpt = get_available_models_for_dataset(dat_nam)["segformer"]["checkpoint"]

        model_name_py = f"{model_cfg_path}/{mod_py}"
        checkpoint_path = f"{model_cfg_path}/{mod_ckpt}"

        self.init_model(model_name_py, checkpoint_path, device=device)


class ADE20kDeepLabV3(CoseModel):
    def __init__(self, device):
        super().__init__(device)

        dat_nam = "ADE20K"
        _paths = load_project_paths(dataset_name=dat_nam)
        model_cfg_path = _paths.MODEL_CONFIG_PATH

        mod_py = get_available_models_for_dataset(dat_nam)["deeplabv3"]["config"]
        mod_ckpt = get_available_models_for_dataset(dat_nam)["deeplabv3"]["checkpoint"]

        model_name_py = f"{model_cfg_path}/{mod_py}"
        checkpoint_path = f"{model_cfg_path}/{mod_ckpt}"
        self.init_model(model_name_py, checkpoint_path, device=device)


class LoveDAPSPNet(CoseModel):
    def __init__(self, device):
        super().__init__(device)

        dat_nam = "LoveDA"
        _paths = load_project_paths(dataset_name=dat_nam)
        model_cfg_path = _paths.MODEL_CONFIG_PATH

        mod_py = get_available_models_for_dataset(dat_nam)["pspnet"]["config"]
        mod_ckpt = get_available_models_for_dataset(dat_nam)["pspnet"]["checkpoint"]

        model_name_py = f"{model_cfg_path}/{mod_py}"
        checkpoint_path = f"{model_cfg_path}/{mod_ckpt}"
        self.init_model(model_name_py, checkpoint_path, device=device)


def get_default_model_for_dataset(
    dataset_name: str, device: Union[str, torch.device]
) -> CoseModel:
    if dataset_name == "Cityscapes":
        return CityscapesPSPNet(device)
    elif dataset_name == "ADE20K":
        return ADE20kSegformer(device)
    elif dataset_name == "LoveDA":
        return LoveDAPSPNet(device)
    else:
        raise ValueError(f"Invalid {dataset_name = }: must be in {__DATASETS}")
