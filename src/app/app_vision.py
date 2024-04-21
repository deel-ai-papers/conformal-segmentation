import os
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from collections import namedtuple
import pandas as pd
import json
import matplotlib as mpl
from PIL import Image


from cose.conformal import aps_multimask, lac_multimask, PredictionHandler

from app.tools import parse_arguments, setup_gpu, setup_mmseg_inference
from app.tools import heatmap_from_multimaks, segmask_from_softmax

import gradio as gr
from cose.utils import load_project_paths


Config = namedtuple(
    "Config", ["dataset", "loss", "prediction_set", "alpha", "lbd", "mincov"]
)

configs = []

# source: old experiments
configs.append(
    Config(
        dataset="Cityscapes",
        loss="binary_loss",
        prediction_set="APS",
        alpha=0.01,
        lbd=0.9956249999999993,
        mincov=0.9,
    )
)

# source: experiments/outputs/Cityscapes/miscoverage_loss/20240302_22h15m07s_Cityscapes__id_101__alpha_0.005__miscoverage_loss.json
configs.append(
    Config(
        dataset="Cityscapes",
        loss="miscoverage_loss",
        prediction_set="LAC",
        alpha=0.005,
        lbd=0.99751,
        mincov=None,
    )
)
# source: experiments/outputs/Cityscapes/miscoverage_loss/20240302_21h57m00s_Cityscapes__id_101__alpha_0.01__miscoverage_loss.json
configs.append(
    Config(
        dataset="Cityscapes",
        loss="miscoverage_loss",
        prediction_set="LAC",
        alpha=0.01,
        lbd=0.96741,
        mincov=None,
    )
)


cose_path = load_project_paths().COSE_PATH.strip()
dir = f"{cose_path}/experiments/outputs/ADE20K/miscoverage_loss"
json_files = [f"{dir}/{file}" for file in os.listdir(dir) if file.endswith(".json")]

RANDOM_SEED = 101
select = ["alpha", "mincov", "dataset", "optimal_lambda", "loss_function"]

for js in json_files:
    with open(js, "r") as j:
        cfg = json.load(j)
        if (
            cfg["experiment_id"] == RANDOM_SEED
            and cfg["early_stopped"] == False
            and cfg["optimal_lambda"] < 1
        ):
            cf = Config(
                dataset=cfg["dataset"],
                loss=cfg["loss_function"],
                prediction_set="LAC",
                alpha=cfg["alpha"],
                lbd=cfg["optimal_lambda"],
                mincov=cfg["mincov"],
            )

            configs.append(cf)


configs_df = pd.DataFrame(configs)
print(f" === {configs_df.head = }")


configs_list = configs_df[
    ["prediction_set", "alpha", "lbd", "mincov", "loss", "dataset"]
].values.tolist()

configs_str = [
    f"Dataset: {cf[5]} ({cf[4][:-5]}), alpha = {cf[1]}"
    for i, cf in enumerate(configs_list)
]


def heatmap_from_input_img_path(
    input_img_path: str,
    user_cfg,
    normalize_by_total_number_of_classes,
):
    torch.cuda.empty_cache()

    for i, cstr in enumerate(configs_str):
        if user_cfg == cstr:
            _config = configs[i]
            mode = _config.prediction_set
            threshold = _config.lbd
            usecase = _config.dataset

    dataset, model, _ = setup_mmseg_inference(
        torch_device=device,
        use_case=usecase,
    )

    with torch.no_grad():
        predictor = PredictionHandler(model["model"].mmseg_model)

        softmax_prediction = predictor.predict(input_img_path)
        segmask = segmask_from_softmax(softmax_prediction).cpu().numpy()

        if mode == "APS":
            multimask = aps_multimask(
                threshold=threshold,
                predicted_softmax=softmax_prediction,
                n_labels=dataset.n_classes,
            )
        elif mode == "LAC":
            multimask = lac_multimask(
                threshold=threshold,
                predicted_softmax=softmax_prediction,
                n_labels=dataset.n_classes,
            )
        else:
            raise ValueError(f"mode must be [LAC] or [APS], not {mode}")

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


def setup_app_vision(list_of_input_paths):
    input_image = gr.Image(label="Input image")
    out_mask = gr.Image(label="Segmentation mask", show_label=True)
    out_heatmap = gr.Image(label="Uncertainty heatmap", show_label=True)
    output_images = [out_mask, out_heatmap]

    examples = [[ex] for ex in list_of_input_paths[:10]]

    title = (
        f"Uncertainty Quantification (UQ) in semantic segmentation via VARISCO heatmaps"
    )
    description = f"""
        - Author: Luca Mossina, IRT Saint Exupéry. DEEL project, www.deel.ai

        Build a statistically valid heatmap for a confidence level chosen by the user. Ex: (1 - α) = 99%.
        
        Guarantee: E(Y_gt in Heatmap(X)) < α.
        
        """

    chosen_config = gr.Radio(
        choices=configs_str,
        label=f"Choose existing conformalized configuration",
        value=configs_str[0],
    )

    normalize_tot_num_classes = gr.Checkbox(
        label="Normalize heatmap by total number of classes (useful if few classes in ground truth)",
        value=False,
    )

    app = gr.Interface(
        live=True,
        title=title,
        description=description,
        fn=heatmap_from_input_img_path,
        inputs=[input_image, chosen_config, normalize_tot_num_classes],
        outputs=output_images,
        examples=examples,
        analytics_enabled=True,
        allow_flagging="never",
    )

    return app


if __name__ == "__main__":
    args = parse_arguments()
    device = setup_gpu(args.gpu)
    cose_path = load_project_paths().COSE_PATH

    _, _, input_paths_1 = setup_mmseg_inference(device, "Cityscapes")
    _, _, input_paths_2 = setup_mmseg_inference(device, "ADE20K")

    input_paths = []
    input_paths.extend([pt for pt in input_paths_1[:6]])
    input_paths.extend(input_paths_2[:6])

    app = setup_app_vision(list_of_input_paths=input_paths)

    try:
        app.launch(
            share=args.share_url,
            debug=True,
            auth=("user", "confiance"),
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        app.close()
    except Exception as e:
        print(f"Exception: {e}")
        app.close()
