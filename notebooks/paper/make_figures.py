# import numpy as np
# import pandas as pd
# import json

import matplotlib.pyplot as plt
import torch
from typing import Literal, Sequence
from PIL import Image
import matplotlib as mpl

import mmcv

from app.tools import setup_mmseg_inference
from app.tools import setup_mmseg_inference
from app.tools import segmask_from_softmax
from cose.conformal import lac_multimask, PredictionHandler


def plot_heatmap_from_input_img_path(
    input_img_path: str,
    expe_config,
    normalize_by_total_number_of_classes,
    device="cuda:0",
):
    torch.cuda.empty_cache()

    threshold = expe_config["optimal_lambda"]

    dataset, model, _ = setup_mmseg_inference(
        torch_device=device,
        use_case=expe_config["dataset"],
        random_seed=expe_config["experiment_id"],
        n_calib=expe_config["n_calib"],
    )

    with torch.no_grad():
        predictor = PredictionHandler(model["model"].mmseg_model)

        softmax_prediction = predictor.predict(input_img_path)
        segmask = segmask_from_softmax(softmax_prediction).cpu().numpy()

        cmap = mpl.colormaps["tab20"]
        segmask = Image.fromarray(cmap(segmask / segmask.max(), bytes=True))  # * 255)

        figure_size = (15, 10)
        fig, axs = plt.subplots(2, 1, figsize=figure_size, dpi=200)
        # fig, ax = plt.subplot((1,2), figsize=figure_size, dpi=200)

        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title("Input image with segmentation mask")
        _input = mmcv.imread(input_img_path)
        ax1.imshow(mmcv.bgr2rgb(_input))
        ax1.imshow(segmask, alpha=0.5, cmap=cmap)
        # plt.show()

        multimask = lac_multimask(
            threshold=threshold,
            predicted_softmax=softmax_prediction,
            n_labels=dataset.n_classes,
        )

        if normalize_by_total_number_of_classes:
            vmax = dataset.n_classes
        else:
            vmax = multimask.sum(dim=0).max().cpu().numpy()

        cmap = mpl.colormaps["turbo"]
        _map = multimask.sum(dim=0).cpu().numpy()

        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("Uncertainty heatmap")
        im = ax2.imshow(_map, cmap=cmap, aspect="equal", vmax=vmax)

        fig.colorbar(im, ax=ax2, label="Number of classes per pixel", pad=0.05)
        plt.rcParams.update({"font.size": 10})
        # plt.legend()
        plt.show()


def plot_threshold_heatmap_from_input_img_path(
    input_img_path: str,
    expe_config,
    normalize_by_total_number_of_classes,
    # n_classes: int,
    lbd: Sequence[float],
    device="cuda:0",
):
    dataset, model, _ = setup_mmseg_inference(
        torch_device=device,
        use_case=expe_config["dataset"],
        random_seed=expe_config["experiment_id"],
        n_calib=expe_config["n_calib"],
    )

    with torch.no_grad():
        figure_size = (22, 10)
        fig, axs = plt.subplots(1, len(lbd), figsize=figure_size, dpi=300)

        for i, lb in enumerate(lbd):
            predictor = PredictionHandler(model["model"].mmseg_model)
            softmax_prediction = predictor.predict(input_img_path)

            multimask = lac_multimask(
                threshold=lb,
                predicted_softmax=softmax_prediction,
                n_labels=dataset.n_classes,
            )
            heatmap = multimask.sum(dim=0).cpu().numpy()
            if normalize_by_total_number_of_classes:
                vmax = dataset.n_classes
            else:
                vmax = heatmap.max().item()

            cmap = mpl.colormaps["turbo"]

            ax = axs[i]
            hm = ax.imshow(
                heatmap,
                cmap=cmap,
                vmax=vmax,
            )

            if i > 0:
                # plt.xticks([])
                ax.set_yticks([])

            ax.set_title(f"$\lambda = $ {lb}")

        # plt.colorbar(shrink=0.5, aspect="equal")
        plt.subplots_adjust(hspace=0)
        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.colorbar(
            hm, ax=axs, label="Number of classes per pixel", pad=0.05, shrink=0.25
        )
        plt.rcParams.update({"font.size": 10})

        plt.show()

    torch.cuda.empty_cache()


def triple_threshold_heatmap_from_input_img_path(
    input_img_path: str,
    expe_config,
    normalize_by_total_number_of_classes,
    # n_classes: int,
    lbd: Sequence[float],
    titles: Sequence[str],
    device="cuda:0",
):
    dataset, model, _ = setup_mmseg_inference(
        torch_device=device,
        use_case=expe_config["dataset"],
        random_seed=expe_config["experiment_id"],
        n_calib=expe_config["n_calib"],
    )

    with torch.no_grad():
        figure_size = (22, 10)
        fig, axs = plt.subplots(1, len(lbd), figsize=figure_size, dpi=300)

        for i, lb in enumerate(lbd):
            predictor = PredictionHandler(model["model"].mmseg_model)
            softmax_prediction = predictor.predict(input_img_path)

            multimask = lac_multimask(
                threshold=lb,
                predicted_softmax=softmax_prediction,
                n_labels=dataset.n_classes,
            )
            heatmap = multimask.sum(dim=0).cpu().numpy()
            if normalize_by_total_number_of_classes:
                vmax = dataset.n_classes
            else:
                vmax = heatmap.max().item()

            cmap = mpl.colormaps["turbo"]

            ax = axs[i]
            hm = ax.imshow(
                heatmap,
                cmap=cmap,
                vmax=vmax,
            )

            if i > 0:
                # plt.xticks([])
                ax.set_yticks([])

            title = titles[i]
            title = title + f"$\lambda = $ {lb:.6f}"
            ax.set_title(title)

        # plt.colorbar(shrink=0.5, aspect="equal")
        plt.subplots_adjust(hspace=0)
        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.colorbar(
            hm, ax=axs, label="Number of classes per pixel", pad=0.05, shrink=0.25
        )
        plt.rcParams.update({"font.size": 10})

        plt.show()

    torch.cuda.empty_cache()
