import torch
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from typing import List, Optional

import mmcv  # type: ignore

from cose.preprocessing import recode_extra_classes
from cose.datasets import CoseDatasetCityscapes
from cose.conformal import pred_from_mmseg_input, aps_multimask, lac_multimask


@torch.no_grad()
def plot_varisco_map(
    idx,
    dataset: CoseDatasetCityscapes,
    mmseg_model,
    alpha_nominal,
    thre,
    mode="LAC",
    plot_classes=False,
    figsize=(12, 6),
    plot_ticks=True,
    colormap="turbo",
    binclasses_cmap="Purples",
):
    img = dataset[idx]["data_samples"]
    pred = pred_from_mmseg_input(
        img.img_path, mmseg_model, softmaxes=True, segmask=True
    )
    sm = pred["softmax"][0]

    implemented_prediction_sets = ["APS", "LAC"]
    if not mode in implemented_prediction_sets:
        print(f"{mode} not implemented. Implemented: {implemented_prediction_sets}")
        raise NotImplementedError

    if mode == "APS":
        one_hot_attribution = aps_multimask(thre, sm, dataset.n_classes)

    if mode == "LAC":
        one_hot_attribution = lac_multimask(thre, sm, dataset.n_classes)

    print(f" --- Notion of error: {mode}")

    plt.figure(figsize=figsize)

    if plot_classes:
        for i, (lab, name) in enumerate(
            [(0, "road"), (11, "person"), (12, "rider"), (13, "car")]
        ):
            plt_coordinate = 320 + i + 1
            plt.subplot(plt_coordinate)
            plt.title(f"Class: {name}")
            plt.imshow(
                one_hot_attribution[lab].cpu().numpy(), cmap=binclasses_cmap
            )  # "Greys")

            selclass = pred["segmask"].cpu().numpy()
            selclass = np.where(selclass == lab, 1, 0)
            plt.imshow(selclass, cmap=binclasses_cmap, alpha=0.5)
            plt.xticks([])
            plt.yticks([])

        plt.title(f"Binary masks for classes: 'road', 'person', 'rider', 'car'")
        plt.show()

    multimask = one_hot_attribution.sum(dim=0)

    activations = multimask.sum() / (
        one_hot_attribution.shape[1] * one_hot_attribution.shape[2]
    )

    plt.imshow(multimask.cpu().numpy(), cmap=colormap, vmin=1, vmax=dataset.n_classes)
    if not plot_ticks:
        plt.xticks([])
        plt.yticks([])

    tt = f" Varisco ({mode}) heatmap. Risk level = {alpha_nominal:.2f}, opt. lambda = {thre:.5f} -- activations (w/ voids): {activations*100:.2f}%"
    # if mode == "lac":
    #     tt = f"Threshold: smax(x_ij) >= 1 - λ = {1-thre:.3f}. Pixels activated: {activations*100:.1f}% "
    plt.title(tt)
    plt.colorbar()
    plt.show()

    del img, pred, sm
    torch.cuda.empty_cache()


from typing import Sequence


@torch.no_grad()
def plot_example_cityscapes(
    dataset: CoseDatasetCityscapes,
    img_idx,
    n_classes: int,  # =19,
    extraclasses: Optional[Sequence[int]] = [255],
    only_pred_mask=False,
    figsize=(14, 6),
):
    example_img = dataset[img_idx]["data_samples"]

    img = mmcv.imread(example_img.img_path)

    plt.figure(figsize=figsize)

    segmask_title = f"Cityscapes: a semantic segmentation mask overlayed on the image."

    if only_pred_mask:
        _mask = mmcv.imread(example_img.seg_map_path)

        _gt_mask = torch.Tensor(_mask)
        if extraclasses is not None:
            _recoded_gt_mask = recode_extra_classes(
                _gt_mask[:, :, 0], list(range(n_classes)), extraclasses
            )
        else:
            _recoded_gt_mask = _gt_mask

        plt.title(segmask_title)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.imshow(_recoded_gt_mask, alpha=0.95)  # , cmap=my_cmap)
        plt.show()

    else:
        plt.subplot(121)
        plt.imshow(mmcv.bgr2rgb(img))

        plt.subplot(122)
        _mask = mmcv.imread(example_img.seg_map_path)

        _gt_mask = torch.Tensor(_mask)
        if extraclasses is not None:
            _recoded_gt_mask = recode_extra_classes(
                _gt_mask[:, :, 0], list(range(n_classes)), extraclasses
            )
        else:
            _recoded_gt_mask = _gt_mask

        plt.title(segmask_title)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.imshow(_recoded_gt_mask, alpha=0.5)  # , cmap=my_cmap)
        plt.show()

    del img, _mask, _gt_mask
    torch.cuda.empty_cache()


def triplot_img_gt_pred(
    image_jpg,
    gt_mask_np,
    prediction_mask_np,
    # nlabs,
    palette: Optional[List],
    cmap="turbo",
    figsize=(15, 8),
):
    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(image_jpg)
    plt.xlabel("input image")

    plt.subplot(1, 3, 2)

    from matplotlib.colors import LinearSegmentedColormap  # type: ignore

    if palette:
        normalized_palette = [np.array(col) / 255 for col in palette]
        cmap_name = "stanford_codes"
        cmap = LinearSegmentedColormap.from_list(
            cmap_name, normalized_palette, N=len(palette)
        )

    plt.imshow(gt_mask_np, cmap=cmap)
    plt.xlabel("ground-truth mask")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction_mask_np, cmap=cmap)
    plt.xlabel("predicted mask")
    plt.show()
