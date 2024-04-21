import random
import numpy as np
import torch
from tqdm import tqdm  # type: ignore
from typing import Literal, Optional, Sequence, Union, List, Any
from collections import namedtuple


from mmseg.apis.inference import _preprare_data  # type: ignore
from mmseg.datasets import BaseSegDataset  # type: ignore
from mmseg.apis.inference import ImageType

from cose.preprocessing import recode_extra_classes
from cose.models import CoseModel

CONFORMAL_SETS_REGISTER = ["lac", "aps"]


LossOutput = namedtuple(
    "LossOutput",
    ["losses", "activations_ratio", "coverage_ratio", "minimum_coverage_ratio"],
)


def check_risk_upper_bound(alpha_risk, B_loss_upper_bound, n_calibs) -> None:
    """[MATH] we need to check that we have enough calibration points.
    [min_calib]: min number of calibration points such that the risk_bound is above
                 the lower limit of the loss. E.g., if L in [0,1], we need
                 min_calib > 0
    """
    min_calib = np.floor((B_loss_upper_bound - alpha_risk) / alpha_risk).astype(int) + 1
    if n_calibs < min_calib:
        raise ValueError(
            f"ERROR: not enough calibration points. You have {n_calibs} but you need at least {min_calib}"
        )


def compute_risk_bound(risk_level, risk_upper_bound_B, num_calib_samples) -> float:
    risk_bound = risk_level - (risk_upper_bound_B - risk_level) / num_calib_samples

    check_risk_upper_bound(risk_level, risk_upper_bound_B, num_calib_samples)

    return risk_bound


def one_hot_encoding_of_2d_int_mask(mask: torch.Tensor, n_labels: int) -> torch.Tensor:
    mask_one_hot = torch.nn.functional.one_hot(mask, num_classes=n_labels)
    mask_one_hot = torch.movedim(mask_one_hot, 2, 0)  # .cuda()
    return mask_one_hot


def one_hot_encoding_of_gt(gt: torch.Tensor, n_labels: int) -> torch.Tensor:
    return one_hot_encoding_of_2d_int_mask(gt, n_labels)


@torch.no_grad()
def aps_multimask(
    threshold: float,
    predicted_softmax: torch.Tensor,
    n_labels: int,
    always_include_top1=True,
) -> torch.Tensor:
    """generate multimask inspired by APS conformal method, BUT ALWAYS INCLUDE
    TOP-1 CLASS. This breaks the tightness guarantees, but it is necessary to give
    reasonable masks, since in practice we do not want empty pixels.
    """
    ordered_softmax, indices = torch.topk(predicted_softmax, k=n_labels, dim=0)

    test_over_threshold = ordered_softmax.cumsum(0) > threshold
    up_to_first_over = test_over_threshold.cumsum(0)
    test_up_to_first = up_to_first_over <= 1

    if always_include_top1:
        test_up_to_first[0, :, :] = 1  # always include top-1 class

    reverse_indices = torch.argsort(indices, dim=0, descending=False)
    one_hot_attribution = test_up_to_first.gather(0, reverse_indices)

    del test_up_to_first, reverse_indices, ordered_softmax, indices
    torch.cuda.empty_cache()

    return one_hot_attribution.long()


@torch.no_grad()
def lac_multimask(
    threshold: float,
    predicted_softmax: torch.Tensor,
    n_labels: int,
    always_include_top1=True,
) -> torch.Tensor:
    cut_threshold = torch.tensor(
        1 - threshold
    )  # all classes above this value are included

    ordered_softmaxes, indices = torch.topk(predicted_softmax, k=n_labels, dim=0)
    test_over_threshold = ordered_softmaxes >= cut_threshold

    if always_include_top1:
        test_over_threshold[0, :, :] = 1  # always include top-1 class

    reverse_indices = torch.argsort(indices, dim=0, descending=False)
    one_hot_multimask = test_over_threshold.gather(0, reverse_indices)

    del ordered_softmaxes, reverse_indices, test_over_threshold, indices
    torch.cuda.empty_cache()

    return one_hot_multimask.long()


def load_loss(loss_str):
    if loss_str == "binary_loss":
        return binary_loss
    elif loss_str == "miscoverage_loss":
        return miscoverage_loss
    else:
        raise NotImplementedError(f"Loss {loss_str} not implemented")


class Conformalizer:

    def __init__(
        self,
        model: CoseModel,
        dataset: BaseSegDataset,
        random_seed: Optional[int],
        n_calib: int,
        device: torch.device = torch.device("cpu"),
        conformal_set: Literal["lac", "aps"] = "aps",
        loss_type: Literal["binary", "miscoverage"] = "binary",
    ):
        self.conformal_set = conformal_set
        self.model = model
        self.dataset = dataset
        self.device = device
        self.random_seed = random_seed  # if None, no shuffling is performed
        self.n_calib = n_calib

    def split_dataset_cal_test(self):  # , random_seed: int):
        self.calibration_indices, self.test_indices = split_dataset_idxs(
            len_dataset=len(self.dataset),
            n_calib=self.n_calib,
            random_seed=self.random_seed,
        )


def split_dataset_idxs(
    len_dataset: int,
    n_calib: int,
    random_seed: Optional[int] = None,
):
    if n_calib > len_dataset:
        raise ValueError(
            f"n_calib [{n_calib}] must be less than dataset size [{len_dataset}]"
        )
    idxs = [i for i in range(len_dataset)]

    if random_seed is not None:
        random.Random(random_seed).shuffle(idxs)

    cal_idx = idxs[:n_calib]
    test_idx = list(set(idxs).difference(cal_idx))

    assert (
        len(set(cal_idx).intersection(test_idx)) == 0
    ), "calibration and test sets are not disjoint"

    return cal_idx, test_idx


def pred_from_mmseg_input(
    mmseg_img_path: ImageType, mmseg_model, softmaxes: bool, segmask: bool
):
    ## > ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]
    data, is_batch = _preprare_data(mmseg_img_path, mmseg_model)
    pred = mmseg_model.test_step(data)
    pred = pred[0]
    pred_softmaxes = torch.nn.functional.softmax(pred.seg_logits.data, dim=0)

    torch.cuda.empty_cache()

    output = {}

    if softmaxes:
        output["softmax"] = (pred_softmaxes,)
    if segmask:
        output["segmask"] = pred.pred_sem_seg.data[0]

    return output


# HACK: n_wild is 1 because ALL THE DATASET WE USE HAVE ONLY ONE EXTRA LABEL (255)
# which is RECODED AS AN ADDITIONAL CHANNEL: [K+1, H, W]
def is_semantic_mask_in_multimask(
    one_hot_semantic_mask, one_hot_multimask, ignore_mask=None
) -> torch.Tensor:
    """[ignore_mask] is ONE whenever pixel was void, hence it must be ignored in the computation of the coverage ratio"""
    if ignore_mask is None:
        return torch.mul(one_hot_semantic_mask, one_hot_multimask)
    else:
        ## (1) whenever 1 * 1 = 1, the pixel is covered
        n_wild = 1  # HACK: n_wild is 1 because ALL THE DATASETS WE USE HAVE ONLY ONE EXTRA LABEL (255)
        coverage_tensor = torch.mul(
            one_hot_semantic_mask[:-n_wild, :, :],
            one_hot_multimask,
        )

        return coverage_tensor


def compute_activable_pixels(one_hot_semantic_mask: torch.Tensor, n_labs: int):
    """WARNING: assumes only last channels can be non activable"""
    ## TODO: makes no sense, ignorable could be any channel, not just the last ones
    ## ignore_mask: [N_LAB]+1, ... channels in one_hot_semantic_mask
    ignored_pixels_mask_one_hot = None

    ## assumes 3 dims: 0-dim is onehot classes, 1-dim and 2-dim are spatial dims
    n_pixels = one_hot_semantic_mask.shape[1] * one_hot_semantic_mask.shape[2]

    if one_hot_semantic_mask.shape[0] > n_labs:  ## contains void channels
        delta = one_hot_semantic_mask.shape[0] - n_labs
        ignored_pixels_mask_one_hot = one_hot_semantic_mask[-delta:, :, :]

    num_ignored_pixels = 0
    if ignored_pixels_mask_one_hot is not None:
        num_ignored_pixels = int(torch.sum(ignored_pixels_mask_one_hot).long())

    num_activable_pixels = n_pixels - num_ignored_pixels

    return num_activable_pixels, ignored_pixels_mask_one_hot


def miscoverage_loss(
    one_hot_semantic_mask,  # can contain extra channel for void pixels
    output_softmaxes,
    lbds: Union[float, Sequence[float], np.ndarray],
    n_labs: int,  # <-- [dataset.n_classes], being the num of non-void classes
    multimask_type: Literal["lac", "aps"] = "lac",
    minimum_coverage_ratio=False,
):
    """Miscoverage loss for CRC. If ground truth mask is entirely contained
    in multimask parametrized by [lbd] then return 1, else 0
    """
    # [delta_labs]: how many "excess" extra labs there are.
    #               Assumes [n_labs] is num of ground truth labs
    delta_labs = one_hot_semantic_mask.shape[0] - n_labs

    assert (
        delta_labs >= 0
    ), f" [{one_hot_semantic_mask.shape[0] = }] must be >= to n_labs [{n_labs}]"

    if multimask_type == "lac":
        multimask_func = lac_multimask
    elif multimask_type == "aps":
        multimask_func = aps_multimask
    else:
        raise ValueError(
            f"multimask_type must be in {CONFORMAL_SETS_REGISTER}, not {multimask_type}"
        )

    # ## ignore_mask: [N_LAB]+1, ... channels in one_hot_semantic_mask
    num_activable_pixels, ignored_pixels = compute_activable_pixels(
        one_hot_semantic_mask=one_hot_semantic_mask, n_labs=n_labs
    )
    if num_activable_pixels == 0:
        raise ValueError(
            f"num_activable_pixels [{num_activable_pixels}] must be > 0 for n_labs [{n_labs}]"
        )

    if isinstance(lbds, float):
        lbds = [lbds]
    elif isinstance(lbds, (Sequence, np.ndarray)):
        pass
    else:
        raise ValueError(f"lbds must be float or sequence of floats, not {type(lbds)}")

    for lbd in lbds:
        output = LossOutput(
            losses=[],
            activations_ratio=[],
            coverage_ratio=[],
            minimum_coverage_ratio=[],
        )
        thresholded_one_hot_multimask = multimask_func(
            threshold=lbd, predicted_softmax=output_softmaxes, n_labels=n_labs
        )

        binary_test = is_semantic_mask_in_multimask(
            one_hot_semantic_mask=one_hot_semantic_mask,
            one_hot_multimask=thresholded_one_hot_multimask,
            ignore_mask=ignored_pixels,
        )  ## pixel_ij == 1: ground truth mask_ij is covered by multimask_ij
        n_captured = binary_test.sum().cpu().numpy().astype(int)

        assert (
            binary_test.shape[0] == n_labs  # .shape[0] should be channels for classes
        ), f" [{binary_test.shape[0] = }] must be equal to n_labs [{n_labs}]"

        # n_voids_cityscapes = one_hot_semantic_mask[-1, :, :].sum()
        n_extra_labs_pxs = one_hot_semantic_mask[-delta_labs:, :, :].sum()
        n_pixels = one_hot_semantic_mask.shape[1] * one_hot_semantic_mask.shape[2]

        assert n_captured <= (
            int(n_pixels) - int(n_extra_labs_pxs)
        ), f"n_captured [{n_captured}] must be <= (n_pixels - n_extra_labs_pxs) [{(n_pixels - n_extra_labs_pxs)}]"

        cov_ratio = binary_test.sum() / num_activable_pixels
        cov_ratio = cov_ratio.cpu().numpy()
        miscoverage_ratio = 1 - cov_ratio

        if ignored_pixels is not None:
            are_not_void = 1 - ignored_pixels
            activable_pixels = thresholded_one_hot_multimask * are_not_void
        else:
            activable_pixels = thresholded_one_hot_multimask

        n_activations = activable_pixels.sum().cpu().numpy()
        activations_ratio = n_activations / num_activable_pixels

        output.losses.append(miscoverage_ratio)
        output.activations_ratio.append(float(activations_ratio))
        output.coverage_ratio.append(float(cov_ratio))
        output.minimum_coverage_ratio.append(minimum_coverage_ratio)

        if np.isnan(miscoverage_ratio):
            raise ValueError(
                f"[COSE] ERROR: got NaN in miscoverage_ratio for lbd {lbd} and n_labs {n_labs}"
            )

        ## as many appends as there are lbds
        output.losses.append(miscoverage_ratio)
        output.activations_ratio.append(activations_ratio)
        output.coverage_ratio.append(float(cov_ratio))
        output.minimum_coverage_ratio.append(None)

    return output


def binary_loss(
    one_hot_semantic_mask,  # can contain extra channel for void pixels
    output_softmaxes,
    lbds: Union[float, Sequence[float], np.ndarray],
    n_labs: int,  # <-- [dataset.n_classes], being the num of ground-truth classes
    minimum_coverage_ratio: float,
    multimask_type: Literal["lac", "aps"] = "lac",
) -> LossOutput:
    """Miscoverage loss for CRC. If ground truth mask is entirely contained
    in multimask parametrized by [lbd] then return 1, else 0
    """
    delta_labs = one_hot_semantic_mask.shape[0] - n_labs

    assert (
        delta_labs >= 0
    ), f" [{one_hot_semantic_mask.shape[0] = }] must be >= to n_labs [{n_labs}]"

    if multimask_type == "lac":
        multimask_func = lac_multimask
    elif multimask_type == "aps":
        multimask_func = aps_multimask
    else:
        raise ValueError(
            f"multimask_type must be in {CONFORMAL_SETS_REGISTER}, not {multimask_type}"
        )

    # ## ignore_mask: [N_LAB]+1, ... channels in one_hot_semantic_mask
    num_activable_pixels, ignored_pixels = compute_activable_pixels(
        one_hot_semantic_mask=one_hot_semantic_mask, n_labs=n_labs
    )

    if num_activable_pixels == 0:
        raise ValueError(
            f"num_activable_pixels [{num_activable_pixels}] must be > 0 for n_labs [{n_labs}]"
        )

    if isinstance(lbds, float):
        lbds = [lbds]
    elif isinstance(lbds, (Sequence, np.ndarray)):
        pass
    else:
        raise ValueError(f"lbds must be float or sequence of floats, not {type(lbds)}")

    output = LossOutput(
        losses=[],
        activations_ratio=[],
        coverage_ratio=[],
        minimum_coverage_ratio=[],
    )
    for lbd in lbds:
        thresholded_one_hot_multimask = multimask_func(
            threshold=lbd, predicted_softmax=output_softmaxes, n_labels=n_labs
        )

        binary_test = is_semantic_mask_in_multimask(
            one_hot_semantic_mask=one_hot_semantic_mask,
            one_hot_multimask=thresholded_one_hot_multimask,
            ignore_mask=ignored_pixels,
        )  ## pixel_ij == 1: ground truth mask_ij is covered by multimask_ij
        n_captured = binary_test.sum().cpu().numpy()

        assert (
            binary_test.shape[0] == n_labs  # .shape[0] should be channels for classes
        ), f" [{binary_test.shape[0] = }] must be equal to n_labs [{n_labs}]"

        n_extra_labs_pxs = one_hot_semantic_mask[-delta_labs:, :, :].sum()
        n_pixels = one_hot_semantic_mask.shape[1] * one_hot_semantic_mask.shape[2]

        assert n_captured <= (
            int(n_pixels) - int(n_extra_labs_pxs)
        ), f"n_captured [{n_captured}] must be <= (n_pixels - n_extra_labs_pxs) [{(n_pixels - n_extra_labs_pxs)}]"

        cov_ratio = binary_test.cpu().numpy().sum() / num_activable_pixels

        if np.isnan(cov_ratio):
            raise ValueError(
                f"[COSE] ERROR: got NaN in miscoverage_ratio for lbd {lbd} and n_labs {n_labs}"
            )

        binary_loss_value = int(cov_ratio < minimum_coverage_ratio)

        cov_ratio = binary_test.sum().cpu().numpy() / num_activable_pixels

        if ignored_pixels is not None:
            are_not_void = 1 - ignored_pixels
            activable_pixels = thresholded_one_hot_multimask * are_not_void
        else:
            activable_pixels = thresholded_one_hot_multimask

        n_activations = activable_pixels.sum().cpu().numpy()
        activations_ratio = n_activations / num_activable_pixels

        output.losses.append(binary_loss_value)
        output.activations_ratio.append(float(activations_ratio))
        output.coverage_ratio.append(float(cov_ratio))
        output.minimum_coverage_ratio.append(minimum_coverage_ratio)

    return output


def compute_empirical_risks(
    dataset,
    model,
    conformalizer: Conformalizer,
    samples_ids,
    lambdas,
    loss_prms,  # = "miscoverage",
    verbose=False,
    batch_size=1,
    mincov: Optional[float] = None,
) -> List[float]:
    if len(lambdas) == 0:
        raise ValueError(
            f"ERROR: expected [lambdas] to be non-empty but got length of: {len(lambdas)}"
        )
    _losses = []

    loss_name = loss_prms["loss_name"]

    for idx in samples_ids:  # , disable=(not verbose)):
        img_path = dataset[idx]["data_samples"].img_path

        sm = model.predict_softmax(img_path)

        # This returns the mask as encoded in the png file,
        # ignoring the config flag [reduce_zero_label=True] that offsets labels and put 0-lab to 255
        # ref: https://github.com/open-mmlab/mmsegmentation/blob/c46cc85cba6f98bb212278769e77bf0cb2bbf9e2/mmseg/datasets/transforms/loading.py#L112
        gt_mask = torch.Tensor(dataset[idx]["data_samples"].gt_sem_seg.data).long()
        gt_mask = gt_mask[0]

        recoded_gt_mask = recode_extra_classes(
            gt_mask, list(range(dataset.n_classes)), extra_labels=dataset.extralabels
        )

        # WARNING: the [+ len(dataset.extralabels)] is here because we recoded the [255] extra label(s)
        # as an additional channel in the one-hot gt mask: we set 255 -> K+1
        # REMARK: there could be more extra labels, e.g. [99, 255, ...]
        # --> torch.[...].one_hot() with arg [num_classes = K+n_extra] wouldn't work if
        #     if labels integers were not contiguous:
        #       - BAD:  0, 1, 3, .., 6, 255
        #       - GOOD: 0, 1, 3, .., 6, _7_
        n_total_recoded_labels = dataset.n_classes + len(dataset.extralabels)
        one_hot_gt = one_hot_encoding_of_gt(recoded_gt_mask, n_total_recoded_labels)

        # TODO: extract checking void pixels from loss to here.
        # ? add a filter mask with pixels to be ignored in loss?
        # ? and if filter is None, just use all pixels
        # MOCK:
        # this_filter_mask = stuff [...]
        # coverage_loss(
        #     one_hot_semantic_mask_gt=one_hot_gt.to(conformalizer.device),
        #     output_softmaxes=sm.to(conformalizer.device),
        #     ignore_filter_mask=this_filter_mask,
        #     lbds=lambdas,
        # )

        if loss_name == "miscoverage_loss":
            try:
                output_losses = miscoverage_loss(
                    one_hot_semantic_mask=one_hot_gt.to(conformalizer.device),
                    output_softmaxes=sm.to(conformalizer.device),
                    lbds=lambdas,
                    n_labs=dataset.n_classes,
                )
                pred_losses = output_losses.losses
                _losses.append(pred_losses)
            except Exception as e:
                raise e
        elif loss_name == "binary_loss":
            if mincov is None:
                raise ValueError(
                    f"ERROR: with [binary_loss], expected [minimum_coverage_ratio] cannot be [None]"
                )
            else:
                try:
                    output_losses = binary_loss(
                        one_hot_semantic_mask=one_hot_gt.to(conformalizer.device),
                        output_softmaxes=sm.to(conformalizer.device),
                        lbds=lambdas,
                        n_labs=dataset.n_classes,
                        minimum_coverage_ratio=mincov,
                    )
                    pred_losses = output_losses.losses
                    _losses.append(pred_losses)
                except Exception as e:
                    raise e
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")

    losses = np.array(_losses)
    empirical_risks = np.mean(losses, axis=0)
    empirical_risks = empirical_risks.tolist()

    return empirical_risks


class PredictionHandler:
    def __init__(self, mmseg_model, save_dir_path=None):
        self.mmseg_model = mmseg_model

    def extract_img_id(self, image_path):
        # Get the part after the last "/". Works also if no "/" is present
        return image_path.split("/")[-1]

    @torch.no_grad()
    def predict(self, input_image_path) -> torch.Tensor:
        pred_softmax = self.predict_softmax(input_image_path)
        return pred_softmax

    def predict_softmax(self, input_image_path):
        pred = pred_from_mmseg_input(
            input_image_path, self.mmseg_model, softmaxes=True, segmask=False
        )
        softmax_prediction = pred["softmax"][0]
        return softmax_prediction


def lambda_optimization(
    dataset,
    mmseg_model,
    conformalizer: Conformalizer,
    calibration_ids,
    loss_parameters,
    search_parameters,
    alpha_risk,
    mincov: Optional[float],
    verbose=False,
) -> tuple[float, dict[Any, Any], float, bool]:
    # TODO: input params validation

    alpha_tolerance = alpha_risk
    B_loss_bound = loss_parameters["B_loss_bound"]

    ## we look for the smallest lambda such that the
    ## empirical risk is above [risk_bound] by a minimal amount
    risk_bound = compute_risk_bound(alpha_tolerance, B_loss_bound, len(calibration_ids))

    lbd_lower = search_parameters["lbd_lower"]
    lbd_upper: float = search_parameters["lbd_upper"]
    n_iter = search_parameters["n_iter"]
    n_mesh = search_parameters["n_mesh"]
    lbd_tolerance = search_parameters["lbd_tolerance"]

    risks: dict[Any, Any] = {
        "lambdas": [],
        "avg_risks": [],
    }  ## output dict

    ## lbd == 0 breaks things when we always include top-1 class in pred set
    ## In that case, we could get that softmax(x) == 1 (math impossible) actually
    ## achieves a good coverage, just because the top-1 class is always included.
    ## In that case, we skip the optimization
    early_stopped = False
    LAMBDA_ZERO = [0.0]
    emp_risks = compute_empirical_risks(
        dataset=dataset,
        model=mmseg_model,
        conformalizer=conformalizer,
        samples_ids=calibration_ids,
        lambdas=LAMBDA_ZERO,
        loss_prms=loss_parameters,
        verbose=verbose,
        mincov=mincov,
    )
    empirical_risk_lambda_zero = emp_risks[0]

    print(f"\n ======= PRELIM CHECK FOR LAMBDA=0 (skip if useless optim) ====== ")
    print(f" ------ {alpha_risk = }")
    print(f" ------ {empirical_risk_lambda_zero = }")
    print(f" ------ {risk_bound = }")

    if empirical_risk_lambda_zero <= risk_bound:
        early_stopped = True
        risks["lambdas"].append([LAMBDA_ZERO])
        risks["avg_risks"].append(list(emp_risks))
        optimal_lambda = LAMBDA_ZERO[0]
        torch.cuda.empty_cache()

        print(f" ======= EARLY STOPPED: lambda=0 is enough ====== ")
        return optimal_lambda, risks, risk_bound, early_stopped
    else:
        print(f" ======= CONTINUE OPTIMIZATION ======\n")

    for _ in tqdm(range(n_iter)):
        if abs(lbd_upper - lbd_lower) < lbd_tolerance:
            break

        step_size = (lbd_upper - lbd_lower) / n_mesh
        lbds = np.concatenate(
            (np.arange(lbd_lower, lbd_upper, step_size), np.array([lbd_upper]))
        )

        print(f" --- {lbds = }")

        emp_risks = compute_empirical_risks(
            dataset=dataset,
            model=mmseg_model,
            conformalizer=conformalizer,
            samples_ids=calibration_ids,
            lambdas=lbds,
            loss_prms=loss_parameters,
            verbose=verbose,
            mincov=mincov,
        )
        print(f" === lambdas:\n\t{lbds}")
        print(f" === emp risks:\n\t {emp_risks = }")

        risks["lambdas"].append(list(lbds))
        risks["avg_risks"].append(list(emp_risks))

        ## DANGER: following code assumes emp_risks ARE SORTED in descending order.
        ## This is true if computations are done sequentially, because of monotonicity of lambda/risks
        ## (1) no lambda s.t. risk is below CRC threshold: [alpha - (B-alpha)/n]
        if emp_risks[-1] > risk_bound:
            lbd_lower = lbds[-1]
        ## (2) all lambdas have risks below threshold:
        elif (emp_risks[0] <= risk_bound) and (lbds[0] != 0.0):
            if lbds[0] <= 0.0:
                print(
                    f" ======= for {lbds[0]=}, we have {emp_risks[0]=} <= {risk_bound=}"
                    f" ======= This should have not happened"
                )
                break
            lbd_upper = lbds[0]
        else:
            for l_, risk in zip(lbds, emp_risks):
                if risk > risk_bound:
                    lbd_lower = l_
                elif risk <= risk_bound:
                    lbd_upper = l_
                    break

    optimal_lambda = lbd_upper
    torch.cuda.empty_cache()
    return optimal_lambda, risks, risk_bound, early_stopped


def compute_losses_on_test(
    dataset,
    model: CoseModel,
    conformalizer: Conformalizer,
    samples_ids,
    lbd: float,
    minimum_coverage,
    loss,
    pred_dump_path=None,
    return_coverage_ratio=True,
    verbose=False,
) -> np.ndarray:
    if not isinstance(lbd, float):
        raise ValueError(
            f"ERROR: expected [lbd] to be a float, but got type: {type(lbd)}"
        )
    losses = []
    activations = []
    empirical_coverage_ratio = []

    predictor = PredictionHandler(model.mmseg_model, save_dir_path=pred_dump_path)

    for i, idx in tqdm(enumerate(samples_ids), disable=(not verbose)):
        img_path = dataset[idx]["data_samples"].img_path
        softmax_prediction = predictor.predict(img_path)

        gt_mask = dataset[idx]["data_samples"].gt_sem_seg.data
        gt_mask = gt_mask[0]  # mono batch: mmseg does not support mini-batches
        recoded_gt_mask = recode_extra_classes(
            gt_mask,
            list(range(dataset.n_classes)),
            extra_labels=dataset.extralabels,  # [255]
        )

        one_hot_gt = one_hot_encoding_of_gt(
            recoded_gt_mask, dataset.n_classes + len(dataset.extralabels)
        )

        output_losses = loss(
            one_hot_semantic_mask=one_hot_gt.to(conformalizer.device),
            output_softmaxes=softmax_prediction.to(conformalizer.device),
            lbds=[lbd],
            minimum_coverage_ratio=minimum_coverage,
            n_labs=dataset.n_classes,
        )

        losses.append(output_losses.losses[0])  # [0] because we only have one lambda
        activations.append(output_losses.activations_ratio[0])
        empirical_coverage_ratio.append(output_losses.coverage_ratio[0])

    loss_np_array = np.array((losses, activations, empirical_coverage_ratio))
    return loss_np_array
