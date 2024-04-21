import os
import json
import torch
import itertools

from mmengine.registry import init_default_scope

from cose.utils import load_project_paths
from app.tools import (
    setup_inference_ADE20K,
    setup_inference_cityscapes,
    setup_inference_LoveDA,
)
from cose.experiments import parse_args_expes, run_crc_experiment
from cose.conformal import Conformalizer

if __name__ == "__main__":
    args = parse_args_expes()
    random_seed = args.seed
    path_to_config_expe = args.config_file

    loss_params = {}
    lbd_search_params = {}

    print(f" === {path_to_config_expe = }")

    with open(path_to_config_expe, "r") as f:
        config = json.load(f)

        expe_name = config["expe_name"]  # <-- redundant
        alphas = config["alpha_risks"]

        conformal_params = {
            "calib_test_ratio": config["calib_test_ratio"],
        }

        loss_params = dict(
            loss_name=config["loss_name"],
            B_loss_bound=config["B_loss_bound"],
        )

        if "minimum_coverage_thresholds" in config.keys():
            loss_params["minimum_coverage_thresholds"] = config[
                "minimum_coverage_thresholds"
            ]
        else:
            loss_params["minimum_coverage_thresholds"] = False

        lbd_search_params = dict(
            lbd_lower=config["lbd_lower"],
            lbd_upper=config["lbd_upper"],
            n_iter=config["n_iter"],
            n_mesh=config["n_mesh"],
            lbd_tolerance=config["lbd_tolerance"],
        )

    PATHS = load_project_paths(dataset_name=config["dataset_name"])

    output_dir = f"{PATHS.COSE_PATH}/experiments/outputs/{config['dataset_name']}/{config['loss_name']}"
    print(f" --- output to: {output_dir = }")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f" --- --- ↳ created output directory")

    device_str = args.gpu
    print(f" --- using cuda device: {device_str = }")
    device = torch.device(device_str)

    init_default_scope("mmseg")  ## setup mmseg stuff

    if config["dataset_name"] == "ADE20K":
        dataset, model, _ = setup_inference_ADE20K(
            torch_device=device,
            recover_calib_test_splits=False,
        )
        model = model["model"]
    elif config["dataset_name"] == "Cityscapes":
        dataset, model, _ = setup_inference_cityscapes(
            torch_device=device,
            recover_calib_test_splits=False,
        )
        model = model["model"]
    elif config["dataset_name"] == "LoveDA":
        dataset, model, _ = setup_inference_LoveDA(
            torch_device=device,
            recover_calib_test_splits=False,
        )
        model = model["model"]
    else:
        raise ValueError(f"Invalid dataset_name: {config['dataset_name']}")

    N_CALIB = int(config["calib_test_ratio"] * len(dataset))

    with torch.no_grad():
        print(f" === {random_seed = }")
        mod = Conformalizer(
            model=model,
            dataset=dataset,
            random_seed=random_seed,
            n_calib=N_CALIB,
            device=device,
        )

        mod.split_dataset_cal_test()
        #
        print(
            f" --- n_calib: {len(mod.calibration_indices)} / {len(dataset)} << vs >> n_test: {len(mod.test_indices)} / {len(dataset)}"
        )

        if loss_params["minimum_coverage_thresholds"]:
            crc_params = itertools.product(
                alphas, loss_params["minimum_coverage_thresholds"]
            )
            for alpha, mincov in crc_params:
                run_crc_experiment(
                    conformal_model=mod,
                    calib_dataset=dataset,
                    calib_ids=mod.calibration_indices,
                    random_seed=random_seed,
                    alpha=alpha,
                    mincov=mincov,
                    loss_params=loss_params,
                    search_params=lbd_search_params,
                    output_directory=output_dir,
                    experiment_name=expe_name,
                )
        else:
            for alpha in alphas:
                run_crc_experiment(
                    conformal_model=mod,
                    calib_dataset=dataset,
                    calib_ids=mod.calibration_indices,
                    random_seed=random_seed,
                    alpha=alpha,
                    loss_params=loss_params,
                    search_params=lbd_search_params,
                    output_directory=output_dir,
                    experiment_name=expe_name,
                )
