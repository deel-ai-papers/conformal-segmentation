import json
import pandas as pd  # type: ignore
import torch
import argparse

from typing import Optional

from cose.models import CoseModel
from cose.conformal import (
    Conformalizer,
    compute_losses_on_test,
    load_loss,
    lambda_optimization,
    split_dataset_idxs,
)


def parse_args_expes():
    parser = argparse.ArgumentParser(
        description="Run CRC experiment: choose loss and dataset"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        help="Default: [cuda:0], but also [cuda:1], if two gpus on server",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Cityscapes", "ADE20K", "LoveDA"],
        help="Available datasets: [Cityscapes], [ADE20K] [LoveDA]",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed: determines how val data are split into _cal_ and _test_",
        required=True,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Config file (JSON) format: expe's params (loss, etc.)",
        required=True,
    )

    args = parser.parse_args()
    return args


def run_crc_experiment(
    conformal_model,
    calib_dataset,
    calib_ids,
    random_seed,
    alpha,
    loss_params,
    search_params,
    output_directory,
    experiment_name: str,
    mincov: Optional[float] = None,
):
    with torch.no_grad():
        try:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
            loss_name = loss_params["loss_name"]
            filename_out = f"{timestamp}_{calib_dataset.name}__id_{random_seed}__alpha_{alpha}__mincov_{mincov}__{loss_name}.json"
            output_path = f"{output_directory}/{filename_out}"
            print(f" ------ {output_path = }")

            optimal_lambda, risks, risk_bound, early_stopped = lambda_optimization(
                dataset=calib_dataset,
                mmseg_model=conformal_model.model,
                conformalizer=conformal_model,
                calibration_ids=calib_ids,
                loss_parameters=loss_params,
                search_parameters=search_params,
                alpha_risk=alpha,
                verbose=True,
                mincov=mincov,
            )

            print(f" ------ end of optim lambda with results:")
            print(f" --- {optimal_lambda = }")
            print(f" --- {risks = }")
            print(f" ----------------------------------------")
            print()

            results = {
                "expe_name": experiment_name,
                "alpha": alpha,
                "alpha_risk_bound": risk_bound,
                "early_stopped": early_stopped,
                "mincov": mincov,  # loss_params["minimum_coverage_threshold"],
                "dataset": calib_dataset.name,
                "experiment_id": random_seed,
                "optimal_lambda": optimal_lambda,
                "loss_function": loss_params["loss_name"],
                "risks": risks,
                "lbd_search_params": search_params,
                "n_calib": conformal_model.n_calib,
                "cal_id": conformal_model.calibration_indices,
            }

            with open(output_path, "w") as f:
                json.dump(results, f)

            torch.cuda.empty_cache()

        except Exception as E:
            torch.cuda.empty_cache()
            print(f"{E = }")


def get_all_configs_in_expe(res_paths_jsons):
    expe_configs = []

    for _, res_path in enumerate(res_paths_jsons):
        with open(f"{res_path}", "r") as f:
            res = json.load(f)
            # > res.keys() = "expe_name",     "alpha",  "alpha_risk_bound", "early_stopped",
            # >              "mincov",        "dataset","experiment_id",    "optimal_lambda",
            # >              "loss_function", "risks",  "lbd_search_params","n_calib", "cal_id"

            # rename keys: experiment_id -> random_seed
            res["random_seed"] = res.pop("experiment_id")

            keep_keys = [
                "expe_name",
                "early_stopped",
                "dataset",
                "alpha",
                "loss_function",
                "mincov",
                "optimal_lambda",
                "random_seed",
                "n_calib",
                "cal_id",
            ]
            dropping_keys = set(res) - set(keep_keys)
            for dropper in dropping_keys:
                del res[dropper]

            res["res_path"] = res_path

        expe_configs.append(res)

    return expe_configs


@torch.no_grad()
def eval_empirical_metrics(
    configs_df,
    dataset,
    pretrained_model: CoseModel,
    save_preds_to=None,
    get_cov_ratio=True,
    csv_to=None,
    n_tests: Optional[int] = None,
):
    results = []

    model = pretrained_model
    configs = configs_df

    # Check if there are multiple n_calib values for the same random seed
    assert (
        len(configs["n_calib"].unique()) == 1
    ), f" --- ERROR: multiple n_calib values for same random seed"

    # Names of default columns
    cols = [
        "random_seed",
        "alpha",
        "optimal_lambda",
        "mincov",
        "loss_function",
        "empirical_risk",
        "activations",
        "id_path",
    ]
    if get_cov_ratio:
        cols.append("empirical_coverage_ratio")

    for _, row in configs.iterrows():
        res = [  # Defaults columns: always present
            row["random_seed"],  # Include random_seed in the results
            row["alpha"],
            row["optimal_lambda"],
            row["mincov"],
            row["loss_function"],
        ]

        mod = Conformalizer(
            model=model,
            dataset=dataset,
            random_seed=row["random_seed"],
            device=model.device,
            n_calib=row["n_calib"],
        )

        cal_ids, test_ids = split_dataset_idxs(
            len_dataset=len(dataset),
            n_calib=row["n_calib"],
            random_seed=row["random_seed"],
        )

        if n_tests is not None:
            test_ids = test_ids[:n_tests]

        # Check if the split is the same as during conformalization
        if row["cal_id"] != str(cal_ids):
            raise RuntimeError(
                f" --- ERROR: splitting did not return same split as during conformalization. {row['cal_id']} != {str(cal_ids)}"
            )

        del cal_ids

        test_losses_np_array = compute_losses_on_test(
            dataset=dataset,
            model=mod.model,
            conformalizer=mod,
            pred_dump_path=save_preds_to,
            samples_ids=test_ids,
            lbd=row["optimal_lambda"],
            minimum_coverage=row["mincov"],
            loss=load_loss(row["loss_function"]),
            return_coverage_ratio=get_cov_ratio,
            verbose=True,
        )

        averages = test_losses_np_array.mean(axis=1)

        res.append(averages[0])  ## empirical risk
        res.append(averages[1])  ## activations
        res.append(row["res_path"])

        if get_cov_ratio:
            res.append(averages[2])  ## empirical coverage ratio

        results.append(res)

        ## HACK: write to disk after each experiment, to check progress
        try:
            partial_tests_df = pd.DataFrame(results, columns=cols)
            partial_path = f"{csv_to}_PARTIAL.csv"
            partial_tests_df.to_csv(f"{partial_path}", index=True, index_label="index")
        except:
            raise RuntimeError(" --- ERROR: could not write to disk")

    # Create a DataFrame from the results
    tests_df = pd.DataFrame(results, columns=cols)
    tests_df.to_csv(csv_to, index=True, index_label="index")

    torch.cuda.empty_cache()
    return tests_df
