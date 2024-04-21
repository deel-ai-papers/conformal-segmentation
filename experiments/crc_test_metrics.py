import os
import torch
import argparse
import datetime
import pandas as pd

from mmengine.registry import init_default_scope

from cose.datasets import load_project_paths, DATASETS
from cose.experiments import eval_empirical_metrics
from cose.models import get_default_model_for_dataset

COSE_PATH = load_project_paths().COSE_PATH


def argparse_metrics_script():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "--gpu",
        type=str,
        help="Default: cuda:0, but also cuda:1 for SODA machine",
        # required=True,
        default="cuda:0",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{COSE_PATH}/experiments/metrics",
        help="Output directory. default: ./experiments/metrics",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input file containing the optimized lambdas and experiment setup",
        required=True,
    )
    parser.add_argument(
        "--dump-pred",
        type=str,
        default=None,
        help="Path to write predictions to disk. Default: None",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    init_default_scope("mmseg")  ## setup mmseg stuff

    args = argparse_metrics_script()
    device = torch.device(args.gpu)
    config_expe_df = pd.read_csv(args.input)

    assert (
        len(set(config_expe_df.dataset)) == 1
    ), f"ERROR: Multiple datasets in config file: {set(config_expe_df.dataset)}"

    assert (
        len(set(config_expe_df.loss_function)) == 1
    ), f"ERROR: Multiple datasets in config file: {set(config_expe_df.dataset)}"

    dataset_name = config_expe_df.dataset[0]  # TODO: ensure string is valid dir name
    loss_name = config_expe_df.loss_function[0]
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H:%M")

    output_dir = f"{args.output_dir}/{dataset_name}"
    output_filename = f"{dataset_name}_{loss_name}_{current_date}.csv"
    default_output = f"{output_dir}/{output_filename}"

    print(f" --- output to:         {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f" --- using cuda device: {args.gpu}")
    print(f" --- metrics, write to: {default_output}")
    print(f" --- input file:        {args.input}")

    N_TESTS = None
    if N_TESTS is None:
        print(f" --- n test samples:    all.")
    else:
        print(f" --- n test samples:    {N_TESTS}.")

    dataset = DATASETS[dataset_name](data_partition="val")

    model = get_default_model_for_dataset(dataset_name, device)

    try:
        test_coverages = eval_empirical_metrics(
            configs_df=config_expe_df,
            pretrained_model=model,
            dataset=dataset,
            save_preds_to=None,
            get_cov_ratio=True,
            csv_to=default_output,
            n_tests=N_TESTS,
        )
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
