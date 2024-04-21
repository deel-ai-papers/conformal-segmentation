import os
import pandas as pd
import argparse

from cose.experiments import get_all_configs_in_expe


def parse_expe_outputs():
    parser = argparse.ArgumentParser(
        description="CRC experiments: process raw json outputs, prep for metrics"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory: parse all json files (one per rnd_seed-ed run, shared CRC config)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments/outputs/processed_lambdas",
        help="Output file name with all optimized lambdas (default: optimized_lambdas.csv)",
    )
    parser.add_argument(
        "--outname",
        type=str,
        default="optimized_lambdas.csv",
        help="Output file name with all optimized lambdas (default: optimized_lambdas.csv)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_expe_outputs()
    input_dir = args.input_dir
    outdir = args.outdir

    json_files = [file for file in os.listdir(input_dir) if file.endswith(".json")]

    all_results_paths = []
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        all_results_paths.append(json_path)

    cfgs = get_all_configs_in_expe(all_results_paths)
    cfgs_df = pd.DataFrame(cfgs)
    cfgs_df = cfgs_df.drop_duplicates(
        subset=["random_seed", "n_calib", "alpha", "mincov", "optimal_lambda"]
    )
    print(f" --- n. experimental setups: {len(cfgs_df)}")

    expe_name = cfgs_df.expe_name[0]
    # to avoid changing this naming mistake in all files
    expe_name = expe_name.replace("ade20k", "ADE20K")
    expe_name = expe_name.replace("loveda", "LoveDA")
    expe_name = expe_name.replace("cityscapes", "Cityscapes")

    output_path = f"{outdir}/{expe_name}_{args.outname}"
    try:
        # Use the specified output file name or the default value
        cfgs_df.to_csv(output_path, index=False)
    except Exception as e:
        raise ValueError(f"Error writing to file: {e}")
