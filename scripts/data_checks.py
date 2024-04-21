## This script runs a series of checks on the dataset, before running the calibration:
## 1. Which/how many classes are for void pixels?
import multiprocessing
import torch
import json

from mmengine.registry import init_default_scope

from cose.datasets import (
    DATASETS,
)
from cose.utils import load_project_paths


def get_dataset_descriptors(datname, DatasetClass, verbose=False):
    descriptors = {}

    dataset = DatasetClass()
    count_labs = 0
    for i, _ in enumerate(dataset.classes):
        count_labs += 1

    assert (
        count_labs == dataset.n_classes
    ), "The count of classes is not consistent with the number of classes in the dataset."

    descriptors[datname] = {
        "n_classes": dataset.n_classes,
        "count_labs": count_labs,
    }

    non_nominal_labels = []
    for i, sample in enumerate(dataset):
        if verbose and i % 200 == 0:
            print(f" --- {datname = }: {i} of {len(dataset)}")

        gt = sample["data_samples"].gt_sem_seg.data

        labels_in_gt = set(gt.unique().tolist())
        nominal = set(c for c in range(dataset.n_classes))
        extralabels = labels_in_gt.difference(labels_in_gt.intersection(nominal))
        non_nominal_labels.extend(extralabels)

    descriptors[datname]["non_nominal_labels"] = list(set(non_nominal_labels))

    return descriptors


if __name__ == "__main__":
    init_default_scope("mmseg")  ## Mmseg Boilerplate

    device_str = "cuda:0"
    device = torch.device(device_str)

    verbose = True

    datasets = [(datname, DatasetClass) for datname, DatasetClass in DATASETS.items()]

    with multiprocessing.Pool(4) as pool:
        descriptors = pool.starmap(get_dataset_descriptors, datasets)

    if verbose:
        print(" ====== SUMMARY ======")
        for desc in descriptors:
            print()
            for dat, stats in desc.items():
                print(f" === {dat}:")
                print(
                    f"     Number of labels declared in class:     {stats['n_classes']}"
                )
                print(
                    f"     Number of labels counted in their list: {stats['count_labs']}"
                )
                print(
                    f"     Non-nominal labels found:               {stats['non_nominal_labels']}"
                )

    # Write descriptors to JSON file
    _path = load_project_paths().COSE_PATH
    with open(f"{_path}/models/datasets_descriptors.json", "w") as f:
        json.dump(descriptors, f)
