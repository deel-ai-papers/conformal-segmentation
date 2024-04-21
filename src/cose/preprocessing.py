import torch

from typing import Sequence, Optional


def recode_extra_classes(
    semantic_map: torch.Tensor,
    good_labels: Sequence[int],
    extra_labels: Optional[Sequence[int]],
    force_recode=False,
) -> torch.Tensor:
    """input: tensor (W,H). Dim[0]: int, 0..(n_labels - 1)
    output: tensor (W,H). Dim[0]: int, 0..(n_labels - 1) + n_extra_labels

    WARNING: we assume that labels are 0-based sequence. 5 good labels = [0,1,2,3,4]

    [good_labels]: must contain ALL ACCEPTABLE LABELS in reference encoding (e.g. benchmark dataset).
    Labels such as "void" or "other" that were not used in training could be coded with wild values such
    as "99" or "255".
    Such values, we want to recode as label "(n_labels - 1) + 1", "(n_labels -1) + 2" etc.


    [extra_labels] are recoded and appended to the list of good classes. This is to
    facilitate some operations with torch, where it assumes that classes are encoded as
    sequentially ordered integers.
    """
    if not isinstance(semantic_map, torch.Tensor):
        raise ValueError(f"Error: semantic_map must be a torch tensor")

    if not semantic_map.ndim == 2:
        raise ValueError(f"ERROR: expected shape=(W,H) but got {semantic_map.shape}")

    if not isinstance(good_labels, list) or not isinstance(extra_labels, list):
        raise ValueError(f"Error: [good_labels] and [extra_labels] must be lists")

    intersection = set(good_labels).intersection(extra_labels)
    if len(intersection) > 0 and not force_recode:
        raise ValueError(
            f"[extra_labels] contains labels in [good_labels]: {intersection}"
        )

    ## Sort extra-labels, so that their new labels with be preserve the original ordering
    extra_labels.sort()

    recoded_semantic_map = torch.clone(semantic_map)

    # TEMP: assume that the "biggest" label is in this input
    max_good_label = max(good_labels)

    for i, extra_lab in enumerate(extra_labels):
        newlabel = max_good_label + i + 1

        ## if extra_lab IS NOT in the segmentation map, nothing should happen,
        ## but the newlabel counter progresses to the next.
        recoded_semantic_map[recoded_semantic_map == extra_lab] = newlabel

    # N_LAB_WITH_RECODED = len(good_labels) + len(extra_labels)

    return recoded_semantic_map  # , N_LAB_WITH_RECODED
