"""Export binary label-space diagnostics without rerunning feature imputation."""

import json
import os

import pandas as pd

from config import LABEL_COLS, ORIG_PROCESSED_DIR, RAW_CSV
from label_metrics import label_space_metrics


def main():
    raw = pd.read_csv(RAW_CSV, usecols=LABEL_COLS).fillna(0).astype(int)
    raw["LET_IS"] = (raw["LET_IS"] > 0).astype(int)
    output = {
        "definition": (
            "LDS is the mean directed conditional probability P(Y_j=1 | Y_i=1) "
            "over all ordered off-diagonal label pairs."
        ),
        "label_order": LABEL_COLS,
        "full_cohort": label_space_metrics(raw.to_numpy()),
    }
    for split in ("train", "validation", "test"):
        file_stem = "val" if split == "validation" else split
        labels = pd.read_csv(os.path.join(ORIG_PROCESSED_DIR, f"y_{file_stem}.csv"))
        output[split] = label_space_metrics(labels.to_numpy(dtype=int))

    destination = os.path.join(ORIG_PROCESSED_DIR, "label_space_metrics.json")
    with open(destination, "w") as handle:
        json.dump(output, handle, indent=2)
    print(destination)
    for split in ("full_cohort", "train", "validation", "test"):
        metrics = output[split]
        print(
            f"{split}: LC={metrics['label_cardinality']:.6f}, "
            f"LD={metrics['label_density']:.6f}, "
            f"LDS={metrics['label_dependency_score']:.6f}"
        )


if __name__ == "__main__":
    main()
