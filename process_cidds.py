import pandas as pd

## We process the cidds dataset to make it ready for training.

import os
from pathlib import Path


PATH = "data_real/cidds/train.csv"
PATH2="data_real/cidds/test.csv"


def process_cidds_file(path: str, chunksize: int = 200_000) -> None:
    src = Path(path)
    tmp = src.with_suffix(src.suffix + ".tmp")

    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    reader = pd.read_csv(src, chunksize=chunksize, low_memory=False)
    wrote_header = False

    for chunk in reader:
        # 1) Preserve any existing label as "original_label"
        if "label" in chunk.columns:
            chunk = chunk.rename(columns={"label": "original_label"})

        # 2) Convert attack_type -> binary label (benign=0, else=1), and rename to "label"
        if "attack_type" not in chunk.columns:
            raise KeyError("Expected column 'attack_type' not found in CIDDS CSV.")

        attack_type = (
            chunk["attack_type"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )
        chunk["label"] = (attack_type != "benign").astype("int8")
        chunk = chunk.drop(columns=["attack_type"])

        chunk.to_csv(tmp, index=False, mode="w" if not wrote_header else "a", header=not wrote_header)
        wrote_header = True

    os.replace(tmp, src)

if __name__ == "__main__":
    process_cidds_file(PATH)
    process_cidds_file(PATH2)
