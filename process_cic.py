import pandas as pd
import os
from pathlib import Path

# Paths defined based on your request
PATH = "data_real/cic/train.csv"
PATH2 = "data_real/cic/test.csv"

def process_cic_file(path: str, chunksize: int = 200_000) -> None:
    """
    Reads a CSV in chunks, converts multi-class labels to binary, 
    and overwrites the file.
    
    Logic:
    - If label == 0 (BENIGN) -> 0
    - If label > 0 (Attack)  -> 1
    """
    src = Path(path)
    tmp = src.with_suffix(src.suffix + ".tmp")

    if not src.exists():
        print(f"Skipping {src}: File not found.")
        return

    print(f"Processing {src}...")
    
    # Read in chunks to handle large file sizes efficiently
    reader = pd.read_csv(src, chunksize=chunksize, low_memory=False)
    wrote_header = False

    for chunk in reader:
        # 1) Clean column names (CIC-IDS often has spaces like " Label" or " Flow Duration")
        chunk.columns = chunk.columns.str.strip()
        
        # 2) Identify the label column (usually 'label' or 'Label')
        if "label" in chunk.columns:
            label_col = "label"
        elif "Label" in chunk.columns:
            label_col = "Label"
        else:
            # If not found by name, assume it's the last column (standard for this dataset)
            label_col = chunk.columns[-1]
        
        # 3) Convert to Binary
        # The JSON shows the data is already numeric (0=Benign, 1..14=Attacks).
        # We ensure it's treated as numeric, then apply the binary logic.
        # Logic: True (Attack) becomes 1, False (Benign) becomes 0.
        chunk[label_col] = pd.to_numeric(chunk[label_col], errors='coerce').fillna(0)
        chunk[label_col] = (chunk[label_col] != 0).astype("int8")
        
        # 4) Standardize column name to "label" (lowercase)
        if label_col != "label":
            chunk = chunk.rename(columns={label_col: "label"})

        # Write to temp file
        chunk.to_csv(tmp, index=False, mode="w" if not wrote_header else "a", header=not wrote_header)
        wrote_header = True

    # Atomic replacement of the original file
    os.replace(tmp, src)
    print(f"Completed {src}: Labels converted to binary (0=Benign, 1=Attack).")

if __name__ == "__main__":
    process_cic_file(PATH)
    process_cic_file(PATH2)