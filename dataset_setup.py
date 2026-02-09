import os
import pandas as pd
from datasets import load_dataset

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def process_split(split_name, dataset_split):
    df = pd.DataFrame(dataset_split)
    print(f"{split_name} columns:", df.columns.tolist())

    # This is the actual schema
    required = {"reactants", "product"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Unexpected schema in {split_name}: {df.columns}")

    # Construct reaction SMILES (no reagents in this dataset)
    df["reaction_smiles"] = df["reactants"] + ">>" + df["product"]

    # Keep class label too — useful later
    df = df[["reaction_smiles", "class"]].dropna()

    df.to_csv(f"{DATA_DIR}/uspto_{split_name}.csv", index=False)
    print(f"Saved {split_name} with {len(df)} rows")

def download_and_process():
    print("Downloading USPTO-50K (bisectgroup)...")
    ds = load_dataset("bisectgroup/USPTO_50K")
    print(ds)
    for split in ds:
        process_split(split, ds[split])

if __name__ == "__main__":
    download_and_process()

