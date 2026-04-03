import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer
)

TRAIN_PATH = "data/uspto_train.csv"
VAL_PATH = "data/uspto_val.csv"
TEST_PATH = "data/uspto_test.csv"
OUT_PATH = "data/route_embeddings.pt"

BATCH_SIZE = 64

def embed_reactions(rxnfp_generator, rxns):
    """Generate embeddings for a list of reaction SMILES."""
    all_embeddings = []
    for i in tqdm(range(0, len(rxns), BATCH_SIZE), desc="Generating embeddings"):
        batch = rxns[i:i + BATCH_SIZE]
        fps = rxnfp_generator.convert_batch(batch)
        all_embeddings.extend(fps)
    return torch.tensor(np.array(all_embeddings), dtype=torch.float32)

def main():
    # Load splits separately
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Initialize rxnfp generator
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    # Compute frequency features based on TRAINING set only
    class_counts = Counter(df_train['class'])

    def compute_freq_features(df):
        df = df.copy()
        freq_features = []
        for cls in df['class']:
            freq = class_counts.get(cls, 0)  # 0 if not seen in training
            freq_features.append(np.log(freq + 1))
        return torch.tensor(freq_features, dtype=torch.float32).unsqueeze(1)

    # Generate embeddings for each split
    print("\nEmbedding training set...")
    train_embeddings = embed_reactions(rxnfp_generator, df_train['rxn_smiles'].tolist())
    train_freq = compute_freq_features(df_train)

    print("\nEmbedding validation set...")
    val_embeddings = embed_reactions(rxnfp_generator, df_val['rxn_smiles'].tolist())
    val_freq = compute_freq_features(df_val)

    print("\nEmbedding test set...")
    test_embeddings = embed_reactions(rxnfp_generator, df_test['rxn_smiles'].tolist())
    test_freq = compute_freq_features(df_test)

    # Save with split information
    torch.save({
        "train": {
            "embeddings": train_embeddings,
            "freq_features": train_freq,
            "smiles": df_train['rxn_smiles'].tolist()
        },
        "val": {
            "embeddings": val_embeddings,
            "freq_features": val_freq,
            "smiles": df_val['rxn_smiles'].tolist()
        },
        "test": {
            "embeddings": test_embeddings,
            "freq_features": test_freq,
            "smiles": df_test['rxn_smiles'].tolist()
        }
    }, OUT_PATH)

    print(f"\nSaved embeddings:")
    print(f"  Train: {train_embeddings.shape}")
    print(f"  Val: {val_embeddings.shape}")
    print(f"  Test: {test_embeddings.shape}")

if __name__ == "__main__":
    main()
