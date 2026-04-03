"""
Demo script showing novelty scoring on example reactions.
Compares common reaction patterns vs unusual transformations.
"""

import torch
import numpy as np
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer
)
from model import ConditionalScoreNet
from collections import Counter

def load_model():
    """Load trained model and normalization stats."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    metrics = torch.load("metrics.pt", map_location=device)
    mean = metrics["mean"]
    std = metrics["std"]

    model = ConditionalScoreNet(dim=256, cond_dim=1)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)
    model.eval()

    rxnfp_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxnfp_model, tokenizer)

    # Load training class frequencies for conditioning
    data = torch.load("data/route_embeddings.pt")
    train_freq = data["train"]["freq_features"]
    mean_freq = train_freq.mean().item()

    return model, mean, std, rxnfp_generator, mean_freq, device

def score_reactions(reactions, model, mean, std, rxnfp_generator, mean_freq, device):
    """Score a list of reaction SMILES."""
    embeddings = rxnfp_generator.convert_batch(reactions)
    X = torch.tensor(np.array(embeddings), dtype=torch.float32)
    # Normalize using stats on same device
    X = (X - mean.to(X.device)) / std.to(X.device)

    # Use mean training frequency as conditioning for unknown reactions
    C = torch.ones((X.size(0), 1), dtype=torch.float32) * mean_freq

    X, C = X.to(device), C.to(device)
    t = torch.ones((X.size(0), 1), device=device) * 0.5

    with torch.no_grad():
        score = model(X, t, C)
        novelty = torch.norm(score, dim=1)

    return novelty.cpu().numpy()

def main():
    print("Loading model...")
    model, mean, std, rxnfp_generator, mean_freq, device = load_model()
    print(f"Using device: {device}\n")

    # Load example reactions
    low_novelty = []
    high_novelty = []
    current_group = None

    with open("example_reactions.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Low"):
                current_group = "low"
            elif line.startswith("# High"):
                current_group = "high"
            elif ">>" in line:
                if current_group == "low":
                    low_novelty.append(line)
                elif current_group == "high":
                    high_novelty.append(line)

    # Score reactions
    print("Scoring reactions...\n")

    low_scores = score_reactions(low_novelty, model, mean, std, rxnfp_generator, mean_freq, device)
    high_scores = score_reactions(high_novelty, model, mean, std, rxnfp_generator, mean_freq, device)

    # Display results
    print("="*70)
    print("COMMON REACTION PATTERNS (expected low novelty)")
    print("="*70)
    for i, (rxn, score) in enumerate(zip(low_novelty, low_scores)):
        reactants, product = rxn.split(">>")
        print(f"\nReaction {i+1}: Novelty Score = {score:.2f}")
        print(f"  Reactants: {reactants[:60]}..." if len(reactants) > 60 else f"  Reactants: {reactants}")
        print(f"  Product:   {product[:60]}..." if len(product) > 60 else f"  Product:   {product}")

    print("\n")
    print("="*70)
    print("UNUSUAL REACTION PATTERNS (expected high novelty)")
    print("="*70)
    for i, (rxn, score) in enumerate(zip(high_novelty, high_scores)):
        reactants, product = rxn.split(">>")
        print(f"\nReaction {i+1}: Novelty Score = {score:.2f}")
        print(f"  Reactants: {reactants[:60]}..." if len(reactants) > 60 else f"  Reactants: {reactants}")
        print(f"  Product:   {product[:60]}..." if len(product) > 60 else f"  Product:   {product}")

    # Summary statistics
    print("\n")
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nCommon patterns:  mean={low_scores.mean():.2f}, std={low_scores.std():.2f}")
    print(f"Unusual patterns: mean={high_scores.mean():.2f}, std={high_scores.std():.2f}")
    print(f"\nDifference: {high_scores.mean() - low_scores.mean():.2f} (higher = more novel)")

    if high_scores.mean() > low_scores.mean():
        print("\n✓ Model correctly assigns higher novelty to unusual reaction patterns")
    else:
        print("\n✗ Unexpected: common patterns scored higher than unusual ones")

if __name__ == "__main__":
    main()
