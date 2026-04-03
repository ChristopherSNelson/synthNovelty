"""
Score novelty of individual reaction SMILES.

Usage:
    python score_reaction.py "reactants>>product"
    python score_reaction.py --file reactions.txt
"""

import torch
import numpy as np
import argparse
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer
)
from model import ConditionalScoreNet

def load_model():
    """Load trained model and normalization stats."""
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load normalization stats
    metrics = torch.load("metrics.pt", map_location=device)
    mean = metrics["mean"]
    std = metrics["std"]

    # Load model
    model = ConditionalScoreNet(dim=256, cond_dim=1)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Load rxnfp
    rxnfp_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxnfp_model, tokenizer)

    # Load training class frequencies for conditioning
    data = torch.load("data/route_embeddings.pt")
    train_freq = data["train"]["freq_features"]
    mean_freq = train_freq.mean().item()

    return model, mean, std, rxnfp_generator, mean_freq, device

def score_reactions(reactions, model, mean, std, rxnfp_generator, mean_freq, device):
    """Score a list of reaction SMILES."""
    # Generate embeddings
    embeddings = rxnfp_generator.convert_batch(reactions)
    X = torch.tensor(np.array(embeddings), dtype=torch.float32)

    # Normalize using stats on same device
    X = (X - mean.to(X.device)) / std.to(X.device)

    # Use mean training frequency as conditioning for unknown reactions
    C = torch.ones((X.size(0), 1), dtype=torch.float32) * mean_freq

    # Score
    X, C = X.to(device), C.to(device)
    t = torch.ones((X.size(0), 1), device=device) * 0.5

    with torch.no_grad():
        score = model(X, t, C)
        novelty = torch.norm(score, dim=1)

    return novelty.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Score reaction novelty")
    parser.add_argument("reaction", nargs="?", help="Reaction SMILES (reactants>>product)")
    parser.add_argument("--file", "-f", help="File with one reaction SMILES per line")
    args = parser.parse_args()

    if not args.reaction and not args.file:
        parser.print_help()
        return

    print("Loading model...")
    model, mean, std, rxnfp_generator, mean_freq, device = load_model()

    if args.file:
        with open(args.file) as f:
            reactions = [line.strip() for line in f if line.strip() and ">>" in line]
    else:
        reactions = [args.reaction]

    print(f"Scoring {len(reactions)} reaction(s)...")
    scores = score_reactions(reactions, model, mean, std, rxnfp_generator, mean_freq, device)

    print("\n" + "="*70)
    print("NOVELTY SCORES")
    print("="*70)
    for rxn, score in zip(reactions, scores):
        # Truncate long SMILES for display
        display_rxn = rxn if len(rxn) < 60 else rxn[:57] + "..."
        print(f"\n{display_rxn}")
        print(f"  Novelty: {score:.4f}")

    if len(reactions) > 1:
        print(f"\nMean novelty: {scores.mean():.4f}")
        print(f"Std novelty:  {scores.std():.4f}")

if __name__ == "__main__":
    main()
