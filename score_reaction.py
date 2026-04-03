"""
Score novelty of individual reaction SMILES or entire retrosynthetic routes.

Usage:
    python score_reaction.py "reactants>>product"
    python score_reaction.py --file reactions.txt
    python score_reaction.py --route route.json
"""

import torch
import numpy as np
import argparse
import json
import os
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
    if not os.path.exists("metrics.pt"):
        raise FileNotFoundError("metrics.pt not found. Please run train.py first.")
    
    metrics = torch.load("metrics.pt", map_location=device)
    mean = metrics["mean"]
    std = metrics["std"]

    # Load model
    if not os.path.exists("model.pt"):
        raise FileNotFoundError("model.pt not found. Please run train.py first.")
        
    model = ConditionalScoreNet(dim=256, cond_dim=1)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Load rxnfp
    rxnfp_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxnfp_model, tokenizer)

    # Load training class frequencies for conditioning
    if not os.path.exists("data/route_embeddings.pt"):
        raise FileNotFoundError("data/route_embeddings.pt not found. Please run precompute_routes.py first.")
        
    data = torch.load("data/route_embeddings.pt")
    train_freq = data["train"]["freq_features"]
    mean_freq = train_freq.mean().item()

    return model, mean, std, rxnfp_generator, mean_freq, device

def score_reactions(reactions, model, mean, std, rxnfp_generator, mean_freq, device):
    """Score a list of reaction SMILES."""
    if not reactions:
        return np.array([])
        
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
    parser.add_argument("--route", "-r", help="JSON file containing a retrosynthetic route/tree")
    args = parser.parse_args()

    if not args.reaction and not args.file and not args.route:
        parser.print_help()
        return

    print("Loading model...")
    model, mean, std, rxnfp_generator, mean_freq, device = load_model()

    is_route = False
    if args.route:
        from route_scorer import RouteScorer
        with open(args.route) as f:
            tree_data = json.load(f)
        
        scorer = RouteScorer()
        if isinstance(tree_data, list):
            results = scorer.score_route(tree_data)
            reactions = tree_data
        else:
            results = scorer.score_tree(tree_data)
            # Re-extract for display
            reactions = []
            def _ext(n):
                if isinstance(n, dict):
                    if n.get("is_reaction") or n.get("type") == "reaction":
                        if "smiles" in n: reactions.append(n["smiles"])
                    for c in n.get("children", []): _ext(c)
                elif isinstance(n, list):
                    for i in n: _ext(i)
            _ext(tree_data)
        
        scores = np.array(results["step_scores"])
        route_meta = results
        is_route = True
    elif args.file:
        with open(args.file) as f:
            reactions = [line.strip() for line in f if line.strip() and ">>" in line]
        scores = score_reactions(reactions, model, mean, std, rxnfp_generator, mean_freq, device)
    else:
        reactions = [args.reaction]
        scores = score_reactions(reactions, model, mean, std, rxnfp_generator, mean_freq, device)

    print("\n" + "="*70)
    print("NOVELTY SCORES")
    print("="*70)
    for i, (rxn, score) in enumerate(zip(reactions, scores)):
        label = f"Step {i+1}" if is_route else f"Reaction {i+1}"
        # Truncate long SMILES for display
        display_rxn = rxn if len(rxn) < 60 else rxn[:57] + "..."
        print(f"\n{label}: {display_rxn}")
        print(f"  Novelty: {score:.4f}")

    if is_route:
        print("\n" + "="*70)
        print("ROUTE SUMMARY")
        print("="*70)
        print(f"Number of steps: {route_meta['num_steps']}")
        print(f"Mean Novelty:    {route_meta['mean_novelty']:.4f}")
        print(f"Max Novelty:     {route_meta['max_novelty']:.4f} (Bottleneck)")
    elif len(reactions) > 1:
        print(f"\nMean novelty: {scores.mean():.4f}")
        print(f"Std novelty:  {scores.std():.4f}")

if __name__ == "__main__":
    main()
