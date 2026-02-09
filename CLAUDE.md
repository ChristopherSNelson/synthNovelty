# Project Context for Claude

## What This Is

A reaction-space novelty evaluator that scores chemical reactions based on how unusual their transformation patterns are. Uses conditional diffusion (DDPM) to model the density of RXNFP embeddings from USPTO-50K reactions.

**Key insight:** Novelty is measured in transformation space, not molecular structure space.

## Current State

- **Working pipeline:** `dataset_setup.py` → `precompute_routes.py` → `train.py` → `evaluate.py`
- **Demo works:** `python demo.py` shows common reactions score ~4, unusual ones score ~11
- **Outputs:** CSV files with SMILES + novelty scores, comparison plots
- **Hardware:** Supports CUDA, MPS (Apple Silicon), CPU

## Architecture

- `model.py`: `ConditionalScoreNet` (sinusoidal time embedding + 3-layer MLP) wrapped by `DiffusionModel`
- Embeddings: 256-dim RXNFP vectors
- Conditioning: log(frequency + 1) of crude reaction class (first 10 chars of SMILES)
- Novelty score: L2 norm of predicted noise at t=0.5

## Known Limitations

1. **Random splits don't show novelty difference** - Train/val/test are randomly sampled from same distribution, so no significant difference in novelty scores between splits. Time-based splits would be more meaningful.

2. **Crude reaction class** - Using first 10 chars of SMILES is a rough approximation. Could use actual reaction templates or atom-mapping based classification.

3. **Single-step only** - Doesn't handle multi-step retrosynthetic routes.

## Possible Next Steps

### High Impact
- **Time-split evaluation**: Train on pre-2015 reactions, test on post-2015. Should show test reactions have higher novelty. Requires dataset with timestamps.
- **Score real novel drugs**: Get SMILES for recently approved drugs (2023+), synthesize their reaction routes, score against older drugs.

### Medium Impact
- **Better reaction classes**: Use reaction templates from RDChiral or atom-mapping instead of crude string prefix.
- **Visualization**: t-SNE/UMAP of embeddings colored by novelty score.
- **Baseline comparisons**: Compare against kNN density, GMM, isolation forest.

### Lower Priority
- Exact likelihood via probability flow ODE instead of score norm approximation
- Web interface for scoring arbitrary reactions

### Reach Goal
- **Multi-step retrosynthetic routes**: Instead of scoring individual reactions, score entire synthesis trees. Would require: (1) retrosynthesis prediction model (e.g., from ASKCOS or similar), (2) route embedding strategy (aggregate/pool step embeddings, or graph neural network over route tree), (3) new training data with full routes. This would measure novelty of synthesis strategies, not just individual transformations.

## File Overview

| File | Purpose |
|------|---------|
| `dataset_setup.py` | Downloads USPTO-50K from HuggingFace |
| `precompute_routes.py` | Generates RXNFP embeddings, saves per-split |
| `model.py` | Diffusion model architecture |
| `train.py` | Training loop, saves model.pt + metrics.pt |
| `evaluate.py` | Scores all splits, outputs CSVs + plots |
| `demo.py` | Quick demo of common vs unusual reactions |
| `score_reaction.py` | CLI to score individual reactions |

## Commands

```bash
# Full pipeline
python dataset_setup.py && python precompute_routes.py && python train.py && python evaluate.py

# Quick demo
python demo.py

# Score a reaction
python score_reaction.py "CCBr.NC>>CCNC"
```

## Dependencies Note

`rxnfp` has compatibility issues with newer transformers. We import only `RXNBERTFingerprintGenerator` and `get_default_model_and_tokenizer` from `rxnfp.transformer_fingerprints` - avoid importing from `rxnfp.models`.
