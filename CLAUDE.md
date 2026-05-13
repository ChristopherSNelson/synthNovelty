# Project Context for Claude

## What This Is

A reaction-space novelty evaluator that scores chemical reactions and multi-step retrosynthetic routes based on how unusual their transformation patterns are. Uses conditional diffusion (DDPM) to model the density of RXNFP embeddings from USPTO-50K reactions.

**Key insight:** Novelty is measured in transformation space, not molecular structure space. Strategic innovation is captured via multi-step route scoring.

## Current State

- **Working pipeline:** `dataset_setup.py` → `precompute_routes.py` → `train.py` → `evaluate.py`
- **Time-split Evaluation:** Trained on Pre-2016, tested on 2016 data. Test set shows significantly higher novelty (p=0.0418).
- **Multi-step Routes:** `route_scorer.py` supports scoring entire synthesis trees/routes via JSON.
- **Demos:** `demo.py` (single-step) and `route_demo.py` (multi-step) show clear differentiation.
- **Hardware:** Supports CUDA, MPS (Apple Silicon), CPU.

## Architecture

- `model.py`: `ConditionalScoreNet` (sinusoidal time embedding + 3-layer MLP) wrapped by `DiffusionModel`.
- Embeddings: 256-dim RXNFP vectors.
- Conditioning: log(frequency + 1) of 10 standard Schneider reaction classes.
- Novelty score: L2 norm of predicted noise at t=0.5.
- Route metrics: Mean and Max (bottleneck) novelty across steps.

## Recent Improvements

1.  **Time-split Dataset:** Transitioned to `pingzhili/uspto-50k` with patent IDs; created 2016 cutoff split.
2.  **Standard Classes:** Replaced crude SMILES prefix hack with 10 standard reaction classes.
3.  **Route Evaluation:** Added framework for scoring retrosynthesis trees from JSON.

## File Overview

| File | Purpose |
|------|---------|
| `dataset_setup.py` | Downloads USPTO-50K and creates Time-Split |
| `precompute_routes.py` | Generates RXNFP embeddings and class features |
| `model.py` | Diffusion model architecture |
| `train.py` | Training loop, saves model.pt + metrics.pt |
| `evaluate.py` | Scores all splits, time-split statistical analysis |
| `route_scorer.py` | Logic for multi-step synthesis route scoring |
| `demo.py` / `route_demo.py` | Single-step and multi-step novelty demos |
| `score_reaction.py` | CLI for reactions, files, and routes |

## Known Limitations

- **Inference conditioning bug**: all new reactions receive `mean_freq` conditioning, making the conditioning branch useless at inference. Fix requires a reaction classifier at scoring time.
- **demo.py is broken**: references `example_reactions.txt` which does not exist in the repo.
- **Score formula**: `||predicted_noise||_2` is an empirical proxy, not a proper likelihood estimate.

## Mistakes Log

- Do not claim the conditioning is "class-aware" at inference - it always uses `mean_freq`.
- Do not assume `demo.py` runs without `example_reactions.txt`.

## Commands

```bash
# Full pipeline
python dataset_setup.py && python precompute_routes.py && python train.py && python evaluate.py

# Evaluation (now includes k-NN baseline, effect size CI, multi-t sweep)
python evaluate.py

# Face-validity benchmark
python benchmark.py

# CLI Usage (single t or multi-t uncertainty)
python score_reaction.py "CCBr.NC>>CCNC"
python score_reaction.py --timesteps 0.3,0.5,0.7 "CCBr.NC>>CCNC"
python score_reaction.py --route route.json

# REST API (requires: pip install fastapi uvicorn)
uvicorn api:app --reload --port 8000
```
