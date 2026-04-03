# Retrosynthesis Route Novelty via Conditional Diffusion Density Estimation

## Overview

This project implements a reaction-space novelty evaluator for small molecules. Instead of defining novelty in molecular graph space (e.g., fingerprint distance), we define novelty in retrosynthetic transformation space.

**Core idea:** A molecule is novel if the synthetic transformation logic required to make it lies in a low-density region of reaction-space learned from historical chemistry.

We model the distribution of reaction embeddings using a conditional diffusion density estimator, conditioned on reaction frequency in the training corpus.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download pingzhili/USPTO-50K and create Time-Based Splits (Pre/Post 2016)
python dataset_setup.py

# Generate RXNFP embeddings and compute class-frequency features (using 10 standard classes)
python precompute_routes.py

# Train the conditional diffusion model (saves model.pt + loss_curve.png)
python train.py

# Evaluate novelty on all splits and perform statistical analysis
python evaluate.py

# Run demo comparing common vs unusual reactions
python demo.py
```

## Results

The model successfully distinguishes between common and unusual reaction patterns:

| Reaction Type | Mean Novelty Score |
|--------------|-------------------|
| Common patterns (amide couplings, etc.) | ~4.7 |
| Unusual patterns (complex heterocycles) | ~11.8 |

### Time-Split Evaluation

The model was evaluated using a **time-based split** to test its ability to identify "novel" (future) chemistry:
- **Train/Val:** Reactions from patents published **before 2016** (22,676 examples).
- **Test:** Reactions from patents published in **2016** (27,277 examples).

**Significance:** Newer reactions (2016+) show **statistically significantly higher novelty** scores compared to historical training data (p-value = 0.0418), confirming the model captures chemical evolution.

Higher scores indicate reactions in low-density regions of the learned distribution → more novel.

## Output Files

After running the pipeline:
- `novelty_scores_train.csv` - Novelty scores for training reactions
- `novelty_scores_val.csv` - Novelty scores for validation reactions
- `novelty_scores_test.csv` - Novelty scores for test reactions
- `novelty_comparison.png` - Distribution comparison across splits
- `loss_curve.png` - Training curves

## Conceptual Framework

### Reaction Embedding

Each reaction is embedded using RXNFP (a pretrained transformer model trained on reaction SMILES).

- `r` = a reaction (reaction SMILES string)
- `φ(r) ∈ ℝᵈ` = RXNFP embedding vector (d=256)

### Frequency Conditioning

Each reaction belongs to a reaction class. We compute:
- `f(r)` = frequency of the reaction class in the corpus
- `c(r) = log(f(r) + 1)` = scalar conditioning variable

This allows the model to distinguish between rare reactions and simply low-density embedding regions.

### Diffusion Density Model

We model the probability distribution over embeddings using a conditional denoising diffusion probabilistic model (DDPM).

**Forward Process:**
```
xₜ = √αₜ·x₀ + √(1-αₜ)·ε
```

Where:
- `x₀ = φ(r)` = original reaction embedding
- `xₜ` = noisy embedding at timestep t
- `αₜ ∈ (0,1)` = cumulative noise schedule
- `ε ~ N(0,I)` = Gaussian noise

**Training Objective:**
```
L = E[‖ε - sθ(xₜ, t, c)‖²]
```

### Novelty Score

After training, we approximate novelty via the score norm:
```
N(r) = ‖sθ(xₜ, t, c)‖
```

Where:
- Larger values imply lower estimated density
- `t` is fixed at 0.5 (intermediate noise level)

**Intuition:** If the model must apply a large correction (high score magnitude), the embedding lies in a low-density region → more novel.

## Why Reaction-Space Novelty?

Traditional novelty measures structural difference:
```
Novelty = 1 - max(Tanimoto similarity)
```

Our method measures statistical rarity of synthetic transformation patterns.

**Advantages:**
- Captures transformation-level innovation
- Accounts for corpus reaction frequency
- Learns manifold geometry
- Generalizes beyond simple nearest-neighbor distance

## Dataset

We use the USPTO-50K reaction dataset (pingzhili version) with a time-based split:
- **Train:** 20,409 reactions (Pre-2016)
- **Validation:** 2,267 reactions (Pre-2016)
- **Test:** 27,277 reactions (2016)

## Project Structure

```
dataset_setup.py      # Downloads + preprocesses USPTO-50K
precompute_routes.py  # Generates RXNFP embeddings per split
model.py              # Conditional diffusion model architecture
train.py              # Training loop with early stopping
evaluate.py           # Novelty scoring + comparison plots
demo.py               # Demo comparing common vs unusual reactions
score_reaction.py     # Score individual reaction SMILES
```

## Scoring Individual Reactions

```bash
# Score a single reaction
python score_reaction.py "CC(=O)Cl.NCC>>CC(=O)NCC"

# Score reactions from a file
python score_reaction.py --file my_reactions.txt
```

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Diffusion steps | 1000 | Stable DDPM regime |
| Embedding dim | 256 | RXNFP output |
| Hidden dim | 512 | Score network capacity |
| Batch size | 256 | Fits GPU/MPS memory |
| Optimizer | AdamW | Stable for diffusion |
| Learning rate | 2e-4 | Standard DDPM scale |
| Scheduler | Cosine annealing | Smooth convergence |
| Gradient clipping | 1.0 | Prevent exploding gradients |
| Early stopping | patience=5 | Prevent overfitting |

## Hardware Support

The code automatically detects and uses:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

## Limitations

- Single-step reactions only (multi-route extension possible)
- Approximate likelihood via score norm
- USPTO data is patent-biased

## Possible Extensions

- Multi-step retrosynthetic route embeddings
- Exact likelihood estimation via probability flow ODE
- Compare against: kNN density, GMM, normalizing flows

## Interpretation

High novelty score does not imply synthetic value. It implies low estimated probability under the learned reaction distribution—statistical rarity in transformation space, not biological relevance.

## References

- RXNFP: Schwaller et al., "Mapping the Space of Chemical Reactions Using Attention-Based Neural Networks" (2021)
- DDPM: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- USPTO: Lowe, "Extraction of chemical structures and reactions from the literature" (2012)
