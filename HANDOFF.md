# Session Handoff

## What was done this session

- Added MIT License and pushed to GitHub
- **`evaluate.py`**: Added k-NN baseline comparison (critical scientific validity check), bootstrap 95% CI for rank-biserial effect size, Cohen's d, and a multi-t sweep (t=0.1 to 0.9) to test whether t=0.5 is the optimal scoring timestep
- **`score_reaction.py`**: Added `score_reactions_multit()` function and `--timesteps` CLI flag for multi-t uncertainty estimation (averages scores across timesteps, reports std as heuristic uncertainty)
- **`benchmark.py`** (new): Curated face-validity benchmark with 10 reactions spanning routine, uncommon, and novel categories; includes honest caveats about conditioning limitations
- **`api.py`** (new): FastAPI REST server with `/health`, `/score`, and `/score_route` endpoints; model loaded at startup via lifespan event
- **`README.md`**: Added ASCII retrosynthesis route diagram with per-step novelty scores showing what multi-step scoring looks like in practice; updated project structure table

## What's left / next steps (priority order)

1. **Run evaluate.py and check k-NN baseline result** - This is the most important outstanding question. If k-NN |r| >= diffusion |r|, the diffusion model is not adding value over a lookup table. If true, need to investigate why (conditioning, architecture, score formula).
2. **Run benchmark.py** - Check face validity. If routine reactions don't score lower than novel ones, the mean_freq conditioning at inference is likely the culprit (see Known Limitations below).
3. **Fix the inference-time conditioning bug** - Currently all scored reactions receive `mean_freq` as conditioning, making the conditioning branch useless at inference. Fix: use a reaction classifier (Schneider classifier or RXNFP-based) to assign the class at inference time, then look up the correct log-frequency. This is the highest-priority model improvement.
4. **Fix demo.py** - It references `example_reactions.txt` which does not exist in the repo. The file needs to be created or the demo needs to be rewritten to use inline reactions.
5. **Multiple training seeds** - p=0.0418 is borderline. Run train.py 3-5 times with different seeds and report mean/std p-value to assess replicability.
6. **One-hot conditioning** - Replacing the scalar log-freq conditioning with a 10-dim one-hot Schneider class embedding would preserve class identity. Requires changing `precompute_routes.py` (generate one-hot instead of scalar), `model.py` (`cond_dim=10`), and retraining.

## Key decisions made and why

- **k-NN baseline added to evaluate.py**: Without it, there's no way to know if diffusion modeling adds anything over nearest-neighbor distance. This is a critical scientific validity check that was missing.
- **Bootstrap CI over p-value CI**: Bootstrap on rank-biserial correlation is more informative than a CI for the p-value (which has less intuitive interpretation).
- **Multi-t ensemble as "uncertainty"**: Honest in docstring that this is a heuristic, not Bayesian uncertainty. The std across timesteps reflects score sensitivity to the t choice.
- **Did NOT change the core score formula** (`||model(x_t, t, c)||_2`): Theoretically impure (not a proper density estimate) but empirically validated. Changing it would require retraining and re-evaluation.
- **Did NOT implement one-hot conditioning yet**: Requires full pipeline rerun. Code is clean enough to add as next step without breaking existing behavior.

## Current blockers

- **Cannot run evaluate.py, benchmark.py, or train.py** without the precomputed data files (`data/route_embeddings.pt`, `model.pt`, `metrics.pt`). These are not in git (large files).
- **demo.py is broken**: references `example_reactions.txt` which does not exist.
- **api.py requires `pip install fastapi uvicorn`** - not in any requirements file.

## Known limitations (worth noting in any writeup)

- At inference time, all reactions receive `mean_freq` conditioning - the model is effectively unconditioned on reaction class for new reactions.
- The novelty score `||predicted_noise||_2` is not a proper likelihood estimate. It's an empirical proxy that happens to work (p<0.05) but lacks theoretical grounding.
- USPTO data is patent-biased; reactions from academic literature may behave differently.
