import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import ConditionalScoreNet
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy import stats

# ---------- Device ----------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- Load Metrics (includes normalization stats) ----------
metrics = torch.load("metrics.pt")
train_losses = metrics["train_losses"]
val_losses = metrics["val_losses"]
mean = metrics["mean"]
std = metrics["std"]

# ---------- Plot Training Curves ----------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Val Loss", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("Training vs Validation Loss", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig("loss_curve.png", dpi=150, bbox_inches='tight')
print("Saved loss_curve.png")

# ---------- Load Embeddings ----------
data = torch.load("data/route_embeddings.pt")

# ---------- Load Model ----------
dim = data["train"]["embeddings"].shape[1]
model = ConditionalScoreNet(dim=dim, cond_dim=1)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

def compute_novelty_scores(embeddings, freq_features, smiles_list, split_name):
    """Compute novelty scores for a set of embeddings."""
    # Normalize using training statistics
    X = (embeddings - mean) / std
    C = freq_features

    dataset = TensorDataset(X, C)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    novelty_scores = []

    with torch.no_grad():
        for x, c in tqdm(loader, desc=f"Scoring {split_name}"):
            x, c = x.to(device), c.to(device)

            # Use fixed mid-level diffusion time
            t = torch.ones((x.size(0), 1), device=device) * 0.5

            score = model(x, t, c)
            novelty = torch.norm(score, dim=1)

            novelty_scores.extend(novelty.cpu().numpy())

    # Create DataFrame with results
    df = pd.DataFrame({
        'reaction_smiles': smiles_list,
        'novelty_score': novelty_scores
    })

    return df

# ---------- Score Each Split ----------
results = {}

for split in ["train", "val", "test"]:
    print(f"\nProcessing {split} set...")
    df = compute_novelty_scores(
        data[split]["embeddings"],
        data[split]["freq_features"],
        data[split]["smiles"],
        split
    )
    results[split] = df

    # Save CSV
    csv_path = f"novelty_scores_{split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df)} reactions)")

# ---------- Compare Distributions ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram comparison
ax1 = axes[0]
colors = {'train': '#2ecc71', 'val': '#3498db', 'test': '#e74c3c'}
labels = {'train': f'Train (n={len(results["train"])})',
          'val': f'Val (n={len(results["val"])})',
          'test': f'Test (n={len(results["test"])})'}

for split in ["train", "val", "test"]:
    scores = results[split]['novelty_score']
    ax1.hist(scores, bins=50, alpha=0.6, label=labels[split], color=colors[split], density=True)

ax1.set_xlabel("Novelty Score", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.set_title("Novelty Score Distribution by Split", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Box plot comparison
ax2 = axes[1]
box_data = [results[split]['novelty_score'].values for split in ["train", "val", "test"]]
bp = ax2.boxplot(box_data, labels=["Train", "Val", "Test"], patch_artist=True)

for patch, split in zip(bp['boxes'], ["train", "val", "test"]):
    patch.set_facecolor(colors[split])
    patch.set_alpha(0.6)

ax2.set_ylabel("Novelty Score", fontsize=12)
ax2.set_title("Novelty Score Comparison", fontsize=14)
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("novelty_comparison.png", dpi=150, bbox_inches='tight')
print("\nSaved novelty_comparison.png")

# ---------- Statistical Summary ----------
print("\n" + "="*60)
print("NOVELTY SCORE STATISTICS")
print("="*60)

for split in ["train", "val", "test"]:
    scores = results[split]['novelty_score']
    print(f"\n{split.upper()}:")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Median: {scores.median():.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")

# Statistical tests
print("\n" + "-"*60)
print("STATISTICAL TESTS (Mann-Whitney U)")
print("-"*60)

train_scores = results["train"]['novelty_score'].values
test_scores = results["test"]['novelty_score'].values

stat, pval = stats.mannwhitneyu(test_scores, train_scores, alternative='greater')
print(f"\nTest vs Train (test > train):")
print(f"  U-statistic: {stat:.0f}")
print(f"  p-value: {pval:.2e}")
if pval < 0.05:
    print(f"  Result: Test reactions have SIGNIFICANTLY HIGHER novelty scores")
else:
    print(f"  Result: No significant difference")

# Effect size + Bootstrap CI
def _rank_biserial(a, b):
    """Rank-biserial correlation: +1 means a always > b, -1 means never."""
    s, _ = stats.mannwhitneyu(a, b, alternative='greater')
    return (2 * s) / (len(a) * len(b)) - 1

n1, n2 = len(test_scores), len(train_scores)
r = _rank_biserial(test_scores, train_scores)
print(f"  Effect size (r): {abs(r):.3f}")

rng = np.random.default_rng(42)
boot_r = [
    _rank_biserial(
        rng.choice(test_scores, size=n1, replace=True),
        rng.choice(train_scores, size=n2, replace=True)
    )
    for _ in range(2000)
]
ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
print(f"  Bootstrap 95% CI for r: [{ci_lo:.3f}, {ci_hi:.3f}]")
if ci_lo > 0:
    print("  -> CI excludes zero: effect is consistently positive.")
else:
    print("  -> CI crosses zero: effect may not replicate. Interpret p-value with caution.")

pooled_std = np.sqrt(
    ((n1 - 1) * test_scores.std()**2 + (n2 - 1) * train_scores.std()**2) / (n1 + n2 - 2)
)
d = (test_scores.mean() - train_scores.mean()) / pooled_std
interp = "small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large")
print(f"  Cohen's d: {d:.3f} ({interp})")

# ---- k-NN BASELINE ----
# Critical check: if k-NN distance in RXNFP space separates train/test equally well,
# the diffusion model adds no value over a simple lookup table.
print("\n" + "-"*60)
print("k-NN BASELINE: Does diffusion add value over nearest-neighbor?")
print("-"*60)

try:
    from sklearn.neighbors import NearestNeighbors as _KNN

    _tr = ((data["train"]["embeddings"] - mean) / std).numpy()
    _te = ((data["test"]["embeddings"] - mean) / std).numpy()

    # LOO distance for train: k=2 to skip self (dist~0)
    _knn2 = _KNN(n_neighbors=2, metric="euclidean").fit(_tr)
    _tr_knn = _knn2.kneighbors(_tr)[0][:, 1]

    # Nearest training neighbor for test (k=1)
    _knn1 = _KNN(n_neighbors=1, metric="euclidean").fit(_tr)
    _te_knn = _knn1.kneighbors(_te)[0][:, 0]

    _, _knn_p = stats.mannwhitneyu(_te_knn, _tr_knn, alternative='greater')
    _knn_r = _rank_biserial(_te_knn, _tr_knn)

    print(f"\n  {'Method':<26} {'p-value':>12}  {'|r|':>6}")
    print(f"  {'Diffusion model':<26} {pval:>12.4e}  {abs(r):>6.3f}")
    print(f"  {'k-NN (L2, k=1) baseline':<26} {_knn_p:>12.4e}  {abs(_knn_r):>6.3f}")

    margin = abs(r) - abs(_knn_r)
    if margin > 0.01:
        print(f"\n  Diffusion outperforms k-NN by delta|r|={margin:.3f}.")
        print("  The model captures structure beyond nearest-neighbor distance.")
    elif margin > -0.01:
        print(f"\n  Diffusion and k-NN perform similarly (delta|r|={margin:.3f}).")
        print("  Unclear whether diffusion adds value; consider richer conditioning.")
    else:
        print(f"\n  k-NN is stronger by delta|r|={abs(margin):.3f}.")
        print("  The diffusion model underperforms simple distance - investigate why.")

except ImportError:
    print("  scikit-learn not installed; skipping. Run: pip install scikit-learn")

# ---- MULTI-t SWEEP ----
# Justifies (or challenges) the fixed t=0.5 choice for novelty scoring.
print("\n" + "-"*60)
print("MULTI-t SWEEP: Is t=0.5 the best diffusion timestep?")
print("-"*60)

def _score_at_t(embs, feats, t_val, batch_size=512):
    X = (embs - mean) / std
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        cb = feats[i:i + batch_size].to(device)
        tb = torch.ones((xb.size(0), 1), device=device) * t_val
        with torch.no_grad():
            out.append(torch.norm(model(xb, tb, cb), dim=1).cpu().numpy())
    return np.concatenate(out)

_t_results = []
for _t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    _tr_s = _score_at_t(data["train"]["embeddings"], data["train"]["freq_features"], _t)
    _te_s = _score_at_t(data["test"]["embeddings"], data["test"]["freq_features"], _t)
    _, _tp = stats.mannwhitneyu(_te_s, _tr_s, alternative='greater')
    _t_results.append((_t, _tp, abs(_rank_biserial(_te_s, _tr_s)), _tr_s.mean(), _te_s.mean()))

_best_idx = max(range(len(_t_results)), key=lambda i: _t_results[i][2])
print(f"\n  {'t':>5}  {'p-value':>12}  {'|r|':>6}  {'train mean':>10}  {'test mean':>10}")
for i, (_t, _tp, _tr_val, _trm, _tem) in enumerate(_t_results):
    tag = " <- current" if _t == 0.5 else (" <- best" if i == _best_idx and _t != 0.5 else "")
    print(f"  {_t:>5.1f}  {_tp:>12.4e}  {_tr_val:>6.3f}  {_trm:>10.4f}  {_tem:>10.4f}{tag}")

_best_t = _t_results[_best_idx][0]
if _best_t != 0.5:
    _best_r = _t_results[_best_idx][2]
    _cur_r = _t_results[4][2]
    print(f"\n  t={_best_t:.1f} shows stronger effect (|r|={_best_r:.3f} vs {_cur_r:.3f} at t=0.5).")
    print(f"  If stable across seeds, update the default timestep in score_reaction.py.")
else:
    print(f"\n  t=0.5 is optimal - current default is well-justified.")

print("\n" + "="*60)
print("Evaluation complete.")
