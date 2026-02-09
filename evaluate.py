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

# Effect size (rank-biserial correlation)
n1, n2 = len(test_scores), len(train_scores)
effect_size = 1 - (2 * stat) / (n1 * n2)
print(f"  Effect size (r): {abs(effect_size):.3f}")

print("\n" + "="*60)
print("Evaluation complete.")
