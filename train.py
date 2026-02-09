import torch
from torch.utils.data import DataLoader, TensorDataset
from model import DiffusionModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Load Data ----------
data = torch.load("data/route_embeddings.pt")

train_X = data["train"]["embeddings"]
train_C = data["train"]["freq_features"]
val_X = data["val"]["embeddings"]
val_C = data["val"]["freq_features"]

print(f"Train: {train_X.shape[0]}, Val: {val_X.shape[0]}")

# ---------- Normalize Embeddings (TRAIN SET ONLY) ----------
mean = train_X.mean(dim=0)
std = train_X.std(dim=0) + 1e-6

train_X = (train_X - mean) / std
val_X = (val_X - mean) / std

# Build dataloaders
train_dataset = TensorDataset(train_X, train_C)
val_dataset = TensorDataset(val_X, val_C)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# ---------- Device ----------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- Model ----------
model = DiffusionModel(dim=train_X.shape[1], cond_dim=1)
model.model.to(device)

optimizer = torch.optim.AdamW(model.model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

epochs = 50

train_losses = []
val_losses = []

best_val_loss = float("inf")
patience = 5
patience_counter = 0

# ---------- Training Loop ----------
for epoch in range(epochs):
    model.model.train()
    train_loss = 0

    for x, c in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        x, c = x.to(device), c.to(device)
        loss = model.loss(x, c)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ---------- Validation ----------
    model.model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, c in val_loader:
            x, c = x.to(device), c.to(device)
            loss = model.loss(x, c)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step()

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # ---------- Early Stopping ----------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.model.state_dict(), "model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# ---------- Save Metrics + Normalization ----------
torch.save({
    "train_losses": train_losses,
    "val_losses": val_losses,
    "mean": mean,
    "std": std
}, "metrics.pt")

# ---------- Plot ----------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Diffusion Training Curve")
plt.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches='tight')

print("Training complete. Model + metrics saved.")
