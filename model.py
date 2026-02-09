import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) *
            -(math.log(10000) / (half_dim - 1))
        )
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class ConditionalScoreNet(nn.Module):
    def __init__(self, dim, cond_dim, hidden=512):
        super().__init__()
        self.time_mlp = SinusoidalPosEmb(64)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + 64, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t, cond):
        # t may be (batch, 1) or (batch,) - ensure it's (batch,) for time_mlp
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = self.time_mlp(t)
        x_in = torch.cat([x, cond, t_emb], dim=-1)
        return self.net(x_in)

class DiffusionModel:
    def __init__(self, dim, cond_dim, timesteps=1000):
        self.model = ConditionalScoreNet(dim, cond_dim)
        self.timesteps = timesteps

        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def loss(self, x0, cond):
        device = x0.device
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)

        # Index on CPU, then move result to device
        alpha_hat_t = self.alpha_hat[t.cpu()].unsqueeze(1).to(device)
        xt = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise

        t = t.float() / self.timesteps
        pred_noise = self.model(xt, t.unsqueeze(1), cond)
        return nn.functional.mse_loss(pred_noise, noise)

