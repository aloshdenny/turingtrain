"""
mixture_cn_predictor.py
=======================
Forward property model: predicts cetane number (CN) from a mixture's
volume-weighted mean latent vector.

The mixture latent vector is:
    z_mix = Σᵢ vᵢ · μᵢ
where μᵢ is the posterior mean for component i and vᵢ is its volume fraction.

This formulation preserves the linear blending intuition of ideal mixtures while
operating entirely in a continuous latent space learned by the VAE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNPredictor(nn.Module):
    """MLP that maps a mixture latent vector to a scalar CN prediction.

    Parameters
    ----------
    latent_dim : int   Dimensionality of z_mix (must match VAE latent_dim).
    hidden_dims : tuple[int, ...]  Hidden layer widths.
    dropout : float
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_mix: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_mix : (batch, latent_dim)

        Returns
        -------
        cn_pred : (batch,)
        """
        return self.net(z_mix).squeeze(-1)


class MixtureCNModel(nn.Module):
    """Wraps the VAE encoder + CN predictor for end-to-end mixture CN inference.

    During Stage 1 (VAE pre-training), this class is NOT used — the VAE is
    trained on individual molecules in unsupervised fashion.

    During Stage 2 (predictor training), the VAE encoder weights are FROZEN
    and only the predictor is trained.

    During Stage 3 (fine-tuning), both encoder and predictor weights are
    updated jointly with a combined loss.

    Parameters
    ----------
    vae_encoder : SELFIESEncoder  (from selfies_vae.py)
    predictor   : CNPredictor
    n_components: int  Maximum number of mixture components (default 10).
    """

    def __init__(
        self,
        vae_encoder: nn.Module,
        predictor: CNPredictor,
        n_components: int = 10,
    ) -> None:
        super().__init__()
        self.encoder = vae_encoder
        self.predictor = predictor
        self.n_components = n_components

    def forward(
        self,
        component_tokens: torch.Tensor,
        volume_fractions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        component_tokens : (batch, n_components, seq_len)  padded token IDs.
            Absent components should be fully-padded rows (all pad_idx).
        volume_fractions : (batch, n_components)  normalised volume fractions
            (sum to 1 per row; 0 for absent components).

        Returns
        -------
        cn_pred  : (batch,)           predicted cetane numbers
        z_mix    : (batch, latent_dim) mixture latent vector (for introspection)
        """
        batch, n_comp, seq_len = component_tokens.shape

        # Reshape to (batch * n_comp, seq_len) for batched encoding
        flat_tokens = component_tokens.reshape(batch * n_comp, seq_len)
        mu, _ = self.encoder(flat_tokens)             # (batch*n_comp, latent_dim)

        # Reshape back and blend
        mu = mu.view(batch, n_comp, -1)               # (batch, n_comp, latent_dim)
        vf = volume_fractions.unsqueeze(-1)            # (batch, n_comp, 1)
        z_mix = (mu * vf).sum(dim=1)                  # (batch, latent_dim)

        cn_pred = self.predictor(z_mix)
        return cn_pred, z_mix


# ─────────────────────────────────────────────────────────────────────────────
# Weighted MSE loss with high-CN up-weighting
# ─────────────────────────────────────────────────────────────────────────────

def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    high_cn_threshold: float = 80.0,
    high_cn_weight: float = 5.0,
) -> torch.Tensor:
    """MSE loss with optional up-weighting for high-CN samples.

    Samples with ``target > high_cn_threshold`` receive *high_cn_weight* times
    the standard weight (1.0).  This directly targets the sparse, hard-to-fit
    high cetane number regime.

    Parameters
    ----------
    pred               : (batch,)
    target             : (batch,)
    high_cn_threshold  : float    CN above which the higher weight is applied.
    high_cn_weight     : float    Multiplier for high-CN samples.

    Returns
    -------
    loss : scalar
    """
    weights = torch.where(
        target > high_cn_threshold,
        torch.full_like(target, high_cn_weight),
        torch.ones_like(target),
    )
    sq_err = F.mse_loss(pred, target, reduction="none")
    return (weights * sq_err).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combined (joint) loss for Stage 3 fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def joint_loss(
    recon_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    cn_loss: torch.Tensor,
    *,
    lambda_pred: float = 10.0,
    lambda_vae: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Combine VAE ELBO and CN prediction loss for Stage 3.

    L = λ_pred · MSE(CN) + λ_vae · (Recon + β · KL)

    The default λ_pred=10 prioritises CN accuracy during fine-tuning while
    still maintaining the latent structure learned by the VAE.
    """
    return lambda_pred * cn_loss + lambda_vae * (recon_loss + beta * kl_loss)
