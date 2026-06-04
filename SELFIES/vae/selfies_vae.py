"""
selfies_vae.py
==============
Transformer-based Variational Autoencoder (VAE) for SELFIES strings.

The VAE learns a continuous latent space over individual molecule SELFIES.
A compressed latent vector z (μ ± σ) represents each molecule.
For mixture property prediction, per-component latent vectors are mixed
by volume-weighted summation before feeding the CN predictor.

Architecture
------------
Encoder:
    Embedding(vocab_size, d_model)
    Positional Encoding
    TransformerEncoder (n_layers × n_heads)
    Pooling (mean over non-padding positions)
    Linear → [μ, log σ]   (each: latent_dim)

Decoder:
    Linear(latent_dim, d_model)
    Positional Encoding
    TransformerDecoder (n_layers × n_heads)  [teacher-forced during training]
    Linear(d_model, vocab_size) → token logits

Loss
----
    L = CrossEntropy(recon) + β · KL(N(μ,σ) ‖ N(0,1))
    β is annealed from 0 → 1 over `beta_anneal_epochs` epochs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SELFIESEncoder(nn.Module):
    """Map a padded SELFIES token sequence to (μ, log σ) in latent space.

    Parameters
    ----------
    vocab_size : int
    d_model    : int    Transformer hidden dimension.
    latent_dim : int    Size of z (μ and log σ each have this many elements).
    n_heads    : int    Number of attention heads.
    n_layers   : int    Number of Transformer encoder layers.
    d_ff       : int    Feed-forward sub-layer dimension.
    dropout    : float
    pad_idx    : int    Index of the <pad> token; masked from attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        latent_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.fc_mu     = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(
        self, src: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        src : (batch, seq_len)  integer token IDs

        Returns
        -------
        mu     : (batch, latent_dim)
        logvar : (batch, latent_dim)
        """
        pad_mask = (src == self.pad_idx)        # True where padded
        # Prevent completely masked sequences (e.g. absent components) from causing NaNs in transformer
        pad_mask = pad_mask.clone()
        pad_mask[:, 0] = False

        emb = self.embedding(src) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)                  # (batch, seq, d_model)

        enc_out = self.transformer(emb, src_key_padding_mask=pad_mask)  # (batch, seq, d_model)

        # Mean-pool over non-padding positions
        valid = (~pad_mask).unsqueeze(-1).float()          # (batch, seq, 1)
        pooled = (enc_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        # pooled: (batch, d_model)

        return self.fc_mu(pooled), self.fc_logvar(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class SELFIESDecoder(nn.Module):
    """Reconstruct a SELFIES token sequence from a latent vector z.

    Uses teacher-forcing during training: the ground-truth shifted sequence
    is passed as the decoder input (``tgt``).

    Parameters
    ----------
    vocab_size : int
    d_model    : int
    latent_dim : int
    n_heads    : int
    n_layers   : int
    d_ff       : int
    dropout    : float
    pad_idx    : int
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        latent_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Project latent z to a single "memory" token for cross-attention
        self.latent_proj = nn.Linear(latent_dim, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        z: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z   : (batch, latent_dim)
        tgt : (batch, tgt_len)  teacher-forced target sequence
              (typically src shifted right: BOS + tokens, without final EOS)

        Returns
        -------
        logits : (batch, tgt_len, vocab_size)
        """
        batch, tgt_len = tgt.shape

        # Memory: (batch, 1, d_model) — one latent "context" token
        memory = self.latent_proj(z).unsqueeze(1)

        # Decoder embedding
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)    # (batch, tgt_len, d_model)

        # Causal mask so each position only attends to previous positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=z.device
        )
        tgt_pad_mask = (tgt == self.pad_idx)  # (batch, tgt_len)

        dec_out = self.transformer(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=None,
        )  # (batch, tgt_len, d_model)

        return self.output_proj(dec_out)    # (batch, tgt_len, vocab_size)

    @torch.no_grad()
    def greedy_decode(
        self,
        z: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding (inference only).

        Parameters
        ----------
        z       : (batch, latent_dim)
        bos_idx : int
        eos_idx : int
        max_len : int

        Returns
        -------
        token_ids : (batch, max_len)  long tensor
        """
        batch = z.size(0)
        device = z.device

        # Start with BOS
        generated = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self(z, generated)              # (batch, cur_len, vocab)
            next_tok = logits[:, -1, :].argmax(-1)   # (batch,)
            next_tok = next_tok.masked_fill(finished, self.pad_idx)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok.eq(eos_idx)
            if finished.all():
                break

        # Pad to max_len if needed
        if generated.size(1) < max_len:
            pad = torch.full(
                (batch, max_len - generated.size(1)), self.pad_idx,
                dtype=torch.long, device=device
            )
            generated = torch.cat([generated, pad], dim=1)

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# Full VAE wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SELFIESVAE(nn.Module):
    """End-to-end VAE: Encoder + reparameterisation + Decoder.

    Convenience wrapper that wires the encoder and decoder together and
    exposes the standard VAE interface.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 128,
        latent_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        shared_kwargs = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_idx=pad_idx,
            max_len=max_len,
        )
        self.encoder = SELFIESEncoder(**shared_kwargs)
        self.decoder = SELFIESDecoder(**shared_kwargs)

        self.latent_dim = latent_dim
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε · exp(½ logvar),  ε ~ N(0, I)."""
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def forward(
        self, src: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass.

        Parameters
        ----------
        src : (batch, seq_len)  input token IDs (incl. BOS and EOS)

        Returns
        -------
        logits : (batch, seq_len-1, vocab_size)  reconstruction logits
        mu     : (batch, latent_dim)
        logvar : (batch, latent_dim)
        """
        mu, logvar = self.encoder(src)
        z = self.reparameterise(mu, logvar)

        # Teacher-forced decoder input: everything except the last token
        tgt = src[:, :-1]
        logits = self.decoder(z, tgt)

        return logits, mu, logvar

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode src → (μ, logvar) without decoding."""
        return self.encoder(src)

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Greedy-decode a latent vector to token IDs."""
        return self.decoder.greedy_decode(z, self.bos_idx, self.eos_idx, self.max_len)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helper
# ─────────────────────────────────────────────────────────────────────────────

def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    pad_idx: int = 0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ELBO loss = reconstruction loss + β · KL divergence.

    Parameters
    ----------
    logits  : (batch, tgt_len, vocab_size)
    targets : (batch, tgt_len)  ground-truth token IDs (shifted left: src[:, 1:])
    mu      : (batch, latent_dim)
    logvar  : (batch, latent_dim)
    beta    : float  weight on KL term
    pad_idx : int    padding index; excluded from CE loss
    reduction: 'mean' or 'sum'

    Returns
    -------
    total_loss : scalar
    recon_loss : scalar
    kl_loss    : scalar
    """
    batch, tgt_len, vocab = logits.shape

    recon = F.cross_entropy(
        logits.reshape(batch * tgt_len, vocab),
        targets.reshape(batch * tgt_len),
        ignore_index=pad_idx,
        reduction=reduction,
    )

    # KL divergence: -½ Σ (1 + logvar - μ² - exp(logvar))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    kl = kl.mean() if reduction == "mean" else kl.sum()

    total = recon + beta * kl
    return total, recon, kl


# ─────────────────────────────────────────────────────────────────────────────
# Beta annealing schedule
# ─────────────────────────────────────────────────────────────────────────────

def beta_schedule(epoch: int, anneal_epochs: int = 20) -> float:
    """Linear warm-up of β from 0 to 1 over *anneal_epochs* epochs."""
    return min(1.0, epoch / max(anneal_epochs, 1))
