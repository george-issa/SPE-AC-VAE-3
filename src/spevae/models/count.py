r"""Posterior over the number of poles, :math:`p(n \mid G, \sigma, \beta)`.

The flow posterior carries the pole count in a presence channel, and it does
not learn it: measured on held-out pole sets, 9% of its samples have the true
count whatever that count is, and the reported bands are a mixture over
counts that is 20--25% wider than the samples of any one count.  Within the
true count the same samples are close to calibrated.

A classifier is the direct estimator of that marginal: cross-entropy is a
proper scoring rule, so trained on the prior it converges on the true
:math:`p(n \mid G)` rather than on a blurred velocity field.  It is used to
reweight flow samples by ``p(n) / q(n)``, with ``q`` the samples' own count
histogram (:func:`count_weights`).
"""

from __future__ import annotations

import torch
from torch import nn

from .constraints import spectral_norm_from_greens
from .encoder import GreensEncoder

__all__ = ["CountPosterior", "count_weights", "resample_by_count", "save_count", "load_count"]


class CountPosterior(nn.Module):
    """Classifier of the pole count on the encoder's summary token."""

    def __init__(self, num_poles: int = 12, d_model: int = 96, enc_layers: int = 2,
                 n_heads: int = 8, d_ff: int = 512, n_matsubara: int = 24):
        super().__init__()
        self.config = dict(num_poles=num_poles, d_model=d_model, enc_layers=enc_layers,
                           n_heads=n_heads, d_ff=d_ff, n_matsubara=n_matsubara)
        self.P = int(num_poles)
        self.encoder = GreensEncoder(latent_dim=2, d_model=d_model, n_layers=enc_layers,
                                     n_heads=n_heads, d_ff=d_ff, n_matsubara=n_matsubara)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, self.P))
        self.meta: dict = {}

    def logits(self, G: torch.Tensor, x: torch.Tensor, beta: float,
               sigma: torch.Tensor | None = None, mask: torch.Tensor | None = None,
               includes_endpoints: bool = True) -> torch.Tensor:
        """``(B, P)`` logits; class ``k`` is a spectrum of ``k + 1`` poles."""
        norm = spectral_norm_from_greens(G, includes_endpoints)
        denom = torch.where(norm.abs() > 1e-12, norm, torch.ones_like(norm)).reshape(-1, 1)
        s_hat = None if sigma is None else sigma / denom.abs()
        h, _ = self.encoder.tokens(G / denom, x, sigma_hat=s_hat, beta=beta, mask=mask)
        return self.head(h[:, 0])

    def loss(self, batch) -> torch.Tensor:
        n = batch.poles_true["n_true"].to(batch.G.device).long().clamp(1, self.P) - 1
        return nn.functional.cross_entropy(
            self.logits(batch.G, batch.x, batch.beta, sigma=batch.sigma), n)

    @torch.no_grad()
    def probs(self, G, x, beta, sigma=None, **kw) -> torch.Tensor:
        """``(B, P)``: probability of 1, 2, ..., P poles."""
        G = torch.as_tensor(G, dtype=torch.float32).reshape(-1, torch.as_tensor(x).numel())
        if sigma is not None:
            sigma = torch.as_tensor(sigma, dtype=torch.float32).reshape(G.shape)
        return torch.softmax(self.logits(G, torch.as_tensor(x, dtype=torch.float32), float(beta),
                                         sigma=sigma, **kw), -1)


def count_weights(p_n: torch.Tensor, counts: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """Importance weights that give a set of samples the count marginal ``p_n``.

    ``p_n`` is ``(P,)`` (probability of 1..P poles), ``counts`` ``(S,)`` the
    samples' own counts.  A sample of count ``n`` gets ``p_n[n] / q[n]`` with
    ``q`` the fraction of samples at that count, so the weighted histogram of
    counts equals ``p_n`` restricted to the counts that occur (and
    renormalised).  ``floor`` mixes ``p_n`` with the uniform distribution
    first, to bound the weights.  Returns weights summing to one.
    """
    P = p_n.numel()
    p = (1.0 - floor) * p_n.double() + floor / P
    c = counts.long().clamp(1, P) - 1
    q = torch.bincount(c, minlength=P).double() / max(c.numel(), 1)
    w = p[c] / q[c]
    return w / w.sum()


def resample_by_count(p_n: torch.Tensor, counts: torch.Tensor,
                      generator: torch.Generator | None = None, floor: float = 0.02):
    """Indices that turn a set of samples into one whose counts follow ``p_n``.

    Systematic resampling with the weights of :func:`count_weights`; the
    result is unweighted again and as large as the input.  Returns
    ``(take, diagnostics)``: the effective sample size, the number of
    distinct samples kept, and the share of ``p_n`` on counts that occur
    among the samples at all (what is missing cannot be recovered by
    reweighting -- draw more samples).
    """
    w = count_weights(p_n, counts, floor)
    n = int(w.numel())
    u0 = float(torch.rand(1, generator=generator))
    pos = (u0 + torch.arange(n, dtype=torch.float64)) / n
    take = torch.searchsorted(torch.cumsum(w, 0), pos).clamp_max(n - 1)
    P = p_n.numel()
    present = torch.zeros(P, dtype=torch.bool)
    present[counts.long().clamp(1, P) - 1] = True
    return take, {"count_posterior": [float(v) for v in p_n],
                  "count_mass_covered": float(p_n[present].sum()),
                  "count_ess": float(1.0 / (w * w).sum()),
                  "count_distinct_samples": int(torch.unique(take).numel())}


def save_count(model: CountPosterior, path: str, **meta) -> str:
    torch.save({"format": "count-1", "config": dict(model.config),
                "state_dict": model.state_dict(), "meta": meta}, path)
    return path


def load_count(path: str, map_location="cpu") -> CountPosterior:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("format") != "count-1":
        raise ValueError(f"{path} is not a count-posterior checkpoint")
    m = CountPosterior(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.meta = ck.get("meta", {})
    return m.eval()
