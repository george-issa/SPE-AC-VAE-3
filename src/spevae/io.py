r"""Saving and loading trained models.

A ``state_dict`` is not a model.  Rebuilding one needs the pole count, the
latent width, the layer widths, the residue mode and the frequency cutoff, and
getting any of them wrong either fails with a shape error or -- worse -- loads
silently into a differently-shaped architecture.  So a checkpoint here carries
the constructor arguments alongside the weights, and :func:`load_model` needs
nothing but the path.

The format
----------
``{"format": 1, "spevae_version": ..., "config": {...}, "state_dict": {...},
"meta": {...}}``.  ``meta`` is free-form (training metrics, run label, git
revision) and is returned by :func:`load_checkpoint` for provenance -- a
continuation that ends up in a paper should be traceable to the run that
produced the weights.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from . import __version__
from .models.vae import SPEVAE

__all__ = ["save_model", "load_model", "load_checkpoint"]

_FORMAT = 1


def save_model(model: SPEVAE, path: str, **meta: Any) -> str:
    """Write ``model`` plus everything needed to rebuild it.

    Extra keyword arguments are stored under ``meta`` (metrics, config, a run
    label -- anything worth keeping with the weights).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        {
            "format": _FORMAT,
            "spevae_version": __version__,
            "config": dict(model.config),
            "state_dict": model.state_dict(),
            "meta": meta,
        },
        path,
    )
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Read a checkpoint without instantiating anything."""
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model(
    path: str,
    map_location="cpu",
    strict: bool = True,
    **overrides: Any,
) -> SPEVAE:
    """Rebuild a trained :class:`~spevae.models.vae.SPEVAE` from a checkpoint.

    Parameters
    ----------
    path
        Checkpoint written by :func:`save_model`.
    overrides
        Constructor arguments to change on load.  Only for settings that do not
        affect the weights -- ``cache_limit``, ``max_quad_nodes``,
        ``quad_nodes_per_width`` (a *finer* quadrature is always safe).  Changing
        a width or the pole count will fail to load, which is the intended
        behaviour.

    Returns
    -------
    SPEVAE
        In ``eval()`` mode, since the overwhelmingly common reason to load a
        checkpoint is to continue data with it.
    """
    ckpt = load_checkpoint(path, map_location)
    if "state_dict" not in ckpt:
        raise ValueError(f"{path} does not look like a spevae checkpoint")

    cfg = dict(ckpt.get("config") or {})
    if "format" not in ckpt:
        cfg = _upgrade_legacy_config(cfg, path)
    cfg.update(overrides)

    model = SPEVAE(**cfg)
    model.load_state_dict(ckpt["state_dict"], strict=strict)
    model.eval()
    return model


def _upgrade_legacy_config(cfg: dict, path: str) -> dict:
    """Map a pre-format-1 checkpoint's argparse dump onto constructor arguments.

    Early checkpoints stored ``vars(args)`` from the training script, which
    recorded only the knobs that script exposed and left the rest at values
    hardcoded there.  Those defaults are reproduced here so old weights stay
    loadable; new checkpoints carry the full config and do not come through
    this path.
    """
    known = {"latent_dim", "num_poles", "d_model"}
    legacy = {k: v for k, v in cfg.items() if k in known}
    if not legacy:
        raise ValueError(
            f"{path} is an unversioned checkpoint with no recognisable config; "
            f"rebuild the model by hand and load its state_dict directly.")
    return {
        **legacy,
        "omega_c": 20.0,
        "enc_layers": 3, "dec_layers": 2, "n_heads": 4, "d_ff": 256,
        "n_matsubara": 16,
    }
