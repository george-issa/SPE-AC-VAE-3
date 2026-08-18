"""End-to-end CLI: the exact path a new user walks.

Runs ``spevae continue`` / ``finetune`` / ``info`` through ``cli.main(argv)``
on a temp-dir dataset generated from a tiny model, checking the artefacts on
disk rather than the console text -- the .npz with the pole set is the
contract.
"""

import json

import numpy as np
import pytest
import torch

from spevae import save_model
from spevae.cli import main

from test_refine import BETA, decodable_dataset, tiny_model


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    model = tiny_model(seed=11, residue_mode="complex")
    mg, poles_true, _ = decodable_dataset(model, seed=11, n_tau=41, n_bins=200)
    bins_file = d / "bins.txt"
    np.savetxt(bins_file, mg.bins.numpy())
    ckpt = d / "model.pt"
    save_model(model, str(ckpt), stage="test")
    return {"dir": d, "bins": bins_file, "ckpt": ckpt, "beta": BETA}


def test_continue_writes_the_pole_set(workspace, capsys):
    out = workspace["dir"] / "result.npz"
    main(["continue", "--bins", str(workspace["bins"]),
          "--beta", str(workspace["beta"]),
          "--checkpoint", str(workspace["ckpt"]),
          "--samples", "8", "--out", str(out)])
    text = capsys.readouterr().out
    assert "positivity:" in text and "effective poles:" in text

    z = np.load(out)
    for key in ("omega", "A", "A_std", "poles_omega", "poles_gamma",
                "residues_re", "residues_im", "G_recon", "tau"):
        assert key in z, key
    # the physics constraints, as shipped to the user
    assert (z["poles_gamma"] > 0).all()
    assert (z["residues_re"] > 0).all()
    assert z["residues_im"].sum() == pytest.approx(0.0, abs=1e-6)
    assert z["residues_re"].sum() == pytest.approx(1.0, abs=0.05)  # ~ measured norm

    with open(str(out) + ".json") as f:
        meta = json.load(f)
    assert {"chi2", "chi2_target", "certified_positive", "min_A"} <= set(meta)


def test_continue_refuses_a_broken_convention(workspace):
    bad = workspace["dir"] / "bad.txt"
    bins = np.loadtxt(workspace["bins"])
    np.savetxt(bad, 3.7 * bins)          # neither sign satisfies the sum rule
    with pytest.raises(SystemExit, match="Refusing to continue"):
        main(["continue", "--bins", str(bad), "--beta", str(workspace["beta"]),
              "--checkpoint", str(workspace["ckpt"])])


def test_finetune_roundtrip(workspace, capsys):
    out = workspace["dir"] / "tuned.pt"
    main(["finetune", "--bins", str(workspace["bins"]),
          "--beta", str(workspace["beta"]),
          "--checkpoint", str(workspace["ckpt"]), "--out", str(out),
          "--steps", "40", "--batch-size", "16", "--eval-every", "20"])
    assert out.exists()
    from spevae import load_checkpoint, load_model
    tuned = load_model(str(out))
    assert tuned.num_poles == 8
    meta = load_checkpoint(str(out))["meta"]
    assert meta["stage"] == "finetune"
    assert meta["beta"] == workspace["beta"]

    main(["info", "--checkpoint", str(out)])
    text = capsys.readouterr().out
    assert "num_poles" in text and "finetune" in text
