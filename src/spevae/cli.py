r"""The ``spevae`` command-line interface.

Four verbs, in the order a user meets them:

``spevae continue``
    The headline: bins in, spectrum out.  Reads whitespace-delimited
    :math:`(N_{\rm bins} \times L_\tau)` text, detects the sign convention from
    the sum rule (and refuses to guess when neither sign works), runs the
    model, prints the calibrated :math:`\chi^2`, the effective pole count and
    the exact positivity certificate, and writes the *pole set* -- the grid-free
    result -- alongside :math:`A(\omega)` on a convenience grid.

``spevae finetune``
    Stage 2 on your own cell: adapts a pretrained checkpoint to one dataset's
    noise, with no ground truth needed.

``spevae pretrain``
    Stage 1 from scratch on the synthetic family generator.  ``--preset fast``
    is a laptop-scale smoke run; ``--preset full`` is a real training run.

``spevae info``
    What a checkpoint is: architecture, pole count, residue mode, training
    provenance.

Everything the CLI does is a thin call into the library; nothing here is the
only way to do anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch


def _load_measured(args, need_bins: bool = True):
    """Shared --bins/--beta handling with the sum-rule gate."""
    from .data import read_text_bins

    if args.beta is None:
        sys.exit("--bins requires --beta")
    mg = read_text_bins(args.bins, args.beta, sign_flip=args.sign_flip,
                        dtype=torch.float64)
    print(mg.summary(getattr(args, "min_eigenvalue", 0.0)))
    print()
    sr = mg.check_sum_rule()
    if not sr["ok"]:
        sys.exit(f"G(0)+G(beta) = {sr['norm']:.6f}, expected 1. "
                 f"Try --sign-flip, or check that tau runs 0 ... beta "
                 f"inclusive. Refusing to continue: a wrong convention "
                 f"produces a confident, wrong spectrum.")
    return mg


def _add_bins_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--bins", required=True, metavar="FILE",
                    help="whitespace-delimited (N_bins x L_tau) text file, "
                         "tau = 0 ... beta inclusive")
    ap.add_argument("--beta", type=float, required=True,
                    help="inverse temperature of the simulation")
    ap.add_argument("--sign-flip", dest="sign_flip", action="store_true",
                    default=None,
                    help="force G -> -G (default: detect from the sum rule)")
    ap.add_argument("--no-sign-flip", dest="sign_flip", action="store_false",
                    help="force no negation")


def cmd_continue(args) -> None:
    from . import RefineConfig, load_model, refine_latent

    mg = _load_measured(args)
    model = load_model(args.checkpoint)
    print(f"model: {model.num_parameters():,} parameters, "
          f"{model.num_poles} poles ({model.residue_mode} residues), "
          f"latent {model.latent_dim}")

    omega = torch.linspace(args.omega_min, args.omega_max, args.n_omega,
                           dtype=torch.float32)
    G_mean = mg.mean.unsqueeze(0).float()
    # raw sigma for chi^2 (pinned slices are excluded there); floored sigma as
    # the encoder's noise channel (pinned slices would sit ~30 logs out of
    # distribution otherwise)
    sigma = mg.sigma_mean.unsqueeze(0).float()
    sigma_enc = (mg.sigma_encoder / math.sqrt(mg.n_bins)).unsqueeze(0).float()

    if args.refine:
        rr = refine_latent(model, mg, RefineConfig(
            steps=args.refine_steps, min_eigenvalue=args.min_eigenvalue))
        poles = rr.poles                    # RefineResult poles are already physical
        A = rr.spectral(omega)
        A_np, A_std = A[0].double().numpy(), None
        G_recon = rr.G_recon
        chi2, target = rr.chi2, rr.chi2_target
        print(f"latent refinement: {rr.steps_run} steps, "
              f"chi2 {rr.chi2_init:.4g} -> {rr.chi2:.4g} "
              f"(calibrated target {target:.4g})"
              + ("   [prior binding -- treat with care]" if rr.prior_is_binding
                 else ""))
    else:
        out = model.continue_(G_mean, beta=mg.beta, tau=mg.tau.float(),
                              sigma=sigma_enc, omega=omega, n_samples=args.samples)
        poles = out["poles"]
        A_np = out["A"][0].double().numpy()
        A_std = (out["A_std"][0].double().numpy() if "A_std" in out else None)
        G_recon = out["G_recon"]
        from .losses import chi2 as chi2_fn
        cov = mg.covariance(args.min_eigenvalue)
        chi2 = float(chi2_fn(G_recon.double(), G_mean.double(), sigma.double(),
                             inv_sqrt_R=cov["inv_sqrt_R"],
                             slices=cov["slices"]))
        target = float(cov["chi2_target"])
        print(f"chi2 = {chi2:.4g}  (calibrated perfect-fit target {target:.4g}, "
              f"ratio {chi2 / max(target, 1e-300):.3g})")
        pin = mg.pinned_deviation(G_recon.double())
        if pin > 0.0:
            print(f"exactly-pinned slices (PH symmetry): max deviation {pin:.2e}")

    raw_poles = poles
    consolidation = None
    if args.consolidate:
        from .physics import consolidate_poles, refit_poles_to_greens
        res = args.resolution if args.resolution else math.pi / mg.beta
        merged0, (crep,) = consolidate_poles(poles[0:1], res)
        cov0 = mg.covariance(args.min_eigenvalue)
        merged, _, _ = refit_poles_to_greens(
            merged0, mg.beta, mg.x, mg.mean, mg.sigma_mean,
            inv_sqrt_R=cov0["inv_sqrt_R"], slices=cov0["slices"],
            steps=args.refit_steps, ph_symmetric=args.ph_symmetric,
            fold_tolerance=res if args.ph_symmetric else 0.0,
            positivity_weight=args.refit_positivity,
            pair_bound=args.ph_symmetric)   # structural A>=0 comes free there
        # chi2-neutrality is measured, not assumed: push the consolidated set
        # through the exact forward map against the same whitened covariance.
        x = mg.x.float()
        K = model.kernel_matrix(mg.beta, x)
        fwd = model.forward_map(mg.beta, x.device, x.dtype)
        G_cons = fwd(merged.to_dimensionless(mg.beta).to(torch.float32), x,
                     kernel_matrix=K)
        from .losses import chi2 as _chi2_fn
        cov_c = mg.covariance(args.min_eigenvalue)
        chi2_cons = float(_chi2_fn(
            G_cons.double(), mg.mean.unsqueeze(0).double(),
            mg.sigma_mean.unsqueeze(0).double(),
            inv_sqrt_R=cov_c["inv_sqrt_R"], slices=cov_c["slices"]))
        print(f"consolidation: {crep}")
        print(f"  chi2 {chi2:.4g} -> {chi2_cons:.4g}  "
              f"(delta {chi2_cons - chi2:+.3g}; the data "
              f"{'cannot tell them apart' if abs(chi2_cons - chi2) < 0.1 * max(chi2, 1.0) else 'CAN tell -- inspect'})")
        consolidation = {"resolution": res, "n_in": crep.n_in,
                         "n_out": crep.n_out, "chi2_consolidated": chi2_cons,
                         "matsubara_err": crep.matsubara_err_polished}
        poles = merged
        A_np = poles.spectral(omega)[0].double().numpy()

    from .physics.positivity import positivity_certificate
    cert = positivity_certificate(poles[0:1])[0]
    a_unit = poles.a[0] / poles.a[0].sum().clamp_min(1e-30)
    eff = float(1.0 / (a_unit**2).sum())
    print(f"effective poles: {eff:.2f} of {poles.shape[1]}")
    print(f"positivity: {cert}")

    p = poles[0]
    order = torch.argsort(p.eps)
    print("\n  pole   omega        gamma        weight a     asym b")
    for i in order.tolist():
        if float(p.a[i]) == 0.0:
            continue                        # consolidation padding: inert
        print(f"  {i:>4}   {float(p.eps[i]):>+9.4f}   {float(p.gamma[i]):>9.4f}"
              f"   {float(p.a[i]):>9.5f}   {float(p.b[i]):>+9.5f}")

    # pi/beta-broadened spectrum: the honest reporting scale (structure finer
    # than the first Matsubara frequency is not supportable by tau-data).
    w_np = omega.double().numpy()
    _res = math.pi / mg.beta
    _h = float(w_np[1] - w_np[0])
    _n = int(4 * _res / _h) + 1
    _k = np.arange(-_n, _n + 1) * _h
    _ker = np.exp(-0.5 * (_k / _res) ** 2)
    _ker /= _ker.sum()
    A_sm = np.convolve(A_np, _ker, mode="same")

    if args.out:
        np.savez(
            args.out,
            omega=w_np, A=A_np,
            A_smoothed=A_sm,
            **({"A_std": A_std} if A_std is not None else {}),
            poles_omega=p.eps.double().numpy(),
            poles_gamma=p.gamma.double().numpy(),
            residues_re=p.a.double().numpy(),
            residues_im=p.b.double().numpy(),
            G_recon=G_recon.squeeze(0).double().numpy(),
            tau=mg.tau.double().numpy(),
            **({"raw_poles_omega": raw_poles.eps[0].double().numpy(),
                "raw_poles_gamma": raw_poles.gamma[0].double().numpy(),
                "raw_residues_re": raw_poles.a[0].double().numpy(),
                "raw_residues_im": raw_poles.b[0].double().numpy()}
               if consolidation is not None else {}),
        )
        with open(str(args.out) + ".json", "w") as f:
            json.dump({"beta": mg.beta, "chi2": chi2, "chi2_target": target,
                       "effective_poles": eff,
                       "certified_positive": cert.is_positive,
                       "min_A": cert.min_A,
                       "negative_weight": cert.negative_weight,
                       "checkpoint": args.checkpoint,
                       "refined": bool(args.refine),
                       "consolidation": consolidation}, f, indent=2)
        print(f"\nsaved -> {args.out} (+.json)   "
              f"[the pole set is the result; A(omega) is a convenience grid]")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        w = w_np
        ax1.plot(w, A_np, lw=1.0, alpha=0.55, label="spevae (full resolution)")
        ax1.plot(w, A_sm, lw=1.8, label=r"spevae @ $\pi/\beta$ (honest scale)")
        if A_std is not None:
            ax1.fill_between(w, A_np - A_std, A_np + A_std, alpha=0.25,
                             label="latent ensemble")
        ax1.set_xlabel(r"$\omega$"); ax1.set_ylabel(r"$A(\omega)$")
        ax1.legend(frameon=False)
        t = mg.tau.double().numpy()
        ax2.errorbar(t, mg.mean.double().numpy(),
                     yerr=mg.sigma_mean.double().numpy(), fmt=".", ms=3,
                     alpha=0.6, label="data (mean)")
        ax2.plot(t, G_recon.squeeze(0).double().numpy(), lw=1.2,
                 label="reconstruction")
        ax2.set_xlabel(r"$\tau$"); ax2.set_ylabel(r"$G(\tau)$")
        ax2.set_yscale("log"); ax2.legend(frameon=False)
        fig.tight_layout(); fig.savefig(args.figure, dpi=150)
        plt.close(fig)
        print(f"figure -> {args.figure}")

        from pathlib import Path

        from .figures import plot_latent_space, plot_pole_residue_plane
        from .physics.poles import PoleSet as _PS
        stem = Path(args.figure)

        # Pole/residue planes with a latent-ensemble cloud when available.
        cloud = poles[0:1]
        if not args.refine:
            with torch.no_grad():
                mu1, lv1, norm1 = model.encode(G_mean, mg.x.float(), mg.beta,
                                               sigma=sigma_enc)
            sets = []
            with torch.no_grad():
                for _ in range(min(max(args.samples, 1), 64)):
                    z = model.reparameterise(mu1, lv1)
                    sets.append(model.decode(z, norm1, mg.beta))
            cloud = _PS(
                eps=torch.cat([s.eps for s in sets]),
                gamma=torch.cat([s.gamma for s in sets]),
                a=torch.cat([s.a for s in sets]),
                b=torch.cat([s.b for s in sets]),
            ).to_physical(mg.beta)
        pole_file = str(stem.with_name(stem.stem + "-poles" + stem.suffix))
        plot_pole_residue_plane(cloud, bold=poles[0:1], path=pole_file,
                                title=mg.label or "pole set")
        print(f"figure -> {pole_file}")

        # Latent diagnostics over every bin: the collapse check.
        with torch.no_grad():
            mu_all = model.encode(mg.bins.float(), mg.x.float(), mg.beta,
                                  sigma=mg.sigma.unsqueeze(0)
                                  .expand_as(mg.bins).float())[0]
        latent_file = str(stem.with_name(stem.stem + "-latent" + stem.suffix))
        plot_latent_space(mu_all, path=latent_file,
                          title=f"{mg.label or 'dataset'}: {mg.n_bins} bins")
        print(f"figure -> {latent_file}")


def cmd_finetune(args) -> None:
    from . import load_model, save_model
    from .training import FinetuneConfig, finetune

    mg = _load_measured(args)
    model = load_model(args.checkpoint)
    cfg = FinetuneConfig(steps=args.steps, batch_size=args.batch_size,
                         lr=args.lr, train=args.train,
                         min_eigenvalue=args.min_eigenvalue,
                         eval_every=args.eval_every, seed=args.seed,
                         device=args.device)

    def cb(kind, step, rec):
        if kind == "eval":
            print(f"[eval @ {step}] chi2_ratio={rec['val_chi2_ratio']:.4g} "
                  f"min_A={rec['val_min_A']:.2e} "
                  f"eff_poles={rec['effective_poles']:.2f}")

    hist = finetune(model, mg, cfg, callback=cb)
    f = hist["final"]
    print(f"\nbest step {hist['best_step']}: chi2_ratio "
          f"{hist['best_val_chi2_ratio']:.4g}, "
          f"{hist['skipped']} skipped batches"
          + (f", stopped early ({hist['stopped_reason']})"
             if hist["stopped_early"] else ""))
    print(f"final: chi2={f['val_chi2']:.4g} target={f['val_chi2_target']:.4g} "
          f"min_A={f['val_min_A']:.2e} eff_poles={f['effective_poles']:.2f}")
    save_model(model, args.out, stage="finetune", source=args.checkpoint,
               bins=args.bins, beta=mg.beta, history_eval=hist["eval"],
               config=vars(cfg))
    print(f"saved -> {args.out}")


def cmd_pretrain(args) -> None:
    from . import save_model
    from .data.synthetic import SyntheticGenerator
    from .losses import LossWeights
    from .models import SPEVAE
    from .training import PretrainConfig, pretrain

    presets = {
        "fast": dict(steps=4_000, batch_size=128, d_model=96, enc_layers=2,
                     beta_range=(6.0, 10.0)),
        "full": dict(steps=25_000, batch_size=48, d_model=160, enc_layers=3,
                     beta_range=(2.0, 20.0)),
    }
    p = presets[args.preset]
    steps = args.steps or p["steps"]
    # Narrow-distribution overrides for mechanism A/Bs: a fixed beta (and one
    # dtau) pins the eval stream and the kernel conditioning, so a 1-2 minute
    # run answers "does this loss/knob help" before any wide run is paid for.
    beta_range = (args.beta_min or p["beta_range"][0],
                  args.beta_max or p["beta_range"][1])
    dist = {"beta_range": beta_range}
    if args.dtau is not None:
        dist["dtau_choices"] = (args.dtau,)

    torch.manual_seed(args.seed)
    if args.init_from:
        from . import load_model as _lm
        model = _lm(args.init_from)
        print(f"warm start from {args.init_from} "
              f"({model.num_parameters():,} params)")
    else:
        model = SPEVAE(latent_dim=args.latent_dim, num_poles=args.num_poles,
                       omega_c=args.omega_c, d_model=p["d_model"],
                       enc_layers=p["enc_layers"], residue_mode=args.residue_mode,
                       ph_symmetric=args.ph_symmetric)
    if args.data in ("poles", "mixed"):
        from .data.pole_prior import MixedGenerator, PolePriorGenerator
        gen = PolePriorGenerator(omega_c=args.omega_c,
                                 P_max=args.num_poles - 4,
                                 device=args.device, **dist)
        if args.data == "mixed":
            fam = SyntheticGenerator(omega_c=args.omega_c,
                                     families=args.families,
                                     device=args.device, **dist)
            gen = MixedGenerator(gen, fam, p_pole=args.p_pole,
                                 p_pole_final=args.p_pole_final)
    else:
        gen = SyntheticGenerator(omega_c=args.omega_c,
                                 families=args.families,
                                 ph_symmetric=args.ph_symmetric,
                                 device=args.device, **dist)
    weights = LossWeights(w1=args.w_w1, l2=args.w_l2, barrier=args.w_barrier,
                          sparsity=args.w_sparsity, multiscale=args.w_multiscale,
                          highpass=args.w_highpass,
                          pole_coherence=args.w_pole_coh,
                          pole_eps1=args.pole_eps1)
    cfg = PretrainConfig(steps=steps, batch_size=args.batch_size or p["batch_size"],
                         lr=args.lr, eval_every=max(steps // 12, 1),
                         eval_batches=8, seed=args.seed, device=args.device,
                         weights=weights)
    print(f"pretraining {model.num_parameters():,} parameters for {steps} steps "
          f"({args.preset} preset, device={args.device})")

    def cb(kind, step, rec):
        if kind == "train":
            print(f"[{step:>6}] total={rec['total']:.4g} chi2={rec['chi2']:.4g} "
                  f"w1={rec['w1_omega']:.4g} kl={rec['kl']:.3g} "
                  f"({rec['family']}, beta={rec['beta']:.3g})")
        elif kind == "eval":
            print(f"[eval @ {step}] w1_omega={rec['w1_omega']:.4f} "
                  f"l2_rel={rec['l2_rel']:.4f} chi2_geo={rec.get('chi2_geo', 0):.4g} "
                  f"AU={rec['active_units']:.2f}/{int(rec['latent_dim'])}")
            # Rolling crash-recovery checkpoint (raw weights, not the EMA):
            # a multi-hour run must not be losable to one stray kill.
            save_model(model, args.out + ".partial", stage="pretrain-partial",
                       step=step, eval_metrics=rec)

    hist = pretrain(model, gen, cfg, callback=cb)
    save_model(model, args.out, stage="pretrain", preset=args.preset,
               steps=steps, seed=args.seed,
               final_eval=hist["eval"][-1] if hist["eval"] else {})
    print(f"\nsaved -> {args.out}  (wall time {hist['wall_time']:.0f}s)")


def cmd_info(args) -> None:
    from . import load_checkpoint

    ck = load_checkpoint(args.checkpoint)
    print(f"format {ck.get('format')}  spevae {ck.get('spevae_version')}")
    print("config:")
    for k, v in ck.get("config", {}).items():
        print(f"  {k:>22} = {v}")
    meta = ck.get("meta", {})
    if meta:
        print("meta:")
        for k, v in meta.items():
            s = str(v)
            print(f"  {k:>22} = {s[:100] + '...' if len(s) > 100 else s}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="spevae",
        description="Analytic continuation of imaginary-time Green's functions "
                    "via a physics-constrained pole-expansion VAE.")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("continue", help="continue G(tau) bins to A(omega)")
    _add_bins_args(c)
    c.add_argument("--checkpoint", required=True, help="trained model (.pt)")
    c.add_argument("--samples", type=int, default=200,
                   help="latent samples for the uncertainty band (0 = mean only)")
    c.add_argument("--refine", action="store_true",
                   help="refine the latent code against this dataset first")
    c.add_argument("--refine-steps", type=int, default=2000)
    c.add_argument("--consolidate", action="store_true",
                   help="merge sub-resolution poles and refit the survivors "
                        "directly to the data (chi2 delta printed)")
    c.add_argument("--resolution", type=float, default=None,
                   help="merge scale in physical omega (default pi/beta)")
    c.add_argument("--refit-steps", type=int, default=4000)
    c.add_argument("--refit-positivity", type=float, default=1.0,
                   help="positivity pressure in the consolidation refit "
                        "(0 disables)")
    c.add_argument("--ph-symmetric", action="store_true",
                   help="mirror-constrained refit: A(w) = A(-w) identically; "
                        "for half-filled / PH-symmetric data only")
    c.add_argument("--omega-min", type=float, default=-10.0)
    c.add_argument("--omega-max", type=float, default=10.0)
    c.add_argument("--n-omega", type=int, default=801)
    c.add_argument("--min-eigenvalue", type=float, default=0.0)
    c.add_argument("--out", default=None, help="write result to this .npz")
    c.add_argument("--figure", default=None, help="save A(omega) + G(tau) figure")
    c.set_defaults(func=cmd_continue)

    f = sub.add_parser("finetune", help="Stage 2: adapt a checkpoint to one cell")
    _add_bins_args(f)
    f.add_argument("--checkpoint", required=True)
    f.add_argument("--out", required=True, help="where to save the tuned model")
    f.add_argument("--steps", type=int, default=2000)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--train", default="encoder_only",
                   choices=["encoder_only", "decoder_only", "all"])
    f.add_argument("--min-eigenvalue", type=float, default=0.0)
    f.add_argument("--eval-every", type=int, default=100)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cpu")
    f.set_defaults(func=cmd_finetune)

    p = sub.add_parser("pretrain", help="Stage 1: train on the synthetic families")
    p.add_argument("--preset", default="fast", choices=["fast", "full"])
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--latent-dim", type=int, default=12)
    p.add_argument("--num-poles", type=int, default=16)
    p.add_argument("--omega-c", type=float, default=20.0)
    p.add_argument("--residue-mode", default="complex",
                   choices=["complex", "real"])
    p.add_argument("--data", default="families",
                   choices=["families", "poles", "mixed"],
                   help="training data: legacy A-space families, or the "
                        "pole-first motif prior with exact canonical labels")
    p.add_argument("--ph-symmetric", action="store_true",
                   help="mirror-paired decoder: A(w) = A(-w) by construction, "
                        "trained on PH-symmetrised targets; for half-filled / "
                        "PH-symmetric data only")
    p.add_argument("--w-w1", type=float, default=3.0,
                   help="peak-position (Wasserstein) loss weight")
    p.add_argument("--w-l2", type=float, default=1.0,
                   help="shape/height (relative L2) loss weight")
    p.add_argument("--w-barrier", type=float, default=0.0,
                   help="soft width barrier above gamma_min")
    p.add_argument("--w-sparsity", type=float, default=0.0,
                   help="weight-entropy penalty (minimal-pole pressure)")
    p.add_argument("--w-multiscale", type=float, default=0.0,
                   help="broadened-envelope multiscale L2 weight")
    p.add_argument("--w-highpass", type=float, default=3.0,
                   help="sub-thermal mismatch weight (anti-pole-comb)")
    p.add_argument("--w-pole-coh", type=float, default=0.0,
                   help="stack-coherence weight (pole-labeled data)")
    p.add_argument("--pole-eps1", type=float, default=0.01,
                   help="final OT eps (set equal to 0.1 to disable annealing)")
    p.add_argument("--p-pole", type=float, default=0.5,
                   help="fraction of pole-labeled batches under --data mixed")
    p.add_argument("--p-pole-final", type=float, default=None,
                   help="if set, the pole fraction moves linearly from "
                        "--p-pole to this value over the run (curriculum)")
    p.add_argument("--beta-min", type=float, default=None,
                   help="override the preset's beta range (equal min/max "
                        "fixes beta -- the cheap tier for mechanism A/Bs)")
    p.add_argument("--beta-max", type=float, default=None)
    p.add_argument("--dtau", type=float, default=None,
                   help="use a single dtau instead of the preset's choices")
    p.add_argument("--families", type=lambda s: s.split(","), default=None,
                   help="comma-separated subset of spectral families for the "
                        "family stream (e.g. 'gapped,phonon')")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--init-from", default=None,
                   help="warm-start weights from a checkpoint (e.g. a .partial "
                        "rescued from a killed run); architecture comes from "
                        "the checkpoint, --latent-dim/--num-poles are ignored")
    p.set_defaults(func=cmd_pretrain)

    i = sub.add_parser("info", help="describe a checkpoint")
    i.add_argument("--checkpoint", required=True)
    i.set_defaults(func=cmd_info)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
