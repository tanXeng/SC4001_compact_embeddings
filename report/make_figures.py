"""Generate every figure and derived number used in the report.

Reads only the frozen, committed analysis data in
SC4001_compact_embeddings/analysis/data (research/vocab-scaling-fp32-isoflops
branch); no datasets or checkpoints are required.

Usage (from the report/ directory):
    python -m pip install matplotlib pandas numpy
    python make_figures.py --data ../SC4001_compact_embeddings/analysis/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- styling --
plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "lines.linewidth": 1.3, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})
# One colour per representation, one line style / marker per vocabulary size.
COLOR = {"dense": "#4d4d4d", "low_rank": "#1f77b4", "adaptive": "#2ca02c", "hash": "#d62728"}
STYLE = {16384: (":", "s"), 32768: ("--", "o"), 65536: ("-", "^")}
LABEL = {"baseline16k": "Dense 16K (baseline)", "dense32k": "Dense 32K*", "lowrank32k": "Low-rank 32K",
         "adaptive32k": "Adaptive 32K", "dense64k": "Dense 64K*", "lowrank64k": "Low-rank 64K",
         "adaptive64k": "Adaptive 64K", "hash64k": "Hash 64K"}
ORDER = list(LABEL)


def load(data: Path):
    runs = pd.read_csv(data / "run_summary.csv").set_index("arm").loc[ORDER]
    evals = pd.read_csv(data / "evaluation_history.csv")
    freq = pd.read_csv(data / "frequency_metrics.csv")
    diag = json.loads((data / "representation_diagnostics.json").read_text())
    train = pd.read_csv(data / "training_history.csv.gz",
                        usecols=["arm", "estimated_flops", "update_nll", "update_raw_bytes"])
    return runs, evals, freq, diag, train


def style(arm, runs):
    rep, v = runs.loc[arm, "representation"], int(runs.loc[arm, "vocab_size"])
    ls, mk = STYLE[v]
    return dict(color=COLOR[rep], ls=ls, marker=mk, label=LABEL[arm])


def fig_dense_parameters(runs, out):
    """Dense tied model composition from the frozen accounting records."""
    dense = runs.loc[["baseline16k", "dense32k", "dense64k"]]
    backbone = dense.backbone_parameters.to_numpy(dtype=np.int64)
    vocabulary = dense.vocabulary_parameters.to_numpy(dtype=np.int64)
    assert np.all(backbone == backbone[0])
    assert np.all(vocabulary == dense.vocab_size.to_numpy(dtype=np.int64) * 512)
    assert np.all(backbone + vocabulary == dense.total_parameters.to_numpy(dtype=np.int64))
    x = np.arange(len(dense))
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.bar(x, backbone / 1e6, width=.55, color="#4d4d4d", label="Transformer backbone")
    ax.bar(x, vocabulary / 1e6, bottom=backbone / 1e6, width=.55,
           color="#1f77b4", label="Dense tied input/output vocabulary matrix")
    for i, (body, vocab) in enumerate(zip(backbone, vocabulary)):
        ax.text(i, (body + vocab / 2) / 1e6, f"{100 * vocab / (body + vocab):.1f}%",
                ha="center", va="center", color="white")
    ax.set(xticks=x, xticklabels=["16K", "32K", "64K"], xlabel="Vocabulary size",
           ylabel="Parameters (millions)", ylim=(0, 72))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_dense_parameters.pdf")
    plt.close(fig)


def fig_curves(runs, evals, train, out):
    """(a) training BPB (smoothed), (b) held-out screening BPB vs compute, (c) vs training text."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    window = 1500  # updates; a ratio of sums is the exact BPB over the window
    for arm in ORDER:
        st = style(arm, runs)
        d = train[train.arm == arm]
        nll = d.update_nll.rolling(window).sum()
        byt = d.update_raw_bytes.rolling(window).sum()
        bpb = nll / (np.log(2) * byt)
        axes[0].plot(d.estimated_flops / 1e17, bpb, color=st["color"], ls=st["ls"], lw=1.0, label=st["label"])
        e = evals[(evals.arm == arm) & evals.scheduled_curve_point & (evals.compute_fraction > 0)]
        axes[1].plot(e.estimated_flops.astype(float) / 1e17, e.bpb, color=st["color"], ls=st["ls"],
                     marker=st["marker"], ms=3.5, label=st["label"])
        # Same held-out points, re-indexed by the amount of training text consumed (sample efficiency).
        axes[2].plot(e.training_raw_bytes.astype(float) / 1e9, e.bpb, color=st["color"], ls=st["ls"],
                     marker=st["marker"], ms=3.5)
    axes[0].set(xlabel=r"Training compute (×10$^{17}$ FLOPs)", ylabel="Training BPB (1500-update window)",
                xlim=(1.0, 5.05), ylim=(1.29, 1.52), title="(a) Training BPB vs. compute")
    axes[1].set(xlabel=r"Training compute (×10$^{17}$ FLOPs)", ylabel="Screening BPB (held-out)",
                ylim=(1.24, 1.60), title="(b) Held-out BPB vs. compute")
    axes[2].set(xlabel="Training text consumed (GB)", ylabel="Screening BPB (held-out)",
                ylim=(1.24, 1.60), title="(c) Held-out BPB vs. training text")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=8, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.07),
               handlelength=2.6, columnspacing=.9, fontsize=6.3)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig_curves.pdf")
    plt.close(fig)


def fig_tradeoff(runs, out):
    """(a) storage vs confirmation BPB, (b) optimiser updates obtained from the same budget."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    base = runs.loc["baseline16k"]
    y = np.arange(len(ORDER))
    for col, off, alpha, name in (("screening_bpb", -.18, .45, "screening"),
                                  ("confirmation_bpb", .18, .95, "confirmation")):
        delta = (runs[col] - base[col]).values
        shown = np.clip(delta, -0.03, 0.012)  # Hash is far off-scale; value printed instead
        ax.barh(y + off, shown, .34, color=[COLOR[r] for r in runs.representation], alpha=alpha,
                label=name, edgecolor="none")
    for i, arm in enumerate(ORDER):
        ratio = runs.loc[arm, "vocabulary_serialized_inference_bytes"] / base["storage_ceiling_bytes"]
        d = runs.loc[arm, "confirmation_bpb"] - base["confirmation_bpb"]
        txt = f"{ratio:.2f}× storage" + (f"   (Δ = +{d:.3f})" if d > .012 else "")
        ax.text(0.0135 if d > .012 else 0.002, i, txt, va="center", fontsize=6.3)
    ax.axvline(0, color="k", lw=.6)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[a] for a in ORDER]); ax.invert_yaxis()
    ax.set(xlabel="ΔBPB vs. Dense 16K baseline (negative = better)", xlim=(-0.022, 0.034),
           title="(a) Change in held-out BPB and vocabulary storage")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=6.3, handlelength=1.2)

    ax = axes[1]
    y = np.arange(len(ORDER))
    upd = runs.optimizer_updates.values / 1e3
    ax.barh(y, upd, color=[COLOR[r] for r in runs.representation], alpha=.85, height=.65)
    for i, arm in enumerate(ORDER):
        ax.text(upd[i] + .5, i, f"{upd[i]:.1f}K  ({runs.loc[arm, 'raw_bytes'] / 1e9:.2f} GB)",
                va="center", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set(xlabel="Optimiser updates within 5×10$^{17}$ FLOPs (×10$^3$)", xlim=(0, 70),
           title="(b) Updates (and training text) per fixed budget")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)


def fig_frequency_and_diagnostics(runs, freq, diag, out):
    """(a) NLL by training-frequency third, (b) low-rank spectra, (c) hash bucket occupancy."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    arms = ["baseline16k", "dense32k", "lowrank32k", "dense64k", "lowrank64k", "hash64k"]
    groups = ["frequent", "middle", "rare"]
    width = .13
    for j, arm in enumerate(arms):
        f = freq[(freq.arm == arm) & (freq.observation_role == "confirmation_final")
                 & (freq.group_definition == "training_frequency_rank_type_thirds")].sort_values("group")
        st = style(arm, runs)
        ax.bar(np.arange(3) + (j - 2.5) * width, f.token_nll, width, color=st["color"],
               alpha={16384: .35, 32768: .6, 65536: .95}[int(runs.loc[arm, "vocab_size"])],
               hatch={"--": "///", ":": "...", "-": ""}[st["ls"]], edgecolor="white", lw=.3, label=st["label"])
    ax.set_xticks(range(3)); ax.set_xticklabels(groups)
    ax.set(ylabel="Token NLL (nats)", title="(a) NLL by training-frequency third")
    ax.legend(frameon=False, fontsize=5.8, ncol=1, loc="upper left", handlelength=1.2)
    ax.set_ylim(0, 15.5)

    ax = axes[1]
    rec = {(r["arm"], r["observation_role"]): r["values"] for r in diag}
    for arm in ["lowrank32k", "lowrank64k"]:
        s = np.array(rec[(arm, "endpoint")]["full_effective_singular_values"])
        st = style(arm, runs)
        ax.plot(np.arange(1, len(s) + 1), s / s[0], color=st["color"], ls=st["ls"], label=st["label"] + " (full)")
    for arm in ["dense64k", "hash64k"]:
        s = np.array(rec[(arm, "endpoint")]["singular_values"])
        st = style(arm, runs)
        ax.plot(np.arange(1, len(s) + 1), s / s[0], color=st["color"], ls=st["ls"], lw=.9,
                label=st["label"] + " (128 rows)")
    ax.set(yscale="log", xlabel="Singular value index", ylabel=r"$\sigma_i/\sigma_1$",
           title="(b) Effective-embedding spectra")
    ax.legend(frameon=False, fontsize=5.8)

    ax = axes[2]
    h = rec[("hash64k", "endpoint")]["hash"]["occupancy_histogram"]
    k = np.array(sorted(map(int, h))); v = np.array([h[str(i)] for i in k])
    ax.bar(k, v, color=COLOR["hash"], alpha=.8, width=.8)
    ax.axvline(h_mean := rec[("hash64k", "endpoint")]["hash"]["occupancy"]["mean"], color="k", lw=.7, ls="--")
    ax.text(h_mean + 3.2, v.max() * .9, f"mean {h_mean:.2f}\n(= 2V/B)", fontsize=6.5)
    ax.set(xlabel="Token slots per shared component", ylabel="Components",
           title="(c) Hash 64K component occupancy")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig_frequency_diagnostics.pdf")
    plt.close(fig)


def print_numbers(runs, train):
    """Derived numbers quoted in the text (gain recovered, bytes/token, etc.)."""
    base = runs.loc["baseline16k"]
    print("arm, screen, confirm, dConfirm vs base, bytes/token, FLOPs/update(e12), active h")
    for arm in ORDER:
        r = runs.loc[arm]
        print(f"{arm:12s} {r.screening_bpb:.4f} {r.confirmation_bpb:.4f} "
              f"{r.confirmation_bpb - base.confirmation_bpb:+.4f} {r.raw_bytes / r.scored_tokens:.3f} "
              f"{r.estimated_flops / r.optimizer_updates / 1e12:.2f} {r.active_hours:.1f}")
    for v, (d, c) in {"32K": ("dense32k", ["lowrank32k", "adaptive32k"]),
                      "64K": ("dense64k", ["lowrank64k", "adaptive64k", "hash64k"])}.items():
        for col in ("screening_bpb", "confirmation_bpb"):
            gain = base[col] - runs.loc[d, col]
            print(v, col, {a: round((base[col] - runs.loc[a, col]) / gain, 2) for a in c})
    last = train.groupby("arm").tail(2000).groupby("arm")
    print("final-2000-update training BPB:",
          (last.update_nll.sum() / (np.log(2) * last.update_raw_bytes.sum())).round(4).to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("../SC4001_compact_embeddings/analysis/data"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    runs, evals, freq, diag, train = load(args.data)
    fig_dense_parameters(runs, args.out)
    fig_curves(runs, evals, train, args.out)
    fig_tradeoff(runs, args.out)
    fig_frequency_and_diagnostics(runs, freq, diag, args.out)
    print_numbers(runs, train)
