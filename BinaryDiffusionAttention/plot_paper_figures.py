"""Rebuild paper figures from saved measurements; no GPU or training required.

Run: uv run --with matplotlib python BinaryDiffusionAttention/plot_paper_figures.py
Outputs: editable SVG, vector PDF, and 220-dpi PNG under figures/.
"""
from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
BLUE, TEAL, GOLD, INK = "#4169A1", "#008C82", "#D89526", "#24364B"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#CBD5DF",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
    "grid.color": "#E5EBF0", "grid.linewidth": .7,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def save(fig, name, note):
    fig.text(.04, .015, note, fontsize=8, color="#607084")
    fig.subplots_adjust(bottom=.19, top=.88, wspace=.32)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(name)


def mechanism():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.65, 1]})
    a.set_title("A   A few routes, composed across steps", loc="left")
    # Each row reads only the preceding row: no serial within-step updates.
    for layer in range(1, 5):
        jump = 2 ** (layer - 1)
        offsets = [0, jump] if layer <= 2 else [0, 1, 2, jump]
        for dst in range(16):
            for offset in offsets:
                src = dst - offset
                if src >= 0:
                    c = TEAL if offset == jump else (GOLD if offset else "#AAB7C5")
                    a.plot([src, dst], [layer - 1, layer], color=c,
                           alpha=.36 if offset else .2, lw=.75, zorder=1)
    # Concrete binary path: source 0 reaches destination 15 via 1+2+4+8.
    path = [0, 1, 3, 7, 15]
    for layer in range(1, 5):
        a.add_patch(FancyArrowPatch((path[layer-1], layer-1), (path[layer], layer),
                                   arrowstyle="-|>", mutation_scale=12, lw=2.2,
                                   color=TEAL, zorder=3))
    for layer in range(5):
        a.scatter(range(16), [layer] * 16, s=21, color=INK, zorder=4)
    a.set(xticks=[0, 1, 3, 7, 15], yticks=range(5),
          yticklabels=["Input", "Step 1", "Step 2", "Step 3", "Step 4"],
          xlabel="Sequence position (time axis)", xlim=(-.5, 15.7), ylim=(-.2, 5.1))
    a.legend(handles=[Line2D([0], [0], color=TEAL, lw=2, label="Dyadic transport"),
                      Line2D([0], [0], color=GOLD, lw=2, label="Local transport")],
             loc="upper left", frameon=False, fontsize=8)
    b.set_title("B   Reach expands exponentially", loc="left")
    steps = list(range(0, 21))
    b.plot(steps, [2**x for x in steps], color=TEAL, lw=2.5)
    for x, label in [(10, "10 steps / 1K"), (16, "16 steps / 65K"), (20, "20 steps / 1M")]:
        b.scatter(x, 2**x, color=TEAL, s=38)
        b.annotate(label, (x, 2**x), xytext=(-8, 9), textcoords="offset points",
                   ha="right", fontsize=9)
    b.set_yscale("log", base=2)
    b.set(xlabel="Propagation steps", ylabel="Dyadic coverage span (positions)",
          yticks=[1, 1024, 65536, 1048576], yticklabels=["1", "1K", "65K", "1M"],
          ylim=(.7, 4e6), xticks=[0, 5, 10, 15, 20])
    b.grid(axis="y")
    save(fig, "fig01_temporal_diffusion",
         "Structural illustration, not learned attention weights. All positions update in parallel from the previous step.")


def learning():
    dense = load("wikitext_65536_l16_causal_20pass_dev.json")
    sparse = load("wikitext_65536_l16_sbda_20pass_dev.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for a, key, title in zip(axes, ["mean_loss", "dev_loss"],
                              ["A   Training fit", "B   Held-out validation"]):
        for d, color, label in [(dense, BLUE, "Dense causal"), (sparse, TEAL, "Staged binary-local")]:
            records = d["pass_records"]
            xs, ys = [r["pass"] for r in records], [r[key] for r in records]
            a.plot(xs, ys, color=color, lw=2.4, marker="o", ms=3.5, label=label)
            a.annotate(f"{ys[-1]:.4f}", (xs[-1], ys[-1]), xytext=(6, 0),
                       textcoords="offset points", color=color, va="center", weight="bold")
        a.set(title=title, xlabel="Training passes", xlim=(1, 23), xticks=[1, 5, 10, 15, 20])
        a.grid(axis="y")
    axes[0].set_ylabel("Cross-entropy (nats / character; lower is better)")
    axes[0].legend(frameon=False)
    save(fig, "fig02_learning_curves",
         "WikiText-2 | 65,536 positions | 16 layers | 722,880 parameters | one seed. Train: online pass mean; validation: after each pass.")


def topology():
    files = list((ROOT / "results" / "wikitext_internal_3seed").glob("*.json"))
    assert len(files) == 1, "Select the intended topology record explicitly if additional runs are added."
    rows = json.loads(files[0].read_text())
    modes = ["full", "h0", "h1", "h4", "h8"]
    names = ["Dense", "Dyadic only", "Local at every step", "Greedy coverage", "Staged binary-local"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for a, key, title in zip(axes, ["mean_last_10_loss", "val_loss"],
                              ["A   Training fit", "B   Validation"]):
        for i, mode in enumerate(modes):
            vals = [r[key] for r in rows if r["mode"] == mode]
            assert len(vals) == 3
            mean = statistics.mean(vals)
            color = TEAL if mode == "h8" else BLUE
            # Individual seed points and sample SD, not a confidence interval.
            a.errorbar(mean, i, xerr=statistics.stdev(vals), fmt="o", color=color,
                       capsize=4, ms=7, lw=1.5, zorder=3)
            a.scatter(vals, [i-.13]*3, s=16, color=color, alpha=.45)
        a.set(title=title, xlabel="Cross-entropy (nats / character)",
              yticks=range(5), yticklabels=names, xlim=(1.85, 2.14))
        a.grid(axis="x")
    axes[0].invert_yaxis()
    save(fig, "fig03_topology_screening",
         "WikiText-2 | 256 positions | 8 layers | 500 updates | GTX 1080 Ti. Large dots: mean; whiskers: sample SD; small dots: three seeds.")


def systems():
    data = load("sbda_vs_causal_4090_forward.json")["results"]
    groups = {mode: sorted([r for r in data if r["mode"] == mode and r["status"] == "ok"],
                           key=lambda r: r["length"]) for mode in ["causal", "sbda"]}
    lengths = [r["length"] for r in groups["causal"]]
    assert lengths == [r["length"] for r in groups["sbda"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for mode, color, label in [("causal", BLUE, "Dense / FlashAttention"), ("sbda", TEAL, "Sparse / fused Triton")]:
        rows = groups[mode]
        axes[0].plot(lengths, [r["tokens_per_second"]/1e6 for r in rows],
                     color=color, marker="o", lw=2.3, ms=4, label=label)
        axes[2].plot(lengths, [r["peak_allocated_gib"] for r in rows],
                     color=color, lw=2.3, marker="o" if mode == "causal" else "x", ms=5, label=label)
    ratios = [s["tokens_per_second"]/d["tokens_per_second"]
              for d, s in zip(groups["causal"], groups["sbda"])]
    axes[1].plot(lengths, ratios, color=TEAL, marker="o", lw=2.3, ms=4)
    axes[1].axhline(1, color=GOLD, lw=1, ls="--")
    axes[1].set_ylim(-2, max(ratios)*1.2)
    axes[1].annotate(f"{ratios[-1]:.2f}×", (lengths[-1], ratios[-1]),
                     xytext=(-8, 8), textcoords="offset points", ha="right", color=TEAL, weight="bold")
    for a, title, ylabel in zip(axes,
            ["A   Hidden-state forward", "B   Measured speedup", "C   Almost identical memory"],
            ["Million characters / second", "Sparse / dense throughput", "Peak allocated memory (GiB)"]):
        a.set_xscale("log", base=2)
        a.set(title=title, xlabel="Input positions", ylabel=ylabel,
              xticks=[4096, 16384, 65536, 262144, 1048576],
              xticklabels=["4K", "16K", "64K", "256K", "1M"])
        a.grid(axis="y")
    axes[0].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0, .63))
    save(fig, "fig04_forward_scaling",
         "Reported RTX 4090 / 49,140 MiB | BF16 | batch 1 | 8 layers. No vocabulary head or backward; eight steps do not cover 1M.")


def theory():
    powers = list(range(10, 21))
    ns = [2**l for l in powers]
    ratios = [l*n*(n+1)/2 / ((4*l-5)*n-3*l+7) for l, n in zip(powers, ns)]
    dense = load("wikitext_65536_l16_causal_20pass_dev.json")
    sparse = load("wikitext_65536_l16_sbda_20pass_dev.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    a = axes[0]
    a.plot(ns, ratios, color=GOLD, lw=2.5)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set(title="A   Structural interaction reduction", xlabel="Positions (depth = log₂ N)",
          ylabel="Dense / sparse edge count", xticks=[1024, 32768, 1048576],
          xticklabels=["1K", "32K", "1M"], ylim=(80, 6e5))
    a.annotate(f"{ratios[-1]:,.0f}×", (ns[-1], ratios[-1]), xytext=(-8, 10),
               textcoords="offset points", ha="right", color=GOLD, weight="bold")
    a.grid(axis="y")
    for ax, key, divisor, title, ylabel, fmt in [
        (axes[1], "total_seconds", 60, "B   65K training-loop time", "Minutes (lower is better)", ".2f"),
        (axes[2], "peak_allocated_gib", 1, "C   65K training memory", "Peak allocated memory (GiB)", ".3f")]:
        vals = [dense[key]/divisor, sparse[key]/divisor]
        bars = ax.bar(["Dense", "Staged\nbinary-local"], vals, color=[BLUE, TEAL], width=.55)
        ax.bar_label(bars, labels=[format(v, fmt) for v in vals], padding=5, color=INK)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(vals)*1.22))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    save(fig, "fig05_structure_and_execution",
         "A: analytical edge counts, not runtime. B–C: 16-layer / 65K / 20-pass training; loading, validation and checkpoint IO excluded.")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    mechanism()
    learning()
    topology()
    systems()
    theory()
