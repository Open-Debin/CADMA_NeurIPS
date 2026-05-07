#!/usr/bin/env python
"""Minimal Gaussian experiments for Fig. 3 row 1.

Runs the two simultaneous Gaussian mode-dropping settings used in the paper:
``density_shift`` and ``outlier_pollution``.  The default run uses 100 samples
per Gaussian mode as a lightweight reviewer test.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics.capacity_recall import compute_capacity_aware_distribution_matching_analysis
from metrics.clipped_density_coverage import ClippedDensityCoverage
from metrics.prdc import prdc


MODE_LOCS = np.array([0.0, 30.0, 20.0, 10.0, -10.0, -20.0, -30.0])
RATIOS = [i / 10.0 for i in range(11)]
BASELINES = ["Recall", "Coverage", "ClippedCoverage"]
CADMA_CORE = ["ReachableRealRatio", "CapacityRecall"]
CADMA_DECOMP = ["UnderRepresentationRatio", "OverRepresentationRatio"]

CSV_NAMES = {
    "ReachableRealRatio": "Coverage_our",
    "CapacityRecall": "R_cap",
    "UnderRepresentationRatio": "D_drop",
    "OverRepresentationRatio": "D_collapse",
}
DISPLAY_NAMES = {
    "ReachableRealRatio": r"Coverage$_{\mathrm{our}}$",
    "CapacityRecall": r"$R_{\mathrm{cap}}$",
    "UnderRepresentationRatio": r"$D_{\mathrm{drop}}$",
    "OverRepresentationRatio": r"$D_{\mathrm{collapse}}$",
}
STYLE = {
    "Recall": dict(color="#1f77b4", linestyle="-", marker="o"),
    "Coverage": dict(color="#ff7f0e", linestyle="-", marker="s"),
    "ClippedCoverage": dict(color="#7f7f7f", linestyle="-", marker="X"),
    "ReachableRealRatio": dict(color="#17becf", linestyle="-", marker="<", linewidth=5.5, alpha=0.95),
    "CapacityRecall": dict(color="#d62728", linestyle="-", marker=">", linewidth=5.8, alpha=0.98),
    "UnderRepresentationRatio": dict(color="#283593", linestyle="-", marker="s"),
    "OverRepresentationRatio": dict(color="#F9A825", linestyle="-", marker="p"),
}
DOMAINS = {
    "density_shift": "density shift",
    "outlier_pollution": "outlier pollution",
}

plt.rcParams.update({"font.size": 28, "lines.linewidth": 5, "lines.markersize": 20})


def sample_gaussian_modes(n_per_mode: int, feature_dim: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [
        rng.normal(loc=loc, scale=1.0, size=(n_per_mode, feature_dim))
        for loc in MODE_LOCS
    ]


def make_real_data(n_per_mode: int, feature_dim: int, seed: int) -> np.ndarray:
    return np.vstack(sample_gaussian_modes(n_per_mode, feature_dim, np.random.default_rng(seed)))


def make_fake_data(
    domain: str,
    ratio: float,
    n_per_mode: int,
    feature_dim: int,
    seed: int,
    ood_loc: float,
    ood_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    modes = sample_gaussian_modes(n_per_mode, feature_dim, rng)
    anchor = modes[0]
    dropped_pool = np.vstack(modes[1:])

    keep_count = int(round(len(dropped_pool) * (1.0 - ratio)))
    missing_count = len(dropped_pool) - keep_count
    keep_idx = rng.choice(len(dropped_pool), size=keep_count, replace=False)
    kept = dropped_pool[keep_idx] if keep_count else np.empty((0, feature_dim))
    kept_data = np.vstack([anchor, kept])

    if missing_count == 0:
        return kept_data
    if domain == "density_shift":
        idx = rng.choice(len(anchor), size=missing_count, replace=True)
        replacement = anchor[idx] + rng.normal(0.0, 0.1, size=(missing_count, feature_dim))
    elif domain == "outlier_pollution":
        replacement = rng.normal(ood_loc, ood_scale, size=(missing_count, feature_dim))
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return np.vstack([kept_data, replacement])


def evaluate(
    real: np.ndarray,
    fake: np.ndarray,
    baseline_k: int,
    cadma_k: int,
    cadma_radius_mode: str,
    cadma_matching_backend: str,
) -> dict[str, float]:
    prdc_res = prdc(real, fake, nearest_k=baseline_k, n_jobs=1)
    cadma_res = compute_capacity_aware_distribution_matching_analysis(
        real,
        fake,
        nearest_k=cadma_k,
        radius_mode=cadma_radius_mode,
        matching_backend=cadma_matching_backend,
        n_jobs=1,
    )["metrics"]
    return {
        "Recall": float(prdc_res["Recall"]),
        "Coverage": float(prdc_res["Coverage"]),
        "ClippedCoverage": float(ClippedDensityCoverage(real, K=baseline_k, n_jobs=1).ClippedCoverage(fake)),
        **{name: float(cadma_res[name]) for name in CADMA_CORE + CADMA_DECOMP},
    }


def run_domain(args: argparse.Namespace, domain: str) -> list[dict[str, float]]:
    rows = []
    for point_index, ratio in enumerate(RATIOS):
        print(f"[{DOMAINS[domain]}] point {point_index}: ratio={ratio:.1f}", flush=True)
        seed = args.seed + point_index * 100
        real = make_real_data(args.n_per_mode, args.feature_dim, seed)
        fake = make_fake_data(
            domain,
            ratio,
            args.n_per_mode,
            args.feature_dim,
            seed + 1,
            args.ood_loc,
            args.ood_scale,
        )
        rows.append(
            {
                "point_index": point_index,
                "ratio_input": ratio,
                "x": ratio,
                **evaluate(real, fake, args.baseline_k, args.cadma_k, args.cadma_radius_mode, args.cadma_matching_backend),
            }
        )
    return rows


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    fields = ["point_index", "ratio_input", "x", *BASELINES, "Coverage_our", "R_cap", "D_drop", "D_collapse"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row[key] for key in ["point_index", "ratio_input", "x", *BASELINES]}
            out.update({csv_name: row[metric] for metric, csv_name in CSV_NAMES.items()})
            writer.writerow(out)


def save_plot(rows: list[dict[str, float]], metrics: list[str], prefix: Path, with_reference: bool) -> None:
    x = np.array([row["x"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 8))
    handles = []

    for metric in metrics:
        style = STYLE[metric]
        (line,) = ax.plot(
            x,
            [row[metric] for row in rows],
            label=DISPLAY_NAMES.get(metric, metric),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style.get("linewidth"),
            alpha=style.get("alpha", 0.8),
        )
        handles.append(line)

    if with_reference:
        (ref_line,) = ax.plot(x, np.linspace(1.0, 0.143, len(x)), ":", color="black", label="Reference")
        handles.append(ref_line)

    ax.set_xlabel("Outlier Ratio")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(f"{prefix}_PLOT.png", dpi=300, bbox_inches="tight")

    legend = plt.figure(figsize=(5, max(1, len(handles) * 0.3)))
    legend.legend(handles, [handle.get_label() for handle in handles], loc="center", frameon=False)
    plt.axis("off")
    legend.savefig(f"{prefix}_LEGEND.png", dpi=300, bbox_inches="tight")
    plt.close(legend)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS.keys(), default=list(DOMAINS.keys()))
    parser.add_argument("--n-per-mode", type=int, default=100)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--baseline-k", type=int, default=5)
    parser.add_argument("--cadma-k", type=int, default=30)
    parser.add_argument("--cadma-radius-mode", default="min_mean")
    parser.add_argument("--cadma-matching-backend", choices=["scipy", "faiss"], default="scipy")
    parser.add_argument("--ood-loc", type=float, default=80.0)
    parser.add_argument("--ood-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gaussian_mode_dropping"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for domain in args.domains:
        rows = run_domain(args, domain)
        stem = f"gaussian_{domain}_n{args.n_per_mode}_cadmak{args.cadma_k}"
        save_csv(rows, args.out_dir / f"{stem}.csv")
        save_plot(rows, BASELINES + CADMA_CORE, args.out_dir / f"{stem}_baselines_plus_cadma_core", True)
        save_plot(rows, CADMA_DECOMP, args.out_dir / f"{stem}_cadma_decomposition", False)
        print(f"Saved outputs for {DOMAINS[domain]}: {args.out_dir / stem}*")


if __name__ == "__main__":
    main()
