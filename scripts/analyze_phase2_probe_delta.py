"""Analyze probe deltas between two Phase-2 result files.

Primary use: quantify Original - BlindRandom delta with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_gt0": float("nan")}
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(values[idx].mean())
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "p_gt0": float((boot_means > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze probe deltas with bootstrap CIs")
    parser.add_argument("--original_json", type=str, required=True)
    parser.add_argument("--control_json", type=str, required=True)
    parser.add_argument("--post_start", type=int, default=24)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    with open(args.original_json) as f:
        orig = json.load(f)
    with open(args.control_json) as f:
        ctrl = json.load(f)

    targets = list(orig.get("probe_targets", []))
    if not targets:
        targets = list(orig.get("results", {}).keys())

    summary: dict[str, Any] = {
        "original_json": args.original_json,
        "control_json": args.control_json,
        "original_image_mode": orig.get("image_mode"),
        "control_image_mode": ctrl.get("image_mode"),
        "post_start": args.post_start,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "targets": {},
    }

    for target in targets:
        o = np.asarray(orig["results"][target]["mean_accuracy"], dtype=np.float64)
        c = np.asarray(ctrl["results"][target]["mean_accuracy"], dtype=np.float64)
        if o.shape != c.shape:
            raise ValueError(f"Shape mismatch for {target}: {o.shape} vs {c.shape}")

        delta = o - c
        post_delta = delta[args.post_start:]
        post_stats = bootstrap_mean_ci(post_delta, n_boot=args.n_boot, seed=args.seed)

        summary["targets"][target] = {
            "num_layers": int(delta.shape[0]),
            "delta_by_layer": delta.tolist(),
            "delta_mean_all_layers": float(delta.mean()),
            "delta_mean_post": post_stats["mean"],
            "delta_post_ci95_low": post_stats["ci_low"],
            "delta_post_ci95_high": post_stats["ci_high"],
            "delta_post_bootstrap_p_gt0": post_stats["p_gt0"],
            "delta_layer46": float(delta[46]) if delta.shape[0] > 46 else None,
            "delta_layer44": float(delta[44]) if delta.shape[0] > 44 else None,
        }

    out_json = args.out_json
    if out_json is None:
        out_dir = os.path.dirname(args.original_json)
        out_json = os.path.join(out_dir, "phase2_probe_delta_bootstrap.json")

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()

