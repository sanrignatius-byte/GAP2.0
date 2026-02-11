"""Phase 1 Consolidation: Evaluate All Checkpoints.

Loads results from causal tracing, truncation, and geometric analysis,
and produces a unified go/no-go assessment for each checkpoint.

Usage:
    python scripts/run_checkpoint_eval.py --results_dir ./results
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Evaluation")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    results_dir = args.results_dir
    report = []

    def log_and_record(msg, level="info"):
        report.append(msg)
        getattr(logger, level)(msg)

    log_and_record("=" * 70)
    log_and_record("GAP 2.0 Phase 1 — Checkpoint Evaluation Report")
    log_and_record("=" * 70)

    # ---- Checkpoint 1: Modality Cliff Causes Real Failures ----
    log_and_record("\n--- CHECKPOINT 1: Modality Cliff Causes Real Performance Failures ---")

    causal_path = os.path.join(results_dir, "phase1_causal_results.json")
    trunc_path = os.path.join(results_dir, "phase1_truncation_results.json")

    cp1_go = False
    if os.path.exists(causal_path):
        with open(causal_path) as f:
            causal = json.load(f)

        cliff = causal.get("cliff_boundary")
        log_and_record(f"  Cliff boundary: layer {cliff}")
        log_and_record(f"  Hard EVD (mean): {causal['hard_evd_mean']:.1f}")
        log_and_record(f"  Easy EVD (mean): {causal['easy_evd_mean']:.1f}")

        evd_diff = causal['hard_evd_mean'] - causal['easy_evd_mean']
        log_and_record(f"  EVD gap (hard - easy): {evd_diff:.1f} layers")

        if os.path.exists(trunc_path):
            with open(trunc_path) as f:
                trunc = json.load(f)

            hard_drops = trunc.get("hard_accuracy_drops", {})
            easy_drops = trunc.get("easy_accuracy_drops", {})

            if str(cliff) in hard_drops:
                hard_drop = hard_drops[str(cliff)]
                easy_drop = easy_drops.get(str(cliff), 0)
                log_and_record(f"  Hard accuracy drop at cliff: {hard_drop:.2%}")
                log_and_record(f"  Easy accuracy drop at cliff: {easy_drop:.2%}")

                if hard_drop > 0.05 and easy_drop < 0.01:
                    log_and_record("  ✓ GO: Clear differential impact")
                    cp1_go = True
                elif hard_drop > 0.03:
                    log_and_record("  ⚠ MARGINAL: Evidence present but not strong")
                    cp1_go = True  # Proceed with caution
                else:
                    log_and_record("  ✗ NO-GO: No differential impact detected", "warning")
        else:
            log_and_record("  [Truncation results not found]", "warning")
    else:
        log_and_record("  [Causal tracing results not found]", "warning")

    # ---- Checkpoint 2: EVD Correlates with Performance ----
    log_and_record("\n--- CHECKPOINT 2: EVD Correlates with Downstream Performance ---")

    cp2_go = False
    if os.path.exists(causal_path):
        # This requires per-sample EVD and accuracy data
        # For now, check if the EVD difference between hard/easy tasks exists
        log_and_record("  [Full per-sample correlation requires running evd_performance_correlation()]")
        log_and_record("  Proxy check: EVD differs between task difficulty levels")
        if causal['hard_evd_mean'] > causal['easy_evd_mean']:
            log_and_record("  ✓ Positive signal: Hard tasks have higher EVD")
            cp2_go = True
        else:
            log_and_record("  ⚠ Unexpected: Easy tasks have higher EVD", "warning")

    # ---- Checkpoint 3: Geometric Collapse is Observable ----
    log_and_record("\n--- CHECKPOINT 3: Geometric Collapse is Observable ---")

    geo_path = os.path.join(results_dir, "phase1_geometry_results.json")
    cp3_go = False
    if os.path.exists(geo_path):
        with open(geo_path) as f:
            geo = json.load(f)

        layers = np.array(geo["layers"])
        visual_er = np.array(geo["visual_effective_rank"])
        text_er = np.array(geo["text_effective_rank"])
        visual_icc = np.array(geo["visual_cosine_concentration"])
        text_icc = np.array(geo["text_cosine_concentration"])
        cka = np.array(geo["cross_modal_cka"])

        cliff = geo.get("cliff_boundary")
        if cliff is not None:
            cliff_idx = np.argmin(np.abs(layers - cliff))

            # Compute indicators
            v_er_ratio = visual_er[cliff_idx:].mean() / visual_er[:cliff_idx].mean() if cliff_idx > 0 else 1.0
            t_er_ratio = text_er[cliff_idx:].mean() / text_er[:cliff_idx].mean() if cliff_idx > 0 else 1.0
            icc_change = visual_icc[cliff_idx:].mean() - visual_icc[:cliff_idx].mean() if cliff_idx > 0 else 0.0
            cka_change = cka[cliff_idx:].mean() - cka[:cliff_idx].mean() if cliff_idx > 0 else 0.0

            log_and_record(f"  Visual ER post/pre ratio: {v_er_ratio:.3f} (want <0.7)")
            log_and_record(f"  Text ER post/pre ratio: {t_er_ratio:.3f} (want >0.85)")
            log_and_record(f"  Visual ICC change: +{icc_change:.3f} (want positive)")
            log_and_record(f"  CKA change: +{cka_change:.3f} (want positive)")

            checks = sum([
                v_er_ratio < 0.7,
                t_er_ratio > 0.85,
                icc_change > 0.05,
                cka_change > 0.05,
            ])

            if checks >= 3:
                log_and_record(f"  ✓ GO ({checks}/4 criteria met)")
                cp3_go = True
            elif checks >= 2:
                log_and_record(f"  ⚠ MARGINAL ({checks}/4 criteria met)")
            else:
                log_and_record(f"  ✗ NO-GO ({checks}/4 criteria met)", "warning")
    else:
        log_and_record("  [Geometry results not found]", "warning")

    # ---- Overall Decision ----
    log_and_record("\n" + "=" * 70)
    log_and_record("OVERALL GO/NO-GO DECISION")
    log_and_record("=" * 70)

    critical_pass = cp1_go and cp2_go
    if critical_pass and cp3_go:
        log_and_record("✓ FULL GO: All critical checkpoints passed. Proceed to Phase 2 (Geometric Adapter design).")
    elif critical_pass:
        log_and_record("⚠ PARTIAL GO: Critical checkpoints passed, geometric evidence is weak. Consider reframing narrative.")
    elif cp1_go:
        log_and_record("⚠ PARTIAL GO: Cliff causes failures, but EVD correlation unclear. Reframe EVD as efficiency metric.")
    else:
        log_and_record("✗ NO-GO: Critical checkpoints failed. Pivot direction or reframe as analysis-only paper.")

    # Save report
    report_path = os.path.join(results_dir, "checkpoint_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
