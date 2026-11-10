"""PaDiM baseline runner: pooled and per-client, over several seeds.

The unsupervised arm of Stage 1. PaDiM is training-free (see experiments/padim.py),
so a "run" is a forward pass plus a mean and a covariance per patch position.

    python experiments/padim_experiment.py --partitions data/partitions/autovi_official
    python experiments/padim_experiment.py --partitions data/partitions/autovi_official \\
        --seeds 1 2 3 --name padim-autovi

Two arms, both reported:
  pooled      - one PaDiM on all clients' normal training images (the literal
                "centralised" baseline Stage 1 asks for)
  per-client  - one PaDiM per client, scored on its own slice of the test set
                (what the paper effectively does: it reports per-category AUROC)

Requires the OFFICIAL protocol partitions (normal-only train). Pointed at the
supervised re-split it raises instead of quietly discarding the labelled defects:
those images are test positives here, not rubbish.

Primary metric is AUROC, which needs no threshold and is what the AutoVI paper
reports. The thresholded numbers use the 95th percentile of the VALIDATION
normals' scores - no anomalies and no test data, so nothing leaks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from padim import PaDiM  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "fl_app"))
from aitdm_fl.task import get_device, load_npz_arrays, set_seed  # noqa: E402

# A priori false-alarm rate on known-good parts; fixed, never tuned.
NORMAL_QUANTILE = 95.0


def check_normal_only(y: np.ndarray, partitions: Path, what: str) -> None:
    """The protocol guarantee, enforced loudly rather than silently repaired."""
    n_anom = int((y == 1).sum())
    if n_anom:
        raise SystemExit(
            f"{n_anom} anomalies in the {what} under {partitions}. "
            "PaDiM models NORMAL appearance: training on defects teaches it that defects "
            "are normal and collapses the AUROC. You are probably pointing at the "
            "supervised partitions - use data/partitions/autovi_official (--mode official)."
        )


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray,
                        threshold: float) -> dict[str, float]:
    """AUROC and AP are threshold-free; the other two need the threshold."""
    out: dict[str, float] = {}
    if len(np.unique(y_true)) < 2:
        # A single-class slice has no ROC. Report what is defined, flag the rest.
        out["auroc"] = float("nan")
        out["average_precision"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["average_precision"] = float(average_precision_score(y_true, scores))
        # TPR at 95% TNR: the operating point the paper discusses.
        tnr_threshold = float(np.percentile(scores[y_true == 0], NORMAL_QUANTILE))
        out["tpr_at_95_tnr"] = float((scores[y_true == 1] > tnr_threshold).mean())

    y_pred = (scores > threshold).astype(np.int64)
    out["balanced_accuracy_at_threshold"] = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) >= 2 else float("nan"))
    out["threshold"] = float(threshold)
    out["n_images"] = int(len(y_true))
    out["n_anomalies"] = int((y_true == 1).sum())
    return out


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(arr):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(len(arr))}


def save_heatmaps(run_dir: Path, x: np.ndarray, maps: np.ndarray, y: np.ndarray,
                  scores: np.ndarray, per_class: int = 2) -> int:
    """A few image/heatmap pairs per class. Free interpretability, costs nothing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib missing, skipping heatmaps)")
        return 0

    out = run_dir / "heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for label, tag in ((0, "normal"), (1, "anomaly")):
        idx = np.flatnonzero(y == label)
        if not len(idx):
            continue
        # The most anomalous-looking of each class: the informative ones.
        for rank, i in enumerate(idx[np.argsort(-scores[idx])][:per_class]):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(x[i].transpose(1, 2, 0).clip(0, 1))
            axes[0].set_title(f"{tag} (score {scores[i]:.1f})", fontsize=9)
            axes[1].imshow(x[i].transpose(1, 2, 0).clip(0, 1))
            axes[1].imshow(maps[i], cmap="jet", alpha=0.5)
            axes[1].set_title("PaDiM anomaly map", fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out / f"{tag}_{rank}.png", dpi=110)
            plt.close(fig)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaDiM unsupervised baseline (AutoVI official protocol).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--partitions", type=Path, required=True,
                        help="official-protocol partitions (normal-only train)")
    parser.add_argument("--n-dims", type=int, default=100,
                        help="random channel subset kept out of 448")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Gaussian blur on the anomaly map")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--name", help="experiment name (default: padim-<partitions>)")
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--allow-contaminated-train", action="store_true",
                        help="Option B fallback: filter anomalies out of train with a "
                             "warning instead of raising. Not comparable with the paper.")
    args = parser.parse_args()

    partitions = args.partitions.resolve()
    name = args.name or f"padim-{partitions.name}"
    exp_dir = REPO_ROOT / "outputs" / "experiments" / name
    device = get_device()

    files = sorted(partitions.glob("client_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no client_*.npz in {partitions}")

    # --- training images, per client, normal-only ---------------------------
    client_train = []
    for path in files:
        x_tr, y_tr = load_npz_arrays(path, "train")
        if args.allow_contaminated_train:
            keep = y_tr == 0
            if (~keep).sum():
                print(f"WARNING {path.name}: dropping {int((~keep).sum())} anomalies from "
                      "train (--allow-contaminated-train). NOT comparable with the paper.")
            x_tr = x_tr[keep]
        else:
            check_normal_only(y_tr, partitions, f"training split of {path.name}")
        client_train.append(x_tr)

    # --- validation normals, for the threshold ------------------------------
    client_val = []
    for path in files:
        x_va, y_va = load_npz_arrays(path, "val")
        client_val.append(x_va[y_va == 0])

    # --- test set, with owner ids for the per-client arm --------------------
    test = np.load(partitions / "test.npz")
    y_test = test["y_test"].reshape(-1).astype(np.int64)
    x_test = test["x_test"].astype(np.float32)
    if x_test.ndim == 4 and x_test.shape[-1] in (1, 3):
        x_test = x_test.transpose(0, 3, 1, 2)
    x_test = x_test / 255.0
    if x_test.shape[1] == 1:
        x_test = np.repeat(x_test, 3, axis=1)
    # load_npz_arrays returns only (x, y), so owner_test is read directly.
    owner = (test["owner_test"].reshape(-1).astype(np.int64)
             if "owner_test" in test.files else None)
    if owner is None:
        print(f"WARNING: no owner_test in {partitions}/test.npz - per-client arm skipped. "
              "Rebuild with data_splitting/autovi_official_partitions.py.")

    pooled_train = np.concatenate(client_train)
    print(f"{name}: {len(files)} clients, pooled train {len(pooled_train)} normal images, "
          f"test {len(y_test)} images ({int((y_test == 1).sum())} anomalies), device {device}")

    per_seed: list[dict] = []
    for seed in args.seeds:
        set_seed(seed)
        run_dir = exp_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_metrics: dict[str, dict] = {}

        # --- pooled arm ----------------------------------------------------
        model = PaDiM(n_dims=args.n_dims, seed=seed, device=device)
        model.fit(pooled_train, batch_size=args.batch_size)
        val_scores, _ = model.score(np.concatenate(client_val),
                                    batch_size=args.batch_size, sigma=args.sigma,
                                    return_maps=False)
        threshold = float(np.percentile(val_scores, NORMAL_QUANTILE))
        scores, maps = model.score(x_test, batch_size=args.batch_size, sigma=args.sigma,
                                   return_maps=not args.no_heatmaps)
        seed_metrics["pooled"] = metrics_from_scores(y_test, scores, threshold)
        print(f"  seed {seed} pooled      AUROC {seed_metrics['pooled']['auroc']:.3f}  "
              f"AP {seed_metrics['pooled']['average_precision']:.3f}")

        if maps is not None:
            n = save_heatmaps(run_dir, x_test, maps, y_test, scores)
            print(f"    {n} heatmaps -> {run_dir / 'heatmaps'}")
        del maps

        # --- per-client arm ------------------------------------------------
        client_scores = np.full(len(y_test), np.nan)
        if owner is not None:
            for client_id, (x_tr, x_va) in enumerate(zip(client_train, client_val)):
                mask = owner == client_id
                if not mask.any():
                    continue
                m = PaDiM(n_dims=args.n_dims, seed=seed, device=device)
                m.fit(x_tr, batch_size=args.batch_size)
                v, _ = m.score(x_va, batch_size=args.batch_size, sigma=args.sigma,
                               return_maps=False)
                thr_i = float(np.percentile(v, NORMAL_QUANTILE))
                s_i, _ = m.score(x_test[mask], batch_size=args.batch_size,
                                 sigma=args.sigma, return_maps=False)
                client_scores[mask] = s_i
                key = f"client_{client_id}"
                seed_metrics[key] = metrics_from_scores(y_test[mask], s_i, thr_i)
                print(f"  seed {seed} {key}    AUROC {seed_metrics[key]['auroc']:.3f}  "
                      f"AP {seed_metrics[key]['average_precision']:.3f}")

            client_keys = [k for k in seed_metrics if k.startswith("client_")]
            seed_metrics["per_client_mean"] = {
                metric: float(np.nanmean([seed_metrics[k][metric] for k in client_keys]))
                for metric in ("auroc", "average_precision",
                               "balanced_accuracy_at_threshold", "tpr_at_95_tnr")
            }
            print(f"  seed {seed} per-client  AUROC "
                  f"{seed_metrics['per_client_mean']['auroc']:.3f} (mean over clients)")

        run_config = {"partitions-dir": str(partitions), "seed": seed, "mode": "padim",
                      "n-dims": args.n_dims, "sigma": args.sigma,
                      "cov-eps": 0.01, "backbone": "resnet18-imagenet1k-v1",
                      "threshold-rule": f"{NORMAL_QUANTILE}th pct of validation normals",
                      "protocol": "official (normal-only train)"}
        (run_dir / "metrics.json").write_text(json.dumps(
            {"run_config": run_config, "metrics": seed_metrics}, indent=2))
        # Raw per-image scores, so Member 3 can build ROC curves without re-running.
        (run_dir / "scores.json").write_text(json.dumps({
            "label": y_test.tolist(),
            "owner": owner.tolist() if owner is not None else None,
            "pooled_score": scores.tolist(),
            "per_client_score": [None if np.isnan(v) else float(v) for v in client_scores],
            "pooled_threshold": threshold,
        }, indent=2))
        per_seed.append(seed_metrics)

    # --- mean +/- std over seeds -------------------------------------------
    arms = sorted({k for m in per_seed for k in m})
    summary = {
        "experiment": name,
        "partitions": str(partitions),
        "seeds": args.seeds,
        "protocol": "official (AutoVI unsupervised: normal-only train, all defects in test)",
        "arms": {
            arm: {
                metric: mean_std([m[arm][metric] for m in per_seed
                                  if arm in m and metric in m[arm]])
                for metric in ("auroc", "average_precision",
                               "balanced_accuracy_at_threshold", "tpr_at_95_tnr")
            }
            for arm in arms
        },
    }
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'arm':<18}{'AUROC':>16}{'AP':>16}")
    for arm in arms:
        a = summary["arms"][arm]
        print(f"{arm:<18}"
              f"{a['auroc']['mean']:>9.3f} ± {a['auroc']['std']:<4.3f}"
              f"{a['average_precision']['mean']:>9.3f} ± {a['average_precision']['std']:<4.3f}")
    print(f"\nWrote {exp_dir}")


if __name__ == "__main__":
    main()
