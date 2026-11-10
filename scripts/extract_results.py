"""Pull the arm A (centralised) and arm B (federated) numbers out of outputs/.

    python scripts/extract_results.py                  # all three views
    python scripts/extract_results.py --csv            # also fill the heterogeneity CSV

Writes nothing unless --csv is given (then outputs/tables/stage1_ab_results.csv).
Reads only summary.json / results.json / metrics.json, so it is safe to run
while other jobs are still going: splits that have not finished are skipped and
listed at the end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "outputs" / "experiments"

# split dir -> (label, mean_EMD) ordered by heterogeneity, read from Member 1's CSV
LABELS = {"autovi_iid": "IID", "autovi": "Group (real)",
          "autovi_dir_a1.0": "Dirichlet a=1.0", "autovi_dir_a0.1": "Dirichlet a=0.1"}


def emds() -> dict[str, float]:
    csv = REPO / "outputs" / "tables" / "stage1_split_heterogeneity.csv"
    out: dict[str, float] = {}
    if not csv.exists():
        return out
    import csv as _csv
    for row in _csv.DictReader(csv.open()):
        if row.get("partition_dir"):
            out[row["partition_dir"]] = float(row["mean_EMD"])
    return out


def stat(d: dict | None, *keys):
    """Nested get that tolerates a missing run."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(s) -> str:
    if s is None:
        return "—"
    if isinstance(s, dict):
        return f"{s['mean']:.3f} ± {s['std']:.3f}"
    return f"{s:.3f}"


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fl_rounds(exp: Path) -> tuple[dict | None, int, int]:
    """Mean/std of the final-round server balanced accuracy, + round bookkeeping."""
    vals, cfg_rounds, last = [], 0, 0
    for m in sorted(exp.glob("seed*/metrics.json")):
        d = load(m)
        server = stat(d, "evaluate_metrics_serverapp") or {}
        if not server:
            continue
        cfg_rounds = int(stat(d, "run_config", "num-server-rounds") or 0)
        rounds = sorted(int(r) for r in server)
        last = rounds[-1]
        vals.append(server[str(last)]["balanced_accuracy"])
    if not vals:
        return None, cfg_rounds, last
    import statistics
    return ({"mean": statistics.fmean(vals),
             "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}, cfg_rounds, last)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="store_true", help="write outputs/tables/stage1_ab_results.csv")
    ap.add_argument("--expect-rounds", type=int, default=10,
                    help="the round count arm B was supposed to train for")
    args = ap.parse_args()

    e = emds()
    order = sorted(LABELS, key=lambda k: e.get(k, 9))
    missing: list[str] = []
    rows = []

    for split in order:
        label, emd = LABELS[split], e.get(split)
        a = load(EXP / f"central-{split}" / "summary.json")
        b = load(EXP / f"fedavg-{split}" / "summary.json")
        b_dir = EXP / f"fedavg-{split}"
        b_final, cfg_r, last_r = fl_rounds(b_dir) if b_dir.exists() else (None, 0, 0)

        if a is None:
            missing.append(f"central-{split} (no summary.json)")
        if b is None:
            missing.append(f"fedavg-{split} (no summary.json)")
        if b_dir.exists() and cfg_r and last_r != cfg_r:
            missing.append(f"fedavg-{split}: {last_r} of {cfg_r} rounds -> stopped early")
        # A run can COMPLETE 5 of 5 rounds and still be the wrong run: that is the
        # stale-config case, invisible to a last-vs-configured comparison.
        if cfg_r and cfg_r != args.expect_rounds:
            missing.append(f"fedavg-{split}: trained {cfg_r} rounds, expected "
                           f"{args.expect_rounds} -> stale config, re-run with --force")

        a_val = stat(a, "val", "balanced_accuracy")
        b_val = stat(b, "val", "balanced_accuracy")
        gap = (a_val["mean"] - b_val["mean"]) if a_val and b_val else None
        rows.append({
            "split": split, "label": label, "mean_EMD": emd,
            "A_val": a_val, "A_test": stat(a, "test", "balanced_accuracy"),
            "B_val": b_val, "B_test": stat(b, "test", "balanced_accuracy"),
            "gap": gap, "rounds": f"{last_r}/{cfg_r}" if cfg_r else "—",
            "A_worst": stat(a, "val_worst_client_balanced_accuracy"),
            "B_worst": stat(b, "val_worst_client_balanced_accuracy"),
            "B_final_round": b_final,
        })

    # ---------------------------------------------------------- view 1: the headline
    print("\n### A vs B by split (validation balanced accuracy, mean ± std over seeds)\n")
    print("| Split | mean_EMD | A centralised | B federated | A−B gap | rounds |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        g = f"{r['gap']:+.3f}" if r["gap"] is not None else "—"
        emd = f"{r['mean_EMD']:.3f}" if r["mean_EMD"] is not None else "—"
        flag = "" if r["rounds"] in ("—", f"{args.expect_rounds}/{args.expect_rounds}") else " **!**"
        print(f"| {r['label']} | {emd} | {fmt(r['A_val'])} | {fmt(r['B_val'])} "
              f"| {g} | {r['rounds']}{flag} |")

    # ---------------------------------------------------------- view 2: test numbers
    print("\n### Test balanced accuracy (for the final report only)\n")
    print("| Split | A centralised | B federated | A−B gap |")
    print("|---|---|---|---|")
    for r in rows:
        at, bt = r["A_test"], r["B_test"]
        g = f"{at['mean'] - bt['mean']:+.3f}" if at and bt else "—"
        print(f"| {r['label']} | {fmt(at)} | {fmt(bt)} | {g} |")

    # ---------------------------------------------------------- view 3: fairness
    print("\n### Worst-client validation balanced accuracy (fairness)\n")
    print("| Split | A worst client | B worst client |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {fmt(r['A_worst'])} | {fmt(r['B_worst'])} |")

    if args.csv:
        import csv as _csv
        out = REPO / "outputs" / "tables" / "stage1_ab_results.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["partition_dir", "label", "mean_EMD", "centralised_bal_acc",
                        "centralised_bal_acc_std", "federated_bal_acc",
                        "federated_bal_acc_std", "fl_gap", "rounds",
                        "centralised_test_bal_acc", "federated_test_bal_acc",
                        "worst_client_bal_acc_A", "worst_client_bal_acc_B"])
            for r in rows:
                w.writerow([
                    r["split"], r["label"], r["mean_EMD"],
                    stat(r["A_val"], "mean"), stat(r["A_val"], "std"),
                    stat(r["B_val"], "mean"), stat(r["B_val"], "std"),
                    r["gap"], r["rounds"],
                    stat(r["A_test"], "mean"), stat(r["B_test"], "mean"),
                    stat(r["A_worst"], "mean"), stat(r["B_worst"], "mean")])
        print(f"\nWrote {out.relative_to(REPO)}")
        print("(a NEW file: Member 1's stage1_split_heterogeneity.csv is left untouched,")
        print(" because their notebook regenerates it from the sweep)")

    if missing:
        print("\nNot ready yet / needs attention:")
        for m in dict.fromkeys(missing):
            print("  -", m)
    else:
        print("\nAll four splits present for both arms.")


if __name__ == "__main__":
    main()
