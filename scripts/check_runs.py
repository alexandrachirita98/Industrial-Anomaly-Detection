"""Check that the Stage-1 runs finished and that their numbers are sane.

    python scripts/check_runs.py                      # everything under outputs/
    python scripts/check_runs.py --expect-rounds 10   # also require 10 FL rounds
    python scripts/check_runs.py --expect-seeds 3     # also require 3 seeds per experiment

Exit code 0 = every check passed, 1 = at least one FAIL. WARN never fails the
run: it marks something to look at by eye (a chance-level AUROC is a result,
not a broken job).

Three kinds of check, in order of how much they can tell you:
  completeness - did the job write the files it writes only when it finishes?
  consistency  - do the numbers agree with each other (sample counts, seeds)?
  plausibility - are they in the range a working model produces?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
rows: list[tuple[str, str, str]] = []


def add(status: str, what: str, detail: str = "") -> None:
    rows.append((status, what, detail))


def size(arm: dict) -> int | None:
    """Images in one arm. Newer runs write samples, older ones only n_images."""
    for key in ("n_images", "samples"):
        if key in arm:
            return int(arm[key])
    return None


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # unreadable or truncated JSON is a failure
        add(FAIL, str(path.relative_to(REPO)), f"unreadable: {type(exc).__name__}")
        return None


# --------------------------------------------------------------- CNN arms (A, B)
def check_supervised(exp: Path, expect_seeds: int | None) -> None:
    """outputs/experiments/<name>/ written by centralised.py or run_seeds.py.

    metrics.json is written BEFORE the evaluation, results.json after it, so
    metrics without results means the job died between the two (or was killed).
    """
    name = exp.name
    seeds = sorted(exp.glob("seed*/"))
    if not seeds:
        add(FAIL, name, "no seed*/ directories at all")
        return

    done = []
    for seed in seeds:
        tag = f"{name}/{seed.name}"
        has_metrics = (seed / "metrics.json").exists()
        has_results = (seed / "results.json").exists()
        has_model = (seed / "final_model.pt").exists()
        if has_results:
            done.append(seed)
        elif has_metrics or has_model:
            add(FAIL, tag, "metrics.json/final_model.pt but NO results.json "
                           "-> killed before evaluation, re-run this seed")
            continue
        else:
            add(FAIL, tag, "empty seed directory")
            continue

        res = load(seed / "results.json")
        if res is None:
            continue

        # ---- consistency: per-client sample counts must add up to the val set
        per_client = res.get("val_per_client", {})
        if not per_client:
            add(FAIL, tag, "results.json has no val_per_client")
        else:
            total = sum(c["samples"] for c in per_client.values())
            add(OK, tag, f"{len(per_client)} clients, {total} val images, "
                         f"val bal.acc {res['val']['balanced_accuracy']:.3f}")

        # ---- plausibility: chance is exactly 0.5 for balanced accuracy
        for split in ("val", "test"):
            if split not in res:
                add(FAIL, tag, f"no {split} block in results.json")
                continue
            bal = res[split]["balanced_accuracy"]
            if not 0.0 <= bal <= 1.0:
                add(FAIL, tag, f"{split} balanced_accuracy={bal} outside [0,1]")
            elif bal <= 0.5:
                add(WARN, tag, f"{split} balanced_accuracy={bal:.3f} <= 0.5 "
                               "-> at or below chance, the model learnt nothing")
            elif bal > 0.995:
                add(WARN, tag, f"{split} balanced_accuracy={bal:.3f} -> suspiciously "
                               "perfect, check for a leak between train and test")

    if expect_seeds is not None and len(done) != expect_seeds:
        add(FAIL, name, f"{len(done)} finished seeds, expected {expect_seeds}")

    # ---- summarize.py only writes summary.json when it has read every seed
    summ = exp / "summary.json"
    if not summ.exists():
        add(WARN, name, "no summary.json -> run python experiments/summarize.py "
                        f"outputs/experiments/{name}")
    else:
        s = load(summ)
        if s and "seeds" in s:
            if len(s["seeds"]) != len(done):
                add(WARN, name, f"summary.json covers {len(s['seeds'])} seeds but "
                                f"{len(done)} are finished -> summary is stale, re-run summarize.py")
            else:
                add(OK, f"{name}/summary.json", f"seeds {', '.join(s['seeds'])}")


# ------------------------------------------------------------------ PaDiM (C, D)
def check_padim(exp: Path, expect_seeds: int | None) -> None:
    """outputs/experiments/<name>/ written by padim_experiment.py."""
    name = exp.name
    seeds = sorted(exp.glob("seed*/"))
    for seed in seeds:
        tag = f"{name}/{seed.name}"
        if not (seed / "metrics.json").exists():
            add(FAIL, tag, "no metrics.json -> the seed did not finish")
            continue
        m = load(seed / "metrics.json")
        if m is None:
            continue
        met = m.get("metrics", {})
        if "pooled" not in met:
            add(FAIL, tag, "metrics.json has no pooled arm")
            continue

        pooled = met["pooled"]
        clients = [k for k in met if k.startswith("client_")]
        # ---- consistency: the client slices partition the test set exactly.
        # Older runs record only n_images, newer ones also record samples, and
        # the *_mean arms are averages over arms with neither: use whichever is
        # there and skip the arms that have no size of their own.
        sizes = [size(met[c]) for c in clients]
        client_sum = sum(x for x in sizes if x is not None)
        if None in sizes:
            add(WARN, tag, "a client arm records no size (n_images/samples) -> "
                           "cannot check that the slices partition the test set")
        elif client_sum != pooled["n_images"]:
            add(FAIL, tag, f"client samples sum to {client_sum} but pooled n_images="
                           f"{pooled['n_images']} -> the slices do not partition the test set")
        else:
            add(OK, tag, f"{len(clients)} clients, {pooled['n_images']} test images "
                         f"({pooled['n_anomalies']} anomalous), pooled AUROC {pooled['auroc']:.4f}")

        # ---- consistency: a test set with no anomalies makes AUROC undefined
        if pooled["n_anomalies"] == 0:
            add(FAIL, tag, "n_anomalies=0 -> AUROC is undefined, wrong split "
                           "(val is normals-only; AUROC needs the test slices)")

        # ---- plausibility: 0.5 is chance, below 0.5 means the sign is flipped
        for arm, vals in met.items():
            if not isinstance(vals, dict) or "auroc" not in vals:
                continue
            a = vals["auroc"]
            if not 0.0 <= a <= 1.0:
                add(FAIL, tag, f"{arm} auroc={a} outside [0,1]")
            elif a < 0.45 and arm.startswith(("client_", "pooled")):
                add(WARN, tag, f"{arm} auroc={a:.4f} < 0.45 -> worse than chance; "
                               "if many arms do this the score sign may be inverted")

    # ---- the summary is the file the notebook reads, so check its seed count
    summ = exp / "summary.json"
    if not summ.exists():
        add(FAIL, name, "no summary.json -> padim_experiment.py did not reach the end")
        return
    s = load(summ)
    if s is None:
        return
    arms = s.get("arms", {})
    if not arms:
        add(FAIL, f"{name}/summary.json", "no arms block")
        return

    ns = {arm: v["auroc"]["n"] for arm, v in arms.items() if "auroc" in v}
    if ns and len(set(ns.values())) > 1:
        add(FAIL, f"{name}/summary.json", f"arms disagree on seed count: {ns}")
    elif ns:
        n = next(iter(ns.values()))
        if expect_seeds is not None and n != expect_seeds:
            add(FAIL, f"{name}/summary.json", f"n={n} seeds, expected {expect_seeds}")
        else:
            add(OK, f"{name}/summary.json", f"{len(arms)} arms, n={n} seeds each")

    # ---- the headline claim of the federated story, checked not assumed
    if "per_client_mean" in arms and "central_on_own_slice_mean" in arms:
        pc = arms["per_client_mean"]["auroc"]["mean"]
        co = arms["central_on_own_slice_mean"]["auroc"]["mean"]
        verdict = "local wins -> real feature skew" if pc > co else "central wins -> no local advantage"
        add(OK, f"{name}/feature-skew", f"per_client_mean {pc:.4f} vs "
                                        f"central_on_own_slice_mean {co:.4f}: {verdict}")


# ------------------------------------------------------- federated CNN (sim dirs)
def check_sim(run: Path, expect_rounds: int | None) -> None:
    """outputs/sim-*/ written by server_app.py at the very end of the run.

    metrics.json only exists if every round finished, so its presence is the
    completeness check and the round count is the "did it run 10?" check.
    """
    name = run.name
    m = load(run / "metrics.json")
    if m is None:
        return
    if not (run / "final_model.pt").exists():
        add(FAIL, name, "metrics.json but no final_model.pt")

    server = m.get("evaluate_metrics_serverapp", {})
    if not server:
        add(FAIL, name, "no evaluate_metrics_serverapp -> nothing was evaluated; "
                        "set test-every-round=true to get a curve")
        return

    rounds = sorted(int(r) for r in server)
    cfg = m.get("run_config", {})
    want = int(cfg.get("num-server-rounds", 0)) or None
    last = rounds[-1]

    # ---- consistency: the rounds recorded must match the rounds configured
    if want and last != want:
        add(FAIL, name, f"last evaluated round {last} but num-server-rounds={want} "
                        "-> the run stopped early")
    elif expect_rounds and last != expect_rounds:
        add(FAIL, name, f"last round {last}, expected {expect_rounds}")
    elif len(rounds) == 1 and (want or 1) > 1:
        add(WARN, name, f"only round {last} recorded of {want} -> test-every-round was off, "
                        "you get a final number but no learning curve")
    else:
        add(OK, name, f"{len(rounds)} evaluated rounds, last={last}")

    # ---- plausibility: the final number, and whether it moved at all
    final = server[str(last)] if str(last) in server else server[last]
    bal = final.get("balanced_accuracy")
    if bal is not None:
        if bal <= 0.5:
            add(WARN, name, f"final balanced_accuracy={bal:.3f} <= 0.5 -> at chance")
        else:
            add(OK, f"{name}/final", f"balanced_accuracy {bal:.3f}, "
                                     f"loss {final.get('loss', float('nan')):.4f}")
    if len(rounds) > 1:
        first = server[str(rounds[0])] if str(rounds[0]) in server else server[rounds[0]]
        d = final.get("loss", 0) - first.get("loss", 0)
        if d > 0:
            add(WARN, name, f"loss rose over training ({first['loss']:.4f} -> "
                            f"{final['loss']:.4f}) -> diverging, lower the learning rate")


# -------------------------------------------------- federated PaDiM (one round)
def check_padim_federated(run: Path) -> None:
    """outputs/padim-federated-*/ : aggregation of sufficient statistics."""
    name = run.name
    m = load(run / "metrics.json")
    if m is None:
        return
    test = m.get("test")
    if not test:
        add(FAIL, name, "no test block in metrics.json")
        return
    if test.get("n_anomalies", 0) == 0:
        add(FAIL, name, "n_anomalies=0 -> AUROC undefined")
    a = test.get("auroc")
    if a is None or not 0.0 <= a <= 1.0:
        add(FAIL, name, f"auroc={a} missing or outside [0,1]")
    else:
        add(OK, name, f"AUROC {a:.4f} over {test['n_images']} images from "
                      f"{test.get('n_clients', '?')} clients, {test.get('n_dims', '?')} dims")
    if not (run / "padim_federated_model.npz").exists():
        add(WARN, name, "no padim_federated_model.npz (fine if you did not keep it; "
                        "never commit it, it is ~66 MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect-rounds", type=int, default=None,
                    help="require this many FL server rounds (e.g. 10)")
    ap.add_argument("--expect-seeds", type=int, default=None,
                    help="require this many finished seeds per experiment (e.g. 3)")
    args = ap.parse_args()

    if not OUT.exists():
        print(f"No {OUT} - nothing has run yet.")
        return 1

    for exp in sorted((OUT / "experiments").glob("*/")) if (OUT / "experiments").exists() else []:
        # a PaDiM experiment is the one whose seeds carry scores.json
        if list(exp.glob("seed*/scores.json")) or (exp / "summary.json").exists() and \
                "arms" in (load(exp / "summary.json") or {}):
            check_padim(exp, args.expect_seeds)
        else:
            check_supervised(exp, args.expect_seeds)

    for run in sorted(OUT.glob("sim-*/")):
        check_sim(run, args.expect_rounds)
    for run in sorted(OUT.glob("deploy-*/")):
        check_sim(run, args.expect_rounds)
    for run in sorted(OUT.glob("padim-federated-*/")):
        check_padim_federated(run)

    width = max((len(w) for _, w, _ in rows), default=10)
    for status, what, detail in rows:
        print(f"[{status}] {what:<{width}}  {detail}")

    n_fail = sum(1 for s, _, _ in rows if s == FAIL)
    n_warn = sum(1 for s, _, _ in rows if s == WARN)
    print(f"\n{len(rows)} checks: {len(rows) - n_fail - n_warn} OK, {n_warn} WARN, {n_fail} FAIL")
    if n_fail:
        print("FAIL means a run is incomplete or self-inconsistent: re-run it.")
    elif n_warn:
        print("No failures. WARN lines are results to interpret, not broken jobs.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
