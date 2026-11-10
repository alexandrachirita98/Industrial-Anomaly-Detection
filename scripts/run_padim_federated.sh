#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Federated PaDiM: ONE round, no gradients, exact aggregation.
#
# The per-patch means and covariances are sufficient statistics, so summing the
# clients' partial sums rebuilds the pooled model EXACTLY - the federated AUROC
# should equal the centralised pooled AUROC to three decimals.
# See brainstorming/stage2_federated_padim_implementation_guide.md.
#
# Usage:
#   bash scripts/run_padim_federated.sh [PARTITIONS_DIR] [RUN_CONFIG_OVERRIDES]
#
# Examples:
#   bash scripts/run_padim_federated.sh
#   bash scripts/run_padim_federated.sh data/partitions/autovi_official "seed=2"
#   bash scripts/run_padim_federated.sh data/partitions/autovi_official "padim-n-dims=50"
#
# Needs the OFFICIAL protocol partitions (normal-only train): PaDiM models
# NORMAL appearance, and the clients refuse to run on contaminated splits.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

PARTITIONS="$(abspath "${1:-$REPO_ROOT/data/partitions/autovi_official}")"
EXTRA_CONFIG="${2:-}"

if [ ! -f "$PARTITIONS/test.npz" ]; then
    echo "No test.npz in $PARTITIONS." >&2
    echo "Build the official partitions first:" >&2
    echo "  python data_splitting/autovi_official_partitions.py" >&2
    exit 1
fi

# run_simulation.sh already sets min-nodes, partitions-dir and save-dir, and
# handles Ray / the local SuperLink. Do NOT add min-nodes here: passing it twice
# makes Flower complain about something else entirely.
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/outputs/padim-federated-$(basename "$PARTITIONS")-$(date +%Y%m%d-%H%M%S)}"
export SAVE_DIR

echo ">>> Federated PaDiM (1 round, no gradients) on $PARTITIONS"
bash "$HERE/run_simulation.sh" "$PARTITIONS" \
    "padim-mode=true num-server-rounds=1 fraction-evaluate=0.0 $EXTRA_CONFIG"
