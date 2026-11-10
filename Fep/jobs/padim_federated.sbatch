#!/usr/bin/env bash
#SBATCH --job-name=padim_fed
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# ---------------------------------------------------------------------------
# Federated PaDiM: ONE round, no gradients, exact aggregation. The federated
# AUROC should equal the centralised pooled AUROC to three decimals - see
# brainstorming/stage2_federated_padim_implementation_guide.md.
#
#   bash Fep/submit.sh padim_federated
#   bash Fep/submit.sh padim_federated data/partitions/autovi_official "seed=2"
#
# Needs the OFFICIAL protocol partitions (normal-only train). No training loop,
# so CPU is plenty; the only cost is one ResNet18 forward pass per client.
# Results: outputs/padim-federated-<partition-name>-<timestamp>/
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh run_padim_federated.sh "$@"
bash scripts/run_padim_federated.sh "$@"
