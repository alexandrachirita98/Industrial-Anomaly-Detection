#!/usr/bin/env bash
#SBATCH --job-name=padim
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# ---------------------------------------------------------------------------
# PaDiM unsupervised baseline (AutoVI official protocol), as
# experiments/padim_experiment.py. No training loop, so CPU is plenty.
#   bash Fep/submit.sh padim --partitions data/partitions/autovi_official --seeds 1 2 3
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh padim_experiment.py "$@"
python experiments/padim_experiment.py "$@"
