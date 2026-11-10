#!/usr/bin/env bash
#SBATCH --job-name=autovi_data
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
# ---------------------------------------------------------------------------
# Build every AutoVI partition the team needs, in one pass. Pure CPU work
# (unzip, resize, split), so no GPU - but the 400px npz is large, hence --mem.
#
#   bash Fep/submit.sh autovi_data              # all of it
#   bash Fep/submit.sh autovi_data supervised   # steps 1+2 only (A and B)
#   bash Fep/submit.sh autovi_data official     # step 3 only (C, PaDiM)
#
# Expects the three zips in data/raw/autovi/ (data/ -> ~/AITDM_Project/data).
# Everything is idempotent: finished outputs are skipped, so re-submitting
# after a TIMEOUT resumes instead of starting over.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh "${1:-all}"

WHAT="${1:-all}"
CATEGORIES=(pipe_staple pipe_clip engine_wiring)
SEED=42
NUM_CLIENTS=3

ZIPS=()
for c in "${CATEGORIES[@]}"; do ZIPS+=("data/raw/autovi/$c.zip"); done
for z in "${ZIPS[@]}"; do
    [ -f "$z" ] || { echo "ERROR: $z not found. Download the AutoVI zips into data/raw/autovi/ first." >&2; exit 1; }
done

# --- 1. zips -> npz, 400px, for experiments A and B ------------------------
# --keep-extracted is REQUIRED: the data-lead notebook's first code cell
# asserts data/raw/autovi_npz/_extracted exists and reads the PNGs from it for
# the EDA half. Without the flag autovi.py deletes it and that cell fails.
# --size 400 is REQUIRED too: cell 39 asserts summary["image_size"] == 400.
if [ "$WHAT" = all ] || [ "$WHAT" = supervised ]; then
    if [ -f data/raw/autovi_npz/autovi_all.npz ] && [ -d data/raw/autovi_npz/_extracted ]; then
        echo ">>> [1/3] skip conversion (autovi_all.npz and _extracted already there)"
    else
        echo ">>> [1/3] converting zips -> npz (400px, keeping _extracted)"
        srun python data_splitting/autovi.py --zips "${ZIPS[@]}" \
            --out data/raw/autovi_npz --size 400 --keep-extracted \
            --merged data/raw/autovi_npz/autovi_all.npz
    fi

    # --- 2. the four supervised partition sets ----------------------------
    # autovi/ is what A and B are graded on; the other three are the
    # heterogeneity sweep the notebook builds and analyses in cells 63-67.
    echo ">>> [2/3] supervised partitions"
    build_split() {   # build_split <out-name> <extra split_data.py args...>
        local name="$1"; shift
        if [ -f "data/partitions/$name/meta.json" ]; then
            echo "    skip  $name (already built)"
            return
        fi
        echo "    build $name"
        srun python data_splitting/split_data.py \
            --input data/raw/autovi_npz/autovi_all.npz \
            --num-clients "$NUM_CLIENTS" --class-names "normal,anomaly" \
            --seed "$SEED" --out "data/partitions/$name" "$@"
    }
    build_split autovi          --strategy group --group-col category
    build_split autovi_iid      --strategy iid
    build_split autovi_dir_a1.0 --strategy dirichlet --alpha 1.0
    build_split autovi_dir_a0.1 --strategy dirichlet --alpha 0.1
fi

# --- 3. the official-protocol partitions, for experiment C (PaDiM) --------
# NOT split_data.py: the merged npz loses AutoVI's normal-only train split and
# a stratified re-split would put anomalies back into training. Same grouping
# as step 2 (one client per production part), different train/test assignment.
# Separate --out, so --size 128 here does not touch the 400px npz above.
if [ "$WHAT" = all ] || [ "$WHAT" = official ]; then
    echo ">>> [3/3] official-protocol partitions"
    if [ -f data/raw/autovi_official_npz/"${CATEGORIES[0]}".npz ]; then
        echo "    skip  conversion (official npz already there)"
    else
        srun python data_splitting/autovi.py --zips "${ZIPS[@]}" \
            --out data/raw/autovi_official_npz --mode official --size 128
    fi
    if [ -f data/partitions/autovi_official/meta.json ]; then
        echo "    skip  autovi_official (already built)"
    else
        srun python data_splitting/autovi_official_partitions.py \
            --npz-dir data/raw/autovi_official_npz --out data/partitions/autovi_official
    fi
fi

echo
echo "=== done, $(date). Partitions:"
for d in autovi autovi_iid autovi_dir_a1.0 autovi_dir_a0.1 autovi_official; do
    [ -f "data/partitions/$d/meta.json" ] && echo "    data/partitions/$d"
done
