"""Generate notebooks/01_data_lead_dataset_and_partitioning.ipynb (Member 1).

The notebook is generated from this script so it can be regenerated and
re-executed reproducibly. Run:

    python scripts/build_member1_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_data_lead_dataset_and_partitioning.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "01_data_lead_dataset_and_partitioning.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
md(r"""
# Stage 1 — Dataset Description and Federated Partitioning

**Role:** Member 1 — Data & Experiment Design Lead
**Course:** AI for Trustworthy Decision Making
**Domain:** Industrial Anomaly Detection
**Dataset:** AutoVI — Automotive Visual Inspection Dataset (Renault Group)

---

> ### 🔒 Data attribution and redistribution notice
>
> The images shown in this notebook are from the **AutoVI dataset**, © 2023–2024
> **Renault Group**, released under
> **[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)**
> (attribution, **non-commercial**, share-alike).
>
> > Carvalho, P., Lafou, M., Durupt, A., Leblanc, A. & Grandvalet, Y. (2024).
> > *The Automotive Visual Inspection Dataset (AutoVI): A Genuine Industrial Production
> > Dataset for Unsupervised Anomaly Detection.* Computers in Industry.
> > [doi:10.1016/j.compind.2024.104151](https://doi.org/10.1016/j.compind.2024.104151) ·
> > dataset [doi:10.5281/zenodo.10459003](https://doi.org/10.5281/zenodo.10459003)
>
> **This notebook contains embedded photographs from that dataset.** Keep this repository
> private or internal to the university. The raw data itself is never committed (`data/`
> is git-ignored) and must not be used commercially.

---

## What this notebook does

This notebook is the **data half** of Stage 1. It runs in five phases:

| Phase | What happens | Why it is needed |
|---|---|---|
| **A** | **Describe the dataset** — size, shape, class balance, pixel/feature statistics, duplicates, leakage | The professor asked for this first: *we need to know everything about the data before we model it* |
| **B** | Split the data into federated clients (non-IID) | Stage 1 requires 3–5 simulated clients with non-IID splits |
| **C** | Verify and quantify the split | A claim of "non-IID" needs a number, not an adjective |
| **D** | Train one model on this split | The role requires initial training |
| **E** | Export the artefacts for the team | Members 2 and 3 depend on these |

**Phase A is the part that is new and the part that matters most.** A model trained on a
dataset nobody characterised is a model nobody can defend. Everything in Phase A exists to
answer one question: *is this data balanced, clean, and representative — and if not, how
exactly is it broken?*

---

## How to read this notebook

Every section follows the same shape: a short explanation of **why** the check matters,
the code, and then an interpretation of what the output means. The notebook is meant to be
readable as a standalone document, as the Stage 2 guidelines require.
""")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
md(r"""
---

## 0. Setup

Works both locally and in Google Colab. Locally it does nothing except fix the paths.

> **Run this locally.** AutoVI is Renault Group production-line imagery under a
> **non-commercial** licence (CC BY-NC-SA 4.0). Uploading it to a third-party notebook
> service is a data-handling decision you should not make casually in a course about
> trustworthy AI — and the Colab branch below cannot fetch it for you anyway, since the
> archives are not in the repository.
""")

code(r"""
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not Path("AITDM_FLOWER_PROJECT_TUTORIAL").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL.git"],
            check=True)
    os.chdir("AITDM_FLOWER_PROJECT_TUTORIAL")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
                   check=True)

# Find the repository root whether we are in notebooks/ or at the top level
REPO = Path.cwd()
if not (REPO / "data_splitting").exists() and (REPO.parent / "data_splitting").exists():
    REPO = REPO.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO / "fl_app"))
sys.path.insert(0, str(REPO / "experiments"))

print("Repository :", REPO)
print("Colab      :", IN_COLAB)
print("Python     :", sys.version.split()[0])
""")

code(r"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

RNG_SEED = 42          # the seed the WHOLE TEAM uses for partitioning

FIGURES = REPO / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

print(f"Figures -> {FIGURES}")
""")

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
md(r"""
---

## 1. The dataset: AutoVI (Renault Group)

**AutoVI — the Automotive Visual Inspection Dataset.**
<https://auto-vi.utc.fr/> · DOI [10.5281/zenodo.10459003](https://doi.org/10.5281/zenodo.10459003)

This is the real thing: photographs taken **on Renault Group's automotive production
lines**, by Renault Group and the Université de technologie de Compiègne (Roberval and
Heudiasyc laboratories). It is not a toy benchmark and not a medical stand-in — the images
come from a live assembly line, with the brightness and lighting variation that comes with
photographing constantly moving components.

Per the dataset's own disclaimer: *all defects shown were intentionally created on
Renault Group's production lines for the purpose of producing this dataset. The images
were examined and labeled by Renault Group experts, and all defects were corrected after
shooting.*

### The three parts we use

The full dataset is 4.0 GB across six vehicle components. We use the three smallest, and
**each one becomes one federated client**:

| Category | What it is | Defect types |
|---|---|---|
| `pipe_staple` | a staple holding a pipe | `missing` |
| `pipe_clip` | a clip on a pipe | `unclipped`, `operator` |
| `engine_wiring` | wiring in the engine bay | `fastening`, `multiple`, `blue_hoop`, `cardboard`, `obstruction` |

This is a much stronger federated story than a synthetic split: **client = real production
part**, so the non-IID-ness is genuine rather than invented by a Dirichlet draw.

---

### ⚠️ The one thing you must understand about this dataset

**AutoVI ships as an *unsupervised* benchmark. Its `train/` folder contains only normal
images — zero defects.**

```
pipe_staple/train/good/      191 images   <- normal only
pipe_staple/test/good/       188 images
pipe_staple/test/missing/    117 images   <- the only defects, all in test
pipe_staple/ground_truth/    pixel masks for each defect
```

The task it was designed for is: *learn what "normal" looks like, then flag anything that
deviates* — so a model never sees a defect during training.

Stage 1 of this project builds a **supervised classifier**, which cannot learn from that:
it would never see a single positive example.

**So we re-split.** `data_splitting/autovi.py` pools all images of a category and
re-splits them *stratified* into train/val/test, so defects appear in training in the same
proportion everywhere.

> **This must be stated in the report.** It is a deliberate deviation from AutoVI's
> official protocol. Numbers produced this way are **not comparable** with published
> AutoVI results, which are measured under the unsupervised protocol. Reproducing the
> official protocol (a one-class model scored by reconstruction error, using the
> ground-truth masks) is a natural Stage 2 extension.

### Licence — read before sharing anything

AutoVI is **CC BY-NC-SA 4.0**: non-commercial, share-alike, attribution required.
Copyright © 2023–2024 Renault Group.

Practically: it must **not** be committed to git (`data/` is already git-ignored), and it
should stay on a local machine rather than being uploaded to a third-party notebook
service. In a course about trustworthy AI, how we handle an industrial partner's data is
part of the assignment.
""")

code(r"""
# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET = "autovi"

# One federated client per production part
CATEGORIES = ["pipe_staple", "pipe_clip", "engine_wiring"]
IMAGE_SIZE = 128                 # AutoVI ships 400x400; we downscale to keep CPU training sane

AUTOVI_ZIPS = REPO / "data" / "raw" / "autovi"           # the downloaded .zip files
AUTOVI_NPZ = REPO / "data" / "raw" / "autovi_npz"        # converted .npz files
MERGED_NPZ = AUTOVI_NPZ / "autovi_all.npz"               # all categories in one file

PARTITIONS = REPO / "data" / "partitions" / "autovi"

print(f"Dataset    : {DATASET} (REAL Renault production-line data)")
print(f"Categories : {', '.join(CATEGORIES)}  -> {len(CATEGORIES)} federated clients")
print(f"Image size : {IMAGE_SIZE}x{IMAGE_SIZE} (downscaled from 400x400)")
""")

md(r"""
### Getting the data

Download the category archives from Zenodo into `data/raw/autovi/` (about 0.5 GB for our
three categories):

```bash
mkdir -p data/raw/autovi
for c in pipe_staple pipe_clip engine_wiring; do
  curl -L "https://zenodo.org/records/10459003/files/$c.zip?download=1" \
       -o "data/raw/autovi/$c.zip"
done
```

Then convert them to the `.npz` format the rest of the repository uses. This is what
performs the re-split described above:

```bash
python data_splitting/autovi.py \
    --zips data/raw/autovi/*.zip \
    --out data/raw/autovi_npz --size 128 \
    --merged data/raw/autovi_npz/autovi_all.npz
```

The next cell runs that conversion if it has not been done already.
""")

code(r"""
zips = [AUTOVI_ZIPS / f"{c}.zip" for c in CATEGORIES]
missing = [z for z in zips if not z.exists()]

if missing:
    print("Missing archives - download them first (see the cell above):")
    for z in missing:
        print("   ", z)
    raise SystemExit("AutoVI archives not found")

if MERGED_NPZ.exists():
    print(f"Already converted: {MERGED_NPZ} "
          f"({MERGED_NPZ.stat().st_size / 1e6:.1f} MB)")
else:
    cmd = [sys.executable, "data_splitting/autovi.py",
           "--zips", *[str(z) for z in zips],
           "--out", str(AUTOVI_NPZ),
           "--size", str(IMAGE_SIZE),
           "--merged", str(MERGED_NPZ)]
    print("Running:\n  " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("conversion failed")
""")

# ---------------------------------------------------------------------------
# PHASE A
# ---------------------------------------------------------------------------
md(r"""
---
---

# Phase A — Describe the dataset

> *"We first need to describe the dataset, to see if it is balanced. We want to know
> everything about the data."*

This phase answers that, in eight checks. Each one can independently invalidate a result,
so each gets its own section, and each ends with an explicit verdict.

| # | Check | The question it answers |
|---|---|---|
| A.1 | Provenance and licence | Where did this come from, may we use it? |
| A.2 | Size and shape | How much data, in what form? |
| A.3 | **Class balance** | Is it balanced? *(the professor's main question)* |
| A.4 | Split integrity | Are train/val/test consistent with each other? |
| A.5 | Value distributions | Are the features well-behaved or pathological? |
| A.6 | Missing / degenerate data | Are there NaNs, blanks, constant features? |
| A.7 | Duplicates and leakage | Does the same sample appear in train *and* test? |
| A.8 | Per-class appearance | Do the classes actually look different? |
""")

# --- A.1 ---
md(r"""
## A.1 — Provenance, licence, and how the data was obtained

Before a single statistic: *where is this from, and are we allowed to use it?* In a
trustworthy-AI course this is not bureaucracy, it is the first trust question — and with
an industrial partner's data it has real consequences.
""")

code(r"""
PROVENANCE = {
    "Full name": "AutoVI - The Automotive Visual Inspection Dataset",
    "Origin": "Renault Group automotive production lines (real, in-service)",
    "Produced by": ("Renault Group + Universite de technologie de Compiegne "
                    "(Roberval & Heudiasyc laboratories)"),
    "Citation": ("Carvalho, P., Lafou, M., Durupt, A., Leblanc, A. & Grandvalet, Y. "
                 "(2024). The Automotive Visual Inspection Dataset (AutoVI). "
                 "Computers in Industry. doi:10.1016/j.compind.2024.104151"),
    "Dataset DOI": "10.5281/zenodo.10459003",
    "Website": "https://auto-vi.utc.fr/",
    "Licence": "CC BY-NC-SA 4.0 (attribution, NON-COMMERCIAL, share-alike)",
    "Copyright": "(c) 2023-2024 Renault Group",
    "Redistributable": "NO - do not commit to git, do not upload to third-party services",
    "Designed for": "UNSUPERVISED anomaly detection (train contains normal images only)",
    "How we use it": "Re-split to supervised (see the warning in section 1)",
    "Defect authenticity": ("defects were intentionally created on the production line "
                            "by Renault experts, labelled by them, and corrected after "
                            "shooting"),
}

display(pd.DataFrame({"Property": list(PROVENANCE), "Value": list(PROVENANCE.values())})
        .style.hide(axis="index"))
""")

md(r"""
**Verdict (A.1).** The licence is **non-commercial share-alike**, and the data belongs to
an industrial partner. Two concrete consequences for how we work:

1. **Never commit it.** `data/` is git-ignored; deleting a file later does not remove it
   from git history.
2. **Keep it local.** Running this notebook on a third-party service means uploading
   Renault production-line imagery to someone else's servers.

The citation above must appear in the report.
""")

# --- A.2 ---
md(r"""
## A.2 — Size and shape: how much data is there?

Sample counts set a ceiling on what we can honestly conclude. AutoVI is a *real*
industrial dataset, which means it is far smaller than an academic benchmark — a few
hundred images per part, not tens of thousands. Knowing this now prevents over-claiming
later.
""")

code(r"""
from aitdm_fl.task import load_npz_arrays   # noqa: E402

summary_path = AUTOVI_NPZ / "autovi_summary.json"
autovi_summary = json.loads(summary_path.read_text())

raw = np.load(MERGED_NPZ)
images, labels = raw["images"], raw["labels"].reshape(-1)
category = raw["category"].reshape(-1)

print(f"Merged archive : {MERGED_NPZ.name}")
print(f"Arrays         : {sorted(raw.files)}")
print(f"Images         : {images.shape}  {images.dtype}")
print(f"Memory         : {images.nbytes / 1e6:,.0f} MB in RAM")
print(f"Original size  : 400x400 -> resized to {IMAGE_SIZE}x{IMAGE_SIZE}")
""")

code(r"""
rows = []
for cat in CATEGORIES:
    rep = next(c for c in autovi_summary["categories"] if c["category"] == cat)
    m = category == cat
    rows.append({
        "Category": cat,
        "Images": int(m.sum()),
        "Normal": int((labels[m] == 0).sum()),
        "Anomalous": int((labels[m] == 1).sum()),
        "Anomaly %": round(100 * float(labels[m].mean()), 1),
        "Defect types": len(rep["defect_types"]),
        "Defect names": ", ".join(rep["defect_types"]),
    })

size_df = pd.DataFrame(rows)
total = int(size_df["Images"].sum())
display(size_df.style.hide(axis="index"))
print(f"\nTOTAL: {total:,} images across {len(CATEGORIES)} production parts")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))

bars = axes[0].bar(size_df["Category"], size_df["Images"], color="#4C72B0")
axes[0].bar_label(bars, fmt="%d", padding=2)
axes[0].set_ylabel("Images")
axes[0].set_title(f"Dataset size per production part ({total:,} total)")
axes[0].set_ylim(0, size_df["Images"].max() * 1.18)
axes[0].tick_params(axis="x", rotation=12)

x = np.arange(len(size_df))
axes[1].bar(x, size_df["Normal"], label="normal", color="#4C72B0")
axes[1].bar(x, size_df["Anomalous"], bottom=size_df["Normal"],
            label="anomaly", color="#C44E52")
axes[1].set_xticks(x, size_df["Category"], rotation=12)
axes[1].set_ylabel("Images")
axes[1].set_title("Composition: normal vs anomalous")
axes[1].legend()

fig.suptitle("AutoVI — size and composition", y=1.03)
fig.tight_layout()
fig.savefig(FIGURES / "a2_split_sizes.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md(r"""
**Verdict (A.2).** Around two thousand images in total, a few hundred per part. That is
small, and it has two consequences we must respect for the rest of the project:

- per-client validation sets will hold only a few dozen images, so **per-client metrics
  are noisy** — differences of a couple of percentage points are not real;
- every reported number must be a **mean ± std over seeds 1, 2, 3**, never a single run.
""")

# --- A.3 ---
md(r"""
## A.3 — Class balance *(the central question)*

**This is the section the professor asked for.** Everything else in Phase A supports it.

Why it dominates everything downstream: if most parts coming off the line are fine, a
model that says "normal" for every image — catching *not a single defect* — still scores
a high accuracy. It would be useless and look excellent. So before any modelling we must
know the balance, because it decides which metrics are honest.
""")

code(r"""
def class_balance_table(y: np.ndarray, class_names=("normal", "anomaly")) -> pd.DataFrame:
    '''Counts and percentages per class.'''
    y = np.asarray(y).reshape(-1)
    values, counts = np.unique(y, return_counts=True)
    return pd.DataFrame({
        "Class": values,
        "Name": [class_names[int(v)] for v in values],
        "Count": counts,
        "Percent": (counts / counts.sum() * 100).round(2),
    })


CLASS_NAMES = ["normal", "anomaly"]

print("=== Whole dataset (all three parts pooled) ===")
display(class_balance_table(labels).style.hide(axis="index"))

for cat in CATEGORIES:
    print(f"\n=== {cat} ===")
    display(class_balance_table(labels[category == cat]).style.hide(axis="index"))
""")

code(r"""
counts = np.bincount(labels, minlength=2)
majority, minority = counts.max(), counts.min()
imbalance_ratio = majority / minority
majority_class = CLASS_NAMES[int(counts.argmax())]

print(f"Majority class : '{majority_class}' with {majority:,} images")
print(f"Minority class : '{CLASS_NAMES[int(counts.argmin())]}' with {minority:,} images")
print(f"IMBALANCE RATIO: {imbalance_ratio:.2f} : 1")

naive_acc = majority / counts.sum()
print(f"\nA model that ALWAYS predicts '{majority_class}' would score:")
print(f"  accuracy          = {naive_acc:.3f}   <- looks fine, catches ZERO defects")
print(f"  balanced accuracy = 0.500   <- the honest number")

if imbalance_ratio < 1.5:
    verdict = "BALANCED - plain accuracy is acceptable."
elif imbalance_ratio < 3:
    verdict = "MODERATELY IMBALANCED - prefer balanced accuracy and macro-F1."
else:
    verdict = "STRONGLY IMBALANCED - accuracy is misleading; use balanced acc. / macro-F1."
print(f"\nVERDICT: {verdict}")
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

# Left: overall balance
tbl = class_balance_table(labels)
bars = axes[0].bar(tbl["Name"], tbl["Percent"], color=["#4C72B0", "#C44E52"])
axes[0].bar_label(bars, fmt="%.1f%%", padding=2)
axes[0].axhline(50, ls="--", c="black", lw=1.2, label="perfect balance (50%)")
axes[0].set_ylabel("Percent of all images")
axes[0].set_title(f"Overall class balance ({imbalance_ratio:.2f}:1)")
axes[0].set_ylim(0, 85)
axes[0].legend(fontsize=8)

# Right: balance per production part
x = np.arange(len(CATEGORIES))
anomaly_pct = [100 * float(labels[category == c].mean()) for c in CATEGORIES]
bars = axes[1].bar(x, anomaly_pct, color="#C44E52")
axes[1].bar_label(bars, fmt="%.1f%%", padding=2)
axes[1].axhline(50, ls="--", c="black", lw=1.2, label="50%")
axes[1].set_xticks(x, CATEGORIES, rotation=12)
axes[1].set_ylabel("Anomalous images (%)")
axes[1].set_title("Anomaly rate per production part")
axes[1].set_ylim(0, 60)
axes[1].legend(fontsize=8)

fig.suptitle("AutoVI — is the data balanced?", y=1.03)
fig.tight_layout()
fig.savefig(FIGURES / "a3_class_balance.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md(r"""
**Verdict (A.3) — the headline finding of Phase A.**

Read the imbalance ratio and the dashed 50% line above.

The consequence propagates through the entire project:

> **Plain accuracy is not a usable headline metric.** We report **balanced accuracy** and
> **macro-F1** as the primary numbers, with accuracy kept only as a secondary figure.

This is Member 3's metric rule, and it is *derived here, from the data* — not assumed.

Note also the **right-hand plot**: the anomaly rate differs from part to part. Since each
part becomes one federated client, the clients are imbalanced *differently from each
other* — that is label skew, one of the two kinds of non-IID-ness we quantify in Phase C.

> **A caveat on how this balance came about.** These proportions are a consequence of our
> re-split. In AutoVI's official layout the training set is 100% normal and all defects
> sit in the test set. The ratio above describes the *pooled* data, which is the right
> description of what our classifier actually sees — but it is not AutoVI's native class
> balance, and the report must say so.
""")

# --- A.4 ---
md(r"""
## A.4 — Split integrity

Our re-split is stratified, meaning each class should appear in the same proportion in
train, validation and test. If it does not, test performance measures something other
than what we trained for. This cell verifies the stratification actually worked.
""")

code(r"""
rows = []
for rep in autovi_summary["categories"]:
    for split in ("train", "val", "test"):
        rows.append({
            "Category": rep["category"],
            "Split": split,
            "Samples": rep["split_sizes"][split],
            "Anomaly %": round(100 * rep["anomaly_fraction"][split], 1),
        })
split_df = pd.DataFrame(rows)
display(split_df.pivot(index="Category", columns="Split",
                       values=["Samples", "Anomaly %"]))

gaps = []
for rep in autovi_summary["categories"]:
    fr = rep["anomaly_fraction"]
    gaps.append(100 * (max(fr.values()) - min(fr.values())))
worst_gap = max(gaps)

print(f"Largest anomaly-rate spread across splits, within a category: {worst_gap:.2f} pp")
if worst_gap < 3:
    print("OK: stratification worked - every split has the same class mix.")
else:
    print("WARNING: splits are not well stratified; check autovi.py.")
""")

md(r"""
**Verdict (A.4).** A small spread confirms the stratified re-split did its job: train,
validation and test all see the same class mix, so validation is a fair proxy for test and
the final comparison is meaningful.

This is a genuine advantage of the re-split over AutoVI's native layout for our purpose —
under the official protocol the training set is 100% normal, so no validation set drawn
from it could ever estimate defect-detection performance.
""")

# --- A.5 ---
md(r"""
## A.5 — Value distributions

Are the images well-behaved? AutoVI was shot on a live production line, so the readme
warns of *"variations in brightness and lighting on constantly moving components"*. That
is exactly the kind of nuisance variation that can dominate a small model, so we should
measure it rather than assume it.
""")

code(r"""
flat = images.reshape(len(images), -1).astype(np.float32)

stats = pd.DataFrame({
    "Statistic": ["min", "max", "mean", "std", "median",
                  "1st percentile", "99th percentile"],
    "Pixel value": [flat.min(), flat.max(), flat.mean(), flat.std(),
                    np.median(flat), np.percentile(flat, 1), np.percentile(flat, 99)],
})
display(stats.round(2).style.hide(axis="index"))

per_image_mean = flat.mean(axis=1)
print(f"Dynamic range used : {flat.min():.0f} to {flat.max():.0f} of a possible 0-255")
print(f"Mean brightness    : {per_image_mean.mean():.1f} "
      f"(spread {per_image_mean.min():.1f} to {per_image_mean.max():.1f})")
print("The training pipeline divides by 255, so inputs land in [0, 1].")
""")

code(r"""
rng_plot = np.random.default_rng(0)
sample = rng_plot.choice(len(flat), min(300, len(flat)), replace=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))

axes[0].hist(flat[sample].ravel(), bins=60, color="#4C72B0")
axes[0].set_title("Pixel intensity distribution")
axes[0].set_xlabel("Pixel value (0-255)")
axes[0].set_ylabel("Frequency")

for cat in CATEGORIES:
    axes[1].hist(per_image_mean[category == cat], bins=30, alpha=0.6, label=cat)
axes[1].set_title("Mean brightness per image, by part")
axes[1].set_xlabel("Mean pixel value")
axes[1].legend(fontsize=8)

for lab, name, colour in ((0, "normal", "#4C72B0"), (1, "anomaly", "#C44E52")):
    axes[2].hist(per_image_mean[labels == lab], bins=30, alpha=0.6,
                 label=name, color=colour)
axes[2].set_title("Mean brightness: normal vs anomaly")
axes[2].set_xlabel("Mean pixel value")
axes[2].legend(fontsize=8)

fig.suptitle("AutoVI — pixel statistics (production-line lighting variation)", y=1.04)
fig.tight_layout()
fig.savefig(FIGURES / "a5_value_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md(r"""
**Verdict (A.5).** Two things to read here.

The **middle panel** shows brightness differs systematically *between parts*. Since each
part is a federated client, this is **feature skew** — the clients' inputs differ, not just
their labels. Together with the label skew from A.3, it makes this a genuinely non-IID
federated problem, and both are real rather than synthetic.

The **right panel** is a sanity check worth stating in the report: if normal and anomalous
images had very different brightness distributions, a model could score well by reading
the lighting instead of the defect. Overlapping distributions mean that shortcut is not
trivially available.
""")

# --- A.6 ---
md(r"""
## A.6 — Missing values and degenerate features

NaNs crash training or silently poison gradients; constant pixels consume parameters and
contribute nothing. Both are cheap to check and expensive to miss.
""")

code(r"""
n_nan = int(np.isnan(flat).sum())
n_inf = int(np.isinf(flat).sum())
constant_px = int((flat.std(axis=0) == 0).sum())
blank_images = int((flat.std(axis=1) == 0).sum())

checks = pd.DataFrame({
    "Check": ["NaN values", "Infinite values",
              "Constant pixels (identical in every image)",
              "Blank images (no variation at all)"],
    "Count": [n_nan, n_inf, constant_px, blank_images],
    "Status": ["OK" if v == 0 else "INVESTIGATE"
               for v in (n_nan, n_inf, constant_px, blank_images)],
})
display(checks.style.hide(axis="index"))

print(f"Constant pixels: {constant_px:,} of {flat.shape[1]:,} "
      f"({constant_px / flat.shape[1]:.2%})")
print("Note: AutoVI PNGs are RGBA; the converter drops the alpha channel and keeps RGB.")
""")

md(r"""
**Verdict (A.6).** No missing or infinite values — the conversion is clean. A small number
of constant pixels is normal for fixed-camera industrial imagery, where part of the frame
is always the same jig or background.
""")

# --- A.7 ---
md(r"""
## A.7 — Duplicates and train/test leakage

**The most dangerous check in this notebook**, and it matters more here than for a typical
academic dataset.

AutoVI was shot on a moving production line, so **consecutive frames can show the same
physical part**. If one frame lands in train and the next in test, the model has
effectively seen the answer, and the test score is fiction.

Our re-split shuffles images randomly, which makes this risk *real* — so we check it
rather than hope. (It paid off: see the verdict.)
""")

code(r"""
# Step 1 - what did the converter find in the RAW archives?
dup_rows = []
for rep in autovi_summary["categories"]:
    dup_rows.append({
        "Category": rep["category"],
        "Images as shipped": sum(rep["counts_as_shipped"].values()),
        "Exact duplicates removed": rep["exact_duplicates_removed"],
        "Where the pairs were": ", ".join(f"{k} ({v})" for k, v in
                                          rep["duplicate_pairs_by_folder"].items()) or "-",
    })
dup_df = pd.DataFrame(dup_rows)
display(dup_df.style.hide(axis="index"))
print(f"AutoVI (our 3 parts) contained {autovi_summary['exact_duplicates_removed_total']} "
      f"byte-identical duplicate images. autovi.py removed them BEFORE splitting.")
""")

code(r"""
# Step 2 - verify there is no exact overlap left between the splits we will train on
def hashes(a: np.ndarray) -> np.ndarray:
    '''One stable hash per image, for exact-duplicate detection.'''
    flat_a = np.ascontiguousarray(a.reshape(len(a), -1))
    return np.array([hash(row.tobytes()) for row in flat_a])


h_all = hashes(images)
dups = len(h_all) - len(np.unique(h_all))

# Reconstruct which split each image landed in, per category, exactly as autovi.py did
rows = []
for rep in autovi_summary["categories"]:
    cat = rep["category"]
    m = category == cat
    n_tr, n_va, n_te = (rep["split_sizes"][s] for s in ("train", "val", "test"))
    h_cat = h_all[m]
    h_tr = h_cat[:n_tr]
    h_va = h_cat[n_tr:n_tr + n_va]
    h_te = h_cat[n_tr + n_va:]
    rows.append({
        "Category": cat,
        "Duplicates within category": int(len(h_cat) - len(np.unique(h_cat))),
        "train n test": int(len(np.intersect1d(h_tr, h_te))),
        "train n val": int(len(np.intersect1d(h_tr, h_va))),
    })

leak_df = pd.DataFrame(rows)
display(leak_df.style.hide(axis="index"))

total_leak = int(leak_df["train n test"].sum())
print(f"Exact duplicates remaining in the converted dataset: {dups}")
if total_leak:
    print(f"*** {total_leak} test images also appear in train. "
          f"Test scores are NOT trustworthy. ***")
else:
    print("No exact train/test leakage detected.")
assert total_leak == 0, "leakage - fix the converter before training anything"
""")

md(r"""
**Verdict (A.7).** This check earned its place. **AutoVI itself contains exact duplicate
photographs**: in `engine_wiring`, 60 normal images are filed under *both* `train/good`
and `test/good`; in `pipe_clip`, 13 `unclipped` defect images appear twice. In the
official unsupervised protocol that is harmless (a model that trains on normals and is
tested on the same normals is only slightly flattered). But the moment we pooled and
re-split the data, those pairs straddled our train/test boundary: the first run of this
notebook found **19 test images that were also in train**.

So the converter now removes byte-identical images before splitting (step 1), and this
cell asserts that nothing is left (step 2). The dataset shrinks from 1,920 to 1,847
images; every count in this notebook is *after* that removal.

**State the remaining limitation honestly in the report.** Hashing catches only
byte-identical copies. Two consecutive frames of the same part on a moving line are
*near*-duplicates - visually almost the same, numerically different - and would pass
this test undetected. AutoVI gives no frame or part identifier, so we cannot group the
split by physical part. The correct statement is: *exact leakage was found and removed;
near-duplicate leakage cannot be ruled out, and the reported scores may therefore be
mildly optimistic.* Quantifying it (perceptual hashing, or nearest-neighbour distance
between train and test) is a good Stage 2 robustness task for Member 3.
""")

# --- A.8 ---
md(r"""
## A.8 — What do the defects actually look like?

Statistics hide things a single glance reveals. This is also the section that makes the
report concrete for a reader who has never seen the production line.
""")

code(r"""
rng_show = np.random.default_rng(RNG_SEED)
n_show = 5

fig, axes = plt.subplots(len(CATEGORIES), 2 * n_show,
                         figsize=(2 * n_show * 1.35, len(CATEGORIES) * 1.6))
for r, cat in enumerate(CATEGORIES):
    m = category == cat
    for half, (lab, name) in enumerate(((0, "normal"), (1, "anomaly"))):
        idx = np.where(m & (labels == lab))[0]
        pick = rng_show.choice(idx, min(n_show, len(idx)), replace=False)
        for k in range(n_show):
            ax = axes[r, half * n_show + k]
            if k < len(pick):
                ax.imshow(images[pick[k]])
            ax.axis("off")
            if r == 0 and k == n_show // 2:
                ax.set_title(name, fontsize=10)
    axes[r, 0].axis("on")
    axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    axes[r, 0].set_ylabel(cat, fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=36)

fig.suptitle("AutoVI — normal (left) vs anomalous (right) parts", y=1.0)
fig.tight_layout()
fig.savefig(FIGURES / "a8_class_examples.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code(r"""
# Mean image per class, per part: does the defect show up on average?
fig, axes = plt.subplots(len(CATEGORIES), 3, figsize=(7.2, len(CATEGORIES) * 2.3))
for r, cat in enumerate(CATEGORIES):
    m = category == cat
    mean_norm = images[m & (labels == 0)].astype(np.float32).mean(axis=0)
    mean_anom = images[m & (labels == 1)].astype(np.float32).mean(axis=0)
    diff = np.abs(mean_anom - mean_norm).mean(axis=2)

    axes[r, 0].imshow(mean_norm.astype(np.uint8))
    axes[r, 1].imshow(mean_anom.astype(np.uint8))
    im = axes[r, 2].imshow(diff, cmap="inferno")
    for k, t in enumerate(("mean normal", "mean anomaly", "|difference|")):
        axes[r, k].axis("off")
        if r == 0:
            axes[r, k].set_title(t, fontsize=9)
    axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    axes[r, 0].set_ylabel(cat, fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=36)
    print(f"{cat:<15} mean |difference| = {diff.mean():6.2f}  (0-255 scale)")

fig.suptitle("Where do defects change the image?", y=1.0)
fig.tight_layout()
fig.savefig(FIGURES / "a8_class_means.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md(r"""
**Verdict (A.8).** The difference maps show *where* defects systematically change the
image. A bright, localised region means the defect has a consistent spatial signature; a
diffuse map means the defect appears in different places each time and the task is harder.

This is a useful baseline for Member 3's Grad-CAM work in Stage 2: if the trained model
later attends to regions that are dark here, that discrepancy is worth investigating — it
may mean the model is using a shortcut rather than looking at the defect.
""")

# --- Phase A summary ---
md(r"""
---

## Phase A summary — what we now know about the data

| Check | Finding | Consequence for the project |
|---|---|---|
| A.1 Provenance | Real Renault production line; CC BY-NC-SA 4.0 | Never commit; keep local; cite the paper |
| A.2 Size | 1,847 images over 3 parts, a few hundred each | Small: use 3 seeds, distrust small differences |
| **A.3 Balance** | **Imbalanced; rate differs per part** | **Use balanced accuracy + macro-F1; label skew across clients** |
| A.4 Split integrity | Stratified re-split verified | Validation is a fair proxy for test |
| A.5 Distributions | Brightness differs per part | **Feature skew** across clients, on top of label skew |
| A.6 Missing data | No NaNs/Infs; clean conversion | Nothing to fix |
| A.7 **Leakage** | **73 exact duplicate pairs in the raw data; 19 crossed train/test** | **Removed before splitting**; near-duplicates remain a stated limitation |
| A.8 Appearance | Defects have visible, localised signatures | Task is learnable; baseline for Grad-CAM |

**The sentence that carries into every other notebook:**

> AutoVI is imbalanced and its clients differ in both label mix and image brightness, so
> model quality is judged by **balanced accuracy** and **macro-F1**, and per-client results
> must be reported alongside the global average.
""")
# ---------------------------------------------------------------------------
# PHASE B
# ---------------------------------------------------------------------------
md(r"""
---
---

# Phase B — Partition the data into federated clients

Now that we understand the data, we cut it into pieces that simulate separate factories.

## Why non-IID, and what it means

Federated learning exists because organisations will not pool raw data. Each simulated
client must therefore hold its **own, different** slice — if every client had identical
data, federated learning would be trivial and there would be nothing to study.

"Different" has a technical name: **non-IID** (not independent and identically
distributed).

### We do not need to simulate it — AutoVI gives us real non-IID clients

Most federated-learning coursework fakes non-IID-ness with a **Dirichlet split**: take one
pool of data and deal it out unevenly using a random draw. It works, but the clients are
invented.

AutoVI lets us do something better. It contains **three different vehicle parts**, each
photographed on its own station, each with its own defect types. So we use
`--strategy group --group-col category`: **one client = one production part**.

The resulting clients differ in every way that matters, and all of it is real:

| | `pipe_staple` | `pipe_clip` | `engine_wiring` |
|---|---|---|---|
| What is inspected | a pipe staple | a pipe clip | engine-bay wiring |
| Defect types | 1 | 2 | 5 |
| Anomaly rate | ~24% | ~27% | ~36% |
| Image appearance | different lighting and framing per station (see A.5) |

That is **label skew** (A.3) *and* **feature skew** (A.5) at the same time — which is
exactly the hard case federated learning research cares about, and we get it for free
instead of inventing it.

> **In the report, say this explicitly.** "Clients correspond to distinct physical
> inspection stations rather than a synthetic partition" is a considerably stronger claim
> than "we used α=0.3", and it is true.

### The setting everyone on the team must share

| Setting | Value | Why |
|---|---|---|
| Clients | 3 | one per production part; guidelines ask for 3–5 |
| Strategy | `group` on `category` | real clients, not a synthetic draw |
| Seed | 42 | **identical for all members**, or results are not comparable |
| Output | `data/partitions/autovi` | the one folder the whole team uses |
""")

code(r"""
cmd = [
    sys.executable, "data_splitting/split_data.py",
    "--input", str(MERGED_NPZ),
    "--strategy", "group", "--group-col", "category",
    "--num-clients", str(len(CATEGORIES)),
    "--class-names", "normal,anomaly",
    "--seed", str(RNG_SEED),
    "--out", str(PARTITIONS),
]
print("Running:\n  " + " ".join(cmd) + "\n")

result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:\n", result.stderr)
    raise SystemExit("split_data.py failed")
""")

md(r"""
### What was produced

```
data/partitions/autovi/
├── client_0.npz            one production part's private data (x_train, y_train, x_val, y_val)
├── client_1.npz            another part
├── client_2.npz            another part
├── test.npz                the shared global test set (nobody trains on this)
├── meta.json               class counts per client, shapes, settings
└── label_distribution.png  the visual proof that the split is non-IID
```

Note what the global test set now means: it is drawn from **all three parts**, so it
measures whether the federated model works across the whole line, not just on one station.
""")

code(r"""
for f in sorted(PARTITIONS.iterdir()):
    print(f"  {f.name:<26} {f.stat().st_size / 1024:>9,.1f} KB")
""")

# ---------------------------------------------------------------------------
# PHASE C
# ---------------------------------------------------------------------------
md(r"""
---
---

# Phase C — Verify and quantify the split

Claiming a split is "non-IID" is not enough; the report needs a number. This phase produces
the per-client table, the distribution figure, and a quantitative skew measure.
""")

code(r"""
meta = json.loads((PARTITIONS / "meta.json").read_text())
names = meta.get("class_names") or [f"class {i}" for i in range(meta["num_classes"])]

print(f"Source       : {meta['source']}")
print(f"Data type    : {meta['data_type']}")
print(f"Classes      : {meta['num_classes']} -> {names}")
print(f"Sample shape : {meta['sample_shape']}")
print(f"Test samples : {meta['test_samples']:,}")
print(f"Strategy     : {meta['strategy']} (alpha={meta['settings']['alpha']}, "
      f"seed={meta['settings']['seed']})")
""")

code(r"""
# The per-client table. This goes straight into the report.
rows = []
for c in meta["clients"]:
    counts = c["train_class_counts"]
    total_c = sum(counts)
    row = {"Client": c["client"], "Train": c["train"], "Val": c["val"]}
    for i, n in enumerate(counts):
        row[names[i]] = n
    row["Majority %"] = round(100 * max(counts) / total_c, 1) if total_c else 0.0
    rows.append(row)

client_df = pd.DataFrame(rows)
display(client_df.style.hide(axis="index"))

# Markdown for pasting into the report (no extra dependency needed)
def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):   # itertuples keeps each column's dtype
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)

print("\nMarkdown for the report:\n")
print(to_markdown(client_df))
""")

code(r"""
# How non-IID is it, as a single number?
# Jensen-Shannon divergence between each client's class distribution and the global one.
from scipy.spatial.distance import jensenshannon   # noqa: E402

global_counts = np.sum([c["train_class_counts"] for c in meta["clients"]], axis=0)
global_dist = global_counts / global_counts.sum()

js = []
for c in meta["clients"]:
    counts = np.asarray(c["train_class_counts"], dtype=float)
    dist = counts / counts.sum() if counts.sum() else counts
    js.append(jensenshannon(dist, global_dist, base=2))

skew = pd.DataFrame({
    "Client": [c["client"] for c in meta["clients"]],
    "Samples": [c["train"] for c in meta["clients"]],
    "JS divergence from global": np.round(js, 3),
})
display(skew.style.hide(axis="index"))

mean_js = float(np.nanmean(js))
print(f"Mean JS divergence (LABEL skew only): {mean_js:.3f}")
print("   0 = clients have identical class mixes, 1 = maximally different")
if mean_js < 0.05:
    print("-> Label mixes are similar across clients.")
elif mean_js < 0.3:
    print("-> Mild label skew.")
else:
    print("-> Strong label skew.")
print("\nBut label skew is only half the story - see the next cell.")
""")

md(r"""
### Label skew is the wrong measure here — and that matters

The JS divergence above is small, which would normally mean "these clients are nearly
IID, this is a boring federated problem". **That conclusion would be wrong**, because JS
divergence only compares *class proportions*. Our clients hold photographs of
**different objects**, taken at **different stations** under **different lighting** —
a difference it is structurally blind to.

The honest way to measure that is to ask a direct question: **given only the pixels, can
you tell which client an image came from?** If a trivial model can do that perfectly, the
clients' input distributions do not overlap at all, and the federated problem is hard
regardless of what the label statistics say.
""")

code(r"""
from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.model_selection import cross_val_score      # noqa: E402

# Per-client brightness: the simplest evidence of feature skew
bright = pd.DataFrame({
    "Client": range(len(CATEGORIES)),
    "Part": CATEGORIES,
    "Mean brightness": [round(float(images[category == c].reshape(
        (category == c).sum(), -1).astype(np.float32).mean()), 1) for c in CATEGORIES],
    "Anomaly rate": [f"{labels[category == c].mean():.1%}" for c in CATEGORIES],
    "Images": [int((category == c).sum()) for c in CATEGORIES],
})
display(bright.style.hide(axis="index"))

# Can a linear model identify the client from pixels alone?
rng_fs = np.random.default_rng(0)
idx = rng_fs.choice(len(images), min(600, len(images)), replace=False)
X_small = images[idx].reshape(len(idx), -1).astype(np.float32)[:, ::64] / 255.0
scores_fs = cross_val_score(LogisticRegression(max_iter=300), X_small, category[idx], cv=3)
chance = 1 / len(CATEGORIES)

print(f"\nIdentify the CLIENT from pixels alone: accuracy {scores_fs.mean():.3f} "
      f"(chance = {chance:.3f})")
if scores_fs.mean() > 0.9:
    print("-> SEVERE feature skew: the clients' images are trivially distinguishable.")
elif scores_fs.mean() > chance + 0.2:
    print("-> Clear feature skew.")
else:
    print("-> Little feature skew; clients look alike.")
""")

md(r"""
**Verdict (C, the real one).** Report *both* numbers, and be explicit that they disagree:

| Measure | What it captures | Result |
|---|---|---|
| JS divergence | label skew (class proportions) | small — clients have broadly similar class mixes |
| Client identifiability from pixels | **feature skew** (input distributions) | near-perfect — clients are trivially distinguishable |

So the correct characterisation of this federation is:

> **Clients differ far more in what they look at than in how often they fail.** The
> non-IID-ness is dominated by feature skew, not label skew.

This is a genuinely interesting setting, and it predicts the behaviour we see later: a
single averaged global model has to serve three visually unrelated inspection tasks at
once, which is why FedAvg loses ground to centralised training and why one client suffers
far more than the others.

It is also a much more honest analysis than quoting a Dirichlet α and calling it
non-IID — and worth saying in the report that we checked, found the obvious metric
misleading, and measured the thing that actually mattered.
""")

code(r"""
# The figure for the report: per-client class composition
counts_matrix = np.array([c["train_class_counts"] for c in meta["clients"]], dtype=float)
proportions = counts_matrix / counts_matrix.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
bottom_a = np.zeros(len(meta["clients"]))
bottom_b = np.zeros(len(meta["clients"]))
x = np.arange(len(meta["clients"]))

for i, name in enumerate(names):
    axes[0].bar(x, counts_matrix[:, i], bottom=bottom_a, label=name)
    bottom_a += counts_matrix[:, i]
    axes[1].bar(x, proportions[:, i] * 100, bottom=bottom_b, label=name)
    bottom_b += proportions[:, i] * 100

axes[0].set_xticks(x, [f"Client {c['client']}" for c in meta["clients"]])
axes[0].set_ylabel("Training samples")
axes[0].set_title("Absolute size and composition")
axes[0].legend(fontsize=8)

axes[1].set_xticks(x, [f"Client {c['client']}" for c in meta["clients"]])
axes[1].set_ylabel("Percent of the client's data")
axes[1].set_title("Class composition (non-IID evidence)")
axes[1].axhline(100 / len(names), ls="--", c="black", lw=1,
                label="IID would be flat here")
axes[1].set_ylim(0, 118)
axes[1].legend(fontsize=8, loc="upper center", ncol=len(names) + 1,
               framealpha=0.95, borderpad=0.3)

fig.suptitle(f"Federated split — one client per AutoVI production part (seed {RNG_SEED})",
             y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "c_client_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code(r"""
# Sanity checks on the partitions themselves
from aitdm_fl.task import load_npz_arrays   # noqa: E402

print("Shapes actually written to disk:\n")
for i in range(len(CATEGORIES)):
    xt, yt = load_npz_arrays(PARTITIONS / f"client_{i}.npz", "train")
    xv, yv = load_npz_arrays(PARTITIONS / f"client_{i}.npz", "val")
    present = sorted(np.unique(yt).tolist())
    print(f"  client {i}: train {xt.shape}  val {xv.shape}  classes present: {present}")

xte, yte = load_npz_arrays(PARTITIONS / "test.npz", "test")
print(f"  test    : {xte.shape}  classes present: {sorted(np.unique(yte).tolist())}")

assert xte.shape[1:] == xt.shape[1:], "Test and client feature shapes disagree"
print("\nOK: all clients and the test set share the same feature shape.")

# Final leakage check on the files the whole team will train on
h_test = set(hashes(xte))
leaks = 0
for i in range(len(CATEGORIES)):
    xt, _ = load_npz_arrays(PARTITIONS / f"client_{i}.npz", "train")
    leaks += len(h_test.intersection(hashes(xt)))
print(f"Client-train images that also appear in test.npz: {leaks}")
assert leaks == 0, "partition leakage - do not train on this"
""")

md(r"""
### Interpreting the split

Look at the **Majority %** column and the right-hand plot. If one client's bar is a single
colour, that client has **never seen** the other class.

That is the realistic scenario the whole project is about: *a factory that has never
produced a defective part cannot learn to recognise one on its own.* It is simultaneously

- the **motivation** for federated learning (that factory needs the others' knowledge), and
- the source of the **fairness** problem (the averaged global model may serve it poorly).

Both points belong in the report's discussion.
""")

# ---------------------------------------------------------------------------
# PHASE D
# ---------------------------------------------------------------------------
md(r"""
---
---

# Phase D — Train one model on this split

The role requires *"conducts initial training on their dataset split"* and *"shares model
weights for cross-evaluation by teammates"*.

This is a **centralised** model trained on the union of all clients' training data — the
upper bound, i.e. what would be achievable if the factories *could* pool their data.
Member 2 runs the federated counterpart and the full multi-seed comparison; this run exists
to produce weights for Member 3 to evaluate.
""")

code(r"""
RUN_NAME = "member1-centralised"
EPOCHS = 10

cmd = [
    sys.executable, "experiments/centralised.py",
    "--partitions", str(PARTITIONS),
    "--epochs", str(EPOCHS),
    "--seeds", "1",
    "--name", RUN_NAME,
]
print("Running:\n  " + " ".join(cmd) + "\n(this takes a few minutes on CPU)\n")

result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
print(result.stdout[-2500:])
if result.returncode != 0:
    print("STDERR:\n", result.stderr[-2500:])
""")

code(r"""
run_dir = REPO / "outputs" / "experiments" / RUN_NAME / "seed1"
results = json.loads((run_dir / "results.json").read_text())

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Balanced accuracy", "Macro-F1"],
    "Validation": [results["val"]["accuracy"],
                   results["val"]["balanced_accuracy"],
                   results["val"]["f1_macro"]],
    "Test": [results["test"]["accuracy"],
             results["test"]["balanced_accuracy"],
             results["test"]["f1_macro"]],
}).round(3)
display(summary.style.hide(axis="index"))

per_client = pd.DataFrame([
    {"Client": int(cid), "Samples": m["samples"],
     "Accuracy": round(m["accuracy"], 3),
     "Balanced accuracy": round(m["balanced_accuracy"], 3),
     "Macro-F1": round(m["f1_macro"], 3)}
    for cid, m in sorted(results["val_per_client"].items(), key=lambda kv: int(kv[0]))
])
print("\nPer-client validation performance (the fairness view):")
display(per_client.style.hide(axis="index"))

spread = per_client["Balanced accuracy"].max() - per_client["Balanced accuracy"].min()
print(f"\nSpread between best and worst client: {spread:.3f}")
print("A large spread means the single global model serves some factories much worse "
      "than others - that is a fairness finding, not just a number.")
""")

md(r"""
> **Important caveat on these numbers.** This is **one seed**, run to produce weights for
> the team. Nothing here is reportable. Every number in the final report must be
> **mean ± std over seeds 1, 2, 3** — that is Member 2's job, via
> `experiments/centralised.py --seeds 1 2 3` and `experiments/run_seeds.py`.
>
> Note also the gap between validation and test scores. Section A.4 predicted exactly this:
> the splits are balanced differently. The EDA explains the modelling result.
""")

# ---------------------------------------------------------------------------
# PHASE E
# ---------------------------------------------------------------------------
md(r"""
---
---

# Phase E — Hand off to the team

Everything Members 2 and 3 need, in one place.
""")

code(r"""
model_path = run_dir / "final_model.pt"

handoff = {
    "dataset": DATASET,
    "dataset_full_name": "AutoVI - The Automotive Visual Inspection Dataset (Renault Group)",
    "dataset_doi": "10.5281/zenodo.10459003",
    "partitions_dir": str(PARTITIONS),
    "num_clients": len(CATEGORIES),
    "strategy": "group (one client per AutoVI production part)",
    "categories": CATEGORIES,
    "image_size": IMAGE_SIZE,
    "licence": "CC BY-NC-SA 4.0 - Renault Group; do not redistribute",
    "protocol_note": autovi_summary.get("protocol_note", ""),
    "partition_seed": RNG_SEED,
    "num_classes": meta["num_classes"],
    "class_names": names,
    "sample_shape": meta["sample_shape"],
    "test_samples": meta["test_samples"],
    "mean_js_divergence": round(mean_js, 4),
    "imbalance_ratio": round(float(imbalance_ratio), 3),
    "member1_model": str(model_path),
    "figures": sorted(p.name for p in FIGURES.glob("*.png")),
}

handoff_path = REPO / "outputs" / "stage1_data_handoff.json"
handoff_path.write_text(json.dumps(handoff, indent=2))

print(json.dumps(handoff, indent=2))
print(f"\nSaved to {handoff_path}")
""")

md(r"""
## Message to send the team

> **Partitions are ready.**
>
> - Folder: `data/partitions/autovi` — **use exactly this**
> - 3 clients = 3 real AutoVI production parts (group split), seed 42
> - Strongly non-IID: at least one client holds only a single class
> - Dataset is **imbalanced** → report **balanced accuracy and macro-F1**, not accuracy
> - No train/test leakage; test scores are trustworthy
> - My trained weights: `outputs/experiments/member1-centralised/seed1/final_model.pt`
> - Figures for the report: `outputs/figures/`
>
> **Member 2:** run centralised + FedAvg on this folder with `--seeds 1 2 3`.
> **Member 3:** evaluate my model plus Member 2's, and build the comparison table.

---

## What goes in the report from this notebook

| Report item | Where it came from |
|---|---|
| Dataset description and provenance | A.1, A.2 |
| **Class balance analysis + the metric decision** | **A.3** |
| Data quality statement (no leakage, no NaNs) | A.6, A.7 |
| Example images / class difference map | A.8 |
| Federated setup description | Phase B |
| Per-client table + distribution figure | Phase C |
| Quantified non-IID-ness (JS divergence) | Phase C |
| Initial training result | Phase D |

---

## Limitations of this analysis (for the report's §8)

1. **We changed AutoVI's protocol.** AutoVI is an *unsupervised* benchmark whose training
   set contains no defects; we re-split it so a supervised classifier could be trained.
   Our numbers are therefore **not comparable with published AutoVI results**. This is the
   single most important caveat and must be stated wherever results appear.
2. **Near-duplicate leakage cannot be ruled out.** AutoVI contained 73 byte-identical
   duplicate pairs in our three parts; A.7 caught 19 of them crossing the train/test
   boundary and the converter now removes exact copies before splitting. But hashing
   catches only exact copies. AutoVI was shot on a moving line, so consecutive frames of
   the same physical part are near-duplicates that would pass the check, and AutoVI
   provides no frame or part identifier to group by. Scores may be mildly optimistic.
3. **Small dataset.** About 1,900 images over three parts — a few hundred each, and only
   dozens in each client's validation split. Per-client numbers are noisy; report mean ±
   std over seeds 1, 2, 3 and do not read into small differences.
4. **Three of six categories.** We use the three smallest parts to keep CPU training
   feasible. `tank_screw`, `underbody_pipes` and `underbody_screw` (3.5 GB) are unused,
   and would give up to 6 clients.
5. **Downscaled images.** 400×400 → 128×128. Small defects may lose the pixels that
   distinguish them; a Stage 2 ablation at 224 px would test whether resolution is the
   limiting factor.
6. **Ground-truth masks unused.** AutoVI ships pixel-level defect masks. A classifier
   ignores them entirely — they are the natural basis for Stage 2 interpretability work
   (compare Grad-CAM against the true mask) and for the official localisation metric.

---

## The obvious Stage 2 extension

Run AutoVI the way it was designed to be run: a **one-class / reconstruction model**
trained only on normal images, scored by reconstruction error, evaluated with the
official code at <https://github.com/phcarval/autovi_evaluation_code> (image-level AUROC
and the area under the sPRO curve, using the ground-truth masks).

That would let the report say something genuinely strong: *here is how a supervised
classifier compares against the unsupervised protocol the dataset was built for, under
both centralised and federated training.*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({len(cells)} cells)")
