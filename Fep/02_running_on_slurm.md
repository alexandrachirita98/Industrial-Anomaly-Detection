# 2. Running Flower on fep with Slurm

This is the fep version of [tutorial 2](../Local/Linux/02_running_flower.md).
Read its first part, [Flower vocabulary](../Local/Linux/02_running_flower.md#0-flower-vocabulary), first: SuperLink, SuperNode, ServerApp and ClientApp are the same here.
What changes on fep is **how you start things**: you do not run the federation in your terminal, you ask Slurm to run it on a compute node, as a *job*.

> Before you start: [set up fep](01_environment_setup.md) and create a split, e.g.
>
> ```bash
> source .venv/bin/activate
> python data_splitting/split_data.py --dataset dermamnist --num-clients 3 --strategy dirichlet --alpha 0.3
> ```

---

## 0. Slurm in five commands

| Command | What it does |
|---|---|
| `bash Fep/submit.sh JOB ...` | Submits one of our jobs (a file in [`Fep/jobs/`](jobs/)). Prints the job id and its log file |
| `squeue --me` | Your jobs: `PD` = waiting in the queue (the last column says why), `R` = running |
| `tail -f ~/AITDM_Project/slurm/<job>-<id>.out` | Follows the log of a job live (Ctrl+C stops *watching*, not the job) |
| `scancel <id>` | Stops a job (`scancel -u $USER` stops all of yours) |
| `sacct -X -o jobid,jobname,state,elapsed` | Your finished jobs and how long they took |

`Fep/submit.sh` is a thin wrapper around `sbatch`.
It starts the job in the repository root and writes the log to `~/AITDM_Project/slurm/`.
Any `sbatch` option **before** the job name overrides the defaults of that job, e.g. `-p dgxa100 --gres=gpu:1` (partition and GPU), `-t 04:00:00` (time limit), `-c 16` (CPUs).

| Job | Runs | Default resources |
|---|---|---|
| `smoke_test` | `tests/smoke_test.sh` | haswell, 4 CPUs, 30 min |
| `simulation` | `scripts/run_simulation.sh` (server + all clients in one job) | haswell, 8 CPUs, 2 h |
| `deployment` | SuperLink + one SuperNode per client, connected by IP address | haswell, 4 tasks × 2 CPUs, 1 h |
| `run_seeds` | `experiments/run_seeds.py` (seeds 1, 2, 3) | haswell, 8 CPUs, 4 h |
| `centralised` | `experiments/centralised.py` | haswell, 8 CPUs, 2 h |
| `reproduce_tutorials` | `experiments/reproduce_tutorials.sh` | haswell, 16 CPUs, 12 h |
| `autovi_data` | all AutoVI npz + partitions (`autovi.py`, `split_data.py`, `autovi_official_partitions.py`) | haswell, 8 CPUs, 32 GB, 2 h |
| `padim` | `experiments/padim_experiment.py` (unsupervised baseline) | haswell, 8 CPUs, 1 h |
| `command` | any command you give it (`split_data.py`, `gradcam.py`, ...) | haswell, 4 CPUs, 1 h |

## 1. Partitions: which queue can I use?

A **partition** is a group of nodes with the same hardware.
**One job always runs on one partition**: the server and all clients of a run are allocated together, on the same partition, so they can always reach each other.
You choose the partition with `-p`.

Which partitions you may use depends on your **Slurm account** (not on your Unix group): every partition lists the accounts it allows or refuses.
Students have the account `student`, which can use fewer partitions than staff accounts.
Check yours:

```bash
bash Fep/my_partitions.sh          # the partitions you may use, with which account
sinfo -s                           # how busy they are (A/I/O/T = allocated/idle/other/total nodes)
```

```text
Accounts of <you>: student (default: student)
PARTITION    STATE      ACCOUNTS YOU CAN USE
dgxa100      up         student
dgxh100      up         student
h200         up         (no access)
haswell      up         student
...
```

| Partition | Hardware (per node) | Account `student` |
|---|---|---|
| `haswell` (default) | 32 CPUs, no GPU | yes. Enough for MedMNIST at 28 px and for HAR |
| `ucsx` | 3 × A100 | yes |
| `dgxa100` | 8 × A100 | yes, often busy |
| `dgxh100` | 8 × H100 | yes, often busy |
| `xl` | 2 × P100 | yes, but its old GPUs (Pascal) are not supported by the CUDA 13 PyTorch we install: use another partition |
| `ml`, `h200`, `hd`, `sprmcrogpu` | A100 / H200 / A100 / RTX 2080 Ti | no (research accounts only) |

Limits of the `student` account:

* **at most 2 jobs running at the same time**: you can submit more, the others wait in the queue (`squeue --me` shows `(QOSMaxJobsPerUserLimit)`) and start when one of yours finishes.
  This limits experiments, not clients: every job of `Fep/jobs/` holds a **whole federation** (server and all clients), so you can always run 2 complete federated experiments at once,
* at most **12 hours** per job (`-t 12:00:00`),
* at most **3 GPUs** and **3 nodes** per job.

If a submission fails with `Invalid account or account/partition combination`, you may not use that partition: pick another one.
If you have several accounts, choose one with `-A <account>`.
(`sbatch --test-only` cannot tell you this: on fep it accepts combinations that a real submission refuses.)

**Do you need a GPU?** For the 28 px MedMNIST datasets and HAR, no: the network is small and the CPU run is about as fast (5 rounds, 3 clients: 85 s on haswell, 60 s on an A100).
GPU queues are often long, and a CPU job usually *starts* much sooner.
A GPU pays off for larger images (`--size 128`, `--size 224`) and larger networks.

> **CPUs and memory on haswell.** fep gives at most 1.7 GB of memory per CPU there.
> If you ask for more memory (`--mem=16G`), Slurm silently gives you more CPUs (10) to match.
> Our jobs ask only for CPUs: memory follows (8 CPUs = 13.6 GB).

## 2. Simulation: one job, the whole federation

```bash
bash Fep/submit.sh simulation                                       # data/partitions/dirichlet
bash Fep/submit.sh simulation data/partitions/iid
bash Fep/submit.sh simulation data/partitions/dirichlet "num-server-rounds=10 strategy='fedprox'"
bash Fep/submit.sh -p dgxa100 --gres=gpu:1 simulation data/partitions/dirichlet
```

The arguments are exactly those of `bash scripts/run_simulation.sh` on a laptop ([tutorial 2](../Local/Linux/02_running_flower.md#a-simulation-one-command)): the split, then the run-config overrides.
The results are in `~/AITDM_Project/outputs/sim-<split>-<timestamp>-job<id>/` (`metrics.json`, `final_model.pt`).

What the job adds to the laptop version, automatically (see [`scripts/_common.sh`](../scripts/_common.sh) and [`scripts/run_simulation.sh`](../scripts/run_simulation.sh)):

* **Its own ports and Flower folder.**
  On a laptop, `flwr run` starts a hidden local SuperLink on the fixed ports 39091 and 39093.
  On a cluster node shared with other jobs (yours, or another student's), a second job would find that SuperLink and send its run there.
  In a Slurm job, the SuperLink gets free ports, and Flower and Ray keep their files in a folder of that job on the node (`FLWR_HOME`, `RAY_TMPDIR`), not in `~/.flwr`.
  Several jobs can run on the same node at the same time.
* **Only the CPUs and GPUs of the job.**
  Ray would otherwise see all CPUs of the node.
  The log shows what it got, e.g. `(num-supernodes=3 init-args-num-cpus=8 init-args-num-gpus=1 client-resources-num-gpus=0.33)`.
  With a GPU, the clients share it (here a third each), so they can train at the same time.
  Choose other values with `SIM_CPUS`, `SIM_GPUS`, `CLIENT_CPUS`, `CLIENT_GPUS`, e.g. `CLIENT_GPUS=1 bash Fep/submit.sh ...` to give every client a whole GPU, one client at a time.
* **The job id in the results folder**, because two jobs can start in the same second.

The log of a run looks like the laptop output ([sample](../Local/Linux/02_running_flower.md#sample-output)), preceded by a header:

```text
=== job 293022 (simulation) on haswell-wn36.grid.pub.ro, Wed Nov  4 21:09:47 EET 2026
    partition haswell, 8 CPUs, GPUs: none
    data -> /export/home/.../AITDM_Project/data
>>> Simulating 3 clients on .../data/partitions/derma_dir0.3_3 (num-supernodes=3 init-args-num-cpus=8 init-args-num-gpus=0)
...
>>> Done. Model and metrics: .../outputs/sim-derma_dir0.3_3-20261104-210955-job293022
```

## 3. Deployment: every component its own process, connected by IP

In deployment mode, the server and every client are separate processes that talk over the network ([tutorial 2, part B](../Local/Linux/02_running_flower.md#b-deployment-every-component-in-its-own-terminal)).
On fep you do not open terminals for them: the `deployment` job starts them as separate **Slurm tasks** of one job.

```bash
bash Fep/submit.sh deployment data/partitions/dirichlet                           # 3 clients: 4 tasks
bash Fep/submit.sh deployment data/partitions/dirichlet "num-server-rounds=10"
bash Fep/submit.sh -n 6 deployment data/partitions/five_clients                    # 5 clients: 1 + 5 tasks
bash Fep/submit.sh -N 2 deployment data/partitions/dirichlet                       # spread over 2 nodes
bash Fep/submit.sh -p dgxa100 -N 1 --gres=gpu:1 deployment data/partitions/dirichlet   # all on one node, sharing its GPU
```

* `--ntasks` (`-n`) must be the number of clients + 1. The default (4) fits 3 clients.
* The SuperLink and the SuperNodes are *steps* of one job, not separate jobs: the whole federation counts as **one** of your 2 running jobs.
* All tasks are on the **same partition**, but Slurm may place them on **different nodes** of it (force it with `-N 2`).
* **With a GPU** (`--gres=gpu:1`, and `-N 1`), all components run on one node and share its GPU, like the clients of a simulation: every ClientApp trains on it.
  Spreading over several nodes works on CPU only.
  This is a real network federation: the SuperLink listens on its node's IP address and a free port, and every SuperNode connects to `<that IP>:<port>`, never to `127.0.0.1` (which on another node would be *that* node).
* The log says where everything runs, and every component has its own log in `~/AITDM_Project/logs/<jobid>/` (`superlink.log`, `supernode_<i>.log`).

```text
    nodes: haswell-wn[29-30], SuperLink on haswell-wn29 (172.24.13.29:40782)
>>> Where every component runs (step 0 = SuperLink, 1.. = SuperNodes):
    step 293039.0 on haswell-wn29
    step 293039.1 on haswell-wn30
    step 293039.2 on haswell-wn29
    step 293039.3 on haswell-wn29
>>> flwr run (connection 'slurm-293039' -> 172.24.13.29:40782)
...
[round 3] global test: loss=0.9988, accuracy=0.7107, balanced_accuracy=0.2891, f1_macro=0.2854
>>> Done. Model and metrics: .../outputs/deploy-derma_dir0.3_3-20261104-212302-job293039
>>> Stopping SuperNodes and SuperLink
error: *** STEP 293039.1 ON haswell-wn30 CANCELLED AT 2026-11-04T21:25:12 DUE to SIGNAL Killed ***
```

The `CANCELLED ... Killed` lines at the end are normal: once the run is done, the job stops its SuperNodes and SuperLink.
Here SuperNode 0 (step 1) ran on another node than the SuperLink, and the result is the same as with everything on one node.

The menus of `run_deployment_terminals.sh` (restarting a client as an attacker, re-submitting) need interactive windows, which batch jobs do not have.
To do that on fep, get an interactive allocation and use tmux in it:

```bash
salloc -p haswell -c 8 -t 01:00:00          # waits for a node, then gives you a shell on the login node
srun --pty bash                             # a shell on the allocated node
FL_TERMINAL=tmux bash scripts/run_deployment_terminals.sh data/partitions/dirichlet
```

Inside an allocation the scripts behave as in a job: every allocation has its own ports (computed from its job id, not 8000 and 9094...) and its own Flower folder, so this is safe on a shared node.
`bash scripts/run_deployment_local.sh` works there too (one terminal, no menus).
Leave with `exit` twice: the allocation ends and its time stops counting.

## 4. Experiments with seeds, baselines, the whole tutorial

Every command of [tutorial 2's last part](../Local/Linux/02_running_flower.md#centralised-baseline-and-seeds) becomes a job with the same arguments:

```bash
bash Fep/submit.sh run_seeds --partitions data/partitions/dirichlet --config "num-server-rounds=10" --name fedavg
bash Fep/submit.sh run_seeds --partitions data/partitions/dirichlet --config "num-server-rounds=10 loss-weighting='balanced'" --name fedavg-balanced
bash Fep/submit.sh centralised --partitions data/partitions/dirichlet --epochs 10
```

You can submit them all at once: two of them run in parallel (the `student` limit), the others wait and start by themselves.
Summaries are quick, run them on the login node:

```bash
source .venv/bin/activate
python experiments/summarize.py outputs/experiments/central-dirichlet outputs/experiments/fedavg outputs/experiments/fedavg-balanced
```

Finished seeds are skipped, so if a job reaches its time limit (`TIMEOUT` in `sacct`), submit the same command again, with a longer `-t` if needed.

Everything in tutorials 3 to 6:

```bash
bash Fep/submit.sh reproduce_tutorials t3          # or t4, t5, t6; nothing = all (about 2 h on 16 CPUs)
```

Any other command of the tutorials (Grad-CAM, error analysis, feature importance, splitting a large dataset) runs with the `command` job:

```bash
bash Fep/submit.sh command python explainability/gradcam.py --run outputs/<run>
bash Fep/submit.sh command python explainability/error_analysis.py --run outputs/<run>
```

The pictures they write end up in `~/AITDM_Project/outputs/...`.
Copy them to your laptop to look at them:

```bash
scp -r <your.username>@fep.grid.pub.ro:AITDM_Project/outputs/<run> .        # on your laptop
```

## 5. Writing your own job

Copy a job file and change it: the `#SBATCH` lines are the defaults, the rest is ordinary bash run in the repository root, with `.venv` active.

```bash
cp Fep/jobs/simulation.sbatch Fep/jobs/my_sweep.sbatch
bash Fep/submit.sh my_sweep
```

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my_sweep
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
source Fep/jobs/_job_common.sh my_sweep "$@"
for mu in 0.01 0.1 1.0; do
    python experiments/run_seeds.py --partitions data/partitions/dirichlet --name fedprox-mu$mu \
        --config "num-server-rounds=10 strategy='fedprox' proximal-mu=$mu"
done
```

Be fair to the others: ask for the CPUs, GPUs and time you need, not more.
A job that asks for less starts sooner.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Job stays `PD` with `(QOSMaxJobsPerUserLimit)` | You already have 2 jobs running (the `student` limit). It starts when one of them ends |
| Job stays `PD` with `(Priority)` or `(Resources)` | The partition is busy, it is waiting its turn. `sinfo -s` shows idle nodes; a CPU partition usually starts sooner |
| `PD` with `(ReqNodeNotAvail ...)` | The nodes you asked for (`-w`) or the whole partition are down. Choose another partition |
| `Invalid account or account/partition combination specified` | Your account may not use that partition, see [partitions](#1-partitions-which-queue-can-i-use) |
| `TIMEOUT` in `sacct` | The job hit its time limit. Re-submit with a larger `-t` (`run_seeds` skips finished seeds) |
| `OUT_OF_MEMORY` | Ask for more CPUs (`-c 16`): memory grows with them on haswell |
| `The run FAILED - read the error (Traceback) above` | The error is in the job log, above that line. Errors of the ServerApp and ClientApp are Python errors, as on a laptop |
| `CUDA available: False` / clients train on CPU on a GPU partition | Add `--gres=gpu:1` |
| `CUDA error: no kernel image is available` | The GPU is too old for our PyTorch (P100 on `xl`). Use another partition |
| Deployment: `ERROR: 3 clients need --ntasks=4` | Pass `-n <clients + 1>` |
| Deployment: `ERROR: with --gres=gpu the deployment job must run on one node` | Add `-N 1` (with a GPU, all components share one node) |
| Deployment hangs at `Waiting for nodes to connect` | A SuperNode failed: read `~/AITDM_Project/logs/<jobid>/supernode_<i>.log` |
