---
name: hpc
description: SLURM HPC job management — submit, monitor, and cancel batch jobs on Artemis
metadata:
---

# HPC Skill

General-purpose SLURM job management for the Artemis HPC cluster. Provides job submission, status monitoring, queue inspection, and cancellation. The DFT skill builds on top of this for domain-specific workflows.

## Status: IN DEVELOPMENT

## Artemis Cluster Overview

33 nodes total: 25 CPU, 3 large-memory, 3 H100 GPU, 2 A100 GPU. All CPU/largemem/H100 nodes use AMD EPYC 9654 (96 cores, Zen4). A100 nodes use AMD EPYC 7513 (32 cores, Zen3).

### Partitions

| Partition | Wall Time | Nodes | Notes |
|-----------|-----------|-------|-------|
| `venkvis-cpu` | 48h | 25 CPU (96c, 368 GB) | Default for DFT |
| `venkvis-largemem` | 48h | 3 (96c, 768 GB) | Large-memory jobs |
| `venkvis-a100` | 8h | 2 (4× A100 80GB each) | GPU compute |
| `venkvis-h100` | 8h | 3 (4× H100 80GB each) | GPU compute |
| `debug` | 30m | 4 nodes max, 1 job | Quick tests |

### Storage

| Tier | Path | Capacity | Notes |
|------|------|----------|-------|
| Turbo | `/nfs/turbo/coe-venkvis/` | 10 TB (500 GB fair share) | Persistent, backed up |
| Scratch | `/scratch/venkvis_root/venkvis/` | 10 TB (500 GB fair share) | **60-day auto-purge** |
| Home | `/home/<user>` | 80 GB | User home |
| Node Local | `/tmp` | 1.9 TB NVMe | Ephemeral, fast I/O |
| DataDen | Via Globus | 100 TB | Tape archival |

## Scripts

### `slurm_status.py` — Check job status or inspect the queue
```bash
python3 {baseDir}/scripts/slurm_status.py --job-id 12345 --format json
python3 {baseDir}/scripts/slurm_status.py --queue --partition venkvis-cpu --format json
```

### `slurm_submit.py` — Submit an arbitrary batch script
```bash
python3 {baseDir}/scripts/slurm_submit.py --script path/to/job.sh --format json
python3 {baseDir}/scripts/slurm_submit.py --script job.sh --partition venkvis-a100 --format json
```

### `slurm_cancel.py` — Cancel a running or pending job
```bash
python3 {baseDir}/scripts/slurm_cancel.py --job-id 12345 --format json
```

## Parameters

| Parameter | Script | Description |
|-----------|--------|-------------|
| `--job-id` | status, cancel | SLURM job ID |
| `--queue` | status | Show queue overview instead of single job |
| `--partition` | status, submit | Partition name (e.g. `venkvis-cpu`, `venkvis-h100`) |
| `--script` | submit | Path to SLURM batch script |
| `--format` | all | `summary` \| `json` |

## Safety
- Never submit from inside a compute node (checks `SLURM_JOB_ID`)
- Cancel requires explicit `--job-id`; no bulk cancel support
- Write large scratch data to `/scratch/`, not `/nfs/turbo/` or `/home/`
