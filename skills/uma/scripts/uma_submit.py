#!/usr/bin/env python3
"""Submit a UMA relaxation job to SLURM (GPU node).

End-to-end pipeline:
  1. Resolve structure (local file or Materials Project ID)
  2. Generate a SLURM batch script that runs uma_relax.py on a GPU node
  3. Submit via sbatch (or --dry-run to preview)

Safety:
  - Never submits from inside a compute node (checks SLURM_JOB_ID).
  - --dry-run prints the SLURM script without submitting.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCIENCECLAW_DIR = Path.home() / ".scienceclaw"
SKILL_DIR = Path(__file__).resolve().parent.parent
RELAX_SCRIPT = SKILL_DIR / "scripts" / "uma_relax.py"
DEFAULT_PARTITION = "venkvis-h100"
WORK_BASE = SCIENCECLAW_DIR / "uma_jobs"


def check_not_on_compute_node():
    if os.environ.get("SLURM_JOB_ID"):
        print("Error: refusing to submit from inside a compute node "
              "(SLURM_JOB_ID is set). Run from a login node.", file=sys.stderr)
        sys.exit(1)


def resolve_structure_path(args) -> tuple[str, str]:
    """Return (structure_source, description) for the SLURM script.

    If --structure is a local file, return its absolute path.
    If --mp-id, the SLURM script will fetch it at runtime via uma_relax.py --mp-id.
    """
    if args.structure:
        path = Path(args.structure).resolve()
        if not path.exists():
            print(f"Error: structure file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return str(path), path.stem
    if args.mp_id:
        return args.mp_id, args.mp_id.replace("-", "")
    print("Error: provide --structure or --mp-id", file=sys.stderr)
    sys.exit(1)


def build_slurm_script(args, source: str, label: str, job_dir: Path) -> str:
    """Generate a SLURM batch script for UMA relaxation."""
    partition = args.partition
    is_gpu = "a100" in partition or "h100" in partition
    default_walltime = "02:00:00"  # UMA is fast, 2h is plenty

    gpu_line = "#SBATCH --gres=gpu:1" if is_gpu else ""
    ntasks = 1  # UMA uses single GPU

    # Build uma_relax.py command
    relax_cmd_parts = [
        sys.executable, str(RELAX_SCRIPT),
    ]
    if args.mp_id:
        relax_cmd_parts.extend(["--mp-id", args.mp_id])
    else:
        relax_cmd_parts.extend(["--structure", source])

    relax_cmd_parts.extend([
        "--model", args.model,
        "--task", args.task,
        "--device", "cuda" if is_gpu else "cpu",
        "--fmax", str(args.fmax),
        "--steps", str(args.steps),
        "--optimizer", args.optimizer,
        "--output-cif", str(job_dir / f"{label}_relaxed.cif"),
        "--output-traj", str(job_dir / f"{label}_relaxation.traj"),
        "--format", "json",
    ])
    if args.relax_cell:
        relax_cmd_parts.append("--relax-cell")

    relax_cmd = " ".join(relax_cmd_parts)

    # Determine venv path
    venv_path = os.environ.get("VIRTUAL_ENV", "")
    venv_activate = f"source {venv_path}/bin/activate" if venv_path else "# no venv detected"

    script = f"""#!/bin/bash
#SBATCH --job-name=uma-{label}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={args.walltime or default_walltime}
#SBATCH --output={job_dir}/slurm-%j.out
#SBATCH --error={job_dir}/slurm-%j.err
{gpu_line}

# Activate environment
{venv_activate}

# Export API keys for MP fetch and HF model access
export MP_API_KEY="{os.environ.get('MP_API_KEY', '')}"
export HF_TOKEN="{os.environ.get('HF_TOKEN', '')}"

echo "=== UMA Relaxation Job ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Run UMA relaxation
{relax_cmd} | tee {job_dir}/results.json

echo ""
echo "End: $(date)"
"""
    return script


def submit_job(script_path: Path, partition: str = None) -> str:
    """Submit via sbatch, return job ID."""
    cmd = ["sbatch"]
    if partition:
        cmd.extend(["--partition", partition])
    cmd.append(str(script_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    for word in result.stdout.strip().split():
        if word.isdigit():
            return word
    print(f"Error: could not parse job ID from: {result.stdout}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Submit UMA relaxation to SLURM (GPU node)")
    parser.add_argument("--structure", "-s",
                        help="Path to CIF, POSCAR, or XYZ structure file")
    parser.add_argument("--mp-id",
                        help="Materials Project ID (e.g. mp-149)")
    parser.add_argument("--model", "-m", default="uma-m-1p1",
                        choices=["uma-s-1p1", "uma-s-1p2", "uma-m-1p1"],
                        help="UMA checkpoint (default: uma-m-1p1)")
    parser.add_argument("--task", "-t", default="omat",
                        choices=["omat", "omol", "omc", "oc20", "odac"],
                        help="Task / DFT level (default: omat)")
    parser.add_argument("--partition", "-p", default=DEFAULT_PARTITION,
                        help=f"SLURM partition (default: {DEFAULT_PARTITION})")
    parser.add_argument("--walltime", default=None,
                        help="Wall time (default: 02:00:00)")
    parser.add_argument("--relax-cell", action="store_true",
                        help="Enable full cell relaxation")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="Force convergence threshold (default: 0.05)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max optimizer steps (default: 200)")
    parser.add_argument("--optimizer", default="FIRE",
                        choices=["FIRE", "LBFGS"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print SLURM script without submitting")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    args = parser.parse_args()

    source, label = resolve_structure_path(args)

    # Create job working directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = WORK_BASE / f"{label}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)

    slurm_script = build_slurm_script(args, source, label, job_dir)
    script_path = job_dir / "submit.sh"
    script_path.write_text(slurm_script)
    script_path.chmod(0o755)

    if args.dry_run:
        info = {
            "dry_run": True,
            "source": source,
            "label": label,
            "model": args.model,
            "task": args.task,
            "partition": args.partition,
            "job_dir": str(job_dir),
            "slurm_script": slurm_script,
        }
        if args.format == "json":
            print(json.dumps(info, indent=2))
        else:
            print(f"=== DRY RUN — SLURM script (not submitted) ===")
            print(f"Job dir: {job_dir}")
            print(slurm_script)
        return

    check_not_on_compute_node()
    job_id = submit_job(script_path)

    result = {
        "job_id": job_id,
        "source": source,
        "model": args.model,
        "task": args.task,
        "partition": args.partition,
        "status": "PENDING",
        "job_dir": str(job_dir),
        "results_file": str(job_dir / "results.json"),
        "relaxed_cif": str(job_dir / f"{label}_relaxed.cif"),
        "trajectory": str(job_dir / f"{label}_relaxation.traj"),
        "submit_time": datetime.now(timezone.utc).isoformat(),
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
