#!/usr/bin/env python3
"""Retrieve results from a completed UMA SLURM job.

Checks job status via sacct, reads results.json and relaxed CIF from the
job working directory.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORK_BASE = Path.home() / ".scienceclaw" / "uma_jobs"


def get_job_status(job_id: str) -> dict:
    """Query SLURM for job status."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State,Elapsed,ExitCode,NodeList",
             "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {"job_id": job_id, "status": "UNKNOWN",
                    "error": result.stderr.strip()}

        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 5 and parts[0] == job_id:
                return {
                    "job_id": job_id,
                    "status": parts[1],
                    "elapsed": parts[2],
                    "exit_code": parts[3],
                    "node": parts[4],
                }
        return {"job_id": job_id, "status": "NOT_FOUND"}
    except FileNotFoundError:
        return {"job_id": job_id, "status": "UNKNOWN",
                "error": "sacct not available"}
    except Exception as e:
        return {"job_id": job_id, "status": "UNKNOWN", "error": str(e)}


def find_job_dir(job_id: str) -> Path | None:
    """Find the job directory by scanning for slurm output files."""
    if not WORK_BASE.exists():
        return None
    for d in sorted(WORK_BASE.iterdir(), reverse=True):
        if d.is_dir():
            for f in d.glob(f"slurm-{job_id}.*"):
                return d
            # Also check submit.sh for the job
            submit = d / "submit.sh"
            if submit.exists() and job_id in submit.read_text():
                return d
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve results from a completed UMA SLURM job")
    parser.add_argument("--job-id", "-j", required=True,
                        help="SLURM job ID")
    parser.add_argument("--job-dir", "-d",
                        help="Job working directory (auto-detected if omitted)")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    args = parser.parse_args()

    status = get_job_status(args.job_id)

    job_dir = Path(args.job_dir) if args.job_dir else find_job_dir(args.job_id)

    result = {**status}

    if job_dir and job_dir.exists():
        result["job_dir"] = str(job_dir)

        # Read results.json
        results_file = job_dir / "results.json"
        if results_file.exists():
            try:
                results_data = json.loads(results_file.read_text())
                result["relaxation_results"] = results_data
            except json.JSONDecodeError:
                result["relaxation_results_raw"] = results_file.read_text()[:2000]

        # Check for relaxed CIF
        cif_files = list(job_dir.glob("*_relaxed.cif"))
        if cif_files:
            result["relaxed_cif_path"] = str(cif_files[0])
            result["relaxed_cif_content"] = cif_files[0].read_text()[:5000]

        # Check for trajectory
        traj_files = list(job_dir.glob("*.traj"))
        if traj_files:
            result["trajectory_path"] = str(traj_files[0])

        # Read SLURM output
        slurm_out = list(job_dir.glob(f"slurm-{args.job_id}.out"))
        if slurm_out:
            result["slurm_output"] = slurm_out[0].read_text()[-3000:]

        slurm_err = list(job_dir.glob(f"slurm-{args.job_id}.err"))
        if slurm_err:
            err_text = slurm_err[0].read_text().strip()
            if err_text:
                result["slurm_errors"] = err_text[-2000:]
    else:
        result["job_dir"] = None
        result["note"] = "Job directory not found. Provide --job-dir if known."

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"  job_id:   {result['job_id']}")
        print(f"  status:   {result['status']}")
        if result.get("elapsed"):
            print(f"  elapsed:  {result['elapsed']}")
        if result.get("node"):
            print(f"  node:     {result['node']}")
        if result.get("job_dir"):
            print(f"  job_dir:  {result['job_dir']}")
        if result.get("relaxation_results"):
            r = result["relaxation_results"]
            print(f"  formula:  {r.get('formula', 'N/A')}")
            print(f"  energy:   {r.get('final_energy_eV', 'N/A')} eV")
            print(f"  converged: {r.get('converged', 'N/A')}")
            print(f"  steps:    {r.get('steps_taken', 'N/A')}")
        if result.get("relaxed_cif_path"):
            print(f"  CIF:      {result['relaxed_cif_path']}")
        if result.get("note"):
            print(f"  note:     {result['note']}")


if __name__ == "__main__":
    main()
