#!/usr/bin/env python3
"""Query SLURM job status or queue overview on Artemis.

Modes:
  --job-id 12345     → single job status (squeue + sacct fallback)
  --queue            → queue overview for the current user (or --user)
"""

import argparse
import json
import os
import subprocess
import sys


def get_job_status(job_id: str) -> dict:
    """Get status of a single SLURM job."""
    # Try squeue first (active jobs)
    result = subprocess.run(
        ["squeue", "--job", job_id, "--noheader",
         "--format=%i|%T|%M|%N|%P|%j|%l"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("|")
        if len(parts) >= 7:
            return {
                "job_id": parts[0].strip(),
                "status": parts[1].strip(),
                "elapsed": parts[2].strip(),
                "node": parts[3].strip() or None,
                "partition": parts[4].strip(),
                "job_name": parts[5].strip(),
                "time_limit": parts[6].strip(),
            }

    # Fallback to sacct (completed/failed)
    result = subprocess.run(
        ["sacct", "-j", job_id, "--noheader", "--parsable2",
         "--format=JobID,State,Elapsed,NodeList,Partition,JobName,ExitCode"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 7 and "." not in parts[0]:
                return {
                    "job_id": parts[0].strip(),
                    "status": parts[1].strip(),
                    "elapsed": parts[2].strip(),
                    "node": parts[3].strip() or None,
                    "partition": parts[4].strip(),
                    "job_name": parts[5].strip(),
                    "exit_code": parts[6].strip(),
                }

    return {"job_id": job_id, "status": "UNKNOWN",
            "error": "Job not found in squeue or sacct"}


def get_queue_overview(user: str | None = None,
                       partition: str | None = None) -> dict:
    """Get overview of queued/running jobs."""
    cmd = ["squeue", "--noheader",
           "--format=%i|%T|%M|%N|%P|%j|%u|%l"]
    if user:
        cmd.extend(["--user", user])
    if partition:
        cmd.extend(["--partition", partition])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": f"squeue failed: {result.stderr}"}

    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 8:
            jobs.append({
                "job_id": parts[0].strip(),
                "status": parts[1].strip(),
                "elapsed": parts[2].strip(),
                "node": parts[3].strip() or None,
                "partition": parts[4].strip(),
                "job_name": parts[5].strip(),
                "user": parts[6].strip(),
                "time_limit": parts[7].strip(),
            })

    pending = sum(1 for j in jobs if j["status"] == "PENDING")
    running = sum(1 for j in jobs if j["status"] == "RUNNING")

    return {
        "total": len(jobs),
        "running": running,
        "pending": pending,
        "jobs": jobs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Query SLURM job status or queue overview")
    parser.add_argument("--job-id", "-j",
                        help="SLURM job ID for single-job query")
    parser.add_argument("--queue", "-q", action="store_true",
                        help="Show queue overview instead of single job")
    parser.add_argument("--user", "-u", default=None,
                        help="Filter queue by user (default: current user)")
    parser.add_argument("--partition", "-p", default=None,
                        help="Filter by partition (e.g. venkvis-cpu, venkvis-a100, venkvis-h100)")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    args = parser.parse_args()

    if not args.job_id and not args.queue:
        parser.error("provide --job-id or --queue")

    if args.job_id:
        result = get_job_status(args.job_id)
    else:
        user = args.user or os.environ.get("USER")
        result = get_queue_overview(user=user, partition=args.partition)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        if "jobs" in result:
            print(f"  total: {result['total']}  "
                  f"running: {result['running']}  "
                  f"pending: {result['pending']}")
            for j in result["jobs"]:
                print(f"  {j['job_id']:>10}  {j['status']:<10}  "
                      f"{j['job_name']:<20}  {j['elapsed']}")
        else:
            for k, v in result.items():
                if v is not None:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
