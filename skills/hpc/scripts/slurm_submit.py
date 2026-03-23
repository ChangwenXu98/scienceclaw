#!/usr/bin/env python3
"""Submit an arbitrary SLURM batch script to Artemis.

This is the general-purpose submission wrapper. For DFT-specific
submissions, use the dft skill's dft_submit.py instead.

Safety: refuses to run from inside a compute node.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def check_not_on_compute_node():
    if os.environ.get("SLURM_JOB_ID"):
        print("Error: refusing to submit from inside a compute node "
              "(SLURM_JOB_ID is set). Run from a login node.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Submit a SLURM batch script to Artemis")
    parser.add_argument("--script", "-s", required=True,
                        help="Path to SLURM batch script")
    parser.add_argument("--partition", "-p", default=None,
                        help="Override partition (e.g. venkvis-cpu, venkvis-h100, debug)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the script content without submitting")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    script_content = script_path.read_text()

    if args.dry_run:
        if args.format == "json":
            print(json.dumps({"dry_run": True, "script": str(script_path),
                              "content": script_content}, indent=2))
        else:
            print(f"=== DRY RUN: {script_path} ===")
            print(script_content)
        return

    check_not_on_compute_node()

    cmd = ["sbatch"]
    if args.partition:
        cmd.extend(["--partition", args.partition])
    cmd.append(str(script_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = {"error": f"sbatch failed: {result.stderr.strip()}",
               "script": str(script_path)}
        if args.format == "json":
            print(json.dumps(err, indent=2))
        else:
            print(f"Error: {err['error']}", file=sys.stderr)
        sys.exit(1)

    job_id = None
    for word in result.stdout.strip().split():
        if word.isdigit():
            job_id = word
            break

    out = {
        "job_id": job_id,
        "script": str(script_path),
        "status": "PENDING",
        "submit_time": datetime.now(timezone.utc).isoformat(),
    }
    if args.partition:
        out["partition"] = args.partition

    if args.format == "json":
        print(json.dumps(out, indent=2))
    else:
        for k, v in out.items():
            if v is not None:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
