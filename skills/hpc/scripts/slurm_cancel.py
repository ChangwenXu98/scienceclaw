#!/usr/bin/env python3
"""Cancel a SLURM job on Artemis."""

import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Cancel a SLURM job")
    parser.add_argument("--job-id", "-j", required=True,
                        help="SLURM job ID to cancel")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    args = parser.parse_args()

    result = subprocess.run(
        ["scancel", args.job_id],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        err = {"job_id": args.job_id, "status": "ERROR",
               "error": f"scancel failed: {result.stderr.strip()}"}
        if args.format == "json":
            print(json.dumps(err, indent=2))
        else:
            print(f"Error: {err['error']}", file=sys.stderr)
        sys.exit(1)

    out = {"job_id": args.job_id, "status": "CANCELLED"}

    if args.format == "json":
        print(json.dumps(out, indent=2))
    else:
        print(f"  job_id: {out['job_id']}")
        print(f"  status: {out['status']}")


if __name__ == "__main__":
    main()
