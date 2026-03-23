#!/usr/bin/env python3
"""UMA skill entry point — structure relaxation via SLURM GPU submission.

This is the primary script invoked by the ScienceClaw skill executor.
It wraps uma_submit.py to submit a UMA relaxation job to a GPU node,
then optionally polls for completion.

For local (non-SLURM) relaxation, use uma_relax.py directly.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT_SCRIPT = SCRIPT_DIR / "uma_submit.py"
RETRIEVE_SCRIPT = SCRIPT_DIR / "uma_retrieve.py"
MATERIALS_SCRIPT = SCRIPT_DIR.parent.parent / "materials" / "scripts" / "materials_lookup.py"


def _extract_formula(text: str) -> str | None:
    """Extract a chemical formula from a text string.

    Handles cases like:
      "H3S"  -> "H3S"
      "LaH10" -> "LaH10"
      "Search for H3S superconductor" -> "H3S"
      "hydride superconductor candidates (H3S, CaH6)" -> "H3S"
    """
    import re
    text = text.strip()

    # If the text itself looks like a formula (short, starts with uppercase), use it
    if len(text) <= 20 and re.match(r'^[A-Z][a-z]?(\d*[A-Z][a-z]?\d*)*\d*$', text):
        return text

    # Try to extract formulas from the text
    # Pattern: uppercase letter, optional lowercase, optional digits, repeated
    formula_pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b'
    matches = re.findall(formula_pattern, text)
    if matches:
        # Return the first match that looks like a real formula (has at least one digit)
        for m in matches:
            if any(c.isdigit() for c in m):
                return m
        # No digits? Return first match anyway
        return matches[0]

    return None


def _lookup_mp_id(formula: str) -> str:
    """Look up an MP ID from a chemical formula via the materials skill."""
    if not MATERIALS_SCRIPT.exists():
        print(f"Error: materials skill not found at {MATERIALS_SCRIPT}. "
              f"Provide --mp-id directly.", file=sys.stderr)
        sys.exit(1)

    print(f"Looking up MP ID for formula '{formula}'...", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, str(MATERIALS_SCRIPT),
         "--formula", formula, "--format", "json"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"Error: materials lookup failed for '{formula}': {result.stderr}",
              file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
        # Handle both single result and list
        if isinstance(data, list):
            if not data:
                print(f"Error: no materials found for formula '{formula}'",
                      file=sys.stderr)
                sys.exit(1)
            data = data[0]
        mp_id = data.get("material_id") or data.get("mp_id")
        if not mp_id:
            print(f"Error: no MP ID found in lookup result for '{formula}'",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Found: {mp_id} ({data.get('formula', formula)})", file=sys.stderr)
        return mp_id
    except json.JSONDecodeError:
        print(f"Error: could not parse materials lookup output", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run UMA structure relaxation (submits to SLURM GPU node)")
    parser.add_argument("--structure", "-s",
                        help="Path to CIF, POSCAR, or XYZ structure file")
    parser.add_argument("--mp-id",
                        help="Materials Project ID (e.g. mp-149)")
    parser.add_argument("--query", "-q",
                        help="Alias for --mp-id (for skill executor compatibility)")
    parser.add_argument("--model", "-m", default="uma-m-1p1",
                        choices=["uma-s-1p1", "uma-s-1p2", "uma-m-1p1"])
    parser.add_argument("--task", "-t", default="omat",
                        choices=["omat", "omol", "omc", "oc20", "odac"])
    parser.add_argument("--partition", "-p", default="venkvis-h100")
    parser.add_argument("--relax-cell", action="store_true", default=True,
                        help="Enable full cell relaxation (default: True)")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--optimizer", default="FIRE",
                        choices=["FIRE", "LBFGS"])
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job completion and return results")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status checks when --wait (default: 30)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", default="json",
                        choices=["summary", "json"])
    args = parser.parse_args()

    # Handle --query: could be an MP ID (mp-149), a formula (H3S), or a topic string
    if args.query and not args.mp_id and not args.structure:
        query = args.query.strip()
        if query.startswith("mp-") or query.startswith("mvc-"):
            args.mp_id = query
        else:
            # Try to extract a chemical formula from the query
            formula = _extract_formula(query)
            if formula:
                args.mp_id = _lookup_mp_id(formula)
            else:
                print(f"Error: could not extract a chemical formula or MP ID from: {query}",
                      file=sys.stderr)
                sys.exit(1)

    if not args.structure and not args.mp_id:
        print("Error: provide --structure, --mp-id, or --query", file=sys.stderr)
        sys.exit(1)

    # Build submit command
    cmd = [sys.executable, str(SUBMIT_SCRIPT)]
    if args.structure:
        cmd.extend(["--structure", args.structure])
    elif args.mp_id:
        cmd.extend(["--mp-id", args.mp_id])
    cmd.extend([
        "--model", args.model,
        "--task", args.task,
        "--partition", args.partition,
        "--fmax", str(args.fmax),
        "--steps", str(args.steps),
        "--optimizer", args.optimizer,
        "--format", "json",
    ])
    if args.relax_cell:
        cmd.append("--relax-cell")
    if args.dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    try:
        submit_result = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout)
        return

    if args.dry_run or not args.wait:
        if args.format == "json":
            print(json.dumps(submit_result, indent=2))
        else:
            for k, v in submit_result.items():
                print(f"  {k}: {v}")
        return

    # Poll for completion
    job_id = submit_result.get("job_id")
    job_dir = submit_result.get("job_dir")
    print(f"Job {job_id} submitted. Polling every {args.poll_interval}s...",
          file=sys.stderr)

    while True:
        time.sleep(args.poll_interval)
        ret_cmd = [sys.executable, str(RETRIEVE_SCRIPT),
                   "--job-id", job_id, "--format", "json"]
        if job_dir:
            ret_cmd.extend(["--job-dir", job_dir])

        ret = subprocess.run(ret_cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            continue

        try:
            status_data = json.loads(ret.stdout)
        except json.JSONDecodeError:
            continue

        status = status_data.get("status", "UNKNOWN")
        print(f"  Status: {status}", file=sys.stderr)

        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
            if args.format == "json":
                print(json.dumps(status_data, indent=2))
            else:
                for k, v in status_data.items():
                    if k not in ("relaxed_cif_content", "slurm_output", "slurm_errors"):
                        print(f"  {k}: {v}")
            return


if __name__ == "__main__":
    main()
