#!/usr/bin/env python3
"""Execute Python code for computational workflows.

Accepts inline code (--code) or a script file (--file) and executes it,
capturing stdout, stderr, and return code.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Execute Python code for computational workflows")
    parser.add_argument("--code", "-c",
                        help="Inline Python code to execute")
    parser.add_argument("--file", "-f",
                        help="Path to Python script to execute")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Execution timeout in seconds (default: 300)")
    parser.add_argument("--format", default="summary",
                        choices=["summary", "json"])
    args = parser.parse_args()

    if not args.code and not args.file:
        print("Error: provide --code or --file", file=sys.stderr)
        sys.exit(1)

    if args.file:
        script_path = Path(args.file)
        if not script_path.exists():
            print(f"Error: file not found: {script_path}", file=sys.stderr)
            sys.exit(1)
        code = script_path.read_text()
    else:
        code = args.code

    # Write code to a temp file and execute
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    # Execute in the scienceclaw project directory (not /tmp)
    _cwd = str(Path(__file__).resolve().parent.parent.parent)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=args.timeout,
            cwd=_cwd,
            env={**__import__("os").environ},
        )
        elapsed = time.time() - start

        output = {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "execution_time_s": round(elapsed, 2),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        output = {
            "status": "timeout",
            "stdout": "",
            "stderr": f"Execution timed out after {args.timeout}s",
            "return_code": -1,
            "execution_time_s": round(elapsed, 2),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print(f"Status: {output['status']}")
        print(f"Time: {output['execution_time_s']}s")
        if output["stdout"]:
            print(f"Output:\n{output['stdout']}")
        if output["stderr"]:
            print(f"Errors:\n{output['stderr']}", file=sys.stderr)


if __name__ == "__main__":
    main()
