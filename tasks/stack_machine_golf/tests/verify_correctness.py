#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="/app/stack_vm")
    args = parser.parse_args()
    if not os.path.exists(args.binary):
        print(f"missing binary: {args.binary}", file=sys.stderr)
        return 1
    proc = subprocess.run([args.binary], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return 0 if proc.returncode == 0 and "verify=ok" in proc.stdout else 1

if __name__ == "__main__":
    raise SystemExit(main())
