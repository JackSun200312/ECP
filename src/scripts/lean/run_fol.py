#!/usr/bin/env python3
import subprocess
import sys
import json
from src.ecp.utils import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_print_fol.py path/to/File.lean", file=sys.stderr)
        sys.exit(1)

    fol_output = run_print_fol(sys.argv[1])
    print(fol_output)
