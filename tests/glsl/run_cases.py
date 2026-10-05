#!/usr/bin/env python3
"""Run the hand-written GLSL cases through glslc (compile + link + MSL + Metal validation).

Cases are grouped by basename: name.vert + name.frag form a program, name.comp a
compute program. Files starting with "fail_" must be rejected by the compiler.
usage: run_cases.py <glslc> [cases_dir] [-v]
"""
import os
import subprocess
import sys

glslc = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else os.path.join(os.path.dirname(__file__), "cases")
verbose = "-v" in sys.argv
STAGES = {".vert": "vert", ".frag": "frag", ".comp": "comp", ".geom": "geom", ".tesc": "tesc", ".tese": "tese"}
groups = {}
for f in sorted(os.listdir(d)):
    base, ext = os.path.splitext(f)
    if ext in STAGES:
        groups.setdefault(base, []).append(f)
ok = fail = 0
for base, files in sorted(groups.items()):
    args = [glslc, "-q"]
    if base.startswith("fp64"):
        args.append("--no-validate")  # needs the fp64 runtime library (fp64_msl.inc) to link
    for f in files:
        args += ["-S", STAGES[os.path.splitext(f)[1]], os.path.join(d, f)]
    r = subprocess.run(args, capture_output=True, text=True)
    expect_fail = base.startswith("fail_")
    passed = (r.returncode != 0) if expect_fail else (r.returncode == 0)
    if passed:
        ok += 1
        print("PASS", base)
    else:
        fail += 1
        print("FAIL", base)
        print("    " + (r.stderr or r.stdout).strip().replace("\n", "\n    ")[:4000])
print(f"{ok}/{ok + fail} passed")
sys.exit(1 if fail else 0)
