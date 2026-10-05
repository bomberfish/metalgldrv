#!/usr/bin/env python3
"""Run the self-checking GPU execution tests (tests/glsl/exec/*.comp) with glslrun.
usage: run_exec.py <glslrun> [exec_dir]"""
import os
import subprocess
import sys

glslrun = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "exec")
ok = fail = 0
for f in sorted(os.listdir(d)):
    if not f.endswith(".comp"):
        continue
    r = subprocess.run([glslrun, os.path.join(d, f)], capture_output=True, text=True)
    if r.returncode == 0:
        ok += 1
        print("PASS", f, "-", r.stdout.strip().splitlines()[-1].split(": ", 1)[-1])
    else:
        fail += 1
        print("FAIL", f, "(exit %d)" % r.returncode)
        print("    " + (r.stdout + r.stderr).strip().replace("\n", "\n    ")[:3000])
# graphics tests: name.vert + name.frag rendered with --draw
draws = [("fullscreen", [], None), ("lower-half", ["--tri", "-1,-1,1,-1,-1,0"], "rows=0..31")]
for f in sorted(os.listdir(d)):
    if not f.endswith(".vert"):
        continue
    base = os.path.join(d, f[:-5])
    for label, extra, expect in draws:
        r = subprocess.run([glslrun, "--draw", base + ".vert", base + ".frag"] + extra, capture_output=True, text=True)
        good = r.returncode == 0 and (expect is None or expect in r.stdout)
        tag = "%s[%s]" % (f[:-5], label)
        if good:
            ok += 1
            print("PASS", tag, "-", r.stdout.strip().splitlines()[-1])
        else:
            fail += 1
            print("FAIL", tag, (r.stdout + r.stderr).strip()[:2000])
print(f"{ok}/{ok + fail} passed")
sys.exit(1 if fail else 0)
