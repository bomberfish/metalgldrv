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
# multi-pass pipelines (runtime dispatch emulated by glslrun --pipeline)
pd = os.path.join(d, "pipelines")
P = lambda *f: [os.path.join(pd, x) for x in f]
pipes = [
    ("gs_pass", P("gs_pass.vert", "gs_pass.geom", "gs_pass.frag"), [], "fail=0 uncovered=0"),
    ("gs_points", P("gs_points.vert", "gs_points.geom", "gs_points.frag"), ["--verts", "0,0", "--vpp", "1"], "rows=0..31"),
    ("gs_points_inst", P("gs_points.vert", "gs_points.geom", "gs_points.frag"), ["--verts", "0,0,0,0", "--vpp", "1", "--inst", "3"], "pass=1024 fail=0"),
    ("tess_tri", P("tess_tri.vert", "tess_tri.tesc", "tess_tri.tese", "tess_tri.frag"), [], "fail=0 uncovered=0"),
    ("gs_spill", P("gs_pass.vert", "gs_spill.geom", "gs_spill.frag"), [], "fail=0 uncovered=0"),
    ("gs_spill_big", P("gs_pass.vert", "gs_spill_big.geom", "gs_spill.frag"), [], "fail=0 uncovered=0"),
    ("gs_spill_lines", P("gs_pass.vert", "gs_spill_lines.geom", "gs_spill_lines.frag"), [], "pass=912 fail=0"),
    ("tess_spill", P("tess_tri.vert", "tess_tri.tesc", "tess_spill.tese", "gs_spill.frag"), [], "fail=0 uncovered=0"),
    ("tess_quad_notcs", P("tess_quad.vert", "tess_quad.tese", "tess_quad.frag"),
     ["--verts", "-1,-1,1,-1,1,1,-1,1", "--vpp", "4", "--outer", "3.5,6,2.2,9", "--inner", "5.5,7"], "pass=4096 fail=0"),
    ("xfb", P("xfb.vert"), ["--verts", "1,2,3,4,5,6", "--xfb", "twice,id", "--inst", "2", "--expect-prims", "2"],
     "buffer 0 (stride 12): 2 4 0 6 8 1 10 12 2 2 4 100 6 8 101 10 12 102"),
    ("xfb_separate", P("xfb.vert"), ["--verts", "1,2,3,4,5,6", "--xfb", "id,twice", "--xfb-separate", "--expect-prims", "1"],
     "buffer 1 (stride 8): 2 4 6 8 10 12"),
    ("gs_xfb", P("gs_xfb.vert", "gs_xfb.geom"), ["--verts", "0,0,0,0,0,0,1,1,1,1,1,1", "--xfb", "val", "--expect-prims", "4"],
     "buffer 0 (stride 4): 0 1 2 2 1 3 10 11 12 12 11 13"),
    ("gs_xfb_overflow", P("gs_xfb.vert", "gs_xfb.geom"), ["--verts", "0,0,0,0,0,0,1,1,1,1,1,1", "--xfb", "val", "--xfb-capacity", "28",
     "--expect-prims", "2"], "written=2 generated=4"),
    ("xfb_skip", P("xfb.vert"), ["--verts", "1,2,3,4,5,6", "--xfb", "id,gl_SkipComponents2,gl_Position", "--expect-prims", "1"],
     "buffer 0 (stride 28):"),
]
for name, files, extra, expect in pipes:
    r = subprocess.run([glslrun, "--pipeline"] + files + extra, capture_output=True, text=True)
    good = r.returncode == 0 and expect in r.stdout
    if good:
        ok += 1
        print("PASS", name, "-", r.stdout.strip().splitlines()[-1][:120])
    else:
        fail += 1
        print("FAIL", name, (r.stdout + r.stderr).strip()[:3000])
r = subprocess.run([glslrun, "--tesscheck"], capture_output=True, text=True)
if r.returncode == 0:
    ok += 1
    print("PASS tesscheck -", r.stdout.strip().splitlines()[-1])
else:
    fail += 1
    print("FAIL tesscheck", r.stdout[-2000:])
print(f"{ok}/{ok + fail} passed")
sys.exit(1 if fail else 0)
