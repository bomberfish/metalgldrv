#!/usr/bin/env python3
"""Real-world GLSL corpus runner for the metalgldrv compiler.

Resolves Iris/OptiFine-style (#include "/lib/x.glsl") and Sodium-style
(#include <sodium:x.glsl>) includes, injects the defines Iris passes, then runs
glslc on every program (vsh+fsh(+gsh) pairs, csh alone) and reports results by
phase: compile / link / msl / metal / unsupported.

usage: corpus.py <glslc> pack <shaders_dir> [--jobs N] [--show N] [--filter substr]
       corpus.py <glslc> sodium <sodium_shaders_dir>
       corpus.py <glslc> files <dir>          (standalone .vert/.frag/... files, compile-only)
"""
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile

IRIS_DEFINES = [
    "MC_VERSION 12101", "MC_GL_VERSION 460", "MC_GLSL_VERSION 460", "IRIS_VERSION 10800", "MC_OS_MAC",
    "MC_GL_VENDOR_OTHER", "MC_GL_RENDERER_OTHER", "MC_RENDER_QUALITY 1.0", "MC_SHADOW_QUALITY 1.0",
    "MC_NORMAL_MAP", "MC_SPECULAR_MAP", "MC_HAND_DEPTH 0.125", "IS_IRIS", "MC_TEXTURE_FORMAT_LAB_PBR",
    "MC_TEXTURE_FORMAT_LAB_PBR_1_3", "IRIS_TAG_SUPPORT 2", "DISTANT_HORIZONS_SHADOWMAP",
]

PHASES_IRIS = ["NONE", "SKY", "SUNSET", "CUSTOM_SKY", "SUN", "MOON", "STARS", "VOID", "TERRAIN_SOLID", "TERRAIN_CUTOUT_MIPPED",
               "TERRAIN_CUTOUT", "ENTITIES", "BLOCK_ENTITIES", "DESTROY", "OUTLINE", "DEBUG", "HAND_SOLID", "TERRAIN_TRANSLUCENT",
               "TRIPWIRE", "PARTICLES", "CLOUDS", "RAIN_SNOW", "WORLD_BORDER", "HAND_TRANSLUCENT"]
IRIS_DEFINES += ["MC_RENDER_STAGE_%s %d" % (n, i) for i, n in enumerate(PHASES_IRIS)]

# Declarations Iris injects (via glsl-transformer) when a program references them without declaring them.
IRIS_INJECT = {
    "projectionMatrix": "uniform mat4 projectionMatrix;", "modelViewMatrix": "uniform mat4 modelViewMatrix;",
    "projectionMatrixInverse": "uniform mat4 projectionMatrixInverse;", "modelViewMatrixInverse": "uniform mat4 modelViewMatrixInverse;",
    "normalMatrix": "uniform mat3 normalMatrix;", "textureMatrix": "uniform mat4 textureMatrix;", "chunkOffset": "uniform vec3 chunkOffset;",
    "dhProjection": "uniform mat4 dhProjection;", "dhProjectionInverse": "uniform mat4 dhProjectionInverse;",
    "dhPreviousProjection": "uniform mat4 dhPreviousProjection;", "dhNearPlane": "uniform float dhNearPlane;",
    "dhFarPlane": "uniform float dhFarPlane;", "dhRenderDistance": "uniform int dhRenderDistance;", "renderStage": "uniform int renderStage;",
}
IRIS_INJECT_VS = {
    "vaPosition": "in vec3 vaPosition;", "vaColor": "in vec4 vaColor;", "vaUV0": "in vec2 vaUV0;", "vaUV1": "in ivec2 vaUV1;",
    "vaUV2": "in ivec2 vaUV2;", "vaNormal": "in vec3 vaNormal;", "dhMaterialId": "int dhMaterialId;",
}


def iris_inject(src, stage):
    decls = []
    table = dict(IRIS_INJECT)
    if stage == "vert":
        table.update(IRIS_INJECT_VS)
    for name, decl in table.items():
        if re.search(r"\b%s\b" % name, src) and not re.search(r"(uniform|in|attribute)\s+\w+\s+%s\b" % name, src) and not re.search(r"^\s*int\s+%s\s*;" % name, src, re.M):
            decls.append(decl)
    return decls


INCLUDE_RE = re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]')


def resolve(path, root, kind, stack, missing):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        missing.add(path)
        return ""
    out = []
    for line in text.splitlines():
        m = INCLUDE_RE.match(line)
        if not m:
            out.append(line)
            continue
        inc = m.group(1)
        if kind in ("sodium", "minecraft"):
            ns, _, rel = inc.partition(":")
            target = None
            if ns == kind:
                target = os.path.join(root, "include", rel)
            elif ns == "minecraft" and MC_INCLUDE:
                target = os.path.join(MC_INCLUDE, rel)
        else:
            target = os.path.join(root, inc.lstrip("/")) if inc.startswith("/") else os.path.join(os.path.dirname(path), inc)
        if not target or not os.path.exists(target):
            missing.add(inc)
            out.append("// missing include " + inc)
            continue
        target = os.path.normpath(target)
        if target in stack:
            continue
        out.append(resolve(target, root, kind, stack | {target}, missing))
    return "\n".join(out)


def inject(src, defines, decls=()):
    lines = src.split("\n")
    vi = next((i for i, l in enumerate(lines) if l.strip().startswith("#version")), -1)
    ins = ["#define " + d for d in defines] + list(decls)
    if vi < 0:
        return "\n".join(ins + lines)
    return "\n".join(lines[: vi + 1] + ins + lines[vi + 1:])


MC_INCLUDE = os.path.expanduser("~/metalgldrv-deps/corpus/minecraft/assets/minecraft/shaders/include")
if not os.path.isdir(MC_INCLUDE):
    MC_INCLUDE = None

PHASES = {0: "ok", 1: "compile", 3: "link", 4: "msl", 5: "metal"}


def run_program(glslc, stages, tmpdir, tag):
    args = [glslc, "-q"]
    for st, src in stages:
        fn = os.path.join(tmpdir, tag.replace("/", "_") + "." + st)
        open(fn, "w").write(src)
        args += ["-S", st, fn]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "timeout", "timeout"
    phase = PHASES.get(r.returncode, "crash(%d)" % r.returncode)
    err = (r.stderr or "").strip()
    if phase == "msl" and ("not supported yet" in err or "not supported by Metal" in err):
        phase = "unsupported"
    return phase, err


def pack_programs(shaders):
    progs = {}
    dirs = [shaders] + [os.path.join(shaders, d) for d in os.listdir(shaders) if d.startswith("world") and os.path.isdir(os.path.join(shaders, d))]
    for d in dirs:
        for f in os.listdir(d):
            base, ext = os.path.splitext(f)
            st = {".vsh": "vert", ".fsh": "frag", ".gsh": "geom", ".csh": "comp", ".tcs": "tesc", ".tes": "tese"}.get(ext)
            if not st:
                continue
            key = os.path.relpath(os.path.join(d, base), shaders)
            if st == "comp":
                progs[key + ":" + f] = [(st, os.path.join(d, f))]
            else:
                progs.setdefault(key, []).append((st, os.path.join(d, f)))
            # compute shaders named name_a.csh etc are separate programs
    return progs


def main():
    glslc, mode, root = sys.argv[1], sys.argv[2], sys.argv[3]
    jobs = 8
    show = 15
    filt = None
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
    if "--show" in sys.argv:
        show = int(sys.argv[sys.argv.index("--show") + 1])
    if "--filter" in sys.argv:
        filt = sys.argv[sys.argv.index("--filter") + 1]
    tmpdir = tempfile.mkdtemp(prefix="mglcorpus")
    work = []
    if mode == "pack":
        for key, files in sorted(pack_programs(root).items()):
            if filt and filt not in key:
                continue
            if os.path.basename(key).startswith(("clrwl_", "voxy", "dh_")):
                continue  # programs for other mods (Colorwheel / Voxy / Distant Horizons) depend on mod-injected APIs
            missing = set()
            stages = []
            for st, path in sorted(files, key=lambda x: ["vert", "tesc", "tese", "geom", "frag", "comp"].index(x[0])):
                defs = IRIS_DEFINES + [{"vert": "VERTEX_SHADER", "frag": "FRAGMENT_SHADER", "geom": "GEOMETRY_SHADER",
                                        "comp": "COMPUTE_SHADER", "tesc": "TESS_CONTROL_SHADER", "tese": "TESS_EVALUATION_SHADER"}[st]]
                text = resolve(path, root, "pack", {path}, missing)
                stages.append((st, inject(text, defs, iris_inject(text, st))))
            work.append((key, stages))
    elif mode == "sodium":
        variants = [["USE_VERTEX_COMPRESSION"], ["USE_VERTEX_COMPRESSION", "USE_FOG"], ["USE_VERTEX_COMPRESSION", "USE_FOG", "ALPHA_CUTOUT 0.5"]]
        for d, _, fs in os.walk(os.path.join(root, "blocks")):
            for f in fs:
                if f.endswith(".vsh"):
                    base = os.path.join(d, f[:-4])
                    for vi, v in enumerate(variants):
                        missing = set()
                        stages = [("vert", inject(resolve(base + ".vsh", root, "sodium", set(), missing), v))]
                        if os.path.exists(base + ".fsh"):
                            stages.append(("frag", inject(resolve(base + ".fsh", root, "sodium", set(), missing), v)))
                        work.append((os.path.basename(base) + "#" + str(vi), stages))
    elif mode == "minecraft":
        base_defs = ["PORTAL_LAYERS 15", "OIT_TRANSMITTANCE_TARGET_COUNT 2", "OIT_NUMBER_OF_DEPTH_BINS 8"]
        # define sets the game actually compiles (normal, cutout, lighting variants, OIT passes)
        variants = [base_defs + v for v in [[], ["ALPHA_CUTOUT 0.1"], ["PER_FACE_LIGHTING", "APPLY_TEXTURE_MATRIX", "GLINT"],
                                            ["EMISSIVE", "NO_OVERLAY", "NO_CARDINAL_LIGHTING"],
                                            ["OIT", "OIT_ALPHA_ONLY", "OIT_TRANSMITTANCE"], ["OIT", "OIT_ACCUMULATE"]]]
        dummy_vs = "#version 330\nlayout(location = 0) in vec3 Position;\nvoid main() { gl_Position = vec4(Position, 1.0); }\n"
        for sub in ("core", "post"):
            d = os.path.join(root, sub)
            for f in sorted(os.listdir(d)):
                if not f.endswith(".fsh") and not (f.endswith(".vsh") and not os.path.exists(os.path.join(d, f[:-4] + ".fsh"))):
                    continue
                base = os.path.join(d, f[:-4])
                for vi, v in enumerate(variants):
                    if f.startswith("oit_"):
                        v = base_defs + (["OIT", "OIT_DEPTH_BOUNDS"] if "depth_bounds" in f else ["OIT"]) if vi == 0 else None
                        if v is None:
                            continue
                    missing = set()
                    stages = []
                    if os.path.exists(base + ".vsh"):
                        stages.append(("vert", inject(resolve(base + ".vsh", root, "minecraft", set(), missing), v)))
                    else:
                        stages.append(("vert", dummy_vs))
                    if os.path.exists(base + ".fsh"):
                        stages.append(("frag", inject(resolve(base + ".fsh", root, "minecraft", set(), missing), v)))
                    work.append((sub + "/" + os.path.basename(base) + "#" + str(vi), stages))
    elif mode == "files":
        ext_map = {".vert": "vert", ".frag": "frag", ".comp": "comp", ".vsh": "vert", ".fsh": "frag", ".csh": "comp"}
        for d, _, fs in os.walk(root):
            for f in fs:
                st = ext_map.get(os.path.splitext(f)[1])
                if st:
                    p = os.path.join(d, f)
                    work.append((os.path.relpath(p, root), [(st, open(p, errors="replace").read())]))
    results = {}
    errors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_program, glslc, stages, tmpdir, key): key for key, stages in work}
        for fu in concurrent.futures.as_completed(futs):
            key = futs[fu]
            phase, err = fu.result()
            results[key] = phase
            if phase != "ok":
                errors[key] = err
    counts = {}
    for k, ph in results.items():
        counts[ph] = counts.get(ph, 0) + 1
    total = len(results)
    print("== %s: %d programs ==" % (root, total))
    for ph, n in sorted(counts.items(), key=lambda x: -x[1]):
        print("  %-12s %4d  (%.1f%%)" % (ph, n, 100.0 * n / max(1, total)))
    # group first error lines to find common causes
    shown = 0
    for key in sorted(errors):
        if results[key] == "unsupported":
            continue
        if shown >= show:
            break
        shown += 1
        first = "\n      ".join([l for l in errors[key].splitlines() if l.strip()][:4])
        print("  [%s] %s\n      %s" % (results[key], key, first[:600]))


if __name__ == "__main__":
    main()
