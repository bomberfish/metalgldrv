#!/usr/bin/env python3
"""Lean replacement for VK-GL-CTS external/fetch_sources.py.

Fetches only the external sources the GL CTS (glcts) build needs, using
shallow single-commit git fetches instead of full clones.  Revisions are read
from the CTS checkout's own fetch_sources.py so they always match the pinned
CTS commit.  Vulkan-only packages (validation layers, Vulkan-Docs, video
samples/generator, renderdoc) are skipped.
"""
import os, subprocess, sys

cts = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
ext = os.path.join(cts, "external")
sys.path.insert(0, ext)
os.chdir(ext)
import fetch_sources as fs  # noqa: E402

WANT_GIT = {"spirv-tools", "glslang", "spirv-headers", "amber", "jsoncpp"}
WANT_PKG = {"libpng"}  # zlib comes from the macOS SDK

def run(*args, cwd=None):
    subprocess.check_call(list(args), cwd=cwd)

for pkg in fs.PACKAGES:
    if isinstance(pkg, fs.GitRepo) and pkg.baseDir in WANT_GIT:
        dst = os.path.join(ext, pkg.baseDir, pkg.extractDir)
        stamp = os.path.join(dst, ".cts-rev")
        if os.path.exists(stamp) and open(stamp).read().strip() == pkg.revision:
            print(f"[fetch] {pkg.baseDir} up to date ({pkg.revision})")
            continue
        print(f"[fetch] {pkg.baseDir} @ {pkg.revision}")
        os.makedirs(dst, exist_ok=True)
        if not os.path.isdir(os.path.join(dst, ".git")):
            run("git", "init", "-q", cwd=dst)
        run("git", "fetch", "-q", "--depth", "1", pkg.httpsUrl, pkg.revision, cwd=dst)
        run("git", "checkout", "-q", "--force", "FETCH_HEAD", cwd=dst)
        if pkg.patch:
            run("git", "apply", os.path.join(ext, pkg.patch), cwd=dst)
        with open(stamp, "w") as f:
            f.write(pkg.revision + "\n")
    elif isinstance(pkg, fs.SourcePackage) and pkg.baseDir in WANT_PKG:
        stamp = os.path.join(ext, pkg.baseDir, pkg.extractDir, ".cts-rev")
        if os.path.exists(stamp) and open(stamp).read().strip() == pkg.checksum:
            print(f"[fetch] {pkg.baseDir} up to date")
            continue
        print(f"[fetch] {pkg.baseDir} (tarball)")
        fs.pkg = pkg  # fetch_sources.py refers to module-globals "pkg" and "args"
        fs.args = type("A", (), {"insecure": False})()
        pkg.update("https", False)
        with open(stamp, "w") as f:
            f.write(pkg.checksum + "\n")
