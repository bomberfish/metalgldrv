#!/usr/bin/env python3
"""Add an LC_LOAD_DYLIB (or LC_LOAD_WEAK_DYLIB) to a Mach-O executable, in the header padding.

For apps that DYLD_INSERT_LIBRARIES can't reach (launched by Steam or another launcher, or with a
hardened runtime): the executable then loads libmetalgldrv itself, which turns on replacement mode.

  macho-add-dylib.py [--weak] [--sign] [-o OUT] BINARY DYLIB

DYLIB is stored as given; use an absolute path (or @executable_path/...). Every 64-bit slice of a
universal binary gets the command. A slice that already loads DYLIB is left alone. The command goes
into the zero padding between the load commands and the first section, so nothing moves; a binary
without enough padding is refused. Editing the header invalidates the code signature: --sign signs
the result ad hoc (codesign -fs -), otherwise sign it before running it.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC, FAT_MAGIC_64 = 0xCAFEBABE, 0xCAFEBABF
LC_SEGMENT_64 = 0x19
LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_LAZY_LOAD_DYLIB, LC_LOAD_UPWARD_DYLIB = (
    0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023)
DYLIB_CMDS = {LC_LOAD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_LAZY_LOAD_DYLIB, LC_LOAD_UPWARD_DYLIB}
MACH_HEADER_64 = 32
SECTION_64 = 80


def slices(data):
    """(offset, size) of each Mach-O in the file."""
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic in (FAT_MAGIC, FAT_MAGIC_64):
        n = struct.unpack_from(">I", data, 4)[0]
        out = []
        for i in range(n):
            if magic == FAT_MAGIC:
                _, _, off, size, _ = struct.unpack_from(">iiIII", data, 8 + 20 * i)
            else:
                _, _, off, size, _, _ = struct.unpack_from(">iiQQII", data, 8 + 32 * i)
            out.append((off, size))
        return out
    return [(0, len(data))]


def add_to_slice(data, base, dylib, cmd_id):
    """Adds the command to the Mach-O at `base`; returns (changed, message)."""
    magic, cputype, _, _, ncmds, sizeofcmds = struct.unpack_from("<IiIIII", data, base)
    if magic != MH_MAGIC_64:
        return False, f"skipped a slice that isn't 64-bit little-endian Mach-O (magic 0x{magic:08x})"
    arch = {0x01000007: "x86_64", 0x0100000C: "arm64"}.get(cputype, f"cputype 0x{cputype:x}")

    # Where the first section's contents start (the end of the usable padding), and the dylibs already loaded.
    first_content = None
    off = base + MACH_HEADER_64
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmd == LC_SEGMENT_64:
            nsects = struct.unpack_from("<I", data, off + 64)[0]
            for s in range(nsects):
                s_off, = struct.unpack_from("<I", data, off + 72 + SECTION_64 * s + 48)
                if s_off and (first_content is None or s_off < first_content):
                    first_content = s_off
        elif cmd in DYLIB_CMDS:
            name_off = struct.unpack_from("<I", data, off + 8)[0]
            name = bytes(data[off + name_off:off + cmdsize]).split(b"\0")[0].decode("utf-8", "replace")
            if name == dylib:
                return False, f"{arch}: already loads {dylib}"
        off += cmdsize
    if first_content is None:
        raise SystemExit(f"{arch}: no section contents found, can't tell how much header padding there is")

    # struct dylib_command: cmd, cmdsize, name offset, timestamp, current_version, compatibility_version; then the name.
    name = dylib.encode("utf-8") + b"\0"
    cmdsize = (24 + len(name) + 7) & ~7
    new_cmd = struct.pack("<IIIIII", cmd_id, cmdsize, 24, 2, 0x10000, 0x10000) + name
    new_cmd += b"\0" * (cmdsize - len(new_cmd))

    cmds_end = base + MACH_HEADER_64 + sizeofcmds
    room = base + first_content - cmds_end
    if cmdsize > room:
        raise SystemExit(f"{arch}: needs {cmdsize} bytes of header padding, has {room}")
    if any(data[cmds_end:cmds_end + cmdsize]):
        raise SystemExit(f"{arch}: header padding isn't zero, refusing to overwrite it")
    data[cmds_end:cmds_end + cmdsize] = new_cmd
    struct.pack_into("<II", data, base + 16, ncmds + 1, sizeofcmds + cmdsize)
    kind = "LC_LOAD_WEAK_DYLIB" if cmd_id == LC_LOAD_WEAK_DYLIB else "LC_LOAD_DYLIB"
    return True, f"{arch}: added {kind} {dylib} ({cmdsize} bytes, {room - cmdsize} bytes of padding left)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("binary")
    ap.add_argument("dylib", help="install name to load, stored as given")
    ap.add_argument("-o", "--output", help="write here instead of replacing BINARY")
    ap.add_argument("--weak", action="store_true", help="LC_LOAD_WEAK_DYLIB: the app still starts when DYLIB is missing")
    ap.add_argument("--sign", action="store_true", help="sign the result ad hoc (codesign -fs -)")
    args = ap.parse_args()

    with open(args.binary, "rb") as f:
        data = bytearray(f.read())
    cmd_id = LC_LOAD_WEAK_DYLIB if args.weak else LC_LOAD_DYLIB
    changed = False
    for base, _ in slices(data):
        c, msg = add_to_slice(data, base, args.dylib, cmd_id)
        print(msg)
        changed |= c
    if not changed and not args.output:
        return

    out = args.output or args.binary
    mode = os.stat(args.binary).st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(out)), prefix=".macho-add-dylib.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, out)  # a new inode: a running copy of the old file keeps its pages
    except BaseException:
        os.unlink(tmp)
        raise
    if args.sign:
        subprocess.run(["codesign", "-fs", "-", out], check=True)
    else:
        print(f"the signature of {out} is now invalid: codesign -fs - '{out}'", file=sys.stderr)


if __name__ == "__main__":
    main()
