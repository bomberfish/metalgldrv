#!/usr/bin/env python3
"""Crash-resilient glcts runner for metalgldrv.

Runs glcts over a case list in one process; if the process crashes or hangs,
the in-flight case is recorded as Crash/Timeout, the crash report (.ips) is
captured, and a fresh process continues with the remaining cases.  All chunk
logs are merged into one QPA file and a per-group summary is printed/written.

Normally invoked through run.sh.
"""
import argparse, fnmatch, glob, json, os, re, signal, subprocess, sys, threading, time
from collections import Counter, OrderedDict, defaultdict

STATUSES = ["Pass", "Fail", "NotSupported", "QualityWarning", "CompatibilityWarning",
            "InternalError", "ResourceError", "Crash", "Timeout", "Other"]


def load_cases(caselists, patterns, excludes):
    cases = []
    seen = set()
    for path in caselists:
        with open(path) as f:
            for line in f:
                c = line.strip()
                if not c or c.startswith("#") or c in seen:
                    continue
                seen.add(c)
                cases.append(c)
    if patterns:
        cases = [c for c in cases if any(fnmatch.fnmatchcase(c, p) for p in patterns)]
    if excludes:
        cases = [c for c in cases if not any(fnmatch.fnmatchcase(c, p) for p in excludes)]
    return cases


def parse_qpa(text):
    """Yield (case, status, details, raw_block) for each case found in a QPA log."""
    results = OrderedDict()
    blocks = {}
    cur = None
    buf = []
    for line in text.splitlines(keepends=True):
        if line.startswith("#beginTestCaseResult "):
            cur = line.split(" ", 1)[1].strip()
            buf = [line]
            continue
        if cur is None:
            continue
        buf.append(line)
        if line.startswith("#endTestCaseResult"):
            block = "".join(buf)
            m = re.search(r'<Result StatusCode="([^"]+)">(.*?)</Result>', block, re.S)
            status, det = (m.group(1), m.group(2).strip()) if m else ("Other", "no <Result>")
            if status != "Pass" and det in ("Fail", "Failed", "Not supported", "Error", ""):
                # Generic result text: use the first informative log message instead.
                texts = [t.strip() for t in re.findall(r"<Text>(.*?)</Text>", block, re.S)]
                texts = [t for t in texts if t and "state reset failed" not in t and "will terminate" not in t]
                if texts:
                    det = f"{det}: {texts[0][:300]}"
            results[cur] = (status, det)
            blocks[cur] = block
            cur = None
        elif line.startswith("#terminateTestCaseResult"):
            reason = line.split(" ", 1)[1].strip() if " " in line else ""
            results[cur] = ("Timeout" if "Timeout" in reason else "Crash", reason)
            blocks[cur] = "".join(buf)
            cur = None
    return results, blocks, cur, "".join(buf) if cur else ""


def ips_files():
    d = os.path.expanduser("~/Library/Logs/DiagnosticReports")
    return set(glob.glob(os.path.join(d, "glcts*.ips")))


def find_ips(pid):
    """Find the macOS crash report written for process `pid`."""
    for path in sorted(ips_files(), key=os.path.getmtime, reverse=True)[:50]:
        try:
            with open(path) as f:
                f.readline()
                body = json.loads(f.read())
            if body.get("pid") == pid:
                return path
        except Exception:  # noqa: BLE001
            continue
    return None


def lldb_backtrace(a, case, tag):
    log = os.path.join(a.outdir, f"{tag}-lldb.qpa")
    cmd = ["lldb", "--batch", "-o", "run", "-k", "thread backtrace", "-k", "kill", "-k", "quit", "--",
           a.glcts, f"--deqp-case={case}", f"--deqp-log-filename={log}"] + a.extra
    try:
        out = subprocess.run(cmd, cwd=a.workdir, capture_output=True, text=True, timeout=300).stdout
    except Exception as e:  # noqa: BLE001
        return f"(lldb failed: {e})"
    finally:
        try:
            os.remove(log)
        except OSError:
            pass
    m = out.find("(lldb) thread backtrace")
    frames = out[m:].splitlines()[1:41] if m >= 0 else out.splitlines()[-40:]
    return "\n".join("  " + l for l in frames)


def summarize_ips(path, max_frames=40):
    """Return a short text backtrace of the faulting thread from a macOS .ips report."""
    try:
        with open(path) as f:
            header = f.readline()
            body = json.loads(f.read())
    except Exception as e:  # noqa: BLE001
        return f"(could not parse {path}: {e})"
    out = []
    exc = body.get("exception", {})
    term = body.get("termination", {})
    out.append(f"exception: {exc.get('type')} {exc.get('signal')} {exc.get('subtype', '')}".rstrip())
    if term.get("indicator"):
        out.append(f"termination: {term.get('indicator')}")
    asi = body.get("asi")
    if asi:
        for k, v in asi.items():
            out.append(f"asi[{k}]: {' | '.join(v) if isinstance(v, list) else v}")
    images = body.get("usedImages", [])
    threads = body.get("threads", [])
    ft = body.get("faultingThread", 0)
    if ft < len(threads):
        for i, fr in enumerate(threads[ft].get("frames", [])[:max_frames]):
            img = images[fr.get("imageIndex", 0)] if fr.get("imageIndex", 0) < len(images) else {}
            name = img.get("name", "?")
            sym = fr.get("symbol", "0x%x" % (img.get("base", 0) + fr.get("imageOffset", 0)))
            off = fr.get("symbolLocation", 0)
            src = ""
            if fr.get("sourceFile"):
                src = f"  ({fr['sourceFile']}:{fr.get('sourceLine', '?')})"
            out.append(f"  #{i:<2} {name:<24} {sym} + {off}{src}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glcts", required=True)
    ap.add_argument("--workdir", required=True, help="directory containing gl_cts/ data (cwd for glcts)")
    ap.add_argument("--caselist", action="append", default=[])
    ap.add_argument("--filter", action="append", default=[], help="fnmatch pattern(s) of cases to include")
    ap.add_argument("--exclude", action="append", default=[], help="fnmatch pattern(s) of cases to exclude")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--timeout", type=float, default=60.0, help="per-case inactivity timeout (s)")
    ap.add_argument("--top", type=int, default=25, help="number of failure categories to list")
    ap.add_argument("--group-depth", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1, help="parallel glcts processes")
    ap.add_argument("--max-restarts", type=int, default=10000)
    ap.add_argument("extra", nargs="*", help="extra glcts args")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    cases = load_cases(a.caselist, a.filter, a.exclude)
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2
    jobs = max(1, min(a.jobs, len(cases)))
    print(f"[run] {len(cases)} cases, {jobs} job(s) -> {a.outdir}", flush=True)

    results = {}
    merged = open(os.path.join(a.outdir, "results.qpa"), "w")
    crashes = []
    terminations = Counter()  # glcts self-terminations after a case (e.g. state reset errors)
    lock = threading.Lock()
    nproc = [0]
    t0 = time.time()

    def worker(wid, shard):
        remaining = list(shard)
        chunk = 0
        while remaining and chunk <= a.max_restarts:
            chunk += 1
            tag = f"w{wid}-c{chunk}"
            cl = os.path.join(a.outdir, f"{tag}.txt")
            log = os.path.join(a.outdir, f"{tag}.qpa")
            with open(cl, "w") as f:
                f.write("\n".join(remaining) + "\n")
            if os.path.exists(log):
                os.remove(log)
            cmd = [a.glcts, f"--deqp-caselist-file={cl}", f"--deqp-log-filename={log}"] + a.extra
            stdout_path = os.path.join(a.outdir, f"{tag}.stdout")
            with open(stdout_path, "w") as so:
                p = subprocess.Popen(cmd, cwd=a.workdir, stdout=so, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                with lock:
                    nproc[0] += 1
                last_size, last_change, timed_out = -1, time.time(), False
                while p.poll() is None:
                    time.sleep(0.1)
                    try:
                        sz = os.path.getsize(log)
                    except OSError:
                        sz = 0
                    if sz != last_size:
                        last_size, last_change = sz, time.time()
                    elif time.time() - last_change > a.timeout:
                        timed_out = True
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                        break
            rc = p.returncode
            text = open(log, errors="replace").read() if os.path.exists(log) else ""
            res, blocks, inflight, partial = parse_qpa(text)
            crash = None
            if inflight is not None:
                status = "Timeout" if timed_out else "Crash"
                sig = -rc if rc is not None and rc < 0 else None
                detail = (f"process {'killed after %.0fs inactivity' % a.timeout if timed_out else 'died'} (rc={rc}"
                          + (f", signal {signal.Signals(sig).name}" if sig else "") + ")")
                bt = ""
                if not timed_out:
                    ips = None
                    for _ in range(40):  # wait for ReportCrash to write the .ips
                        ips = find_ips(p.pid)
                        if ips:
                            break
                        time.sleep(0.25)
                    if ips:
                        bt = f"report: {ips}\n{summarize_ips(ips)}"
                    else:
                        # ReportCrash throttles repeated reports; re-run the case under lldb instead.
                        bt = "(no .ips crash report; backtrace from lldb re-run)\n" + lldb_backtrace(a, inflight, tag)
                tail = "".join(open(stdout_path, errors="replace").readlines()[-12:])
                bt += "\nglcts output tail:\n" + "".join("  | " + l for l in tail.splitlines(True))
                crash = (inflight, status, detail, bt)
            with lock:
                for c, r in res.items():
                    results[c] = r
                    merged.write(blocks[c])
                    if "test program will terminate" in blocks[c]:
                        texts = re.findall(r"<Text>(.*?)</Text>", blocks[c], re.S)
                        msg = next((t for t in reversed(texts) if "will terminate" not in t), "?")
                        terminations[msg] += 1
                if crash:
                    results[inflight] = (crash[1], crash[2])
                    merged.write(partial + f"#terminateTestCaseResult {crash[1]}\n")
                    crashes.append(crash)
                    print(f"[run] {crash[1]}: {inflight} {crash[2]}", flush=True)
            done = set(res)
            if inflight is not None:
                done.add(inflight)
            elif rc != 0 and not res:
                print(f"[run] glcts exited rc={rc} without running any case; see {stdout_path}",
                      file=sys.stderr, flush=True)
                with lock:
                    for c in remaining:
                        results[c] = ("Other", f"glcts failed to start (rc={rc})")
                return
            before = len(remaining)
            remaining = [c for c in remaining if c not in done]
            if len(remaining) == before:
                with lock:
                    for c in remaining:
                        results[c] = ("Other", "not executed (case not found in glcts?)")
                remaining = []
            for fn in (cl, log) + (() if inflight is not None else (stdout_path,)):
                try:
                    os.remove(fn)
                except OSError:
                    pass

    def progress():
        while any(t.is_alive() for t in threads):
            time.sleep(30)
            with lock:
                n = len(results)
            print(f"[run] {n}/{len(cases)} done, {time.time() - t0:.0f}s", flush=True)

    # Interleave cases over workers so slow groups are spread out.
    threads = [threading.Thread(target=worker, args=(i, cases[i::jobs]), daemon=True) for i in range(jobs)]
    for t in threads:
        t.start()
    pt = threading.Thread(target=progress, daemon=True)
    pt.start()
    for t in threads:
        t.join()
    merged.close()
    elapsed = time.time() - t0
    results = OrderedDict((c, results[c]) for c in cases if c in results)
    chunk = nproc[0]

    # ---- summary
    groups = defaultdict(Counter)
    total = Counter()
    for c, (st, _) in results.items():
        st = st if st in STATUSES else "Other"
        g = ".".join(c.split(".")[: a.group_depth])
        groups[g][st] += 1
        total[st] += 1
    cols = [s for s in STATUSES if total[s]]
    lines = []
    w = max([len(g) for g in groups] + [5])
    hdr = f"{'group':<{w}} " + " ".join(f"{s[:12]:>12}" for s in cols) + f" {'total':>7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for g in sorted(groups):
        lines.append(f"{g:<{w}} " + " ".join(f"{groups[g][s]:>12}" for s in cols) +
                     f" {sum(groups[g].values()):>7}")
    lines.append("-" * len(hdr))
    lines.append(f"{'TOTAL':<{w}} " + " ".join(f"{total[s]:>12}" for s in cols) + f" {sum(total.values()):>7}")
    lines.append(f"\n{len(results)} cases in {elapsed:.1f}s, {chunk} glcts process(es)")
    fails = defaultdict(list)
    for c, (st, det) in results.items():
        if st in ("Fail", "InternalError", "ResourceError", "QualityWarning", "CompatibilityWarning", "Other"):
            key = re.sub(r"0x[0-9a-fA-F]+|\d+(\.\d+)?", "N", det.strip())[:160]
            fails[(st, key)].append(c)
    if fails:
        lines.append("\nTop failure messages (status, normalized message, count, example cases):")
        for (st, key), cs in sorted(fails.items(), key=lambda kv: -len(kv[1]))[:a.top]:
            lines.append(f"  {len(cs):>6}x {st}: {key}")
            for c in cs[:3]:
                lines.append(f"           e.g. {c}")
    if terminations:
        lines.append(f"\n{sum(terminations.values())} case(s) made glcts terminate itself afterwards"
                     " (the case result is kept; the run restarts with the next case):")
        for msg, n in terminations.most_common():
            lines.append(f"  {n:>6}x  {msg}")
    if crashes:
        lines.append(f"\n{len(crashes)} crash(es)/timeout(s):")
        for c, st, det, _ in crashes:
            lines.append(f"  {st:<8} {c}  [{det}]")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(a.outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(a.outdir, "results.csv"), "w") as f:
        f.write("case,status,details\n")
        for c, (st, det) in results.items():
            det1 = det.replace('"', "'").replace("\n", " ")[:300]
            f.write(f'{c},{st},"{det1}"\n')
    if crashes:
        with open(os.path.join(a.outdir, "crashes.txt"), "w") as f:
            for c, st, det, bt in crashes:
                f.write(f"==== {st}: {c}\n{det}\n{bt}\n\n")
        print(f"backtraces: {os.path.join(a.outdir, 'crashes.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
