#!/usr/bin/env python3
"""Archive a CTS run as a conformance receipt.

usage: save_report.py <results dir from run.sh> <case list name> [--surface S] [--platform P]

Copies results.csv (every case: status + details), summary.txt and the gzipped raw
QPA log into tests/cts/reports/<timestamp>-<list>/, records the driver commit, and
appends a line to tests/cts/reports/index.csv. LATEST.md is regenerated with the most
recent run of each case list.
"""
import argparse, csv, datetime, gzip, json, os, shutil, subprocess, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REPORTS = os.path.join(HERE, "reports")
FIELDS = ["date", "commit", "dirty", "caselist", "platform", "surface", "total", "pass", "fail", "not_supported",
          "quality_warning", "compatibility_warning", "crash", "timeout", "other", "pass_pct", "pass_or_not_supported_pct",
          "report_dir"]


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("caselist")
    ap.add_argument("--surface", default="pbuffer")
    ap.add_argument("--platform", default="macOS " + subprocess.run(["sw_vers", "-productVersion"], capture_output=True,
                                                                     text=True).stdout.strip() + " / Apple GPU")
    a = ap.parse_args()
    src = os.path.abspath(a.results)
    rows = list(csv.DictReader(open(os.path.join(src, "results.csv"))))
    st = Counter(r["status"] for r in rows)
    now = datetime.datetime.now()
    name = now.strftime("%Y%m%d-%H%M") + "-" + a.caselist
    dst = os.path.join(REPORTS, name)
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, "results.csv"), dst)
    if os.path.exists(os.path.join(src, "summary.txt")):
        shutil.copy(os.path.join(src, "summary.txt"), dst)
    if os.path.exists(os.path.join(src, "crashes.txt")):
        shutil.copy(os.path.join(src, "crashes.txt"), dst)
    qpa = os.path.join(src, "results.qpa")
    if os.path.exists(qpa):
        with open(qpa, "rb") as fi, gzip.open(os.path.join(dst, "results.qpa.gz"), "wb", 9) as fo:
            shutil.copyfileobj(fi, fo)
    dv = os.path.join(src, "driver-version.txt")
    commit = open(dv).read().split()[0] if os.path.exists(dv) else git("rev-parse", "--short", "HEAD")
    dirty = os.path.exists(dv) and len(open(dv).read().split("\n")) > 2
    total = len(rows)
    passed = st["Pass"]
    ns = st["NotSupported"]
    rec = {
        "date": now.isoformat(timespec="seconds"), "commit": commit, "dirty": int(dirty), "caselist": a.caselist,
        "platform": a.platform, "surface": a.surface, "total": total, "pass": passed, "fail": st["Fail"],
        "not_supported": ns, "quality_warning": st["QualityWarning"], "compatibility_warning": st["CompatibilityWarning"],
        "crash": st["Crash"], "timeout": st["Timeout"],
        "other": total - sum(st[k] for k in ("Pass", "Fail", "NotSupported", "QualityWarning", "CompatibilityWarning",
                                             "Crash", "Timeout")),
        "pass_pct": f"{100.0 * passed / total:.2f}" if total else "0",
        "pass_or_not_supported_pct": f"{100.0 * (passed + ns) / total:.2f}" if total else "0",
        "report_dir": os.path.relpath(dst, HERE),
    }
    json.dump(rec, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    idx = os.path.join(REPORTS, "index.csv")
    new = not os.path.exists(idx)
    with open(idx, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(rec)
    # LATEST.md: most recent run per case list
    latest = {}
    for r in csv.DictReader(open(idx)):
        latest[r["caselist"]] = r
    lines = ["# Latest CTS results per case list", "",
             "Statuses as reported by glcts. A conformant result has no Fail/Crash/Timeout; NotSupported is",
             "allowed only for optional features the implementation doesn't expose.", "",
             "| case list | commit | date | total | pass | not supported | fail | crash/timeout | warnings | pass+NS % | report |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(latest):
        r = latest[k]
        lines.append(f"| {k} | {r['commit']}{'+' if r['dirty'] == '1' else ''} | {r['date']} | {r['total']} | {r['pass']} | "
                     f"{r['not_supported']} | {r['fail']} | {int(r['crash']) + int(r['timeout'])} | "
                     f"{int(r['quality_warning']) + int(r['compatibility_warning'])} | {r['pass_or_not_supported_pct']} | "
                     f"[{os.path.basename(r['report_dir'])}]({os.path.basename(r['report_dir'])}/results.csv) |")
    open(os.path.join(REPORTS, "LATEST.md"), "w").write("\n".join(lines) + "\n")
    print(f"saved {dst}: {passed}/{total} pass, {ns} not supported, {st['Fail']} fail")


if __name__ == "__main__":
    sys.exit(main())
