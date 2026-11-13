"""Run cutmesh over a suite of STL files and report which ones mesh cleanly.

    python benchmark/run_bench.py benchmark/suites/synthetic.json
    python benchmark/run_bench.py benchmark/suites/synthetic.json -j 4 --only "thin_*"
    python benchmark/run_bench.py SUITE --baseline benchmark/baselines/synthetic.json
    python benchmark/run_bench.py SUITE --save-baseline       # -> benchmark/baselines/<suite>.json
    python benchmark/run_bench.py SUITE --set addLayers='{"nLayers": 3}'

Each case runs in its own process (a crash or a hang only loses that case) with a timeout.
Results go to benchmark/results/<suite>-<time>/: results.json, results.csv, summary.md and,
per case, cases/<name>/{meshDict.json, log.txt, overview.png, cut.png, case/constant/polyMesh}
(pictures: render_mesh.py; --no-images skips them).  The polyMesh of a PASS case is deleted
unless --keep is given.

Suite file (paths relative to the suite file):
    {"name": "synthetic",
     "defaults": {"cellsAcross": 24, "meshDict": {"surfaceRefinement": {"default": 2}}},
     "cases": [{"name": "sphere", "stl": "../data/synthetic/sphere.stl", "expect": "pass"},
               {"name": "cube_aligned", "stl": "...", "meshDict": {"baseCellSize": 0.0625}}]}

cellsAcross sets baseCellSize = (longest bounding-box side) / cellsAcross unless the case's
meshDict gives baseCellSize.  A case whose STL is missing is skipped (optional third-party data).

Status of a case:
    PASS     valid mesh, no warnings, no quality flags, volume within --vol-tol of the STL
    WARN     valid mesh, but warnings in the log (leak, separate regions, ...) or quality flags
             (non-orthogonality > 70, skewness > 4; with --strict also negative face-triangle
             tets) or a volume mismatch
    INVALID  mesh written but broken: non-positive volumes, bad face pyramids, open cells,
             faces shared by > 2 cells, boundary far from the STL
    ERROR    exception (message recorded)        CRASH  process died        TIMEOUT
Expectation ("expect" per case): "pass" (PASS only), "valid" (PASS or WARN), "reject" (input is
broken: ERROR or WARN is fine, a silent PASS or an INVALID mesh is not).  The verdict column
says ok / UNEXPECTED (or "-" when no expectation is given).
"""
import argparse
import csv
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RANK = {"PASS": 0, "WARN": 1, "INVALID": 2, "ERROR": 3, "CRASH": 3, "TIMEOUT": 3, "SKIP": -1}
INVALID_WARNINGS = ("faces shared by more than two cells", "boundary faces far from the STL")


# ------------------------------------------------------------------ worker (one case)
def _peak_memory_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 / (1024 if sys.platform == "darwin" else 1)
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 1024 ** 2
    except Exception:
        pass
    return None


def worker(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    wd = job["workdir"]
    res = {"name": job["name"]}
    lines = []
    logf = open(os.path.join(wd, "log.txt"), "w", encoding="utf-8")

    def log(s):
        s = str(s)
        lines.append(s)
        logf.write(s + "\n"); logf.flush()
    t0 = time.time()
    try:
        from cutmesh.mesher import run
        R = run(os.path.join(wd, "meshDict.json"), os.path.join(wd, "case"), log=log, check=True)
        q = {k: v for k, v in R.get("quality", {}).items() if not k.startswith("_")}
        res["quality"] = q
        res["steps"] = {name: round(dt, 3) for name, dt in R["timer"].steps}
        if job.get("images", True):
            try:                          # pictures must never turn a good case into an ERROR
                from render_mesh import render
                t1 = time.time()
                res["images"] = [os.path.basename(p) for p in render(R, wd, job["name"])]
                res["renderTime"] = round(time.time() - t1, 2)
            except Exception as e:        # noqa: BLE001
                res["renderError"] = f"{type(e).__name__}: {e}"
    except BaseException as e:            # noqa: BLE001 - record everything, incl. MemoryError
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        log(res["traceback"])
    res["time"] = round(time.time() - t0 - res.get("renderTime", 0), 3)
    res["peakMemMB"] = _peak_memory_mb()
    res["warnings"] = [s.strip() for s in lines if "WARNING" in s]
    res["repairs"] = [s.strip() for s in lines if s.startswith("REPAIR:")]
    for s in lines:
        if s.startswith("boundary faces: max distance"):
            try:
                res["boundaryDistCells"] = float(s.split("(")[1].split(" x")[0])
            except (IndexError, ValueError):
                pass
    logf.close()
    with open(os.path.join(wd, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    return 0


# ------------------------------------------------------------------ driver
def load_suite(path):
    with open(path, encoding="utf-8") as f:
        suite = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    suite.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    d = suite.get("defaults", {})
    cases = []
    for c in suite["cases"]:
        c = dict(c)
        c["stl"] = os.path.normpath(os.path.join(base, c["stl"]))
        c.setdefault("cellsAcross", d.get("cellsAcross", 24))
        md = json.loads(json.dumps(d.get("meshDict", {})))
        md.update(c.get("meshDict", {}))
        c["meshDict"] = md
        c.setdefault("expect", d.get("expect"))
        c.setdefault("timeout", d.get("timeout"))
        cases.append(c)
    return suite, cases


def classify(res, stl, vol_tol, strict=False):
    if res.get("status") in ("CRASH", "TIMEOUT"):
        return res["status"], [res.get("error", "")]
    if "error" in res:
        return "ERROR", [res["error"]]
    q = res.get("quality", {})
    bad, flags = [], []
    if q.get("nNegVol", 0):
        bad.append(f"{q['nNegVol']} non-positive volumes")
    if q.get("nBadPyramids", 0):
        bad.append(f"{q['nBadPyramids']} bad pyramids")
    if q.get("openCellEdges", 0):
        bad.append(f"{q['openCellEdges']} open cell edges")
    if q and not q.get("upperTriangular", True):
        bad.append("not upper-triangular")
    for w in res.get("warnings", []):
        if any(k in w for k in INVALID_WARNINGS):
            bad.append(w.replace("WARNING: ", ""))
        else:
            flags.append(w.replace("WARNING: ", ""))
    if q.get("nNonOrth70", 0):
        flags.append(f"{q['nNonOrth70']} faces non-orth > 70")
    if q.get("nSkew4", 0):
        flags.append(f"{q['nSkew4']} faces skewness > 4")
    if q.get("nBadTetDecomp", 0) and strict:      # stricter than checkMesh: only with --strict
        flags.append(f"{q['nBadTetDecomp']} negative face-triangle tets")
    ve = res.get("volErr")
    if ve is not None and abs(ve) > vol_tol:
        flags.append(f"volume {ve:+.2%} vs STL")
    if bad:
        return "INVALID", bad + flags
    return ("WARN" if flags else "PASS"), flags


def verdict(expect, status):
    if not expect or status == "SKIP":
        return "-"
    ok = {"pass": ("PASS",), "valid": ("PASS", "WARN"), "reject": ("ERROR", "WARN")}.get(expect)
    if ok is None:
        return "-"
    return "ok" if status in ok else "UNEXPECTED"


def _relpath(p):
    try:
        return os.path.relpath(p, ROOT).replace(os.sep, "/")
    except ValueError:                    # another drive on Windows
        return p


def run_case(c, outdir, args, overrides):
    from stl_check import check_file
    wd = os.path.join(outdir, "cases", c["name"])
    os.makedirs(wd, exist_ok=True)
    res = {"name": c["name"], "stl": _relpath(c["stl"]), "expect": c.get("expect")}
    if not os.path.exists(c["stl"]):
        res.update(status="SKIP", notes=["STL not found"], verdict="-")
        return res
    try:
        inp = check_file(c["stl"])
    except Exception as e:  # noqa: BLE001 - unreadable / empty STL: this case is an ERROR
        inp = {"error": f"{type(e).__name__}: {e}"}
    res["input"] = inp
    if "error" in inp:
        msg = f"input diagnostics failed: {inp['error']}"
        res.update(status="ERROR", error=msg, notes=[msg])
        res["verdict"] = verdict(c.get("expect"), "ERROR")
        return res
    md = dict(c["meshDict"])
    md.update(overrides)
    if "baseCellSize" not in md:
        lo, hi = inp["bbox"]
        md["baseCellSize"] = max(h - l for l, h in zip(lo, hi)) / float(c["cellsAcross"])
    md["domain"] = c["stl"]
    md["case"] = "case"
    with open(os.path.join(wd, "meshDict.json"), "w", encoding="utf-8") as f:
        json.dump(md, f, indent=1)
    job = {"name": c["name"], "workdir": wd, "images": not args.no_images}
    jp = os.path.join(wd, "job.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(job, f)
    rp = os.path.join(wd, "result.json")
    if os.path.exists(rp):
        os.remove(rp)
    timeout = c.get("timeout") or args.timeout
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", jp],
                           capture_output=True, text=True, timeout=timeout, cwd=wd)
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                res.update(json.load(f))
        else:
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-5:]
            res.update(status="CRASH", error=f"exit code {p.returncode}: " + " | ".join(tail))
    except subprocess.TimeoutExpired:
        res.update(status="TIMEOUT", error=f"no result after {timeout} s")
    res.setdefault("time", round(time.time() - t0, 3))
    res["baseCellSize"] = md["baseCellSize"]
    q = res.get("quality") or {}
    if q and inp.get("oriented") and inp.get("signedVolume"):
        res["volErr"] = q["totalVol"] / abs(inp["signedVolume"]) - 1.0
    status, notes = classify(res, inp, args.vol_tol, args.strict)
    notes = notes + [r.replace("REPAIR: ", "repaired: ") for r in res.get("repairs", [])]
    res["status"], res["notes"] = status, notes
    res["verdict"] = verdict(c.get("expect"), status)
    if status == "PASS" and not args.keep:
        shutil.rmtree(os.path.join(wd, "case", "constant", "polyMesh"), ignore_errors=True)
    return res


def input_summary(inp):
    if not inp or "error" in inp:
        return (inp or {}).get("error", "")
    from stl_check import summary
    return summary(inp)


CSV_COLS = ["name", "status", "verdict", "expect", "time", "cells", "maxNonOrth", "maxSkewness",
            "nBadTetDecomp", "volErr", "peakMemMB", "input", "notes"]


def row(r):
    q = r.get("quality") or {}
    return {
        "name": r["name"], "status": r["status"], "verdict": r.get("verdict", "-"),
        "expect": r.get("expect") or "", "time": r.get("time", ""),
        "cells": q.get("nCells", ""),
        "maxNonOrth": round(q["maxNonOrth"], 1) if "maxNonOrth" in q else "",
        "maxSkewness": round(q["maxSkewness"], 2) if "maxSkewness" in q else "",
        "nBadTetDecomp": q.get("nBadTetDecomp", ""),
        "volErr": f"{r['volErr']:+.3%}" if r.get("volErr") is not None else "",
        "peakMemMB": round(r["peakMemMB"]) if r.get("peakMemMB") else "",
        "input": input_summary(r.get("input")),
        "notes": "; ".join(r.get("notes", [])),
    }


def compare(results, baseline):
    """Lines describing cases that got worse (or better) than the baseline."""
    old = {r["name"]: r for r in baseline.get("results", [])}
    out = []
    for r in results:
        o = old.get(r["name"])
        if not o or r["status"] == "SKIP" or o["status"] == "SKIP":
            continue
        a, b = RANK[o["status"]], RANK[r["status"]]
        if b > a:
            out.append(f"WORSE  {r['name']}: {o['status']} -> {r['status']}  ({'; '.join(r.get('notes', []))})")
        elif b < a:
            out.append(f"better {r['name']}: {o['status']} -> {r['status']}")
        qa, qb = o.get("quality") or {}, r.get("quality") or {}
        if qa.get("nCells") and qb.get("nCells") and qa["nCells"] != qb["nCells"]:
            out.append(f"cells  {r['name']}: {qa['nCells']} -> {qb['nCells']}")
        if o.get("time") and r.get("time") and o["time"] > 2 and r["time"] > 1.3 * o["time"]:
            out.append(f"slower {r['name']}: {o['time']:.1f}s -> {r['time']:.1f}s")
    return out


def write_reports(outdir, suite, results, args, cmp_lines):
    payload = {"suite": suite["name"], "date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "python": sys.version.split()[0], "args": vars(args), "results": results}
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=float)
    rows = [row(r) for r in results]
    with open(os.path.join(outdir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CSV_COLS)
        w.writeheader(); w.writerows(rows)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    unexpected = [r for r in results if r.get("verdict") == "UNEXPECTED"]
    md = [f"# {suite['name']} - {payload['date']}", "",
          "  ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: RANK[kv[0]])) +
          f"  |  unexpected {len(unexpected)}", ""]
    if cmp_lines:
        md += ["## vs baseline", ""] + [f"- {s}" for s in cmp_lines] + [""]
    md += ["| case | status | verdict | cells | time [s] | max non-orth | max skew | vol err | input | notes |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['name']} | {x['status']} | {x['verdict']} | {x['cells']} | {x['time']} | "
                  f"{x['maxNonOrth']} | {x['maxSkewness']} | {x['volErr']} | {x['input']} | "
                  f"{x['notes'].replace('|', '/')} |")
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return payload, counts, unexpected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", nargs="?")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--only", action="append", help="fnmatch pattern(s) on case names")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--vol-tol", type=float, default=0.01, help="relative volume mismatch flagged as WARN")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a meshDict key for every case")
    ap.add_argument("--cells-across", type=float, help="override cellsAcross for every case")
    ap.add_argument("--keep", action="store_true", help="keep the polyMesh of PASS cases")
    ap.add_argument("--no-images", action="store_true", help="skip overview.png / cut.png")
    ap.add_argument("--strict", action="store_true",
                    help="also flag negative face-triangle tets (stricter than checkMesh)")
    ap.add_argument("--out", help="output directory (default results/<suite>-<time>)")
    ap.add_argument("--baseline", help="results.json of an earlier run to compare against")
    ap.add_argument("--save-baseline", action="store_true", help="also write baselines/<suite>.json")
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if not args.suite:
        ap.error("suite file required")
    suite, cases = load_suite(args.suite)
    if args.only:
        cases = [c for c in cases if any(fnmatch.fnmatch(c["name"], p) for p in args.only)]
    if args.cells_across:
        for c in cases:
            c["cellsAcross"] = args.cells_across
    overrides = {}
    for s in args.set:
        k, v = s.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    outdir = os.path.abspath(args.out or os.path.join(
        HERE, "results", f"{suite['name']}-{time.strftime('%Y%m%d-%H%M%S')}"))
    os.makedirs(outdir, exist_ok=True)
    print(f"suite {suite['name']}: {len(cases)} cases -> {outdir}")

    def one(c):
        try:
            r = run_case(c, outdir, args, overrides)
        except Exception as e:            # noqa: BLE001 - one broken case never stops the suite
            msg = f"benchmark driver: {type(e).__name__}: {e}"
            r = {"name": c["name"], "stl": _relpath(c["stl"]), "expect": c.get("expect"),
                 "status": "ERROR", "error": msg, "notes": [msg],
                 "verdict": verdict(c.get("expect"), "ERROR")}
        q = r.get("quality") or {}
        print(f"  {r['status']:<8} {r.get('verdict', '-'):<10} {r['name']:<28} "
              f"{q.get('nCells', ''):>8} cells {r.get('time', 0):7.1f}s  {'; '.join(r.get('notes', []))[:110]}",
              flush=True)
        return r
    if args.jobs > 1:
        with ThreadPoolExecutor(args.jobs) as ex:
            results = list(ex.map(one, cases))
    else:
        results = [one(c) for c in cases]
    cmp_lines = []
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            cmp_lines = compare(results, json.load(f))
    payload, counts, unexpected = write_reports(outdir, suite, results, args, cmp_lines)
    if args.save_baseline:
        bdir = os.path.join(HERE, "baselines")
        os.makedirs(bdir, exist_ok=True)
        slim = dict(payload, args={k: v for k, v in vars(args).items() if k not in ("out", "baseline", "suite")},
                    results=[{k: v for k, v in r.items() if k != "traceback"} for r in results])
        with open(os.path.join(bdir, suite["name"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=1, default=float)
    print("  ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: RANK[kv[0]]))
          + f"  |  unexpected {len(unexpected)}")
    for s in cmp_lines:
        print("  " + s)
    print(f"summary: {os.path.join(outdir, 'summary.md')}")
    if args.baseline:                     # known failures are in the baseline: fail only on regressions
        return 1 if any(s.startswith("WORSE") for s in cmp_lines) else 0
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
