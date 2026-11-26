"""Run cutmesh over a suite of STL files and report which ones mesh cleanly.

    python benchmark/run_bench.py benchmark/suites/synthetic.json
    python benchmark/run_bench.py benchmark/suites/synthetic.json -j 4 --only "thin_*"
    python benchmark/run_bench.py SUITE --baseline benchmark/baselines/synthetic.json
    python benchmark/run_bench.py SUITE --save-baseline       # -> benchmark/baselines/<suite>.json
    python benchmark/run_bench.py SUITE --set addLayers='{"nLayers": 3}'
    python benchmark/run_bench.py SUITE --baseline B.json --same-mesh   # a changed mesh fails

Each case runs in its own process (a crash or a hang only loses that case) with a timeout.
Results go to benchmark/results/<suite>-<time>/: results.json, results.csv, summary.md and,
per case, cases/<name>/{meshDict.json, log.txt, overview.png, cut.png, case/constant/polyMesh}
(pictures: render_mesh.py; --no-images skips them).  The polyMesh of a PASS case is deleted
unless --keep is given.  Every written mesh is fingerprinted (mesh_hash.py), so --baseline
reports whether the mesh changed; with --same-mesh a changed mesh is a regression (exit code 1).
Hashes are only compared when the baseline was made on the same platform (CPU, OS, NumPy).

Suite file (paths relative to the suite file):
    {"name": "synthetic",
     "defaults": {"cellsAcross": 24, "meshDict": {"surfaceRefinement": {"default": 2}}},
     "cases": [{"name": "sphere", "stl": "../data/synthetic/sphere.stl", "expect": "pass"},
               {"name": "cube_aligned", "stl": "...", "meshDict": {"baseCellSize": 0.0625}}]}

A case may list "probes": [{"name", "from": [x, y, z], "to": [x, y, z], "minCells": n,
"zone": name, "minZoneCells": n, "maxZoneCells": n}]: line segments (across a narrow gap)
whose cells are counted (cell with the nearest centre at 400 points); fewer cells than
minCells, or zone cells outside the limits, make the case WARN (a closed gap does not show
in the volume or the leak check).

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
import platform
import re
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
def probe_cells(R, probes, n=400):
    """Cells along line segments (e.g. across a narrow gap): each of n points on the segment
    goes to the cell with the nearest centre (mean of its face centres), so a gap that
    closed shows up as too few cells.  Returns [{name, cells, zoneCells}]."""
    import numpy as np
    from scipy.spatial import cKDTree
    P = R["points"]; sizes, flat = R["faces"]; own, nei = R["owner"], R["neighbour"]
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    fc = np.add.reduceat(P[flat], ptr[:-1]) / np.asarray(sizes)[:, None]
    nC = int(max(own.max(), nei.max() if len(nei) else 0)) + 1
    cc = np.zeros((nC, 3)); cnt = np.zeros(nC)
    np.add.at(cc, own, fc); np.add.at(cnt, own, 1)
    np.add.at(cc, nei, fc[:len(nei)]); np.add.at(cnt, nei, 1)
    tree = cKDTree(cc / cnt[:, None])
    out = []
    for pr in probes:
        a, b = np.asarray(pr["from"], float), np.asarray(pr["to"], float)
        t = (np.arange(n) + 0.5) / n
        _, c = tree.query(a + t[:, None] * (b - a))
        cells = np.unique(c)
        r = {"name": pr["name"], "cells": int(len(cells))}
        z = pr.get("zone")
        if z is not None and z in (R.get("cell_zones") or {}):
            r["zoneCells"] = int(np.isin(cells, R["cell_zones"][z]).sum())
        out.append(r)
    return out



def measure(res, R, lines, meshdict_path):
    """the measures of the quality gate that the quality check does not give: the smallest
    cell, the layer coverage (with addLayers) and the feature-edge deviation.  A measure
    that does not apply is stored as None (no layers in the case, no feature edges in the
    domain); a measure that could not be taken is left out, and the gate fails on it."""
    import mesh_metrics as mm
    h = mm.smallest_cell(lines)
    if h:
        res["hmin"] = h
    res["layers"] = mm.layer_coverage(lines)
    if h:
        res["featureEdges"] = mm.feature_edge_deviation(R["points"], R["faces"], len(R["neighbour"]),
                                                        mm.domain_feature_segments(meshdict_path), h)


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
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE       # 64-bit handle: the default int truncates it
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
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
        res["parts"] = [[step_name(st), part, round(dt, 3)] for st, part, dt in getattr(R["timer"], "parts", [])]
        try:
            from mesh_hash import mesh_hash
            res["meshHash"] = mesh_hash(os.path.join(wd, "case"))
        except Exception as e:            # noqa: BLE001 - a missing hash only skips the check
            res["meshHashError"] = f"{type(e).__name__}: {e}"
        if job.get("probes"):
            res["probes"] = probe_cells(R, job["probes"])
        if job.get("metrics", True):
            t1 = time.time()
            try:                          # measures for --gate; never an ERROR of the case
                measure(res, R, lines, os.path.join(wd, "meshDict.json"))
            except Exception as e:        # noqa: BLE001
                res["metricsError"] = f"{type(e).__name__}: {e}"
            res["metricsTime"] = round(time.time() - t1, 2)
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
    res["time"] = round(time.time() - t0 - res.get("renderTime", 0) - res.get("metricsTime", 0), 3)
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
        for z in md.get("rotatingZones", []):     # zone STLs: relative to the suite file
            if isinstance(z, dict) and isinstance(z.get("stl"), str):
                z["stl"] = os.path.normpath(os.path.join(base, z["stl"]))
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
    for pr, spec in zip(res.get("probes", []), res.get("_probeSpecs", [])):
        if pr["cells"] < spec.get("minCells", 1):
            flags.append(f"probe {pr['name']}: {pr['cells']} cells < {spec.get('minCells', 1)}")
        if "minZoneCells" in spec and pr.get("zoneCells", 0) < spec["minZoneCells"]:
            flags.append(f"probe {pr['name']}: {pr.get('zoneCells', 0)} cells in zone {spec.get('zone')}"
                         f" < {spec['minZoneCells']}")
        if "maxZoneCells" in spec and pr.get("zoneCells", 0) > spec["maxZoneCells"]:
            flags.append(f"probe {pr['name']}: {pr.get('zoneCells', 0)} cells in zone {spec.get('zone')}"
                         f" > {spec['maxZoneCells']}")
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
    job = {"name": c["name"], "workdir": wd, "images": not args.no_images, "probes": c.get("probes"),
           "metrics": not getattr(args, "no_metrics", False)}
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
    res["_probeSpecs"] = c.get("probes") or []
    status, notes = classify(res, inp, args.vol_tol, args.strict)
    res.pop("_probeSpecs")
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


CSV_COLS = ["name", "status", "verdict", "expect", "time", "cells", "maxNonOrth", "nNonOrth70",
            "maxSkewness", "nSkew4", "nBadTetDecomp", "volErr", "edgeOver02", "edgeP95", "layersFull",
            "peakMemMB", "input", "notes"]


def row(r):
    q = r.get("quality") or {}
    return {
        "name": r["name"], "status": r["status"], "verdict": r.get("verdict", "-"),
        "expect": r.get("expect") or "", "time": r.get("time", ""),
        "cells": q.get("nCells", ""),
        "maxNonOrth": round(q["maxNonOrth"], 1) if "maxNonOrth" in q else "",
        "maxSkewness": round(q["maxSkewness"], 2) if "maxSkewness" in q else "",
        "nNonOrth70": q.get("nNonOrth70", ""), "nSkew4": q.get("nSkew4", ""),
        "nBadTetDecomp": q.get("nBadTetDecomp", ""),
        "volErr": f"{r['volErr']:+.3%}" if r.get("volErr") is not None else "",
        "edgeOver02": f"{r['featureEdges']['over02']:.2%}" if r.get("featureEdges") else "",
        "edgeP95": f"{r['featureEdges']['p95']:.3f}" if r.get("featureEdges") else "",
        "layersFull": f"{r['layers']['full']:.1f}%" if r.get("layers") else "",
        "peakMemMB": round(r["peakMemMB"]) if r.get("peakMemMB") else "",
        "input": input_summary(r.get("input")),
        "notes": "; ".join(r.get("notes", [])),
    }


# quality counts compared with the baseline: more of these is a regression (WORSE) when the
# count grows by more than COUNT_TOL and by at least COUNT_MIN faces / cells
COUNTS = {"nSkew4": "faces skewness > 4", "nNonOrth70": "faces non-orth > 70",
          "nNegVol": "non-positive volumes", "nBadPyramids": "bad pyramids"}
COUNT_TOL, COUNT_MIN = 0.10, 5
# time: slower than SLOW_LONG (cases over LONG_CASE s) or SLOW_SHORT (cases over 2 s)
LONG_CASE, SLOW_LONG, SLOW_SHORT = 30.0, 1.05, 1.3
STEP_MIN, STEP_SLOW = 1.0, 1.2          # steps of at least STEP_MIN s, slower by STEP_SLOW


# quality gate (--gate, roadmap v0.7.0): a change that may change the mesh is accepted when
# every case meets all of these against the baseline
GATE_CELLS_PCT = 3                      # cell count within +-3 % (inclusive, compared in integers)
GATE_LAYERS = 1.0                       # full layer thickness: not down by 1 point or more
GATE_EDGE_OVER = 0.005                  # feature-edge samples farther than 0.2 hmin: +0.5 points at most
GATE_EDGE_P95 = 0.02                    # their 95th percentile: +0.02 hmin at most


def gate(r, o):
    """reasons why result r fails the quality gate against the baseline result o ([] = passes):
    no faces with non-orthogonality > 70 (if the baseline has some: no more of them), no more
    faces with skewness > 4, no non-positive volumes, cells within 3 %, the share of wall points
    with full layer thickness not down by 1 point or more, the feature edges not farther from
    the mesh edges (share over 0.2 hmin and 95th percentile)"""
    qa, qb = o.get("quality") or {}, r.get("quality") or {}
    if not qb:
        # no mesh in either (an input that is rejected on purpose): the status decides
        return ["no quality report"] if qa else []
    out = []
    x, y = int(qa.get("nNonOrth70", 0)), int(qb.get("nNonOrth70", 0))
    if y > 0 and y > x:
        out.append(f"faces non-orth > 70 {x} -> {y}")
    x, y = int(qa.get("nSkew4", 0)), int(qb.get("nSkew4", 0))
    if y > x:
        out.append(f"faces skewness > 4 {x} -> {y}")
    if int(qb.get("nNegVol", 0)) > 0:
        out.append(f"{qb['nNegVol']} non-positive volumes")
    if qa.get("nCells") and qb.get("nCells"):
        na, nb = int(qa["nCells"]), int(qb["nCells"])
        dc = nb / na - 1.0
        if abs(nb - na) * 100 > GATE_CELLS_PCT * na:
            out.append(f"cells {qa['nCells']} -> {qb['nCells']} ({dc:+.1%})")
    # the layer coverage and the feature edges must have been measured in both runs (None:
    # measured, but not applicable to the case); a missing one is not a pass
    missing = []
    for res, which in ((o, "the baseline"), (r, "this run")):
        if res.get("metricsError"):
            missing.append(f"no measures in {which} ({res['metricsError']})")
        else:
            missing += [f"{what} not measured in {which}"
                        for key, what in (("layers", "layer coverage"), ("featureEdges", "feature-edge deviation"))
                        if key not in res]
    if missing:
        return out + missing
    la, lb = o.get("layers"), r.get("layers")
    if la:
        if not lb:
            out.append("no layer coverage")
        elif la["full"] - lb["full"] >= GATE_LAYERS:
            out.append(f"full layer thickness {la['full']:.1f}% -> {lb['full']:.1f}%")
    ea, eb = o.get("featureEdges"), r.get("featureEdges")
    if ea:
        if not eb:
            out.append("no feature-edge deviation")
        else:
            if eb["over02"] > ea["over02"] + GATE_EDGE_OVER:
                out.append(f"feature edges over 0.2 hmin {ea['over02']:.1%} -> {eb['over02']:.1%}")
            if eb["p95"] > ea["p95"] + GATE_EDGE_P95:
                out.append(f"feature edges p95 {ea['p95']:.3f} -> {eb['p95']:.3f} hmin")
    return out


def step_name(name):
    """Timer step without its counts: "octree (147272 leaves)" -> "octree"."""
    return re.sub(r"\s*\([^()]*\)", "", name).strip()


def steps_by_name(r):
    out = {}
    for k, v in (r.get("steps") or {}).items():
        n = step_name(k)
        out[n] = out.get(n, 0.0) + v
    return out


def platform_id():
    """CPU, OS and NumPy: mesh hashes are only comparable between runs with the same one."""
    try:
        import numpy
        npv = numpy.__version__
    except ImportError:
        npv = "?"
    return (f"{platform.system()} {platform.machine()} {platform.processor()} | "
            f"Python {platform.python_version()} | NumPy {npv}")


def compare(results, baseline, same_mesh=False, use_gate=False):
    """Lines describing cases that got worse (or better) than the baseline.
    Lines starting with WORSE are regressions (exit code 1); the others are information.
    same_mesh: a mesh whose hash differs from the baseline's is a regression too.
    use_gate: apply the quality gate (gate()) to every case: a case that fails it, that has
    no result to judge, or that the baseline judged and this run skipped, is a regression
    (skipped in both: not judged)."""
    from mesh_hash import diff
    old = {r["name"]: r for r in baseline.get("results", [])}
    out = []
    here, there = platform_id(), baseline.get("platform")
    hashes_ok = there == here
    if not hashes_ok:
        out.append(f"note   mesh hashes not compared: baseline made on {there or 'an unknown platform'}, "
                   f"this run on {here}")
    for r in results:
        o = old.get(r["name"])
        if use_gate and r["status"] != "SKIP" and (not o or o["status"] == "SKIP"):
            out.append(f"WORSE  {r['name']}: gate: not in the baseline")
        if use_gate and r["status"] == "SKIP" and o and o["status"] != "SKIP":
            # judged in the baseline, never checked here (an STL missing in this environment)
            out.append(f"WORSE  {r['name']}: gate: skipped in this run (baseline {o['status']})")
        if not o or r["status"] == "SKIP" or o["status"] == "SKIP":
            continue
        if use_gate:
            g = gate(r, o)
            out.append(f"WORSE  {r['name']}: gate: {'; '.join(g)}" if g else f"gate   {r['name']}: ok")
        a, b = RANK[o["status"]], RANK[r["status"]]
        if b > a:
            out.append(f"WORSE  {r['name']}: {o['status']} -> {r['status']}  ({'; '.join(r.get('notes', []))})")
        elif b < a:
            out.append(f"better {r['name']}: {o['status']} -> {r['status']}")
        qa, qb = o.get("quality") or {}, r.get("quality") or {}
        ha, hb = o.get("meshHash"), r.get("meshHash")
        if hashes_ok and ha and hb:
            bad = diff(ha, hb)
            if bad:
                out.append(f"{'WORSE ' if same_mesh else 'mesh  '} {r['name']}: mesh changed ({', '.join(bad)})")
            else:
                out.append(f"same   {r['name']}: same mesh")
        elif hashes_ok and same_mesh and r["status"] != "SKIP" and (o.get("quality") or r.get("quality")):
            out.append(f"WORSE  {r['name']}: no mesh hash to compare "
                       f"({'baseline' if not ha else 'this run'})")
        if qa.get("nCells") and qb.get("nCells") and qa["nCells"] != qb["nCells"]:
            out.append(f"cells  {r['name']}: {qa['nCells']} -> {qb['nCells']}")
        for k, what in COUNTS.items():
            if k not in qa or k not in qb or qa[k] == qb[k]:
                continue
            x, y = int(qa[k]), int(qb[k])
            if y > x and y - x >= COUNT_MIN and y > x * (1 + COUNT_TOL):
                out.append(f"WORSE  {r['name']}: {what} {x} -> {y}")
            else:
                out.append(f"{'more' if y > x else 'fewer'}   {r['name']}: {what} {x} -> {y}")
        if qa.get("minVol") and qb.get("minVol") and qb["minVol"] < 0.1 * qa["minVol"]:
            out.append(f"minVol {r['name']}: {qa['minVol']:.3g} -> {qb['minVol']:.3g}")
        to, tn = o.get("time"), r.get("time")
        if to and tn and to > 2:
            lim = SLOW_LONG if to > LONG_CASE else SLOW_SHORT
            if tn > lim * to:
                out.append(f"slower {r['name']}: {to:.1f}s -> {tn:.1f}s ({tn / to - 1:+.0%})")
            elif tn < to / lim:
                out.append(f"faster {r['name']}: {to:.1f}s -> {tn:.1f}s ({tn / to - 1:+.0%})")
        sa, sb = steps_by_name(o), steps_by_name(r)
        for k, tb in sb.items():
            ta = sa.get(k)
            if ta is not None and max(ta, tb) >= STEP_MIN and tb > STEP_SLOW * ta:
                out.append(f"slower {r['name']} / {k}: {ta:.1f}s -> {tb:.1f}s")
    return out


def steps_table(results):
    """Markdown table: seconds per step (counts dropped from the names) for each case."""
    cols = []
    for r in results:
        for k in steps_by_name(r):
            if k not in cols:
                cols.append(k)
    if not cols:
        return []
    md = ["## time per step [s]", "",
          "| case | " + " | ".join(cols) + " | total |", "|---" * (len(cols) + 2) + "|"]
    for r in results:
        st = steps_by_name(r)
        if st:
            md.append(f"| {r['name']} | " + " | ".join(f"{st[k]:.2f}" if k in st else "" for k in cols)
                      + f" | {r.get('time', '')} |")
    return md + [""]


def parts_table(results):
    """Markdown table: seconds per part of a step (Timer.mark), one row per case, step and part."""
    rows = [(r["name"], st, part, dt) for r in results for st, part, dt in (r.get("parts") or [])]
    if not rows:
        return []
    md = ["## time per part of a step [s]", "", "| case | step | part | time |", "|---|---|---|---|"]
    md += [f"| {c} | {st} | {part} | {dt:.2f} |" for c, st, part, dt in rows]
    return md + [""]


def _cutmesh_version():
    try:
        from cutmesh import __version__
        return __version__
    except ImportError:
        return None


def write_reports(outdir, suite, results, args, cmp_lines):
    payload = {"suite": suite["name"], "date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "python": sys.version.split()[0], "platform": platform_id(), "args": vars(args),
               "cutmesh": _cutmesh_version(),
               "results": results}
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
    md += ["| case | status | verdict | cells | time [s] | max non-orth | non-orth > 70 | max skew | skew > 4 "
           "| vol err | edges > 0.2 hmin | edges p95 | full layers | input | notes |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['name']} | {x['status']} | {x['verdict']} | {x['cells']} | {x['time']} | "
                  f"{x['maxNonOrth']} | {x['nNonOrth70']} | {x['maxSkewness']} | {x['nSkew4']} | "
                  f"{x['volErr']} | {x['edgeOver02']} | {x['edgeP95']} | {x['layersFull']} | {x['input']} | "
                  f"{x['notes'].replace('|', '/')} |")
    md += [""] + steps_table(results) + parts_table(results)
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
    ap.add_argument("--same-mesh", action="store_true",
                    help="with --baseline: a mesh that differs from the baseline's is a regression")
    ap.add_argument("--gate", action="store_true",
                    help="with --baseline: apply the quality gate to every case (a change that may change the mesh)")
    ap.add_argument("--no-metrics", action="store_true",
                    help="skip the feature-edge and layer measures of the quality gate")
    args = ap.parse_args(argv)
    if args.gate and not args.baseline:
        ap.error("--gate needs --baseline")
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
            cmp_lines = compare(results, json.load(f), args.same_mesh, args.gate)
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
