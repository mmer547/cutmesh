"""Do the public examples pass OpenFOAM's checkMesh?  Meshes the stirred-tank examples and the
showcase suite with this checkout and runs checkMesh of OpenFOAM on each mesh.

    python benchmark/check_openfoam.py [--out DIR] [--only NAME ...]
                                       [--docker IMAGE | --native] [--known FILE]

--docker IMAGE (default opencfd/openfoam-default:2512): checkMesh in that image (docker run,
the case folder mounted); --native: the checkMesh on PATH (an OpenFOAM environment sourced).
OpenFOAM reads the polyMesh as cutmesh writes it (binary by default), so this also checks the
binary format.

A case passes when checkMesh ends with "Mesh OK.".  --known (default
benchmark/checkmesh_known.json) lists the cases known to fail and why; a case that fails and
is not listed (or whose meshing fails) is a regression: exit code 1.  A listed case that
passes is reported, to be removed from the list.  Written: DIR/summary.md, and per case
DIR/<name>.checkMesh.log.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from same_mesh import TANK_CASES  # noqa: E402

DEFAULT_IMAGE = "opencfd/openfoam-default:2512"
HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {name};
}}
"""
# checkMesh reads the case's controlDict (and the schemes for some checks): the least it needs
SYSTEM = {
    "controlDict": "application checkMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
                   "endTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n",
    "fvSchemes": "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n"
                 "divSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\n"
                 "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n",
    "fvSolution": "",
}


def write_system(case):
    """the system/ files checkMesh needs, where the case has none"""
    d = os.path.join(case, "system")
    os.makedirs(d, exist_ok=True)
    for name, body in SYSTEM.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write(HEADER.format(name=name) + body)


def mesh_cases(out, only):
    """mesh the tank examples and the showcase suite with this checkout -> {name: case dir or None}"""
    env = dict(os.environ, PYTHONPATH=ROOT)
    src = os.path.join(ROOT, "examples", "stirred_tank")
    cases = {}
    for md in TANK_CASES:
        name = "tank_" + md[len("meshDict"):-len(".json")].strip("_") if md != "meshDict.json" else "tank"
        if only and name not in only:
            continue
        case = os.path.join(out, name)
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "cutmesh", md, "-noCheck", "-case", case], cwd=src, env=env,
                           capture_output=True, text=True)
        with open(case + ".cutmesh.log", "w", encoding="utf-8") as f:
            f.write(p.stdout + p.stderr)
        print(f"  mesh {name:26s} {'ok' if p.returncode == 0 else 'FAILED'} {time.time() - t0:6.1f} s", flush=True)
        cases[name] = case if p.returncode == 0 else None
    suite = os.path.join(HERE, "suites", "showcase.json")
    with open(suite, encoding="utf-8") as f:
        names = [c["name"] for c in json.load(f)["cases"]]
    names = [n for n in names if not only or n in only]
    if names:
        subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic.py")], cwd=ROOT, check=True,
                       capture_output=True)
        sc = os.path.join(out, "showcase")
        cmd = [sys.executable, os.path.join(HERE, "run_bench.py"), suite, "--keep", "--no-images",
               "--no-metrics", "--out", sc]
        for n in names:
            cmd += ["--only", n]
        t0 = time.time()
        subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(f"  mesh showcase ({len(names)} cases) {time.time() - t0:6.1f} s", flush=True)
        for n in names:
            case = os.path.join(sc, "cases", n, "case")
            ok = os.path.isfile(os.path.join(case, "constant", "polyMesh", "owner"))
            cases[n] = case if ok else None
    return cases


def check_mesh(case, docker, image):
    """run checkMesh on the case -> (output, return code)"""
    case = os.path.abspath(case)
    if docker:
        uid = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else None
        cmd = ["docker", "run", "--rm", "-v", f"{case}:/case", "-w", "/case", "--entrypoint", "bash"]
        if uid:
            cmd += ["-u", uid, "-e", "HOME=/tmp"]
        # the image's OpenFOAM environment (its bashrc may trip `set -u`-style checks: not used)
        cmd += [image, "-c", "for f in /usr/lib/openfoam/openfoam*/etc/bashrc; do . \"$f\"; done; "
                             "checkMesh -case /case"]
    else:
        cmd = ["checkMesh", "-case", case]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr, p.returncode


def judge(text, rc):
    """(status, failed checks): "ok" (Mesh OK.), "failed" (Failed n mesh checks), "error"
    (no verdict: OpenFOAM could not read or check the mesh)"""
    failed = [s.strip() for s in text.splitlines() if s.strip().startswith("***")]
    if "FOAM FATAL" in text or "FOAM FATAL IO ERROR" in text:
        m = re.search(r"FOAM FATAL[^\n]*\n(?:[^\n]*\n){0,6}", text)
        return "error", [m.group(0).strip() if m else "FOAM FATAL ERROR"]
    if re.search(r"^\s*Mesh OK\.", text, re.M):
        return "ok", failed
    m = re.search(r"Failed (\d+) mesh checks", text)
    if m:
        return "failed", failed
    return "error", [f"no verdict from checkMesh (exit code {rc})"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "checkmesh"))
    ap.add_argument("--only", action="append", help="case names (default: all)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--docker", default=DEFAULT_IMAGE, metavar="IMAGE", help=f"OpenFOAM image (default {DEFAULT_IMAGE})")
    g.add_argument("--native", action="store_true", help="the checkMesh on PATH")
    ap.add_argument("--known", default=os.path.join(HERE, "checkmesh_known.json"),
                    help="cases known to fail checkMesh ({name: reason})")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    known = {}
    if args.known and os.path.exists(args.known):
        with open(args.known, encoding="utf-8") as f:
            known = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    print(f"meshing -> {out}")
    cases = mesh_cases(out, args.only)
    print(f"checkMesh ({'native' if args.native else args.docker})")
    rows, bad = [], 0
    for name, case in cases.items():
        if case is None:
            status, failed = "mesh failed", []
        else:
            write_system(case)
            t0 = time.time()
            text, rc = check_mesh(case, not args.native, args.docker)
            with open(os.path.join(out, name + ".checkMesh.log"), "w", encoding="utf-8") as f:
                f.write(text)
            status, failed = judge(text, rc)
            print(f"  {name:26s} {status:7s} {time.time() - t0:5.1f} s  {'; '.join(failed)[:150]}", flush=True)
        expected = name in known
        if status == "ok":
            note = "listed as failing: remove it from the known list" if expected else ""
        else:
            note = f"known: {known[name]}" if expected and status == "failed" else "REGRESSION"
            bad += note == "REGRESSION"
        rows.append((name, status, "; ".join(failed), note))
    md = ["# checkMesh of the public examples", "",
          f"OpenFOAM: {'native' if args.native else args.docker}", "",
          "| case | checkMesh | failed checks | note |", "|---|---|---|---|"]
    md += [f"| {n} | {s} | {f.replace('|', '/')} | {note} |" for n, s, f, note in rows]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    nok = sum(r[1] == "ok" for r in rows)
    print(f"{nok} of {len(rows)} meshes pass checkMesh; {bad} regression(s)")
    for n, s, f_, note in rows:
        if note:
            print(f"  {n}: {s} ({note})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
