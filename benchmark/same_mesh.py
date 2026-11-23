"""Do two versions of cutmesh write the same meshes?  Meshes the same inputs with the code
of two checkouts on this machine and compares the polyMesh files byte by byte.

    python benchmark/same_mesh.py BASE_TREE HEAD_TREE [--out DIR] [--only NAME ...]

BASE_TREE / HEAD_TREE are checkouts of the repository (each with its Rust extension built,
if any).  Meshed: the examples/stirred_tank meshDicts listed in TANK_CASES (HEAD_TREE's
inputs for both), and the benchmark showcase suite (each tree with its own run_bench.py and
the STLs of its own make_synthetic.py).  Exit code 1 if any mesh differs or fails.

Mesh hashes depend on the machine (see mesh_hash.py), so this builds both meshes here
instead of comparing against stored hashes: that is what the CI job does for a pull
request (the base branch and the pull request on the same runner).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mesh_hash import mesh_hash  # noqa: E402

# the stirred-tank examples (rotating zone, AMI interface, cylindrical base, boundary
# layers, a coarse rotor, an STL rotor, a 1 mm rotor-stator gap)
TANK_CASES = ["meshDict.json", "meshDict_ami.json", "meshDict_cylindrical.json", "meshDict_layers.json",
              "meshDict_coarseRotor.json", "meshDict_stlRotor.json", "meshDict_stator_1mm.json"]


def run(cmd, cwd, env=None, log=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            f.write(p.stdout + p.stderr)
    return p.returncode, time.time() - t0


def tank(tree, inputs, out, names):
    """mesh the tank examples of the inputs tree with the code of tree -> {name: case dir}"""
    env = dict(os.environ, PYTHONPATH=os.path.abspath(tree))
    src = os.path.join(inputs, "examples", "stirred_tank")
    res = {}
    for md in names:
        name = "tank_" + md[len("meshDict"):-len(".json")].strip("_") if md != "meshDict.json" else "tank"
        case = os.path.join(out, name)
        rc, t = run([sys.executable, "-m", "cutmesh", md, "-noCheck", "-case", case], src, env, case + ".log")
        print(f"  {name:28s} {'ok' if rc == 0 else 'FAILED'} {t:6.1f} s", flush=True)
        res[name] = case if rc == 0 else None
    return res


def showcase(tree, out, only):
    """the showcase suite meshed by tree's own run_bench, on the STLs made by tree's own
    make_synthetic.py (run_bench looks for them in its own tree) -> {name: case dir}"""
    rc, _ = run([sys.executable, os.path.join(tree, "benchmark", "make_synthetic.py")], tree)
    if rc:
        sys.exit(f"make_synthetic.py failed in {tree}")
    suite = os.path.join(tree, "benchmark", "suites", "showcase.json")
    with open(suite, encoding="utf-8") as f:
        names = [c["name"] for c in json.load(f)["cases"]]
    names = [n for n in names if not only or n in only]
    if not names:
        return {}
    cmd = [sys.executable, os.path.join(tree, "benchmark", "run_bench.py"), suite,
           "--keep", "--no-images", "--out", out]
    for n in names:
        cmd += ["--only", n]
    rc, t = run(cmd, tree, log=out + ".log")
    print(f"  showcase ({len(names)} cases) exit {rc} {t:6.1f} s", flush=True)
    return {n: os.path.join(out, "cases", n, "case") for n in names}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--out", default="same_mesh_out")
    ap.add_argument("--only", action="append", help="case names (tank, tank_ami, pipe_elbow, ...)")
    a = ap.parse_args(argv)
    # absolute: the subprocesses run in other directories (the trees, the examples)
    a.base, a.head, a.out = (os.path.abspath(x) for x in (a.base, a.head, a.out))
    os.makedirs(a.out, exist_ok=True)
    tank_names = [m for m in TANK_CASES
                  if not a.only or ("tank_" + m[len("meshDict"):-len(".json")].strip("_") if m != "meshDict.json" else "tank")
                  in a.only]
    got = {}
    for label, tree in (("base", a.base), ("head", a.head)):
        print(f"{label}: {tree}", flush=True)
        out = os.path.join(a.out, label)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        got[label] = tank(tree, a.head, out, tank_names)
        got[label].update(showcase(tree, os.path.join(out, "showcase"), a.only))
    bad = 0
    print(f"\n{'case':28s} result")
    for name in got["head"]:
        b, h = got["base"].get(name), got["head"].get(name)
        try:
            same = b is not None and h is not None and mesh_hash(b)["digest"] == mesh_hash(h)["digest"]
            msg = "same mesh" if same else "MESH DIFFERS"
        except FileNotFoundError:
            same, msg = False, "FAILED (no polyMesh)"
        bad += not same
        print(f"{name:28s} {msg}")
    print(f"\n{len(got['head']) - bad} of {len(got['head'])} meshes the same")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
