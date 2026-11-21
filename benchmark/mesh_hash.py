"""Fingerprint of a written mesh, to check that a change leaves the mesh byte-identical.

    python benchmark/mesh_hash.py CASE                 # print the hash of CASE/constant/polyMesh
    python benchmark/mesh_hash.py CASE_A CASE_B        # same mesh? (exit code 1 if not)
    python benchmark/mesh_hash.py CASE --save ref.json # store the hashes
    python benchmark/mesh_hash.py CASE --check ref.json

CASE is a case directory (or its polyMesh directory).  Every file in polyMesh is hashed as
written (cutmesh writes no dates or versions into the headers, so equal meshes give equal
files).  The hash depends on the platform: NumPy and the Rust kernels add in a fixed order,
but another CPU, NumPy build or OS can round differently, so compare hashes made on the
same machine.
"""
import argparse
import hashlib
import json
import os
import sys


def polymesh_dir(path):
    for d in (os.path.join(path, "constant", "polyMesh"), path):
        if os.path.isfile(os.path.join(d, "owner")):
            return d
    raise FileNotFoundError(f"no polyMesh under {path}")


def mesh_hash(path):
    """{"files": {name: sha256}, "digest": sha256 of all of them} for a case or polyMesh dir."""
    d = polymesh_dir(path)
    files = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        files[name] = h.hexdigest()
    total = hashlib.sha256("".join(f"{k} {v}\n" for k, v in files.items()).encode())
    return {"files": files, "digest": total.hexdigest()}


def diff(a, b):
    """Names of the polyMesh files that differ (or exist in only one of the two)."""
    fa, fb = a["files"], b["files"]
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("other", nargs="?", help="second case to compare with")
    ap.add_argument("--save", metavar="JSON", help="write the hashes to this file")
    ap.add_argument("--check", metavar="JSON", help="compare with hashes saved by --save")
    args = ap.parse_args(argv)
    a = mesh_hash(args.case)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(a, f, indent=1)
    ref = None
    if args.other:
        ref = mesh_hash(args.other)
    elif args.check:
        with open(args.check, encoding="utf-8") as f:
            ref = json.load(f)
    if ref is None:
        print(a["digest"])
        return 0
    bad = diff(a, ref)
    print("same mesh" if not bad else "mesh differs: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
