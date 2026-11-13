"""Input diagnostics for an STL: is it the closed, consistently oriented surface cutmesh expects?

    python benchmark/stl_check.py model.stl [more.stl ...]

Uses the same repair and checks as cutmesh on load (geometry.repair_triangles): zero-area and
duplicated triangles (which cutmesh removes, and logs), then open edges, non-manifold edges and
edges whose two triangles disagree on orientation in what is left (which cutmesh warns about),
plus connected components, the Euler characteristic, the signed volume (negative = normals
point inward) and the bounding box.  Self-intersections are not checked (costly).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cutmesh.geometry import read_stl, repair_triangles  # noqa: E402


def check_tris(tris):
    """Same repair and checks as cutmesh does on load (geometry.repair_triangles), plus
    connected components and the Euler characteristic of what is left."""
    tris = np.asarray(tris, float)
    t2, _, rep = repair_triangles(tris, np.zeros(len(tris), int))
    bbmin, bbmax = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    diag = float(np.linalg.norm(bbmax - bbmin))
    q = np.round((t2.reshape(-1, 3) - bbmin) / max(diag, 1e-300) * 1e9).astype(np.int64)
    _, vid = np.unique(q, axis=0, return_inverse=True)
    G = vid.reshape(-1, 3)
    und = np.sort(np.stack([G.ravel(), np.roll(G, -1, 1).ravel()], 1), 1)
    key, inv = np.unique(und, axis=0, return_inverse=True)
    inv = inv.ravel()
    # components over shared edges (union-find on triangles)
    parent = np.arange(len(G))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    tri_of = np.repeat(np.arange(len(G)), 3)
    order = np.argsort(inv, kind="stable")
    s_inv, s_tri = inv[order], tri_of[order]
    for k in np.nonzero(s_inv[1:] == s_inv[:-1])[0]:
        ra, rb = find(s_tri[k]), find(s_tri[k + 1])
        if ra != rb:
            parent[ra] = rb
    ncomp = len({find(i) for i in range(len(G))})
    V, E = len(np.unique(G)), len(key)
    closed = rep["openEdges"] == 0 and rep["nonManifoldEdges"] == 0
    return dict(
        triangles=len(tris), vertices=int(V), edges=int(E), euler=int(V - E + len(G)),
        openEdges=rep["openEdges"], nonManifoldEdges=rep["nonManifoldEdges"],
        inconsistentEdges=rep["inconsistentEdges"],
        degenerateTris=rep["zeroArea"], duplicateTris=rep["duplicates"] + 2 * rep["membranes"],
        components=int(ncomp), signedVolume=rep["volume"], bbox=[bbmin.tolist(), bbmax.tolist()],
        diag=diag, closed=bool(closed), oriented=bool(closed and rep["inconsistentEdges"] == 0),
        clean=bool(closed and rep["inconsistentEdges"] == 0 and rep["zeroArea"] == 0
                   and rep["duplicates"] == 0 and rep["membranes"] == 0 and rep["volume"] > 0),
    )


def check_file(path):
    solids = read_stl(path)
    tris = np.concatenate([t for _, t in solids]) if solids else np.zeros((0, 3, 3))
    r = check_tris(tris)
    r["solids"] = [n for n, _ in solids]
    return r


def summary(r):
    flags = []
    if r["openEdges"]:
        flags.append(f"{r['openEdges']} open edges")
    if r["nonManifoldEdges"]:
        flags.append(f"{r['nonManifoldEdges']} non-manifold edges")
    if r["inconsistentEdges"]:
        flags.append(f"{r['inconsistentEdges']} inconsistently oriented edges")
    if r["degenerateTris"]:
        flags.append(f"{r['degenerateTris']} zero-area tris (removed by cutmesh)")
    if r["duplicateTris"]:
        flags.append(f"{r['duplicateTris']} duplicate tris (removed by cutmesh)")
    if r["signedVolume"] <= 0:
        flags.append("normals inward (volume <= 0)")
    return "clean" if not flags else ", ".join(flags)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    js = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    out = {}
    for p in argv:
        r = check_file(p)
        out[p] = r
        if not js:
            print(f"{os.path.basename(p)}: {r['triangles']} tris, {r['components']} components, "
                  f"euler {r['euler']}, volume {r['signedVolume']:.4g} -> {summary(r)}")
    if js:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
