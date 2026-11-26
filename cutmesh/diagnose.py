"""Checks of a generated mesh against the STL.

through_wall(): connections of the fluid through a solid wall (a bypass).  For every
cell, the line from its centre to each of its face centres is sampled; where it runs
through the solid, the STL normals near the entry and the exit are compared.  Opposite
normals mean the line crossed a wall (fluid on both sides joined by the mesh); a line
merely cutting a corner of the solid sees normals at about 90 degrees or less.

python -m cutmesh.diagnose meshDict.json [CASE]   (CASE default: the meshDict's case)
"""
import numpy as np

from . import quality
from .geometry import _core


def _face_geometry(points, faces):
    """quality.face_geometry of faces given as FlatFaces or (sizes, vertices), without
    building a FlatFaces (its per-vertex arrays) when the Rust extension is there"""
    if _core is not None and isinstance(faces, tuple):
        return _core.face_geometry(np.ascontiguousarray(points, float), np.ascontiguousarray(faces[0], np.int64),
                                   np.ascontiguousarray(faces[1], np.int64))
    ff = faces if isinstance(faces, quality.FlatFaces) else quality.FlatFaces(faces)
    return quality.face_geometry(points, ff)


def _cell_reach(C, Cf, owner, neighbour, use_rs=True):
    """per cell the longest line from its centre to its face centres (0 without faces)"""
    if use_rs and _core is not None and hasattr(_core, "cell_reach"):
        return _core.cell_reach(np.ascontiguousarray(C, float), np.ascontiguousarray(Cf, float),
                                np.ascontiguousarray(owner, np.int64), np.ascontiguousarray(neighbour, np.int64))
    nI = len(neighbour)
    reach = np.zeros(len(C))
    np.maximum.at(reach, owner, np.linalg.norm(Cf - C[owner], axis=1))
    np.maximum.at(reach, neighbour, np.linalg.norm(Cf[:nI] - C[neighbour], axis=1))
    return reach


def _free_distance(dom, C, Cf, owner, neighbour):
    """per cell centre in the fluid: its distance to the STL less the ray shift of
    inside() (-inf outside).  A line from the centre shorter than that stays inside.
    Where the STL is farther than the longest line from the centre to its face centres
    the distance does not matter (every line is shorter): there it is inf, found
    without the closest point (a search that stops at that radius)."""
    margin = dom.inside_shift + 1e-9 * dom.L
    if hasattr(dom._rs, "distance_below_each"):
        reach = _cell_reach(C, Cf, owner, neighbour)
        dC = dom._rs.distance_below_each(np.ascontiguousarray(C, float), np.ascontiguousarray(reach + margin, float))
    else:
        _, dC, _ = dom.closest(C)
    return np.where(dom.inside(C), dC - margin, -np.inf)


def through_wall(dom, points, faces, owner, neighbour, samples=16, cos_opposite=-0.5, geom=None, mark=None):
    """(number of cell-centre -> face-centre lines through a wall, their midpoints).
    geom (a dict): receives the face centres and area vectors, the cell centres and
    volumes (Cf, Sf, C, V) for a later quality check of the same mesh.
    mark(part): ends a part of the timing (mesher.Timer.mark)"""
    mark = mark or (lambda part: None)
    nI = len(neighbour)
    if getattr(dom, "_rs", None) is None and dom._kd is None:
        dom.build_search(0.01 * dom.L)          # Python closest-point search (no Rust)
    Cf, Sf = _face_geometry(points, faces)
    nC = int(max(owner.max(), neighbour.max() if nI else 0)) + 1
    C, V = quality.cell_geometry(nC, owner, neighbour, Cf, Sf)
    mark("face and cell geometry")
    if geom is not None:
        geom.update(Cf=Cf, Sf=Sf, C=C, V=V)
    T = dom.tris
    tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    tn /= np.maximum(np.linalg.norm(tn, axis=1), 1e-300)[:, None]
    if getattr(dom, "_rs", None) is not None and hasattr(dom._rs, "through_wall_cells"):
        # the lines built, filtered and sampled in Rust (the same lines, in the same order)
        free = _free_distance(dom, C, Cf, owner, neighbour)
        mark("distance of the cell centres to the STL")
        return dom._rs.through_wall_cells(np.ascontiguousarray(C, float), np.ascontiguousarray(Cf, float),
                                          np.ascontiguousarray(owner, np.int64), np.ascontiguousarray(neighbour, np.int64),
                                          np.ascontiguousarray(free, float), np.ascontiguousarray(tn, float),
                                          int(samples), float(cos_opposite))
    cells = np.concatenate([owner, neighbour])
    fcs = np.concatenate([np.arange(len(owner)), np.arange(nI)])
    A, B = C[cells], Cf[fcs]
    ts = (np.arange(samples) + 0.5) / samples
    if getattr(dom, "_rs", None) is not None:
        # a line from a centre in the fluid that stays closer to it than the STL (less the
        # ray shift of inside()) has every sample inside: no need to sample it
        _, dC, _ = dom.closest(C)
        free = np.where(dom.inside(C), dC - dom.inside_shift - 1e-9 * dom.L, -np.inf)
        ln = np.linalg.norm(B - A, axis=1) * ts[-1]
        keep = np.nonzero(ln >= free[cells])[0]
        A, B = A[keep], B[keep]
    if getattr(dom, "_rs", None) is not None and hasattr(dom._rs, "through_wall"):
        # the same samples and tests per line, in parallel
        return dom._rs.through_wall(np.ascontiguousarray(A, float), np.ascontiguousarray(B, float),
                                    np.ascontiguousarray(tn, float), int(samples), float(cos_opposite))
    hits, mids = 0, []
    for s0 in range(0, len(A), 200000):
        a, b = A[s0:s0 + 200000], B[s0:s0 + 200000]
        Q = a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]
        out = ~dom.inside(Q.reshape(-1, 3)).reshape(len(a), samples)
        h = np.nonzero(out.any(1))[0]
        if not len(h):
            continue
        first = np.argmax(out[h], 1)
        last = samples - 1 - np.argmax(out[h][:, ::-1], 1)
        _, _, t1 = dom.closest(Q[h, first])
        _, _, t2 = dom.closest(Q[h, last])
        wall = np.einsum("ij,ij->i", tn[t1], tn[t2]) < cos_opposite
        hits += int(wall.sum())
        mids.append(0.5 * (a[h[wall]] + b[h[wall]]))
    X = np.concatenate(mids) if mids else np.zeros((0, 3))
    return hits, X


def main(argv=None):
    import sys
    from .geometry import Surface, read_stl
    from .mesher import load_config, _rel
    from .readfoam import read_mesh
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cfg = load_config(argv[0])
    case = argv[1] if len(argv) > 1 else _rel(cfg, cfg.get("case", "."))
    dom = Surface(read_stl(_rel(cfg, cfg["domain"])), log=print, what="domain STL")
    m = read_mesh(case)
    n, X = through_wall(dom, m["points"], m["faces"], m["owner"], m["neighbour"])
    print(f"lines from cell centres to face centres through a solid wall: {n}")
    for x in X[:20]:
        print(f"    at ({x[0]:.6g} {x[1]:.6g} {x[2]:.6g})")
    return 1 if n else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
