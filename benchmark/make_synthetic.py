"""Generate the synthetic stress-test STLs of suites/synthetic.json (no download needed).

    python benchmark/make_synthetic.py            # -> benchmark/data/synthetic/*.stl

Each shape targets one way a cut-cell mesher can break: sharp convex/concave edges, faces
lying exactly on octree grid planes, thin fluid channels, thin solid walls, acute spikes,
holes (genus), nested surfaces, several solids, sliver triangles, and dirty input (open
surface, flipped normals, duplicated/degenerate triangles, non-manifold edge).
All shapes fit in a box of about 1 x 1 x 1 (units are arbitrary: the suite sets the cell
size from the bounding box).
"""
import os
import struct
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic")


# ------------------------------------------------------------------ primitives
def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return v[np.array(f)]


def icosphere(r=0.5, sub=3, c=(0, 0, 0)):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
         [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
         [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    tris = np.array(v, float)[np.array(f)]
    for _ in range(sub):
        a, b, cc = tris[:, 0], tris[:, 1], tris[:, 2]
        ab, bc, ca = (a + b) / 2, (b + cc) / 2, (cc + a) / 2
        tris = np.concatenate([np.stack(x, 1) for x in
                               ([a, ab, ca], [ab, b, bc], [ca, bc, cc], [ab, bc, ca])])
    tris /= np.linalg.norm(tris, axis=2, keepdims=True)
    return tris * r + np.asarray(c, float)


def revolve(profile, n=48):
    """Surface of revolution about z of the polyline profile [(r, z), ...] (first/last r may be 0).
    The profile runs from bottom to top; the outward side is +r."""
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    ring = lambda r, z: np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], 1)
    out = []
    for (r0, z0), (r1, z1) in zip(profile[:-1], profile[1:]):
        A, B = ring(r0, z0), ring(r1, z1)
        A2, B2 = np.roll(A, -1, 0), np.roll(B, -1, 0)
        if r0 > 0:
            out.append(np.stack([A, A2, B2 if r1 > 0 else B], 1))
        if r1 > 0:
            out.append(np.stack([A if r0 > 0 else A, B2, B], 1))
    return np.concatenate(out)


def torus(R=0.35, r=0.12, nu=64, nv=24):
    u = np.linspace(0, 2 * np.pi, nu + 1)[:-1]
    v = np.linspace(0, 2 * np.pi, nv + 1)[:-1]
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.stack([(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)], -1)
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    i1, j1 = (i + 1) % nu, (j + 1) % nv
    a, b, c, d = P[i, j], P[i1, j], P[i1, j1], P[i, j1]
    return np.concatenate([np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)])


def _ear_clip(poly):
    """Triangulate a simple CCW polygon (n x 2) -> index triples."""
    idx = list(range(len(poly)))
    tris = []
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-14:
                continue
            if any(cross(a, b, poly[m]) >= 0 and cross(b, c, poly[m]) >= 0 and cross(c, a, poly[m]) >= 0
                   for m in idx if m not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            break
        else:
            raise ValueError("ear clipping failed (polygon not simple / not CCW?)")
    tris.append(tuple(idx))
    return np.array(tris)


def extrude(poly, z0=0.0, z1=1.0):
    """Prism over the CCW polygon poly (n x 2) from z0 to z1, outward normals."""
    poly = np.asarray(poly, float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
    n = len(poly)
    lo = np.c_[poly, np.full(n, z0)]
    hi = np.c_[poly, np.full(n, z1)]
    t = _ear_clip(poly)
    caps = [lo[t[:, ::-1]], hi[t]]
    j = np.roll(np.arange(n), -1)
    sides = [np.stack([lo, lo[j], hi[j]], 1), np.stack([lo, hi[j], hi], 1)]
    return np.concatenate(caps + sides)


def rotate(tris, axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
    return tris @ R.T


def flip(tris):
    return tris[:, ::-1].copy()


# ------------------------------------------------------------------ writers
def write_binary(path, tris, name="walls"):
    tris = np.asarray(tris, np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.where(ln > 0, ln, 1)).astype(np.float32)
    rec = np.zeros(len(tris), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tris
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


def write_ascii(path, solids):
    with open(path, "w") as f:
        for name, tris in solids:
            f.write(f"solid {name}\n")
            for t in np.asarray(tris, float):
                nn = np.cross(t[1] - t[0], t[2] - t[0])
                ln = np.linalg.norm(nn)
                nn = nn / ln if ln > 0 else nn
                f.write(f" facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}\n  outer loop\n")
                for v in t:
                    f.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")


# ------------------------------------------------------------------ shapes
def shapes():
    S = {}
    # --- clean baselines
    S["sphere"] = icosphere(0.5, 4)
    S["cube"] = box([0, 0, 0], [1, 1, 1])
    S["cube_rotated"] = rotate(rotate(box([-.5, -.5, -.5], [.5, .5, .5]), [0, 0, 1], 30), [1, 1, 0], 20)
    # faces exactly on octree grid planes: the suite gives baseCellSize = 1/16 for this one
    S["cube_aligned"] = box([0, 0, 0], [1, 1, 1])
    S["cylinder"] = revolve([(0, 0), (0.4, 0), (0.4, 1), (0, 1)], 64)
    # --- sharp features
    S["l_bracket"] = extrude([(0, 0), (1, 0), (1, .3), (.3, .3), (.3, 1), (0, 1)], 0, .6)
    S["stepped_shaft"] = revolve([(0, 0), (.45, 0), (.45, .3), (.25, .3), (.25, .7), (.12, .7), (.12, 1), (0, 1)], 64)
    star = [(np.cos(a) * (0.5 if k % 2 == 0 else 0.18), np.sin(a) * (0.5 if k % 2 == 0 else 0.18))
            for k, a in enumerate(np.linspace(0, 2 * np.pi, 13)[:-1])]
    S["star_prism"] = extrude(star, 0, .4)                    # 6 acute convex + 6 reflex edges
    S["cone_tip"] = revolve([(0, 0), (.4, 0), (0, 1)], 64)    # point singularity + acute rim
    S["wedge_10deg"] = extrude([(0, 0), (1, 0), (1, np.tan(np.radians(10)))], 0, .5)
    # --- topology
    S["torus"] = torus()
    S["shell_nested"] = np.concatenate([icosphere(0.5, 3), flip(icosphere(0.3, 3))])   # fluid between
    S["two_spheres"] = np.concatenate([icosphere(0.3, 3, (-.4, 0, 0)), icosphere(0.3, 3, (.4, 0, 0))])
    # --- thin features (relative to the default cell size: diag/32 ~ 0.05 base, level 2 ~ 0.013)
    # fluid comb: wide plenum with 5 thin fluid fingers of width 0.02
    w, g = 0.02, 0.18
    comb = [(0, 0), (1, 0), (1, .3)]
    for k in range(5):
        x1 = 1 - .05 - k * g; x0 = x1 - w
        comb += [(x1, .3), (x1, 1), (x0, 1), (x0, .3)]
    comb += [(0, .3)]
    S["thin_channels"] = extrude(comb, 0, .3)
    # thin solid walls: box with 4 slits of width 0.004 cut almost across (solid fins in the fluid)
    sl = [(0, 0), (1, 0), (1, 1)]
    for k in range(4):
        x = .8 - k * .2
        sl += [(x + .002, 1), (x + .002, .1), (x - .002, .1), (x - .002, 1)]
    sl += [(0, 1)]
    S["thin_walls"] = extrude(sl, 0, .3)
    # fluid slot tapering to a zero-width tip (sharp slot end)
    S["tapered_slot"] = extrude([(0, 0), (1, 0), (1, .4), (.53, .4), (.5, .9), (.47, .4), (0, .4)], 0, .3)
    # solid knife edge jutting into the fluid (acute concave edge seen from the fluid)
    S["solid_spike"] = extrude([(0, 0), (1, 0), (1, 1), (.53, 1), (.5, .3), (.47, 1), (0, 1)], 0, .3)
    # --- triangulation quality
    # long sliver triangles: cylinder with a single segment along its length and fan caps
    S["sliver_cylinder"] = revolve([(0, 0), (.1, 0), (.1, 1), (0, 1)], 200)
    # --- dirty input
    S["dirty_open_box"] = box([0, 0, 0], [1, 1, 1])[2:]                # bottom (z = 0) face missing
    S["dirty_flipped"] = flip(icosphere(0.5, 3))                       # all normals inward
    half = icosphere(0.5, 3)
    S["dirty_mixed_orient"] = np.concatenate([half[: len(half) // 2], flip(half[len(half) // 2:])])
    sp = icosphere(0.5, 3)
    S["dirty_duplicates"] = np.concatenate([sp, sp[:50], np.repeat(sp[:20, :1], 3, 1)])  # dup + zero-area
    S["dirty_edge_touch"] = np.concatenate([box([0, 0, 0], [.5, .5, 1]), box([.5, .5, 0], [1, 1, 1])])
    S["dirty_overlap"] = np.concatenate([icosphere(0.35, 3, (-.15, 0, 0)), icosphere(0.35, 3, (.15, 0, 0))])
    return S


def multi_solid():
    """Box with named inlet/outlet/walls patches (ASCII multi-solid STL)."""
    t = box([0, 0, 0], [1, .4, .4])
    xs = t[:, :, 0]
    inlet = np.all(np.isclose(xs, 0), 1)
    outlet = np.all(np.isclose(xs, 1), 1)
    return [("inlet", t[inlet]), ("outlet", t[outlet]), ("walls", t[~inlet & ~outlet])]


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    for name, tris in shapes().items():
        p = os.path.join(OUT, name + ".stl")
        write_binary(p, tris)
        print(f"{p}  ({len(tris)} triangles)")
    p = os.path.join(OUT, "duct_patches.stl")
    write_ascii(p, multi_solid())
    print(f"{p}  (ASCII, 3 solids)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
