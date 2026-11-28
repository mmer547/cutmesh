"""Generate the synthetic stress-test STLs of suites/synthetic.json (no download needed).

    python benchmark/make_synthetic.py            # -> benchmark/data/synthetic/*.stl

Each shape targets one way a cut-cell mesher can break: sharp convex/concave edges, faces
lying exactly on octree grid planes, thin fluid channels, thin solid walls, acute spikes,
holes (genus), nested surfaces, several solids, sliver triangles, and dirty input (open
surface, flipped normals, duplicated/degenerate triangles, non-manifold edge).
It also writes the showcase shapes of suites/showcase.json (pipes, ducts, a body in a wind
tunnel, with named inlet/outlet patches), which are meant to mesh cleanly.
All stress shapes fit in a box of about 1 x 1 x 1 (units are arbitrary: the suite sets the cell
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


# ------------------------------------------------------------------ showcase (suites/showcase.json)
# Shapes close to real CFD problems, with named inlet/outlet patches.  They are meant to mesh
# cleanly (PASS) and serve as worked examples; see benchmark/README.md.
def _disk(ring, centre):
    """Fan of triangles closing the ring of points (n x 3) at centre."""
    c = np.broadcast_to(np.asarray(centre, float), ring.shape)
    return np.stack([c, ring, np.roll(ring, -1, 0)], 1)


def _facing(tris, direction, *, keep_first=False):
    """Flip the triangles whose normal points against direction (3-vector or per triangle)."""
    tris = np.array(tris, float)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    d = np.broadcast_to(np.asarray(direction, float), n.shape)
    bad = np.einsum("ij,ij->i", n, d) < 0
    tris[bad] = tris[bad][:, [0, 2, 1] if keep_first else [2, 1, 0]]
    return tris


def _volume(tris):
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6)


def sweep_tube(path, r, n=40):
    """Circular tube of radius r along the polyline path (m x 3), parallel-transport frames.
    Returns (wall, start cap, end cap), all facing out of the tube."""
    path = np.asarray(path, float)
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    N = np.cross(T[0], a); N /= np.linalg.norm(N)
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    rings, Ns = [], []
    for k in range(len(path)):
        if k:
            N = N - T[k] * (N @ T[k]); N /= np.linalg.norm(N)
        B = np.cross(T[k], N)
        rings.append(path[k] + r * (np.cos(th)[:, None] * N + np.sin(th)[:, None] * B))
    R = np.array(rings)                                   # (m, n, 3)
    A, A2 = R[:-1], np.roll(R[:-1], -1, 1)
    Bq, B2 = R[1:], np.roll(R[1:], -1, 1)
    wall = np.concatenate([np.stack([A, A2, B2], 2).reshape(-1, 3, 3),
                           np.stack([A, B2, Bq], 2).reshape(-1, 3, 3)])
    axis_pt = np.concatenate([((path[:-1] + path[1:]) / 2)[:, None].repeat(n, 1).reshape(-1, 3)] * 2)
    wall = _facing(wall, wall.mean(1) - axis_pt)
    cap0 = _facing(_disk(R[0], path[0]), -T[0])
    cap1 = _facing(_disk(R[-1], path[-1]), T[-1])
    return wall, cap0, cap1


def _revolve_pipe(profile, n=64):
    """Pipe of revolution about z: wall from the profile [(r, z), ...] (all r > 0, bottom to
    top), plus the inlet disk (bottom) and outlet disk (top)."""
    wall = revolve(profile, n)
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    ring = lambda r, z: np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], 1)
    (r0, z0), (r1, z1) = profile[0], profile[-1]
    return (wall, _facing(_disk(ring(r0, z0), (0, 0, z0)), [0, 0, -1]),
            _facing(_disk(ring(r1, z1), (0, 0, z1)), [0, 0, 1]))


def _split(tris, rules, rest="walls"):
    """Group triangles into solids: rules = [(name, test(vertices n x 3 x 3) -> bool mask)]."""
    out, left = [], np.ones(len(tris), bool)
    for name, test in rules:
        m = left & test(tris)
        out.append((name, tris[m])); left &= ~m
    return out + [(rest, tris[left])]


def _plane(axis, value):
    return lambda t: np.all(np.isclose(t[:, :, axis], value), 1)


def _extrude_holes(outer, holes, z0, z1):
    """Prism over the convex polygon outer (n x 2) minus the circular holes [(cx, cy, r, n)],
    from z0 to z1.  Returns (outer side walls, hole walls, caps), facing out of the fluid."""
    from scipy.spatial import Delaunay
    outer = np.asarray(outer, float)
    pts, loops = [outer], []
    for cx, cy, r, n in holes:
        th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        pts.append(np.c_[cx + r * np.cos(th), cy + r * np.sin(th)])
    P = np.concatenate(pts)
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1)
    keep = np.ones(len(tri), bool)
    for cx, cy, r, n in holes:
        keep &= np.hypot(cen[:, 0] - cx, cen[:, 1] - cy) > r * np.cos(np.pi / n) * 0.999
    tri = tri[keep]
    lo, hi = np.c_[P, np.full(len(P), z0)], np.c_[P, np.full(len(P), z1)]
    caps = np.concatenate([_facing(lo[tri], [0, 0, -1]), _facing(hi[tri], [0, 0, 1])])

    def band(loop2d, inward):
        k = len(loop2d)
        i = np.arange(k); j = np.roll(i, -1)
        a, b = np.c_[loop2d, np.full(k, z0)], np.c_[loop2d, np.full(k, z1)]
        q = np.concatenate([np.stack([a[i], a[j], b[j]], 1), np.stack([a[i], b[j], b[i]], 1)])
        c = loop2d.mean(0)
        d = q.mean(1)[:, :2] - c
        return _facing(q, np.c_[-d if inward else d, np.zeros(len(q))])
    sides = band(outer, False)
    pins = np.concatenate([band(p, True) for p in pts[1:]]) if holes else np.zeros((0, 3, 3))
    return sides, pins, caps


def car_wheels(body, gap=0.005, yaw=0):
    """Four tyres penetrate the floor by 2 mm. All tunnel patches use the same
    tile breakpoints, so every edge of a wheel opening has two incident triangles."""
    radius, width, penetration = 0.05, 0.04, 0.002
    zc = radius - penetration
    half_chord = np.sqrt(radius**2 - zc**2)
    angle = np.pi/2 + np.arcsin(zc/radius)
    phi = np.linspace(-angle, angle, 49)
    wheels = []
    for x in (0.6 + 0.1, 0.6 + 0.5 - 0.1):
        for side in (-1, 1):
            ends = sorted((side*(0.1 + gap), side*(0.1 + gap + width)))
            wheels.append((x, *ends))

    def quad(a, b, c, d):
        return [[a, b, c], [a, c, d]]

    tyres = []
    for x, ya, yb in wheels:
        ring = np.c_[x + radius*np.sin(phi), zc + radius*np.cos(phi)]
        ring[[0, -1], 1] = 0
        side_a = np.c_[ring[:, 0], np.full(len(ring), ya), ring[:, 1]]
        side_b = side_a.copy()
        side_b[:, 1] = yb
        tread = np.array([tri for i in range(len(ring)-1)
                          for tri in quad(side_a[i], side_a[i+1], side_b[i+1], side_b[i])])
        centres = tread.mean(axis=1)
        inward = np.c_[x - centres[:, 0], np.zeros(len(tread)), zc - centres[:, 2]]
        tyres.extend(_facing(tread, inward, keep_first=True))
        for side, normal in ((side_a, [0, 1, 0]), (side_b, [0, -1, 0])):
            caps = [[[x, side[0, 1], 0], p, q] for p, q in zip(side[:-1], side[1:])]
            tyres.extend(_facing(caps, normal, keep_first=True))

    xs = sorted({0, 2, *[x + dx for x, _, _ in wheels for dx in (-half_chord, 0, half_chord)]})
    ys = sorted({-0.4, 0.4, *[y for _, ya, yb in wheels for y in (ya, yb)]})
    floor, ceiling, walls, inlet, outlet = [], [], [], [], []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            tile = np.array(quad([x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]))
            ceiling.extend(tile + [0, 0, 0.5])
            xm, ym = (x0+x1)/2, (y0+y1)/2
            if not any(x-half_chord < xm < x+half_chord and ya < ym < yb for x, ya, yb in wheels):
                floor.extend(_facing(tile, [0, 0, -1], keep_first=True))
        for y, normal in ((ys[0], [0, -1, 0]), (ys[-1], [0, 1, 0])):
            wall = quad([x0, y, 0], [x1, y, 0], [x1, y, .5], [x0, y, .5])
            walls.extend(_facing(wall, normal, keep_first=True))
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x, normal, patch in ((xs[0], [-1, 0, 0], inlet), (xs[-1], [1, 0, 0], outlet)):
            cap = quad([x, y0, 0], [x, y1, 0], [x, y1, .5], [x, y0, .5])
            patch.extend(_facing(cap, normal, keep_first=True))
    solids = [("inlet", inlet), ("outlet", outlet), ("walls", walls + ceiling),
              ("floor", floor), ("body", body), ("wheelArea", tyres)]
    return [(name, rotate(np.asarray(tris), [0, 0, 1], yaw) if yaw else np.asarray(tris))
            for name, tris in solids]


def showcase():
    """name -> [(solid, tris)]; every surface closed and facing out of the fluid."""
    S = {}
    # pipe elbow: D = 0.2, bend radius 1.5 D, straight legs of 3 D and 4 D
    r, Rb = 0.1, 0.3
    arc = np.linspace(0, np.pi / 2, 25)
    path = np.concatenate([np.c_[np.linspace(-0.6, 0, 13)[:-1], np.zeros(12), np.zeros(12)],
                           np.c_[Rb * np.sin(arc), Rb * (1 - np.cos(arc)), np.zeros(25)],
                           np.c_[np.full(16, Rb), Rb + np.linspace(0, 0.8, 17)[1:], np.zeros(16)]])
    wall, cin, cout = sweep_tube(path, r, 48)
    S["pipe_elbow"] = [("inlet", cin), ("outlet", cout), ("walls", wall)]
    # helical coil: 2.5 turns of radius 0.3, pitch 0.18, tube radius 0.05
    t = np.linspace(0, 2.5 * 2 * np.pi, 301)
    path = np.c_[0.3 * np.cos(t), 0.3 * np.sin(t), 0.18 * t / (2 * np.pi)]
    wall, cin, cout = sweep_tube(path, 0.05, 32)
    S["helical_coil"] = [("inlet", cin), ("outlet", cout), ("walls", wall)]
    # venturi: D 0.4 -> throat 0.16 (cosine contraction, 7 deg diffuser), 1.6 long
    z = np.concatenate([[0.0], np.linspace(0.3, 0.55, 14), np.linspace(0.62, 1.3, 18), [1.6]])
    rr = np.where(z <= 0.3, 0.2, np.where(z <= 0.55, 0.08 + 0.06 * (1 + np.cos(np.pi * (z - 0.3) / 0.25)),
                  np.where(z <= 0.62, 0.08, np.minimum(0.2, 0.08 + (z - 0.62) * np.tan(np.radians(7))))))
    wall, cin, cout = _revolve_pipe(list(zip(rr, z)), 64)
    S["venturi"] = [("inlet", cin), ("outlet", cout), ("walls", wall)]
    # sharp-edged orifice plate in a pipe: D 0.3, bore 0.15, plate 0.03 thick
    wall, cin, cout = _revolve_pipe([(0.15, 0), (0.15, 0.45), (0.075, 0.45), (0.075, 0.48),
                                     (0.15, 0.48), (0.15, 1.2)], 64)
    S["orifice_pipe"] = [("inlet", cin), ("outlet", cout), ("walls", wall)]
    # backward-facing step: inlet channel 0.1 high over a step of 0.1, span 0.3
    t = extrude([(-0.3, 0.1), (0, 0.1), (0, 0), (1.5, 0), (1.5, 0.2), (-0.3, 0.2)], 0, 0.3)
    S["backward_step"] = _split(t, [("inlet", _plane(0, -0.3)), ("outlet", _plane(0, 1.5)),
                                    ("sides", lambda t: _plane(2, 0)(t) | _plane(2, 0.3)(t))])
    # T-junction of square ducts: main 0.2 x 0.2 along x, branch 0.1 wide from +y
    t = extrude([(0, 0), (1.2, 0), (1.2, 0.2), (0.45, 0.2), (0.45, 0.6), (0.35, 0.6), (0.35, 0.2),
                 (0, 0.2)], 0, 0.2)
    S["t_junction"] = _split(t, [("inletMain", _plane(0, 0)), ("inletBranch", _plane(1, 0.6)),
                                 ("outlet", _plane(0, 1.2))])
    # pin-fin channel: staggered bank of 13 pins (d 0.08) across a 1.4 x 0.5 x 0.15 duct
    holes = [(x, y, 0.04, 32) for i, x in enumerate([0.35, 0.5, 0.65, 0.8, 0.95])
             for y in ([0.1, 0.25, 0.4] if i % 2 == 0 else [0.175, 0.325])]
    sides, pins, caps = _extrude_holes([(0, 0), (1.4, 0), (1.4, 0.5), (0, 0.5)], holes, 0, 0.15)
    S["pin_fins"] = _split(np.concatenate([sides, caps]), [("inlet", _plane(0, 0)), ("outlet", _plane(0, 1.4))]) \
        + [("pins", pins)]
    # bluff body in a wind tunnel: rounded nose, 25 deg rear slant, 0.05 ground clearance
    a = np.linspace(0, np.pi / 2, 7)
    rn, x0, L, z0, H = 0.05, 0.6, 0.5, 0.05, 0.15
    prof = np.concatenate([[(x0 + L, z0), (x0 + L, z0 + H - 0.1 * np.tan(np.radians(25))),
                            (x0 + L - 0.1, z0 + H)],
                           np.c_[x0 + rn - rn * np.sin(a), z0 + H - rn + rn * np.cos(a)],
                           np.c_[x0 + rn - rn * np.cos(a), z0 + rn - rn * np.sin(a)]])
    body = extrude(prof, -0.1, 0.1)[:, :, [0, 2, 1]]       # profile in x-z, width along y
    if _volume(body) > 0:
        body = flip(body)                                  # the body is a hole in the fluid
    tunnel = box([0, -0.4, 0], [2.0, 0.4, 0.5])
    S["body_in_tunnel"] = _split(tunnel, [("inlet", _plane(0, 0)), ("outlet", _plane(0, 2.0)),
                                          ("floor", _plane(2, 0))]) + [("body", body)]
    S["car_wheels"] = car_wheels(body)
    S["car_wheels_yaw7"] = car_wheels(body, yaw=7)
    S["car_wheels_gap1_yaw7"] = car_wheels(body, gap=0.001, yaw=7)
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
    for name, solids in showcase().items():
        p = os.path.join(OUT, name + ".stl")
        write_ascii(p, solids)
        print(f"{p}  (ASCII, {len(solids)} solids, {sum(len(t) for _, t in solids)} triangles)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
