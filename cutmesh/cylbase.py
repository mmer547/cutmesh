"""Cylindrical base mesh: concentric blocks around an axis instead of the Cartesian
root grid, for rotating machinery.

    core     square block around the axis (Cartesian, n x n cells across)
    ring     O-grid block from the square to the first circle (4 n cells around)
    annuli   concentric blocks, each with its own radial and circumferential division

Each block is an octree in its own index space (u, v, w) mapped to physical space;
the circumferential index is periodic.  The usual refinement (surfaces, features,
zones, regions) runs on top, by physical cell size: level L means the target size
baseCellSize / 2**L.  Blocks are joined by overlaying the faces of both sides on the
common surface (square or circle), so any ratio of divisions gives a conformal
polyhedral mesh.  build_cells() returns the same structure as Octree.build_cells().

meshDict:
    "baseMesh": {
        "type": "cylindrical",
        "origin": [0, 0, 0], "axis": [0, 0, 1],
        "zRange": [z0, z1],                 # optional: default STL extent + one cell
        "nz": 40,                           # or "dz"; default dz = baseCellSize
        "maxSizeRatio": 2.5                 # cell size jump allowed across block faces
    }
The core and the blocks are laid out automatically (auto_layout(): a 2 x 2 core and
concentric blocks whose cells point around the axis, with circles of nodes kept off the
coaxial cylinders of the geometry), out to "rOuter" (default: STL radius + 2/3 cell), for
cells "cellSize" across (default: the smaller of baseCellSize and dz).  Or by hand:
        "core": {"halfWidth": 0.01, "n": 8},
        "blocks": [                         # outward; the first one is the O-grid ring
            {"rOuter": 0.03, "nr": 6},                  # nTheta = 4 n
            {"rOuter": 0.15, "nr": 12, "nTheta": 96}],  # default nTheta: square-ish cells
        (a block may give "radii": its nr + 1 radial nodes, default evenly spaced)
"""
import json
import math

import numpy as np

from .geometry import sdf_below
from .octree import Octree, CHILD

TH0 = -0.25 * math.pi          # angle of the square's corner (a, -a): v = 0
# a cell counts as refined to a level once no edge exceeds the target size by more than
# this (block cells are not powers of two of the target, so 1.0 would add a level for
# a few percent)
SIZE_TOL = 1.3
# automatic layout: each block's radii grow by about AUTO_RATIO, so its cells are about 0.8
# to 1.25 times as wide around at its inner and outer radius as at its middle
AUTO_RATIO = 1.5625
# automatic layout: radial layers are at most this times the target size deep (the longest
# edge must stay within SIZE_TOL of baseCellSize, or refinement adds a level)
AUTO_DR = 1.15


def _dyadic(t, levels=3):
    """how far a surface at t (in cell sizes from a node) stays from the nodes of the cell
    refined 0 .. levels - 1 times: at most 1/3, reached at t = 1/3 or 2/3"""
    return min(abs(2 ** L * t - round(2 ** L * t)) for L in range(levels))


def auto_layout(h, r_outer, avoid=()):
    """core {"halfWidth", "n"} and blocks [{"rOuter", "nr", "nTheta"(, "radii")}] out to
    r_outer for cells about h across (no edge longer than about 1.2 h), pointing around
    the axis everywhere but in the 2 x 2 core.

    Refinement halves all edges of a cell together until the longest is within SIZE_TOL
    of the target, so a base cell only a little longer than baseCellSize gets a whole
    level more: hence edges of at most about h (the caller's h <= baseCellSize).  The size
    of a cell's edges is their mean per index direction (MappedBlock._geometry): around
    the axis that is about the arc at the middle radius.

    The core is 2 x 0.5 .. 0.95 h wide and the O-grid ring (8 around) ends at twice its
    half width; outward the blocks grow by radius ratios of about AUTO_RATIO to r_outer,
    with nTheta (a multiple of 4) for cells h wide around at the middle radius.  Blocks
    are joined by overlaying their faces, so any ratio of nTheta is conformal.

    avoid: radii of coaxial cylinders of the geometry (shaft, hub, walls, rotating zones).
    A cylinder close to a circle of nodes leaves slivers of cut cells along it, and since
    refinement halves the cells, the place that stays clear at every level is a third of
    the way through a layer (_dyadic): so each cylinder gets a layer h deep with it at 1/3
    (or 2/3), the layers between are filled evenly (<= AUTO_DR h) and the blocks end at
    the nodes nearest their radii ("radii": the radial nodes, when not even).  The core's
    half width is chosen likewise for cylinders inside it (along the middle of its sides)
    and for the O-grid circle."""
    avoid = sorted(float(x) for x in avoid)

    def score(t):
        """clearance (in cells, at refinement levels 0 .. 2) of the cylinders inside the
        core from the middle of its sides, and of those near the O-grid circle from that"""
        sc = [_dyadic((t - x) / t) for x in avoid if x < t]
        sc += [_dyadic(abs(2 * t - x) / t) for x in avoid if t * math.sqrt(2) < x < 3 * t]
        return min(sc + [1.0])

    cs = [0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95] + [0.775 - 0.025 * k for k in range(12)]
    cands = [min(c * h, 0.5 * r_outer) for c in cs]
    sc = [score(t) for t in cands]
    ok = [t for t, v in zip(cands, sc) if v >= 0.25]
    a = ok[0] if ok else cands[int(np.argmax(sc))]
    r1 = 2 * a
    if r_outer <= r1 * (1 + 1e-9):
        return {"halfWidth": a, "n": 2}, [{"rOuter": r_outer, "nr": 1}]
    # fixed circles: the O-grid circle, the outer one, a layer h deep around each cylinder
    tol = 1e-6 * h
    anchors = [r1, r_outer]
    for x in avoid:
        if not (r1 < x < r_outer):
            continue
        for lo, hi in ((x - h / 3, x + 2 * h / 3), (x - 2 * h / 3, x + h / 3)):
            if lo < r1 - tol or hi > r_outer + tol:
                continue
            new = [v for v in (lo, hi) if min(abs(v - u) for u in anchors) > tol]
            if all(min(abs(v - u) for u in anchors) >= 0.4 * h for v in new):
                anchors += new
                break
    anchors.sort()
    nodes = [r1]
    for A, B in zip(anchors, anchors[1:]):
        n = max(1, int(math.ceil((B - A) / (AUTO_DR * h) - 1e-6)))
        nodes += [A + (B - A) * j / n for j in range(1, n + 1)]
    nodes = np.array(nodes)
    # block ends: the nodes nearest radii growing by AUTO_RATIO
    nb = max(1, int(round(math.log(r_outer / r1) / math.log(AUTO_RATIO))))
    ends = [len(nodes) - 1]
    for k in range(nb - 1, 0, -1):
        i = int(np.argmin(np.abs(nodes - r1 * (r_outer / r1) ** (k / nb))))
        if 0 < i < ends[0]:
            ends.insert(0, i)
    blocks = [{"rOuter": r1, "nr": 1}]
    i0 = 0
    for i1 in ends:
        ri, ro = float(nodes[i0]), float(nodes[i1])
        nr = i1 - i0
        arc = 2 * math.pi * 0.5 * (ri + ro)
        nth = 4 * max(2, int(round(arc / h / 4)))
        if arc / nth > 1.2 * h:
            nth += 4
        blk = {"rOuter": ro, "nr": nr, "nTheta": nth}
        rs = nodes[i0:i1 + 1]
        if nr > 1 and np.abs(rs - np.linspace(ri, ro, nr + 1)).max() > 1e-9 * ro:
            blk["radii"] = rs.tolist()
        blocks.append(blk)
        i0 = i1
    return {"halfWidth": a, "n": 2}, blocks


def coaxial_radii(x, y, tol, min_angles=6):
    """radii of circles of points around the axis (in-plane coordinates x, y): points
    at one radius (within tol) at no fewer than min_angles angles, with no gap over 90
    degrees between them, i.e. a cylinder or a circular edge of the geometry"""
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    key = np.round(r / tol).astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cut = np.concatenate([[0], np.nonzero(np.diff(ks))[0] + 1, [len(ks)]])
    out = []
    for i0, i1 in zip(cut[:-1], cut[1:]):
        if i1 - i0 < min_angles or ks[i0] == 0:
            continue
        sel = order[i0:i1]
        t = np.unique(np.round(th[sel], 6))
        if len(t) < min_angles:
            continue
        if np.diff(np.concatenate([t, [t[0] + 2 * math.pi]])).max() < 0.5 * math.pi:
            out.append(float(r[sel].mean()))
    return out


# ----------------------------------------------------------------------------- blocks
class MappedBlock(Octree):
    """Octree on an index box (level-0 cells) mapped to physical space by mapper(params)."""

    def __init__(self, name, n, max_level, mapper, periodic):
        super().__init__(np.zeros(3), 1.0, n, max_level, periodic)
        self.name = name
        self.mapper = mapper

    def map_lattice(self, lat):
        return self.mapper(np.asarray(lat, float) / float(1 << self.Lmax))

    def _geometry(self, idx=None):
        """centres, edge vectors (mean per index direction) and corner bbox half diagonal"""
        lev = self.lev if idx is None else self.lev[idx]
        ijk = self.ijk if idx is None else self.ijk[idx]
        s = 1.0 / 2.0 ** lev
        par = (ijk[:, None, :] + CHILD[None]) * s[:, None, None]          # (k, 8, 3)
        P = self.mapper(par.reshape(-1, 3)).reshape(-1, 8, 3)
        cen = self.mapper((ijk + 0.5) * s[:, None])
        E = np.zeros((len(P), 3, 3))
        for a in range(3):
            lo = CHILD[:, a] == 0
            E[:, a] = (P[:, ~lo] - P[:, lo]).mean(1)
        hd = 0.5 * np.linalg.norm(P.max(1) - P.min(1), axis=1)
        return cen, E, hd

    def leaf_sizes(self):
        # the shortest edge: tolerances (snapping) must not exceed what the cell allows
        _, E, _ = self._geometry()
        return np.linalg.norm(E, axis=2).min(1)

    def leaf_volumes(self):
        _, E, _ = self._geometry()
        return np.abs(np.einsum("ij,ij->i", E[:, 0], np.cross(E[:, 1], E[:, 2])))

    def refine_physical(self, surface_sources, volume_sources, h0, log, keep_fn=None):
        """Split leaves larger than the target size baseCellSize / 2**level of a source
        within the source's buffer band (as Octree.refine, by physical size)."""
        for it in range(self.Lmax):
            sel = np.nonzero(self.lev < self.Lmax)[0]
            if len(sel) == 0:
                break
            c, E, hd = self._geometry(sel)
            size = np.linalg.norm(E, axis=2).max(1)
            mark = np.zeros(len(sel), bool)
            for kd, Ls, nbuf in surface_sources:
                t = h0 / 2.0 ** Ls
                need = np.nonzero(size > SIZE_TOL * t)[0]
                if len(need) == 0:
                    continue
                R = nbuf * np.maximum(size[need] - 2 * t, 0.0) + hd[need]
                d, _ = kd.query(c[need], distance_upper_bound=float(R.max()), workers=-1)
                mark[need[d < R]] = True
            for sdf, Lv, nbuf in volume_sources:
                t = h0 / 2.0 ** Lv
                need = np.nonzero(size > SIZE_TOL * t)[0]
                if len(need) == 0:
                    continue
                R = nbuf * np.maximum(size[need] - 2 * t, 0.0) + hd[need]
                mark[need[sdf_below(sdf, c[need], R)]] = True
            if keep_fn is not None and mark.any():
                mi = np.nonzero(mark)[0]
                mark[mi[~keep_fn(c[mi], hd[mi])]] = False
            full = np.zeros(len(self.lev), bool)
            full[sel[mark]] = True
            ns = self.split(full)
            if ns:
                log(f"  {self.name}: pass {it + 1}: split {ns} cells, total {len(self.lev)}")
            if ns == 0:
                break
        self.balance(lambda s: None)


# ----------------------------------------------------------------------------- interfaces
class Side:
    """One block's side of a block interface: which leaves/vertices lie on it and their
    interface coordinates (s along the interface, periodic with period S, and z)."""

    def __init__(self, block, axis, at_end, s_of, S):
        self.b = block
        self.axis = axis          # index axis normal to the side (u = 0 / u = n)
        self.at_end = at_end      # True: index n (outer side), False: index 0
        self.s_of = s_of          # param coords (k, 3) -> s
        self.S = S

    def leaf_mask(self):
        b = self.b
        N = b.n[self.axis] * (1 << b.lev)
        q = b.ijk[:, self.axis]
        return (q + 1 == N) if self.at_end else (q == 0)

    def vertex_mask(self, lat):
        n = self.b.n[self.axis] * (1 << self.b.Lmax)
        return lat[:, self.axis] == (n if self.at_end else 0)

    def leaf_rects(self, idx):
        """(s0, s1, z0, z1) of the side faces of leaves idx, s0 < s1 <= S"""
        b = self.b
        lev = b.lev[idx]
        s = 1.0 / 2.0 ** lev
        ijk = b.ijk[idx]
        lo = ijk * s[:, None]
        hi = (ijk + 1) * s[:, None]
        pos = np.where(self.at_end, hi[:, self.axis], lo[:, self.axis])
        other = [a for a in (0, 1) if a != self.axis][0]
        # two corners along the side (other in-plane index axis), at the side's position
        p0 = lo.copy(); p1 = lo.copy()
        p0[:, self.axis] = pos; p1[:, self.axis] = pos
        p1[:, other] = hi[:, other]
        sa, sb = self.s_of(p0), self.s_of(p1)
        s0, s1 = np.minimum(sa, sb), np.maximum(sa, sb)
        wrap = s1 - s0 > 0.5 * self.S             # the face ending at s = S got s = 0
        s0, s1 = np.where(wrap, s1, s0), np.where(wrap, s0 + self.S, s1)
        return s0, s1, lo[:, 2], hi[:, 2]


def _overlaps(A, B, S, tol):
    """pairs (i, j) of rectangles (s0, s1, z0, z1) of A and B with a common area"""
    s0b, s1b, z0b, z1b = B
    order = np.argsort(s0b)
    s0s = s0b[order]
    pairs = []
    s0a, s1a, z0a, z1a = A
    for i in range(len(s0a)):
        hi = np.searchsorted(s0s, s1a[i] - tol)
        cand = order[:hi]
        cand = cand[(s1b[cand] > s0a[i] + tol) & (z1b[cand] > z0a[i] + tol) & (z0b[cand] < z1a[i] - tol)]
        for j in cand.tolist():
            pairs.append((i, j))
    return pairs


# ----------------------------------------------------------------------------- base mesh
class CylindricalBase:
    def __init__(self, spec, dom, h0, hmin, Lmax_target, log=print, cylinders=(), surfaces=()):
        """cylinders: (point, axis, radius) of rotating zones / refinement regions and
        surfaces: triangle arrays of STL zones, whose coaxial radii the automatic layout
        keeps its circles of nodes away from (besides those of the domain STL)"""
        self.spec = spec
        self.h0 = float(h0)
        self.log = log
        o = np.asarray(spec.get("origin", [0, 0, 0]), float)
        ez = np.asarray(spec.get("axis", [0, 0, 1]), float)
        ez /= np.linalg.norm(ez)
        t = np.array([1.0, 0, 0]) if abs(ez[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = t - np.dot(t, ez) * ez
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ez, e1)
        self.o, self.e1, self.e2, self.ez = o, e1, e2, ez
        V = dom.tris.reshape(-1, 3) - o
        zs = V @ ez
        rmax = float(np.hypot(V @ e1, V @ e2).max())
        if "zRange" in spec:
            z0, z1 = map(float, spec["zRange"])
        else:
            z0, z1 = float(zs.min()) - self.h0, float(zs.max()) + self.h0
        if "nz" in spec:
            nz = int(spec["nz"])
        else:
            nz = max(1, int(math.ceil((z1 - z0) / float(spec.get("dz", self.h0)))))
        self.z0, self.dz, self.nz = z0, (z1 - z0) / nz, nz
        if ("core" in spec) != ("blocks" in spec):
            raise ValueError("baseMesh: give both core and blocks, or neither (automatic layout)")
        if "core" in spec:
            core, blocks = spec["core"], spec["blocks"]
        else:
            h = float(spec.get("cellSize", min(self.h0, self.dz)))
            # default: the outermost wall a third of a layer inside the last circle
            r_out = float(spec.get("rOuter", rmax + 2 * h / 3))
            P = np.concatenate([V] + [np.asarray(t, float).reshape(-1, 3) - o for t in surfaces])
            avoid = coaxial_radii(P @ e1, P @ e2, 1e-3 * h)
            for p_, ax, rc in cylinders:
                ax = np.asarray(ax, float) / np.linalg.norm(ax)
                d = np.asarray(p_, float) - o
                if np.linalg.norm(np.cross(ax, ez)) < 1e-6 and \
                        np.linalg.norm(d - np.dot(d, ez) * ez) < 1e-6 * max(rc, h):
                    avoid.append(float(rc))
            core, blocks = auto_layout(h, r_out, avoid)
            if avoid:
                log("cylindrical base: coaxial radii kept off the circles of nodes: "
                    + ", ".join(f"{x:g}" for x in sorted(set(round(x, 9) for x in avoid))))
            log("cylindrical base: automatic layout for cells of " + f"{h:g}: \"core\": "
                + json.dumps(core) + ", \"blocks\": " + json.dumps(
                    [{k: (round(v, 6) if isinstance(v, float) else
                          [round(x, 6) for x in v] if isinstance(v, list) else v) for k, v in b.items()}
                     for b in blocks]))
        a = float(core["halfWidth"])
        nc = int(core["n"])
        if not blocks:
            raise ValueError("baseMesh: at least one block (the O-grid ring) is needed")
        R = [float(b["rOuter"]) for b in blocks]
        if R[0] <= a * math.sqrt(2) * 1.01 or any(r1 <= r0 for r0, r1 in zip(R, R[1:])):
            raise ValueError("baseMesh: radii must grow outward and the first one exceed the core's corner")
        if R[-1] < rmax:
            log(f"WARNING: baseMesh: outer radius {R[-1]:g} < STL radius {rmax:g}; the STL is clipped")
        nth0 = int(blocks[0].get("nTheta", 4 * nc))
        if nth0 != 4 * nc:
            raise ValueError(f"baseMesh: the first block (O-grid ring) needs nTheta = 4 x core n = {4 * nc}")
        self.a, self.nc, self.R = a, nc, R

        z0_, dz = self.z0, self.dz

        def to_phys(x, y, z):
            return o + x[:, None] * e1 + y[:, None] * e2 + z[:, None] * ez

        def square(s):
            """point on the square perimeter, s in [0, 4) counter-clockwise from (a, -a)"""
            s = np.mod(s, 4.0)
            k = np.minimum(np.floor(s), 3)
            f = s - k
            x = np.select([k == 0, k == 1, k == 2], [np.full_like(f, a), a - 2 * a * f, np.full_like(f, -a)], -a + 2 * a * f)
            y = np.select([k == 0, k == 1, k == 2], [-a + 2 * a * f, np.full_like(f, a), a - 2 * a * f], np.full_like(f, -a))
            return x, y

        def core_map(p):
            return to_phys(-a + 2 * a * p[:, 0] / nc, -a + 2 * a * p[:, 1] / nc, z0_ + p[:, 2] * dz)

        nr0 = int(blocks[0]["nr"])

        def ring_map(p):
            tr = p[:, 0] / nr0
            xs, ys = square(4.0 * p[:, 1] / nth0)
            th = TH0 + 2 * math.pi * p[:, 1] / nth0
            x = (1 - tr) * xs + tr * R[0] * np.cos(th)
            y = (1 - tr) * ys + tr * R[0] * np.sin(th)
            return to_phys(x, y, z0_ + p[:, 2] * dz)

        def annulus_map(ri, ro, nr, nth, radii=None):
            nodes = np.linspace(ri, ro, nr + 1) if radii is None else np.asarray(radii, float)

            def m(p):
                r = np.interp(p[:, 0], np.arange(nr + 1), nodes)
                th = TH0 + 2 * math.pi * p[:, 1] / nth
                return to_phys(r * np.cos(th), r * np.sin(th), z0_ + p[:, 2] * dz)
            return m

        # octree depth per block: from its largest base cell to the smallest target size
        tmin = min(hmin, self.h0 / 2.0 ** Lmax_target)

        def depth(size0):
            return max(1, int(math.ceil(math.log2(max(size0 / tmin, 1.0)))) + 1)

        self.blocks = []
        sz_core = max(2 * a / nc, dz)
        self.blocks.append(MappedBlock("core", (nc, nc, nz), depth(sz_core), core_map, (False, False, False)))
        rad = [(R[0] - a) / nr0, 2 * math.pi * R[0] / nth0, dz]
        self.blocks.append(MappedBlock("ring 1", (nr0, nth0, nz), depth(max(rad)), ring_map, (False, True, False)))
        for k, b in enumerate(blocks[1:], start=1):
            nr = int(b["nr"])
            radii = None
            if "radii" in b:            # radial nodes (the automatic layout's, or by hand)
                radii = np.array(b["radii"], float)
                if len(radii) != nr + 1 or np.any(np.diff(radii) <= 0):
                    raise ValueError(f"baseMesh: block {k + 1}: radii must be nr + 1 = {nr + 1} increasing radii")
                radii[0], radii[-1] = R[k - 1], R[k]
            # default: cells about as wide around as across (refinement goes by the
            # longest edge, so elongated base cells end up over-refined)
            dr = (R[k] - R[k - 1]) / nr
            nth = int(b.get("nTheta", max(4, round(math.pi * (R[k] + R[k - 1]) / dr))))
            drmax = dr if radii is None else float(np.diff(radii).max())
            size0 = max(drmax, 2 * math.pi * R[k] / nth, dz)
            self.blocks.append(MappedBlock(f"ring {k + 1}", (nr, nth, nz), depth(size0),
                                           annulus_map(R[k - 1], R[k], nr, nth, radii), (False, True, False)))
        # smallest base cell (tolerances of the mesher)
        self.min_base_size = float(min(np.linalg.norm(mb._geometry()[1], axis=2).min() for mb in self.blocks))

        # ---------------- interfaces: (inner side, outer side, period, position(s, z))
        def s_core(p):
            x = -a + 2 * a * p[:, 0] / nc
            y = -a + 2 * a * p[:, 1] / nc
            return self._square_s(x, y)

        def s_ring_in(p):
            return 4.0 * p[:, 1] / nth0

        def s_angle(nth):
            return lambda p: 2 * math.pi * p[:, 1] / nth

        def circ(Rk):
            return lambda s, z: to_phys(Rk * np.cos(TH0 + s), Rk * np.sin(TH0 + s), z)

        def sq_pos(s, z):
            x, y = square(s)
            return to_phys(x, y, z)

        core_b, ring_b = self.blocks[0], self.blocks[1]
        self.interfaces = [([Side(core_b, 0, True, s_core, 4.0), Side(core_b, 0, False, s_core, 4.0),
                             Side(core_b, 1, True, s_core, 4.0), Side(core_b, 1, False, s_core, 4.0)],
                            Side(ring_b, 0, False, s_ring_in, 4.0), 4.0, sq_pos)]
        # self.blocks[k] (k >= 1) ends at radius R[k - 1]
        for k in range(1, len(self.blocks) - 1):
            inner, outer = self.blocks[k], self.blocks[k + 1]
            self.interfaces.append(([Side(inner, 0, True, s_angle(inner.n[1]), 2 * math.pi)],
                                    Side(outer, 0, False, s_angle(outer.n[1]), 2 * math.pi),
                                    2 * math.pi, circ(R[k - 1])))
        self.max_ratio = float(spec.get("maxSizeRatio", 2.5))
        log(f"cylindrical base: core {nc}x{nc} (half width {a:g}), radii {R}, nz {nz} (dz {self.dz:g}), "
            f"blocks " + ", ".join(f"{b.name} {b.n.tolist()}" for b in self.blocks))

    @staticmethod
    def _square_s(x, y):
        """perimeter coordinate of points on the square (inverse of square())"""
        ax, ay = np.abs(x), np.abs(y)
        a = np.maximum(ax, ay)
        a = np.where(a > 0, a, 1.0)
        s = np.where((x >= a * (1 - 1e-12)) & (y > -a * (1 - 1e-12)), (y + a) / (2 * a),
            np.where((y >= a * (1 - 1e-12)), 1 + (a - x) / (2 * a),
            np.where((x <= -a * (1 - 1e-12)), 2 + (a - y) / (2 * a), 3 + (x + a) / (2 * a))))
        return np.mod(s, 4.0)

    # ------------------------------------------------------------------ refinement
    def refine(self, surface_sources, volume_sources, log=print, keep_fn=None):
        for b in self.blocks:
            b.refine_physical(surface_sources, volume_sources, self.h0, log, keep_fn)
        self._balance_interfaces(log)

    def _leaf_size(self, b, idx):
        _, E, _ = b._geometry(idx)
        return np.linalg.norm(E, axis=2).max(1)

    def _balance_interfaces(self, log):
        for it in range(20):
            nsplit = 0
            for inner_sides, outer, S, _ in self.interfaces:
                oi = np.nonzero(outer.leaf_mask())[0]
                B = outer.leaf_rects(oi)
                sb = self._leaf_size(outer.b, oi)
                mark_o = np.zeros(len(outer.b.lev), bool)
                for side in inner_sides:
                    ii = np.nonzero(side.leaf_mask())[0]
                    if len(ii) == 0 or len(oi) == 0:
                        continue
                    A = side.leaf_rects(ii)
                    sa = self._leaf_size(side.b, ii)
                    mark_i = np.zeros(len(side.b.lev), bool)
                    tol = 1e-9 * S
                    for i, j in _overlaps(A, B, S, tol):
                        if sa[i] > self.max_ratio * sb[j] and side.b.lev[ii[i]] < side.b.Lmax:
                            mark_i[ii[i]] = True
                        elif sb[j] > self.max_ratio * sa[i] and outer.b.lev[oi[j]] < outer.b.Lmax:
                            mark_o[oi[j]] = True
                    nsplit += side.b.split(mark_i)
                    if mark_i.any():
                        side.b.balance(lambda s: None)
                nsplit += outer.b.split(mark_o)
                if mark_o.any():
                    outer.b.balance(lambda s: None)
            if nsplit == 0:
                break
            log(f"  block interfaces, pass {it + 1}: split {nsplit} cells")

    @property
    def lev(self):
        return np.concatenate([b.lev for b in self.blocks])

    # ------------------------------------------------------------------ cells
    def build_cells(self):
        Ms = [b.build_cells() for b in self.blocks]
        voff = np.cumsum([0] + [len(M["points"]) for M in Ms])
        coff = np.cumsum([0] + [len(M["cell_size"]) for M in Ms])
        koff = np.cumsum([0] + [M["nface_keys"] for M in Ms])
        P = np.concatenate([M["points"] for M in Ms])
        nP0 = len(P)
        rep = np.arange(nP0)
        extra_pts = []

        # face occurrences of all blocks, global ids (CSR)
        G_cell = np.concatenate([M["face_cell"] + coff[bi] for bi, M in enumerate(Ms)])
        G_key = np.concatenate([M["face_key"] + koff[bi] for bi, M in enumerate(Ms)])
        G_n = np.concatenate([np.diff(M["fptr"]) for M in Ms])
        G_v = np.concatenate([M["fverts"] + voff[bi] for bi, M in enumerate(Ms)])
        G_ptr = np.concatenate([[0], np.cumsum(G_n)])
        next_key = int(koff[-1])
        remove = np.zeros(len(G_cell), bool)
        on_if = np.zeros(nP0, bool)               # vertices on a block interface
        new_occ = []                  # (cell, key, poly)
        insert = {}                   # (a, b) -> vertices strictly between, from a to b

        bidx = {id(b): i for i, b in enumerate(self.blocks)}
        # occurrence ranges per block
        occ_off = np.cumsum([0] + [len(M["face_cell"]) for M in Ms])

        for inner_sides, outer, S, pos_fn in self.interfaces:
            tol = 1e-9 * S
            zq = 1e-9 * max(self.dz, 1e-300)
            V = {}                                     # (sk, zk) -> (vid, s, z)

            def vkey(s, z):
                return (int(round((s % S) / tol)) % int(round(S / tol)), int(round(z / zq)))

            faces = []                                 # (side 0 inner / 1 outer, occ index, s list, z list)
            for sidx, side in [(0, sd) for sd in inner_sides] + [(1, outer)]:
                bi = bidx[id(side.b)]
                M = Ms[bi]
                lat = M["points_lattice"]
                onv = side.vertex_mask(lat)
                par = lat / float(1 << side.b.Lmax)
                sv = np.full(len(lat), np.nan)
                sv[onv] = side.s_of(par[onv])
                zv = self.z0 + par[:, 2] * self.dz
                fp = M["fptr"]
                fv = M["fverts"]
                allon = np.minimum.reduceat(onv[fv].astype(np.int8), fp[:-1]).astype(bool)
                on_if[np.nonzero(onv)[0] + voff[bi]] = True
                for f in np.nonzero(allon)[0].tolist():
                    vl = fv[fp[f]:fp[f + 1]]
                    gv = (vl + voff[bi]).tolist()
                    faces.append((sidx, int(occ_off[bi] + f), gv, sv[vl].tolist(), zv[vl].tolist()))
                    for g, s_, z_ in zip(gv, sv[vl].tolist(), zv[vl].tolist()):
                        k = vkey(s_, z_)
                        if k in V:
                            rep[g] = V[k][0]
                        else:
                            V[k] = (g, s_ % S, z_)
            if not faces:
                continue
            # rectangles
            rects = []
            for sidx, oi, gv, sl, zl in faces:
                s = np.array(sl)
                if s.max() - s.min() > 0.5 * S:
                    s = np.where(s < 0.5 * S, s + S, s)
                rects.append((s.min(), s.max(), min(zl), max(zl)))
                remove[oi] = True
            rects = np.array(rects)
            side_of = np.array([f[0] for f in faces])
            ia = np.nonzero(side_of == 0)[0]
            ib = np.nonzero(side_of == 1)[0]
            A = tuple(rects[ia].T)
            B = tuple(rects[ib].T)
            pairs = _overlaps(A, B, S, tol)
            # overlay corners become vertices
            pieces = []
            new_sz = []
            for i, j in pairs:
                ra, rb = rects[ia[i]], rects[ib[j]]
                s0, s1 = max(ra[0], rb[0]), min(ra[1], rb[1])
                z0, z1 = max(ra[2], rb[2]), min(ra[3], rb[3])
                for s_, z_ in ((s0, z0), (s1, z0), (s1, z1), (s0, z1)):
                    k = vkey(s_, z_)
                    if k not in V:
                        gid = nP0 + len(extra_pts) + len(new_sz)
                        new_sz.append((s_ % S, z_))
                        V[k] = (gid, s_ % S, z_)
                pieces.append((ia[i], ib[j], s0, s1, z0, z1))
            if new_sz:
                sz_ = np.array(new_sz)
                extra_pts.extend(pos_fn(sz_[:, 0], sz_[:, 1]))
            # lookups of interface vertices along constant z / constant s lines
            by_z, by_s = {}, {}
            for (sk, zk), (gid, s_, z_) in V.items():
                by_z.setdefault(zk, []).append((s_, gid))
                by_s.setdefault(sk, []).append((z_, gid))
            # sorted coordinate arrays for bisection
            for d in (by_z, by_s):
                for k in d:
                    d[k].sort()
                    d[k] = (np.array([x for x, _ in d[k]]), [g for _, g in d[k]])

            def _range(arr, gids, lo, hi):
                i = int(np.searchsorted(arr, lo + tol, side="right"))
                j = int(np.searchsorted(arr, hi - tol, side="left"))
                return gids[i:j]

            def between_s(z, s0, s1):
                """vertices at height z with s0 < s < s1 (s1 may exceed S), ascending"""
                e = by_z.get(int(round(z / zq)))
                if e is None:
                    return []
                arr, gids = e
                if s1 <= S + tol:
                    return _range(arr, gids, s0, s1)
                return _range(arr, gids, s0, S + tol) + _range(arr, gids, -tol, s1 - S)

            def between_z(s, z0, z1):
                e = by_s.get(int(round((s % S) / tol)) % int(round(S / tol)))
                if e is None:
                    return []
                arr, gids = e
                i = int(np.searchsorted(arr, z0 + zq, side="right"))
                j = int(np.searchsorted(arr, z1 - zq, side="left"))
                return gids[i:j]

            def seg(p, q):
                """vertices strictly inside the interface segment p -> q (each (s, z))"""
                (sp, zp), (sq, zq_) = p, q
                if abs(zp - zq_) <= zq:             # along s
                    if sq < sp:
                        return between_s(zp, sq, sp)[::-1]
                    return between_s(zp, sp, sq)
                if sq < sp:                         # along z (ignore wrap: same s)
                    sp, sq = sq, sp
                ins = between_z(sp, min(zp, zq_), max(zp, zq_))
                return ins if zp < zq_ else ins[::-1]

            # edges of the interface faces of both sides: inserted vertices
            for sidx, oi, gv, sl, zl in faces:
                m = len(gv)
                s = np.array(sl)
                if s.max() - s.min() > 0.5 * S:
                    s = np.where(s < 0.5 * S, s + S, s)
                for k in range(m):
                    a_, b_ = gv[k], gv[(k + 1) % m]
                    ins = seg((s[k], zl[k]), (s[(k + 1) % m], zl[(k + 1) % m]))
                    if ins:
                        ra, rb = int(rep[a_]), int(rep[b_])
                        insert[(ra, rb)] = ins
                        insert[(rb, ra)] = ins[::-1]
            # overlay pieces: shared faces (outward from the inner cell: s, then z)
            for fa, fb, s0, s1, z0, z1 in pieces:
                c = [V[vkey(s0, z0)][0], V[vkey(s1, z0)][0], V[vkey(s1, z1)][0], V[vkey(s0, z1)][0]]
                poly = [c[0]] + between_s(z0, s0, s1) + [c[1]] + between_z(s1, z0, z1) + \
                       [c[2]] + between_s(z1, s0, s1)[::-1] + [c[3]] + between_z(s0, z0, z1)[::-1]
                key = next_key
                next_key += 1
                ca = int(G_cell[faces[fa][1]])
                cb = int(G_cell[faces[fb][1]])
                new_occ.append((ca, key, poly))
                new_occ.append((cb, key, poly[::-1]))

        # ---------------- assemble: drop interface faces, insert edge vertices, merge
        nP = nP0 + len(extra_pts)
        rep_all = np.concatenate([rep, np.arange(nP0, nP)])
        # resolve chains of representatives
        for _ in range(4):
            rep_all = rep_all[rep_all]
        # faces without interface vertices pass through as arrays; the others get the
        # inserted vertices (Python, few)
        touch = np.maximum.reduceat(on_if[G_v].astype(np.int8), G_ptr[:-1]).astype(bool)
        plain = np.nonzero(~touch & ~remove)[0]
        cells, keys, polys = [], [], []
        for k in np.nonzero(touch & ~remove)[0].tolist():
            poly = [int(rep_all[v]) for v in G_v[G_ptr[k]:G_ptr[k + 1]]]
            if insert:
                out = []
                m = len(poly)
                for i in range(m):
                    a_, b_ = poly[i], poly[(i + 1) % m]
                    out.append(a_)
                    out.extend(insert.get((a_, b_), ()))
                poly = out
            cells.append(int(G_cell[k])); keys.append(int(G_key[k])); polys.append(poly)
        for c, k, poly in new_occ:
            cells.append(c); keys.append(k); polys.append([int(rep_all[v]) for v in poly])
        # merge the plain faces (arrays) and the processed ones, ordered by cell
        pn_ = np.array([len(p) for p in polys], np.int64)
        pv_ = np.fromiter((v for p in polys for v in p), np.int64, count=int(pn_.sum()))
        sel = np.repeat(plain, G_n[plain])
        off = np.arange(len(sel)) - np.repeat(np.cumsum(G_n[plain]) - G_n[plain], G_n[plain])
        plain_v = rep_all[G_v[G_ptr[sel] + off]] if len(sel) else np.zeros(0, np.int64)
        all_cell = np.concatenate([G_cell[plain], np.array(cells, np.int64)])
        all_key = np.concatenate([G_key[plain], np.array(keys, np.int64)])
        all_n = np.concatenate([G_n[plain], pn_])
        all_v = np.concatenate([plain_v, pv_])
        aptr = np.concatenate([[0], np.cumsum(all_n)])
        order = np.argsort(all_cell, kind="stable")
        cells = all_cell[order]
        keys = all_key[order]
        fn = all_n[order]
        idx = np.repeat(aptr[order], fn) + (np.arange(int(fn.sum())) - np.repeat(np.cumsum(fn) - fn, fn))
        fverts = all_v[idx]
        _, keys = np.unique(keys, return_inverse=True)
        Pall = np.concatenate([P] + ([np.array(extra_pts)] if extra_pts else []))
        used = np.zeros(nP, bool)
        used[fverts] = True
        remap = np.cumsum(used) - 1
        nleaf = int(coff[-1])
        hanging = np.zeros(nP, bool)
        hang_par = np.full((nP, 4), -1, np.int64)
        hang_size = np.zeros(nP)
        corner = []
        for bi, M in enumerate(Ms):
            g = np.arange(len(M["points"])) + voff[bi]
            r = rep_all[g]
            h = M["hanging"]
            hanging[r[h]] = True
            pp = M["hang_par"][h]
            hang_par[r[h]] = np.where(pp >= 0, rep_all[np.maximum(pp, 0) + voff[bi]], -1)
            hang_size[r[h]] = M["hang_size"][h] / float(1 << self.blocks[bi].Lmax)
            corner.append(rep_all[M["corner_vid"] + voff[bi]])
        keep = np.nonzero(used)[0]
        hp = hang_par[keep]
        hp = np.where(hp >= 0, remap[np.maximum(hp, 0)], -1)
        cell_size = np.concatenate([M["cell_size"] for M in Ms])
        return dict(points=Pall[keep], hanging=hanging[keep], hang_par=hp, hang_size=hang_size[keep],
                    face_cell=cells, face_key=keys.ravel(), fptr=np.concatenate([[0], np.cumsum(fn)]),
                    fverts=remap[fverts], cell_fptr=np.searchsorted(cells, np.arange(nleaf + 1)),
                    nface_keys=int(keys.max()) + 1, cell_size=cell_size,
                    cell_volume=np.concatenate([M["cell_volume"] for M in Ms]),
                    cell_level=np.concatenate([M["cell_level"] for M in Ms]),
                    corner_vid=remap[np.concatenate(corner)])
