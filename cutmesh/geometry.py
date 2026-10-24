"""Geometry: STL surfaces and analytic cylinders as signed-distance level sets.

Sign convention everywhere: phi < 0 inside the closed surface, phi > 0 outside.
"""
import os
import struct
import numpy as np
from scipy.spatial import cKDTree

# Rust extension (rust/, built by tools/build_rust.py): exact closest point, inside test
# and signed distance on a BVH, in parallel.  CUTMESH_NO_RUST=1 forces the Python code.
_core = None
if not os.environ.get("CUTMESH_NO_RUST"):
    try:
        from . import _core
    except ImportError:
        _core = None


def rust_available():
    return _core is not None


def paused_gc(fn):
    """Run fn with Python's cyclic garbage collector paused.  The mesher builds millions
    of small lists (faces of cut cells) and hardly any reference cycles; each GC pass
    rescans all of them, which took 20-80% of the time of some stages."""
    import functools
    import gc

    @functools.wraps(fn)
    def wrapper(*a, **k):
        was = gc.isenabled()
        gc.disable()
        try:
            return fn(*a, **k)
        finally:
            if was:
                gc.enable()
    return wrapper


# ----------------------------------------------------------------------------
# STL I/O
# ----------------------------------------------------------------------------
def read_stl(path):
    """Return list of (solid_name, tris[n,3,3]). Supports ASCII (multi-solid) and binary."""
    with open(path, "rb") as f:
        data = f.read()
    is_binary = True
    if data[:5].lower() == b"solid":
        if len(data) >= 84:
            n = struct.unpack("<I", data[80:84])[0]
            if 84 + 50 * n == len(data):
                is_binary = True
            else:
                is_binary = False
        else:
            is_binary = False
    if is_binary:
        n = struct.unpack("<I", data[80:84])[0]
        rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([
            ("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        name = data[:80].split(b"\0")[0].decode(errors="ignore").replace("solid", "").strip()
        return [(name or "walls", rec["v"].astype(float))]
    solids = []
    name, verts = None, []
    for line in data.decode(errors="ignore").splitlines():
        t = line.strip().split()
        if not t:
            continue
        key = t[0].lower()
        if key == "solid":
            name = t[1] if len(t) > 1 else "walls"
            verts = []
        elif key == "vertex":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif key == "endsolid":
            solids.append((name, np.array(verts, float).reshape(-1, 3, 3)))
            name, verts = None, []
    if name is not None and verts:
        solids.append((name, np.array(verts, float).reshape(-1, 3, 3)))
    return solids


def write_stl_ascii(path, solids):
    with open(path, "w") as f:
        for name, tris in solids:
            f.write(f"solid {name}\n")
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.where(ln > 0, ln, 1)
            for t, nn in zip(tris, n):
                f.write(f" facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}\n  outer loop\n")
                for v in t:
                    f.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")


# ----------------------------------------------------------------------------
# point-triangle closest point (vectorised, Ericson RTCD 5.1.5)
# ----------------------------------------------------------------------------
def closest_point_triangles(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        den = np.where(np.abs(den) > 0, den, 1.0)
        v = vb / den
        w = vc / den
        res = a + ab * v[:, None] + ac * w[:, None]
        done = np.zeros(len(p), bool)

        def put(mask, val):
            nonlocal res, done
            m = mask & ~done
            res = np.where(m[:, None], val, res)
            done |= m

        put((d1 <= 0) & (d2 <= 0), a)
        put((d3 >= 0) & (d4 <= d3), b)
        t = d1 / np.where(d1 - d3 != 0, d1 - d3, 1)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t[:, None])
        put((d6 >= 0) & (d5 <= d6), c)
        t = d2 / np.where(d2 - d6 != 0, d2 - d6, 1)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t[:, None])
        e = (d4 - d3) + (d5 - d6)
        t = (d4 - d3) / np.where(e != 0, e, 1)
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + (c - b) * t[:, None])
    return res


def subdivide_triangles(tris, maxedge):
    """Longest-edge bisection until every edge <= maxedge. Returns (tris, parent)."""
    T = np.asarray(tris, float)
    parent = np.arange(len(T))
    doneT, doneP = [], []
    while len(T):
        e = np.stack([((T[:, 1] - T[:, 0]) ** 2).sum(1),
                      ((T[:, 2] - T[:, 1]) ** 2).sum(1),
                      ((T[:, 0] - T[:, 2]) ** 2).sum(1)], 1)
        k = e.argmax(1)
        big = e[np.arange(len(T)), k] > maxedge * maxedge
        doneT.append(T[~big]); doneP.append(parent[~big])
        if not big.any():
            break
        B = T[big]; kb = k[big]
        r = np.stack([kb, (kb + 1) % 3, (kb + 2) % 3], 1)
        B = B[np.arange(len(B))[:, None], r]
        m = 0.5 * (B[:, 0] + B[:, 1])
        T = np.concatenate([np.stack([B[:, 0], m, B[:, 2]], 1), np.stack([m, B[:, 1], B[:, 2]], 1)])
        parent = np.concatenate([parent[big], parent[big]])
    return np.concatenate(doneT), np.concatenate(doneP)


def _sample_triangles(tris, spacing):
    """Points on triangles (vertices + centroids of a subdivision). Returns (pts, tri_index)."""
    T, par = subdivide_triangles(tris, spacing)
    pts = np.concatenate([T.reshape(-1, 3), T.mean(1)])
    ids = np.concatenate([np.repeat(par, 3), par])
    return pts, ids


# ----------------------------------------------------------------------------
# Triangulated closed surface
# ----------------------------------------------------------------------------
class Surface:
    """Closed (watertight) triangulated surface, possibly with several solids/regions."""

    def __init__(self, solids):
        self.names = [s[0] for s in solids]
        self.tris = np.concatenate([s[1] for s in solids])
        self.region = np.concatenate([np.full(len(s[1]), i) for i, s in enumerate(solids)])
        self.bbmin = self.tris.reshape(-1, 3).min(0)
        self.bbmax = self.tris.reshape(-1, 3).max(0)
        self.L = float(np.linalg.norm(self.bbmax - self.bbmin))
        self._kd = None
        self._kd_spacing = None
        self._rs = _core.TriMesh(np.ascontiguousarray(self.tris)) if _core else None
        self._nb = None
        if self._rs is None:
            self._build_ray_bins()

    @classmethod
    def from_file(cls, path):
        return cls(read_stl(path))

    # ---------------- sampling / nearest search ----------------
    def build_search(self, spacing):
        """KD-tree over centroids of a subdivision of the STL (edges <= spacing).
        Closest-point queries are exact: candidates are all sub-triangles whose
        centroid lies within (best distance + max centroid radius)."""
        T, par = subdivide_triangles(self.tris, spacing)
        self._sub = T
        self._sub_parent = par
        c = T.mean(1)
        self._rmax = float(np.linalg.norm(T - c[:, None], axis=2).max())
        self._kd = cKDTree(c)
        self._kd_spacing = spacing

    def samples(self, spacing, regions=None):
        sel = np.arange(len(self.tris)) if regions is None else np.nonzero(np.isin(self.region, regions))[0]
        pts, ids = _sample_triangles(self.tris[sel], spacing)
        return pts, sel[ids]

    def _exact(self, p, cand):
        n, k = cand.shape
        P = np.repeat(p, k, axis=0)
        T = self._sub[cand.ravel()]
        cp = closest_point_triangles(P, T[:, 0], T[:, 1], T[:, 2])
        d = np.linalg.norm(cp - P, axis=1).reshape(n, k)
        j = d.argmin(1)
        r = np.arange(n)
        return cp.reshape(n, k, 3)[r, j], d[r, j], cand[r, j]

    def closest(self, p, k=8, chunk=50000, exact_within=np.inf):
        """Closest point on the STL, guaranteed exact where dist <= exact_within
        (elsewhere error <= sub-triangle radius, usually exact as well).
        Returns (cp, dist, original_triangle_id)."""
        p = np.atleast_2d(p)
        if self._rs is not None:                 # exact everywhere
            return self._rs.closest(np.ascontiguousarray(p, float))
        n = len(p)
        cp = np.zeros((n, 3)); d = np.zeros(n); sid = np.zeros(n, np.int64)
        if n == 0:
            return cp, d, sid
        k = min(k, len(self._sub))
        for s0 in range(0, n, chunk):
            q = p[s0:s0 + chunk]
            dk, idx = self._kd.query(q, k=k, workers=-1)
            idx = idx.reshape(len(q), -1); dk = dk.reshape(len(q), -1)
            c1, d1, s1 = self._exact(q, idx)
            if k < len(self._sub):
                need = np.nonzero((dk[:, -1] <= d1 + self._rmax) & (d1 <= exact_within))[0]
            else:
                need = np.zeros(0, int)
            if len(need):
                rad = d1[need] + self._rmax
                cnt = self._kd.query_ball_point(q[need], rad, workers=-1, return_length=True)
                cum = np.cumsum(cnt)
                b0 = 0
                while b0 < len(need):
                    b1 = int(np.searchsorted(cum, (cum[b0 - 1] if b0 else 0) + 500_000, side="right"))
                    b1 = max(b1, b0 + 1)
                    sub = need[b0:b1]
                    lists = self._kd.query_ball_point(q[sub], rad[b0:b1], workers=-1)
                    ln = np.array([len(l) for l in lists])
                    flat = np.concatenate([np.asarray(l, np.int64) for l in lists])
                    rp = np.repeat(np.arange(len(sub)), ln)
                    T = self._sub[flat]
                    P = q[sub][rp]
                    cc = closest_point_triangles(P, T[:, 0], T[:, 1], T[:, 2])
                    dd = np.linalg.norm(cc - P, axis=1)
                    o = np.lexsort((dd, rp))
                    best = o[np.searchsorted(rp[o], np.arange(len(sub)))]
                    c1[sub] = cc[best]; d1[sub] = dd[best]; s1[sub] = flat[best]
                    b0 = b1
            cp[s0:s0 + chunk] = c1; d[s0:s0 + chunk] = d1; sid[s0:s0 + chunk] = s1
        return cp, d, self._sub_parent[sid]

    def segment_hits(self, a, b, samples=64):
        """Crossings of the segments a[i] -> b[i] with the surface, sorted along each:
        (segment index, parameter t, triangle id).  Exact with the Rust extension; the
        fallback samples the segments (crossings closer than 1/samples merge)."""
        a = np.ascontiguousarray(np.atleast_2d(a), float)
        b = np.ascontiguousarray(np.atleast_2d(b), float)
        if self._rs is not None:
            return self._rs.segment_hits(a, b)
        ts = (np.arange(samples + 1)) / samples
        Q = a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]
        ins = self.inside(Q.reshape(-1, 3)).reshape(len(a), samples + 1)
        sg, k = np.nonzero(ins[:, 1:] != ins[:, :-1])
        mid = 0.5 * (Q[sg, k] + Q[sg, k + 1])
        _, _, tid = self.closest(mid)
        return sg.astype(np.int64), (k + 0.5) / samples, tid.astype(np.int64)

    def approx_distance(self, p, upper=np.inf):
        d, _ = self._kd.query(p, distance_upper_bound=upper, workers=-1)
        return d

    # ---------------- inside / outside by ray parity ----------------
    def _build_ray_bins(self):
        t = self.tris
        self._nb = int(np.clip(np.sqrt(len(t)) * 0.7, 8, 512))
        lo, hi = self.bbmin[1:], self.bbmax[1:]
        self._blo = lo - 1e-9 * self.L
        self._bsz = (hi - lo + 2e-9 * self.L) / self._nb
        ymin = t[:, :, 1].min(1); ymax = t[:, :, 1].max(1)
        zmin = t[:, :, 2].min(1); zmax = t[:, :, 2].max(1)
        i0 = np.clip(((ymin - self._blo[0]) / self._bsz[0]).astype(int), 0, self._nb - 1)
        i1 = np.clip(((ymax - self._blo[0]) / self._bsz[0]).astype(int), 0, self._nb - 1)
        j0 = np.clip(((zmin - self._blo[1]) / self._bsz[1]).astype(int), 0, self._nb - 1)
        j1 = np.clip(((zmax - self._blo[1]) / self._bsz[1]).astype(int), 0, self._nb - 1)
        bins, tl = [], []
        for ti in range(len(t)):
            for i in range(i0[ti], i1[ti] + 1):
                b = i * self._nb
                for j in range(j0[ti], j1[ti] + 1):
                    bins.append(b + j); tl.append(ti)
        bins = np.array(bins); tl = np.array(tl)
        o = np.argsort(bins, kind="stable")
        self._btri = tl[o]
        self._bstart = np.searchsorted(bins[o], np.arange(self._nb * self._nb + 1))

    def inside(self, p, chunk=100000):
        """Parity of +x ray crossings. True where p is inside the closed surface."""
        p = np.atleast_2d(p)
        if self._rs is not None:                 # same ray, perturbation and hit test
            return self._rs.inside(np.ascontiguousarray(p, float))
        if self._nb is None:
            self._build_ray_bins()
        res = np.zeros(len(p), bool)
        pert = self.L * np.array([0.0, 3.14159265e-7, 2.71828183e-7])
        for s in range(0, len(p), chunk):
            q = p[s:s + chunk] + pert
            ib = ((q[:, 1] - self._blo[0]) / self._bsz[0]).astype(int)
            jb = ((q[:, 2] - self._blo[1]) / self._bsz[1]).astype(int)
            ok = (ib >= 0) & (ib < self._nb) & (jb >= 0) & (jb < self._nb) & \
                 (q[:, 0] <= self.bbmax[0])
            pid = np.nonzero(ok)[0]
            b = ib[pid] * self._nb + jb[pid]
            cnt = self._bstart[b + 1] - self._bstart[b]
            rp = np.repeat(pid, cnt)
            off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            rt = self._btri[np.repeat(self._bstart[b], cnt) + off]
            Q = q[rp]
            T = self.tris[rt]
            A, B, C = T[:, 0], T[:, 1], T[:, 2]

            def cr(u, v, w):  # 2D cross in (y,z) of (v-u) x (w-u)
                return (v[:, 1] - u[:, 1]) * (w[:, 2] - u[:, 2]) - (v[:, 2] - u[:, 2]) * (w[:, 1] - u[:, 1])
            w0 = cr(A, B, Q); w1 = cr(B, C, Q); w2 = cr(C, A, Q)
            hit = ((w0 > 0) & (w1 > 0) & (w2 > 0)) | ((w0 < 0) & (w1 < 0) & (w2 < 0))
            sw = w0 + w1 + w2
            sw = np.where(sw != 0, sw, 1)
            x = (w1 * A[:, 0] + w2 * B[:, 0] + w0 * C[:, 0]) / sw
            hit &= x > Q[:, 0]
            par = np.bincount(rp[hit], minlength=len(q)) % 2
            res[s:s + chunk] = par == 1
        return res

    # ---------------- signed distance ----------------
    def sdf(self, p, exact_mask=None, far_value=None, exact_within=np.inf):
        """Signed distance. Magnitude from closest() where exact_mask (default: all)."""
        p = np.atleast_2d(p)
        if self._rs is not None and exact_mask is None:
            return self._rs.sdf(np.ascontiguousarray(p, float))
        sign = np.where(self.inside(p), -1.0, 1.0)
        if exact_mask is None:
            exact_mask = np.ones(len(p), bool)
        d = np.full(len(p), far_value if far_value is not None else self.L)
        if exact_mask.any():
            _, dd, _ = self.closest(p[exact_mask], exact_within=exact_within)
            d[exact_mask] = dd
        return sign * d

    def feature_edges(self, angle_deg, concave_only=False):
        """Feature edges (dihedral angle > angle, open edges, region boundaries) as segments.
        concave_only: only the sharp manifold edges where the solid juts into the domain (the
        surface is concave seen from inside: a re-entrant edge of the fluid).
        Independent of the triangle winding: the angle is taken between the two triangles'
        in-plane directions away from the edge, and the side of the solid by inside()."""
        tol = 1e-9 * self.L
        v = self.tris.reshape(-1, 3)
        key = np.round(v / tol).astype(np.int64)
        _, vid = np.unique(key, axis=0, return_inverse=True)
        vid = vid.reshape(-1, 3)
        e = np.concatenate([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]])
        et = np.tile(np.arange(len(vid)), 3)
        # unit direction in the triangle's plane from its edge towards its third vertex
        T = self.tris
        ea = np.concatenate([T[:, 0], T[:, 1], T[:, 2]])
        eb = np.concatenate([T[:, 1], T[:, 2], T[:, 0]])
        ec = np.concatenate([T[:, 2], T[:, 0], T[:, 1]])
        t = eb - ea
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
        dv = ec - ea
        dv -= (dv * t).sum(1)[:, None] * t
        hgt = np.linalg.norm(dv, axis=1)
        dv /= np.maximum(hgt, 1e-300)[:, None]
        es = np.sort(e, 1)
        _, inv, cnt = np.unique(es, axis=0, return_inverse=True, return_counts=True)
        o = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[o], np.arange(len(cnt)))
        sel = o[starts]
        feat = np.zeros(len(cnt), bool)
        two = cnt == 2
        if not concave_only:
            feat[~two] = True                      # open / non-manifold edges
        g = np.nonzero(two)[0]
        i1, i2 = o[starts[g]], o[starts[g] + 1]
        t1, t2 = et[i1], et[i2]
        # wedge angle between the triangles < 180 - angle (normals of consistently wound
        # triangles: dot(n1, n2) = -dot(d1, d2))
        sharp = (dv[i1] * dv[i2]).sum(1) > np.cos(np.radians(180.0 - angle_deg))
        if concave_only:
            # the wedge (< 180 deg) is solid: probe a point just inside it
            k = np.nonzero(sharp)[0]
            if len(k):
                j1, j2 = i1[k], i2[k]
                m = 0.5 * (ea[j1] + eb[j1])
                w = dv[j1] + dv[j2]
                w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-300)
                # well inside the wedge: far beyond the ray perturbation of inside()
                r = 0.25 * np.minimum(hgt[j1], hgt[j2])
                solid = ~self.inside(m + r[:, None] * w)
                feat[g[k[solid]]] = True
        else:
            feat[g[sharp | (self.region[t1] != self.region[t2])]] = True
        if not feat.any():
            return np.zeros((0, 2, 3))
        vpos = np.zeros((vid.max() + 1, 3))
        vpos[vid.ravel()] = v
        return vpos[e[sel[feat]]]


# ----------------------------------------------------------------------------
# Analytic finite cylinder
# ----------------------------------------------------------------------------
class Cylinder:
    def __init__(self, point1, point2, radius):
        self.p1 = np.asarray(point1, float)
        self.p2 = np.asarray(point2, float)
        self.R = float(radius)
        ax = self.p2 - self.p1
        self.H = float(np.linalg.norm(ax))
        self.a = ax / self.H
        t = np.array([1.0, 0, 0]) if abs(self.a[0]) < 0.9 else np.array([0, 1.0, 0])
        self.e1 = np.cross(self.a, t); self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.a, self.e1)
        self.L = float(np.hypot(2 * self.R, self.H))

    def _local(self, p):
        d = p - self.p1
        t = d @ self.a
        rv = d - t[:, None] * self.a
        r = np.linalg.norm(rv, axis=1)
        safe = r > 1e-14 * self.L
        dirv = np.where(safe[:, None], rv / np.where(safe, r, 1)[:, None], self.e1)
        return t, r, dirv

    def sdf(self, p, exact_mask=None, far_value=None):
        p = np.atleast_2d(p)
        t, r, _ = self._local(p)
        dr = r - self.R
        da = np.maximum(-t, t - self.H)
        return np.minimum(np.maximum(dr, da), 0) + np.hypot(np.maximum(dr, 0), np.maximum(da, 0))

    def inside(self, p):
        return self.sdf(p) < 0

    def closest(self, p):
        p = np.atleast_2d(p)
        t, r, dv = self._local(p)
        dr = r - self.R
        da = np.maximum(-t, t - self.H)
        outside = (dr > 0) | (da > 0)
        tc = np.clip(t, 0, self.H)
        rc = np.minimum(r, self.R)
        cp_out = self.p1 + tc[:, None] * self.a + rc[:, None] * dv
        # inside: move to the nearest of side / bottom / top
        ds, db, dt = self.R - r, t, self.H - t
        k = np.argmin(np.stack([ds, db, dt], 1), 1)
        cp_in = np.where((k == 0)[:, None], self.p1 + t[:, None] * self.a + self.R * dv,
                         np.where((k == 1)[:, None], p - t[:, None] * self.a,
                                  p + (self.H - t)[:, None] * self.a))
        cp = np.where(outside[:, None], cp_out, cp_in)
        return cp, np.linalg.norm(cp - p, axis=1), np.zeros(len(p), int)

    def samples(self, spacing, regions=None):
        R, H = self.R, self.H
        nth = max(12, int(np.ceil(2 * np.pi * R / spacing)))
        nz = max(1, int(np.ceil(H / spacing)))
        th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        z = np.linspace(0, H, nz + 1)
        T, Z = np.meshgrid(th, z)
        side = (self.p1 + Z.ravel()[:, None] * self.a
                + R * (np.cos(T.ravel())[:, None] * self.e1 + np.sin(T.ravel())[:, None] * self.e2))
        caps = []
        nr = max(1, int(np.ceil(R / spacing)))
        for rr in np.linspace(0, R, nr + 1)[:-1]:
            m = max(1, int(np.ceil(2 * np.pi * rr / spacing)))
            tt = np.linspace(0, 2 * np.pi, m, endpoint=False)
            ring = rr * (np.cos(tt)[:, None] * self.e1 + np.sin(tt)[:, None] * self.e2)
            caps.append(self.p1 + ring)
            caps.append(self.p2 + ring)
        pts = np.concatenate([side] + caps)
        return pts, np.zeros(len(pts), int)

    def feature_points(self, spacing):
        nth = max(12, int(np.ceil(2 * np.pi * self.R / spacing)))
        th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        ring = self.R * (np.cos(th)[:, None] * self.e1 + np.sin(th)[:, None] * self.e2)
        return np.concatenate([self.p1 + ring, self.p2 + ring])


class SurfaceZone:
    """Rotating zone given by a closed STL."""

    def __init__(self, surface):
        self.s = surface
        self.L = surface.L

    def sdf(self, p, exact_mask=None, far_value=None, exact_within=np.inf):
        return self.s.sdf(p, exact_mask, far_value, exact_within)

    def inside(self, p):
        return self.s.inside(p)

    def refine_sdf(self, p):
        """cheap signed distance for refinement decisions (sub-triangle centroid distance)"""
        return np.where(self.s.inside(p), -1.0, 1.0) * self.s.approx_distance(p)

    def closest(self, p):
        return self.s.closest(p)

    def samples(self, spacing, regions=None):
        return self.s.samples(spacing)

    def feature_points(self, spacing):
        seg = self.s.feature_edges(30.0)
        if len(seg) == 0:
            return np.zeros((0, 3))
        return sample_segments(seg, spacing)


def sample_segments(seg, spacing):
    if len(seg) == 0:
        return np.zeros((0, 3))
    ln = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    n = np.maximum(1, np.ceil(ln / spacing).astype(int))
    out = []
    for k in np.unique(n):
        s = seg[n == k]
        t = np.linspace(0, 1, k + 1)
        out.append((s[:, None, 0] * (1 - t)[None, :, None] + s[:, None, 1] * t[None, :, None]).reshape(-1, 3))
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Feature edges (polyline network) with closest-point queries and corners
# ----------------------------------------------------------------------------
class FeatureSet:
    def __init__(self, segments, corner_angle_deg=45.0, L=1.0):
        seg = np.asarray(segments, float).reshape(-1, 2, 3)
        ln = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        self.seg = seg[ln > 0]
        self.L = L
        if len(self.seg) == 0:
            self.corners = np.zeros((0, 3)); self._kd = None
            return
        # corners: valence != 2, or sharp turn between the two incident segments
        tol = 1e-9 * L
        ends = self.seg.reshape(-1, 3)
        key = np.round(ends / tol).astype(np.int64)
        _, vid, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        vid = vid.ravel().reshape(-1, 2)
        corners = []
        d = self.seg[:, 1] - self.seg[:, 0]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        inc = {}
        for s, (a, b) in enumerate(vid.tolist()):
            inc.setdefault(a, []).append((s, 1.0))   # leaving a
            inc.setdefault(b, []).append((s, -1.0))  # arriving b
        cosmax = np.cos(np.radians(corner_angle_deg))
        for v, lst in inc.items():
            if len(lst) != 2:
                corners.append(v); continue
            (s1, g1), (s2, g2) = lst
            # directions pointing away from v
            if np.dot(d[s1] * g1, d[s2] * g2) > -cosmax:
                corners.append(v)
        pos = np.zeros((vid.max() + 1, 3)); pos[vid.ravel()] = ends
        self.corners = pos[np.array(corners, int)] if corners else np.zeros((0, 3))
        mid = 0.5 * (self.seg[:, 0] + self.seg[:, 1])
        self._half = 0.5 * float(np.linalg.norm(self.seg[:, 1] - self.seg[:, 0], axis=1).max())
        self._kd = cKDTree(mid)
        self._rs = _core.SegSet(np.ascontiguousarray(self.seg)) if _core else None

    def _seg_closest(self, p, idx):
        """closest points of p[i] on segments idx[i, :] -> (cp[n, k, 3], dist[n, k])"""
        A = self.seg[idx, 0]; B = self.seg[idx, 1]
        AB = B - A
        t = np.clip(np.einsum("nkj,nkj->nk", p[:, None] - A, AB) /
                    np.maximum(np.einsum("nkj,nkj->nk", AB, AB), 1e-300), 0, 1)
        cp = A + t[..., None] * AB
        return cp, np.linalg.norm(cp - p[:, None], axis=2)

    def closest(self, p):
        """Exact closest point on the segment set. Returns (cp, dist)."""
        p = np.atleast_2d(p)
        if self._kd is None or len(p) == 0:
            return np.zeros((len(p), 3)), np.full(len(p), np.inf)
        if self._rs is not None:
            return self._rs.closest(np.ascontiguousarray(p, float))
        k = min(8, len(self.seg))
        dk, idx = self._kd.query(p, k=k, workers=-1)
        idx = idx.reshape(len(p), -1); dk = dk.reshape(len(p), -1)
        cp, d = self._seg_closest(p, idx)
        j = d.argmin(1); r = np.arange(len(p))
        cp, d = cp[r, j], d[r, j]
        # a segment whose midpoint is farther than the k-th one can still be closer:
        # check all midpoints within (best distance + half the longest segment)
        if k < len(self.seg):
            need = np.nonzero(dk[:, -1] <= d + self._half)[0]
            if len(need):
                lists = self._kd.query_ball_point(p[need], d[need] + self._half, workers=-1)
                for i, l in zip(need.tolist(), lists):
                    c2, d2 = self._seg_closest(p[i:i + 1], np.asarray(l, np.int64)[None])
                    m = int(d2[0].argmin())
                    if d2[0, m] < d[i]:
                        cp[i], d[i] = c2[0, m], d2[0, m]
        return cp, d


def cylinder_features(cyl, spacing):
    n = max(64, int(np.ceil(2 * np.pi * cyl.R / spacing)))
    th = np.linspace(0, 2 * np.pi, n + 1)
    segs = []
    for c in (cyl.p1, cyl.p2):
        ring = c + cyl.R * (np.cos(th)[:, None] * cyl.e1 + np.sin(th)[:, None] * cyl.e2)
        segs.append(np.stack([ring[:-1], ring[1:]], 1))
    return np.concatenate(segs)
