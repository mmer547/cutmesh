"""Exact cut of the cells at convex sharp edges.

The level-set cut decides the side of each cell vertex and interpolates the crossings
along the cell edges; the surface inside a cell is whatever those values describe.  A
sharp edge that passes no vertex is chamfered, and where the vertices next to it have
been snapped onto the two walls (phi == 0) no face has a crossing at all: the sliver of
fluid between the edge and the chord through those vertices is lost, cell by cell, which
saws an edge not aligned with the octree.

Near a convex edge the fluid is locally the intersection of the half-spaces of the STL
triangles there.  These cells are cut here by clipping each face with those planes
(Sutherland-Hodgman): the points where an edge leaves the fluid and where a crease line
pierces a face come out exactly, whatever the vertex values.  Cells whose vertices are
all on the solid side but which hold a sliver of fluid are clipped too (the merging
stage joins the slivers to their neighbours).

Faces shared with cells cut by the level set must come out the same both ways; where
they would not, the neighbour is clipped too, or, if its surroundings are not convex,
the cell goes back to the level-set cut.
"""
import numpy as np

from .geometry import closest_point_triangles


class PlaneClip:
    def __init__(self, dom, pts, phi, hmin, feat, ccv=None, feature_angle=30.0):
        self.cos_feat = float(np.cos(np.radians(feature_angle)))
        self.dom = dom
        self.pts = pts
        self.phi = phi
        self.hmin = hmin
        self.feat = feat
        self.ccv = ccv
        T = dom.tris
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        # outward (towards the solid): a point just off the triangle is outside there
        out = ~dom.inside(T.mean(1) + 0.1 * hmin * n)
        n[~out] *= -1
        self.n = n
        self.d = np.einsum("ij,ij->i", n, T[:, 0])
        self.eps = 1e-7 * hmin
        self.face_cache = {}
        self.face_planes = {}
        self.zero_edge = {}          # the level-set cut's classification of zero-zero edges

    # ------------------------------------------------------------ planes near a point
    def tris_near(self, c, r):
        """triangles within distance r of the point c (sorted ids)"""
        dom = self.dom
        rr = r * (1 + 1e-9) + self.eps
        if dom._rs is not None and hasattr(dom._rs, "near"):     # an older build has no near()
            _, t = dom._rs.near(np.ascontiguousarray(c, float).reshape(1, 3), np.array([rr]))
            return t
        sub = dom._kd.query_ball_point(c, r + dom._rmax)
        if not sub:
            return np.zeros(0, np.int64)
        t = np.unique(dom._sub_parent[np.asarray(sub, np.int64)])
        T = dom.tris[t]
        cp = closest_point_triangles(np.repeat(c[None], len(t), 0), T[:, 0], T[:, 1], T[:, 2])
        return t[np.linalg.norm(cp - c, axis=1) <= rr]

    def planes(self, tris):
        """one triangle per plane (the lowest id), in id order"""
        keep = []
        for t in tris.tolist():
            if not any(abs(self.d[t] - self.d[k]) < self.eps and self.n[t] @ self.n[k] > 1 - 1e-12
                       for k in keep):
                keep.append(t)
        return keep

    def groups(self, pl):
        """group index per plane: planes meeting at less than the feature angle (a
        faceted curved wall) are one group, transitively"""
        g = list(range(len(pl)))

        def find(i):
            while g[i] != i:
                g[i] = g[g[i]]
                i = g[i]
            return i
        N = self.n[pl]
        close = N @ N.T > self.cos_feat
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                if close[i, j]:
                    g[find(i)] = find(j)
        roots = [find(i) for i in range(len(pl))]
        u = {r: k for k, r in enumerate(sorted(set(roots)))}
        return np.array([u[r] for r in roots], np.int64)

    def convex_at(self, P, r):
        """The fluid near the points P (a cell's vertices) is the intersection of the
        half-spaces of the triangles within r: the signs of the plane function and of the
        level set agree on a grid over the cell (and at its vertices).
        Not where a faceted curved wall meets another wall (a cylinder's rim): the clip
        follows every facet there, leaving bands and bends a fraction of a cell wide
        along the crease, where the level-set cut with its feature hooks gives the
        layers a cleaner wall.  Flat walls meeting, or one curved wall (the apex of a
        cone), are clipped."""
        c = P.mean(0)
        tris = self.tris_near(c, r)
        if not len(tris):
            return False, []
        pl = self.planes(tris)
        g = self.groups(pl)
        if g.max() > 0 and len(g) > g.max() + 1:
            return False, pl
        lo, hi = P.min(0), P.max(0)
        g = np.linspace(0, 1, 4)
        G = lo + (hi - lo) * np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        X = np.concatenate([P, G])
        s = (X @ self.n[pl].T - self.d[pl]).max(1)
        ph = self.dom.sdf(X)
        tol = 1e-6 * self.hmin
        bad = ((ph < -tol) & (s > tol)) | ((ph > tol) & (s < -tol))
        return not bad.any(), pl

    def on_one_plane(self, ctx, poly, pl):
        """the polygon lies in one of the planes pl"""
        if not len(pl):
            return False
        X = np.array([ctx.pos[v] for v in poly])
        return bool((np.abs(X @ self.n[pl].T - self.d[pl]) < 1e-6 * self.hmin).all(0).any())

    # ------------------------------------------------------------ one face
    def clip_face(self, fid, poly):
        """Kept part of the face: list of records, in the order of poly.  A record is
        ('v', id) an original vertex, ('e', lo, hi, k) where plane k crosses the original
        edge lo-hi, or ('f', fid, k1, k2, ...) a point inside the face on several planes.
        Cached per face (the reversed list for the other cell)."""
        key = fid
        hit = self.face_cache.get(key)
        if hit is not None:
            recs, P, ref = hit
            if ref == poly:
                return recs, P
            # the other cell: the same face reversed
            return recs[::-1], P[::-1]
        X = self.pts[poly]
        c = X.mean(0)
        r = float(np.linalg.norm(X - c, axis=1).max())
        pl = self.planes(self.tris_near(c, r))
        if not pl:
            # no wall within reach: the whole face is on the side of its vertices
            inside = bool((self.phi[poly] <= 0).all())
            recs, P = ([("v", v) for v in poly], [x for x in X]) if inside else ([], [])
            self.face_cache[key] = (recs, P, list(poly))
            self.face_planes[key] = []
            return recs, P
        m = len(poly)
        if self.groups(pl).max() == 0:
            # no sharp crease near the face: the level-set rule (vertices with phi <= 0,
            # a crossing on each edge from phi < 0 to phi > 0), with exact crossings
            ph = self.phi[poly]
            if not (ph > 0).any():
                recs, P = [("v", v) for v in poly], [x for x in X]
            elif not (ph < 0).any():
                recs, P = [], []
            else:
                recs, P = [], []
                Np, Dp = self.n[pl], self.d[pl]
                for i in range(m):
                    j = (i + 1) % m
                    if ph[i] <= 0:
                        recs.append(("v", poly[i])); P.append(X[i])
                    if ph[i] * ph[j] < 0:
                        a, b = (i, j) if ph[i] < 0 else (j, i)
                        sa, sb = Np @ X[a] - Dp, Np @ X[b] - Dp
                        out = sb > 0
                        if out.any():
                            t = np.where(out, np.clip(sa, None, 0) / np.where(out, np.clip(sa, None, 0) - sb, 1), 2.0)
                            k = int(np.argmin(t)); tt = float(np.clip(t[k], 0, 1))
                        else:
                            k, tt = 0, 0.5
                        lo, hi = min(poly[i], poly[j]), max(poly[i], poly[j])
                        recs.append(("e", lo, hi, pl[k])); P.append(X[a] + tt * (X[b] - X[a]))
            self.face_cache[key] = (recs, P, list(poly))
            self.face_planes[key] = pl
            return recs, P
        recs = [("v", v) for v in poly]
        P = [x for x in X]
        edge_of_v = {}
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            edge_of_v[(a, b)] = edge_of_v[(b, a)] = (min(a, b), max(a, b))
        eps = self.eps
        # One clip per group of planes: a faceted curved wall is clipped by the max of its
        # planes, with the exact exit point along each polygon edge, so that only the
        # sharp creases between groups make points inside the face (the creases between
        # the facets of one group would put a vertex on every facet).
        grp = self.groups(pl)
        pl_arr = np.asarray(pl)
        for q in range(int(grp.max()) + 1):
            if not recs:
                break
            ks = pl_arr[grp == q]
            Nq, Dq = self.n[ks], self.d[ks]
            S = np.array(P) @ Nq.T - Dq                                  # (vertex, plane)
            s = [0.0 if abs(v) < eps else float(v) for v in S.max(1)]
            if all(v <= 0 for v in s):
                continue
            nr, nP = [], []
            L = len(recs)
            for i in range(L):
                j = (i + 1) % L
                if s[i] <= 0:
                    nr.append(recs[i]); nP.append(P[i])
                if (s[i] < 0 < s[j]) or (s[j] < 0 < s[i]):
                    a, b = (i, j) if s[i] < 0 else (j, i)
                    sa, sb = np.minimum(S[a], 0.0), S[b]
                    out = sb > 0
                    t = np.where(out, -sa / np.where(out, sb - sa, 1.0), np.inf)
                    kk = int(np.argmin(t))
                    x = P[a] + float(np.clip(t[kk], 0, 1)) * (P[b] - P[a])
                    e = self._common_edge(recs[i], recs[j], edge_of_v)
                    if e is not None:
                        nr.append(("e", e[0], e[1], int(ks[kk]))); nP.append(x)
                    else:
                        on = tuple(sorted(k for k in pl if abs(self.n[k] @ x - self.d[k]) < 1e-6 * self.hmin))
                        nr.append(("f", fid) + on); nP.append(x)
            recs, P = nr, nP
        if len(recs) < 3:
            recs, P = [], []
        elif not (len(recs) == m and all(r[0] == "v" for r in recs)):
            # a part lying in one wall plane (vertices snapped onto it) has no fluid on
            # either side: as the level-set cut, the face is kept only if no vertex is in
            # the solid, and then it is whole (above)
            Q = np.array(P)
            on = np.abs(Q @ self.n[pl].T - self.d[pl]) < 1e-6 * self.hmin
            if on.all(0).any():
                recs, P = [], []
        self.face_cache[key] = (recs, P, list(poly))
        self.face_planes[key] = pl
        return recs, P

    @staticmethod
    def _common_edge(r1, r2, edge_of_v):
        """the original edge both records lie on, or None"""
        def edges(r):
            if r[0] == "e":
                return {(r[1], r[2])}
            if r[0] == "v":
                return None
            return set()
        e1, e2 = edges(r1), edges(r2)
        if e1 is None and e2 is None:
            e = edge_of_v.get((r1[1], r2[1]))
            return e
        if e1 is None:
            return next((e for e in e2 if r1[1] in e), None)
        if e2 is None:
            return next((e for e in e1 if r2[1] in e), None)
        both = e1 & e2
        return next(iter(both), None)

    # ------------------------------------------------------------ level-set outcome
    def ls_compatible(self, fid, poly):
        """The level-set cut gives this face the same kept part as the clip: with no
        vertex in the solid the whole face, with none in the fluid nothing, otherwise the
        vertices with phi <= 0 and the crossings of the edges from phi < 0 to phi > 0,
        joined by straight chords (no crease inside the face)."""
        recs, _ = self.clip_face(fid, poly)
        ph = self.phi[poly]
        if not (ph > 0).any():
            return len(recs) == len(poly) and all(r[0] == "v" for r in recs)
        if not (ph < 0).any():
            return not recs
        if any(r[0] == "f" for r in recs):
            return False
        m = len(poly)
        ze = self.zero_edge
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            if ph[i] == 0 and ph[(i + 1) % m] == 0 and ze and \
                    ze.get((fid, min(a, b), max(a, b))) is not None:
                return False          # classified by surface_contact: not the plain rule
        want_v = {v for v, p in zip(poly, ph) if p <= 0}
        want_e = {(min(poly[i], poly[(i + 1) % m]), max(poly[i], poly[(i + 1) % m]))
                  for i in range(m) if ph[i] * ph[(i + 1) % m] < 0}
        got_v = {r[1] for r in recs if r[0] == "v"}
        got_e = [(r[1], r[2]) for r in recs if r[0] == "e"]
        return got_v == want_v and len(got_e) == len(want_e) and set(got_e) == want_e


def clipped_face(pc, ctx, fid, poly, alloc):
    """The kept part of a face as (face id, vertex ids), None if nothing is kept: the
    face itself if whole, else a sub-face keyed by its vertex set (so the cell on the
    other side, cutting it the same way, gets the same id)."""
    recs, P = pc.clip_face(fid, poly)
    if not recs:
        return None
    if len(recs) == len(poly) and all(r[0] == "v" for r in recs):
        return fid, list(poly)
    ids = []
    for r, x in zip(recs, P):
        v = alloc(r, x)
        if not ids or ids[-1] != v:
            ids.append(v)
    if len(ids) > 1 and ids[0] == ids[-1]:
        ids.pop()
    if len(set(ids)) < 3:
        return None
    return ctx.sub_id((fid, tuple(sorted(ids)))), ids


def clip_cell(pc, ctx, faces, alloc, hook, creases=None, walls=None):
    """Cut one cell (faces [(fid, poly)]) by plane clipping.  Returns the kept faces
    (with the caps) as the level-set cut's neg list.  creases (a set): gets the ids of
    the caps that lie on the walls (bent along a sharp crease - split over wall groups,
    or by the hook - or in one wall plane), not across the facets of a curved wall.
    walls (a set): gets the ids of the faces on the walls: the caps, and kept faces
    lying in one wall plane (vertices snapped onto it)."""
    neg = []
    for fid, poly in faces:
        f = clipped_face(pc, ctx, fid, poly, alloc)
        if f is not None:
            neg.append(f)
    if not neg:
        return []
    from .cutcell import _chain_loops
    de = {}
    for _, p in neg:
        m = len(p)
        for k in range(m):
            e = (p[k], p[(k + 1) % m])
            de[e] = de.get(e, 0) + 1
    segs = []
    for (a, b), c in de.items():
        for _ in range(c - de.get((b, a), 0)):
            segs.append((b, a))
    # one triangle per plane (faces may have picked different triangles of one plane)
    pl = pc.planes(np.array(sorted(set(k for fid, _ in faces for k in pc.face_planes.get(fid, ()))),
                            np.int64))
    if walls is not None:
        walls.update(f for f, p in neg if pc.on_one_plane(ctx, p, pl))
    for loop in _chain_loops(segs):
        polys = split_cap(pc, ctx, loop, pl, alloc)
        if polys is None:
            polys = hook(ctx, loop) if hook is not None else [loop]
        for poly in polys:
            neg.append((ctx.new_face_id(), poly))
            if walls is not None:
                walls.add(neg[-1][0])
            if creases is not None and (len(polys) > 1 or pc.on_one_plane(ctx, poly, pl)):
                creases.add(neg[-1][0])
    return neg


def split_cap(pc, ctx, loop, pl, alloc):
    """The cap loop (on the walls) in one polygon per wall plane: the run of loop
    vertices on the plane, closed through the point where it meets the planes of the
    runs' ends (a corner or apex inside the cell).  None if the loop does not split
    that way."""
    if not pl:
        return None
    X = np.array([ctx.pos[v] for v in loop])
    tol = 1e-6 * pc.hmin
    onp = np.abs(X @ pc.n[pl].T - pc.d[pl]) < tol                      # (vertex, plane)
    if not onp.any(1).all():
        return None
    # planes meeting at less than the feature angle are one wall (a faceted curve): the
    # cap is split only along the sharp creases between such groups
    grp = pc.groups(pl)
    ng = int(grp.max()) + 1
    on = np.zeros((len(loop), ng), bool)
    for k in range(len(pl)):
        on[:, grp[k]] |= onp[:, k]
    cnt = on.sum(0)
    if (cnt == len(loop)).any():
        return [loop]                                    # all on one wall
    rep = [pl[int(np.nonzero(grp == q)[0][0])] for q in range(ng)]

    def plane_at(i, q):
        """a plane of group q through loop vertex i"""
        k = np.nonzero(onp[i] & (grp == q))[0]
        return pl[int(k[0])]
    # Corners inside the cell: a crease (two walls) met by one loop vertex only ends
    # inside, at its corner with a third wall.  For a pair of such creases sharing a
    # wall, that is the point on the three planes.
    act = np.nonzero(cnt > 0)[0]
    crease1 = []
    for ii, a in enumerate(act.tolist()):
        for b in act[ii + 1:].tolist():
            sh = np.nonzero(on[:, a] & on[:, b])[0]
            if len(sh) == 1:
                crease1.append((a, b, int(sh[0])))
    R = float(np.linalg.norm(X - X.mean(0), axis=1).max())
    corners = {}
    for i1 in range(len(crease1)):
        for i2 in range(i1 + 1, len(crease1)):
            (a1, b1, u), (a2, b2, w) = crease1[i1], crease1[i2]
            tri = sorted({a1, b1} | {a2, b2})
            if len(tri) != 3:
                continue
            k3 = sorted({plane_at(u, a1), plane_at(u, b1), plane_at(w, a2), plane_at(w, b2)})
            if len(k3) != 3:
                continue
            A = pc.n[k3]
            if abs(np.linalg.det(A)) < 1e-9:
                continue
            x = np.linalg.solve(A, pc.d[k3])
            if np.linalg.norm(x - X.mean(0)) > 1.5 * R:
                continue
            if (pc.n[pl] @ x - pc.d[pl]).max() > tol:
                continue                                 # outside the fluid
            if np.linalg.norm(X - x, axis=1).min() < tol:
                continue                                 # a loop vertex is there already
            key = ("c",) + tuple(np.round(x / tol).astype(np.int64).tolist())
            corners[key] = x
    if len(corners) > 1:
        return None
    m = len(loop)
    out = []
    corner = next(iter(corners.items()), None)
    for j in act.tolist():
        n = pc.n[rep[j]]
        o = on[:, j]
        if o.sum() < 2:
            continue
        # the run of loop vertices on wall j (cyclically contiguous), in loop order
        starts = [i for i in range(m) if o[i] and not o[i - 1]]
        if len(starts) != 1:
            return None
        run = []
        i = starts[0]
        while o[i % m] and len(run) < m:
            run.append(i % m); i += 1
        poly = [loop[i] for i in run]
        Q = [X[i] for i in run]
        a, b = run[0], run[-1]
        # closed along a crease (the ends share another plane) or through the corner
        if not ((on[a] & on[b]) & ~np.eye(ng, dtype=bool)[j]).any():
            if corner is None or np.abs(pc.n[pl][grp == j] @ corner[1] - pc.d[pl][grp == j]).min() > tol:
                return None
            poly.append(-1); Q.append(corner[1])         # allocated once the pieces are valid
        Q = np.array(Q)
        c = Q.mean(0)
        S = np.cross(Q - c, np.roll(Q, -1, 0) - c).sum(0) @ n
        if S < 1e-6 * pc.hmin ** 2:
            continue                                     # no area: along a crease only
        out.append(poly)
    if not out:
        return None
    # the pieces must close the loop: every loop edge once, every other edge paired
    de = {}
    for p in out:
        for k in range(len(p)):
            e = (p[k], p[(k + 1) % len(p)])
            de[e] = de.get(e, 0) + 1
    for k in range(m):
        e = (loop[k], loop[(k + 1) % m])
        if de.get(e, 0) != 1:
            return None
        de[e] = 0
    for (u, w), c_ in de.items():
        if c_ and de.get((w, u), 0) != c_:
            return None
    if any(-1 in p for p in out):
        vc = alloc(corner[0], corner[1])
        out = [[vc if v == -1 else v for v in p] for p in out]
    return out


def make_alloc(ctx, phi, feat_closest, tol, weld):
    """Vertex ids for the records of clip_face, shared by all faces: an original vertex
    is itself; the crossing of an edge with a sign change of the level set is the
    level-set cut's crossing (moved onto the exact point), so the cells cut by the level
    set share it; any other point gets one new vertex per position (a point on a crease
    is reached from different planes by different faces).
    A point closer than weld to a crease (feature_closest(x) -> (point, distance)) goes
    onto it: a face snapped onto one wall and passing just under the crease would
    otherwise leave a band of the other wall a fraction of a cell wide, which no flat
    cap follows (on a faceted curved wall it tilts towards the first wall)."""
    at = {}
    n0 = len(phi)
    nb = [np.array(d) for d in np.ndindex(3, 3, 3)]

    def find(x):
        g = np.floor(x / tol).astype(np.int64)
        for d in nb:
            v = at.get(tuple((g + d - 1).tolist()))
            if v is not None and np.linalg.norm(np.asarray(ctx.pos[v]) - x) < tol:
                return v
        return None

    def put(v, x):
        at[tuple(np.floor(x / tol).astype(np.int64).tolist())] = v

    def alloc(r, x):
        if r[0] == "v":
            return r[1]
        x = np.asarray(x, float)
        cp, d = feat_closest(x)
        on = d < weld
        if on:
            x = np.asarray(cp, float)
        if r[0] == "e":
            lo, hi = r[1], r[2]
            if lo < n0 and hi < n0 and phi[lo] * phi[hi] < 0:
                v = ctx.cache.get((lo, hi))
                if v is None:
                    v = find(x)
                    if v is not None:
                        ctx.cache[(lo, hi)] = v
                    else:
                        v = ctx.cross(lo, hi)
                ctx.pos[v] = x.tolist()
                if on:
                    ctx.onfeat.add(v)
                put(v, x)
                return v
        v = find(x)
        if v is None:
            v = len(ctx.pos)
            ctx.pos.append(x.tolist())
            ctx.phi.append(0.0)
            ctx.new_vertices.append(v)
            ctx.new_parents.append((-1, -1))
            if on:
                ctx.onfeat.add(v)
            put(v, x)
        return v
    return alloc
