"""Linear octree on a Cartesian root grid, 2:1 balanced (face/edge/vertex neighbours),
converted to conformal polyhedral cells (hanging nodes become extra polygon vertices).
"""
import itertools
import numpy as np

OFF26 = np.array([d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)])
CHILD = np.array(list(itertools.product((0, 1), repeat=3)))   # (8,3)

# outward-oriented hex faces in local (0/1) corner coordinates: (axis, side, 4 corners)
HEX_FACES = [
    (0, 0, [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    (0, 1, [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    (1, 0, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    (1, 1, [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    (2, 0, [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
    (2, 1, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
]


class Octree:
    def __init__(self, origin, h0, n, max_level):
        self.x0 = np.asarray(origin, float)
        self.h0 = float(h0)
        self.n = np.asarray(n, np.int64)
        self.Lmax = int(max_level)
        g = np.indices(self.n).reshape(3, -1).T
        self.ijk = g.astype(np.int64)
        self.lev = np.zeros(len(g), np.int64)

    # ------------------------------------------------------------------ util
    def h(self, l):
        return self.h0 / 2.0 ** l

    def centers(self, idx=None):
        lev = self.lev if idx is None else self.lev[idx]
        ijk = self.ijk if idx is None else self.ijk[idx]
        hh = self.h0 / 2.0 ** lev
        return self.x0 + (ijk + 0.5) * hh[:, None]

    def _key(self, l, q):
        N = self.n * (1 << l)
        return (q[:, 0] * N[1] + q[:, 1]) * N[2] + q[:, 2]

    def _build_lookup(self):
        self._lk = {}
        for l in range(self.Lmax + 1):
            idx = np.nonzero(self.lev == l)[0]
            k = self._key(l, self.ijk[idx])
            o = np.argsort(k)
            self._lk[l] = (k[o], idx[o])

    def _find_exact(self, l, q):
        """Index of leaf at exactly level l with coords q, else -1."""
        keys, idx = self._lk[l]
        out = np.full(len(q), -1, np.int64)
        if len(keys) == 0 or len(q) == 0:
            return out
        k = self._key(l, q)
        p = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        hit = keys[p] == k
        out[hit] = idx[p[hit]]
        return out

    def find_covering(self, l, q):
        """Leaf of level <= l that contains the level-l cell q (or -1: out of box / finer)."""
        out = np.full(len(q), -1, np.int64)
        N = self.n * (1 << l)
        inb = np.all((q >= 0) & (q < N), axis=1)
        todo = np.nonzero(inb)[0]
        for c in range(l, -1, -1):
            if len(todo) == 0:
                break
            r = self._find_exact(c, q[todo] >> (l - c))
            hit = r >= 0
            out[todo[hit]] = r[hit]
            todo = todo[~hit]
        return out

    def split(self, mask):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return 0
        keep = ~mask
        child_ijk = (2 * self.ijk[idx])[:, None, :] + CHILD[None]
        child_lev = np.repeat(self.lev[idx] + 1, 8)
        self.ijk = np.concatenate([self.ijk[keep], child_ijk.reshape(-1, 3)])
        self.lev = np.concatenate([self.lev[keep], child_lev])
        return len(idx)

    # ------------------------------------------------------------ refinement
    def refine(self, surface_sources, volume_sources, log=print, keep_fn=None):
        """surface_sources: list of (cKDTree, level, nbuf)
        volume_sources: list of (sdf_function, level, nbuf)
        keep_fn(centers, half_diag) -> bool: False for cells certainly outside the
        domain; those are never split (they are discarded later anyway)."""
        for l in range(self.Lmax):
            sel = np.nonzero(self.lev == l)[0]
            if len(sel) == 0:
                continue
            c = self.centers(sel)
            hd = 0.5 * np.sqrt(3) * self.h(l)
            mark = np.zeros(len(sel), bool)
            for kd, Ls, nbuf in surface_sources:
                if Ls <= l:
                    continue
                R = sum(nbuf * self.h(q) for q in range(l + 1, Ls))
                d, _ = kd.query(c, distance_upper_bound=R + hd)
                mark |= d < R + hd
            for sdf, Lv, nbuf in volume_sources:
                if Lv <= l:
                    continue
                R = sum(nbuf * self.h(q) for q in range(l + 1, Lv))
                mark |= sdf(c) < R + hd
            if keep_fn is not None and mark.any():
                mi = np.nonzero(mark)[0]
                mark[mi[~keep_fn(c[mi], hd)]] = False
            full = np.zeros(len(self.lev), bool)
            full[sel[mark]] = True
            ns = self.split(full)
            log(f"  level {l}->{l + 1}: split {ns} cells, total {len(self.lev)}")
        self.balance(log)

    def balance(self, log=print):
        it = 0
        while True:
            self._build_lookup()
            mark = np.zeros(len(self.lev), bool)
            for m in range(2, self.Lmax + 1):
                sel = np.nonzero(self.lev == m)[0]
                if len(sel) == 0:
                    continue
                P = np.unique(self.ijk[sel] >> 1, axis=0)            # level m-1 parents
                Q = (P[:, None, :] + OFF26[None]).reshape(-1, 3)
                Q = np.unique(Q, axis=0)
                r = self.find_covering(m - 1, Q)
                ok = r >= 0
                r = r[ok]
                bad = r[self.lev[r] < m - 1]
                mark[bad] = True
            n = self.split(mark)
            it += 1
            if n == 0:
                break
            log(f"  balance pass {it}: split {n}, total {len(self.lev)}")
        self._build_lookup()

    # ----------------------------------------------------------- cell build
    def build_cells(self):
        """Return dict with points, hanging flag, per-cell faces [(face_key_id, poly)], ..."""
        Lmax = self.Lmax
        S = (1 << (Lmax - self.lev)).astype(np.int64)            # cell size in finest units
        base = self.ijk * S[:, None]
        NV = self.n * (1 << Lmax) + 1

        def vkey(x):
            return (x[..., 0] * NV[1] + x[..., 1]) * NV[2] + x[..., 2]

        corners = base[:, None, :] + CHILD[None] * S[:, None, None]
        ck = vkey(corners).ravel()
        ukeys = np.unique(ck)
        pts_lat = np.stack([ukeys // (NV[1] * NV[2]), (ukeys // NV[2]) % NV[1], ukeys % NV[2]], 1)
        points = self.x0 + pts_lat * (self.h0 / (1 << Lmax))

        def vid_of(keys):
            p = np.clip(np.searchsorted(ukeys, keys), 0, len(ukeys) - 1)
            return np.where(ukeys[p] == keys, p, -1)

        # hanging vertices = edge midpoints / face centres of some leaf that exist as vertices
        big = S >= 2
        hb = base[big]; hs = S[big][:, None]
        # hanging nodes and their parents (ends of the coarse edge, or corners of the
        # coarse face); hang_size = coarse cell size in finest units
        hanging = np.zeros(len(ukeys), bool)
        hang_par = np.full((len(ukeys), 4), -1, np.int64)
        hang_size = np.zeros(len(ukeys), np.int64)
        for d in itertools.product((0, 1, 2), repeat=3):
            free = [a for a in range(3) if d[a] == 1]
            if len(free) not in (1, 2):
                continue
            hv = vid_of(vkey(hb + np.array(d) * hs // 2))
            ok = hv >= 0
            if not ok.any():
                continue
            hv = hv[ok]
            hanging[hv] = True
            pars = []
            for choice in itertools.product((0, 2), repeat=len(free)):
                dd = np.array(d)
                for a, c in zip(free, choice):
                    dd[a] = c
                pars.append(vid_of(vkey(hb[ok] + dd * hs[ok] // 2)))
            if len(pars) == 4:       # face corners in cyclic order
                pars = [pars[0], pars[1], pars[3], pars[2]]
            P_ = np.stack(pars + [pars[0]] * (4 - len(pars)), 1)
            upd = hs[ok, 0] > hang_size[hv]
            hang_par[hv[upd]] = P_[upd]
            hang_size[hv[upd]] = hs[ok, 0][upd]

        # ---------------- faces
        nleaf = len(self.lev)
        quad_cell, quad_axis, quad_lat = [], [], []
        for axis, side, loc in HEX_FACES:
            loc = np.array(loc)
            dirv = np.zeros(3, np.int64); dirv[axis] = 1 if side else -1
            nb = self.ijk + dirv
            cov = np.full(nleaf, -1, np.int64)
            Nl_in = np.zeros(nleaf, bool)
            for l in range(Lmax + 1):
                sel = np.nonzero(self.lev == l)[0]
                if len(sel) == 0:
                    continue
                N = self.n * (1 << l)
                q = nb[sel]
                inb = np.all((q >= 0) & (q < N), axis=1)
                Nl_in[sel] = inb
                cov[sel] = self.find_covering(l, q)
            finer = Nl_in & (cov < 0)
            Q = base[:, None, :] + loc[None] * S[:, None, None]           # (n,4,3)
            own = ~finer
            quad_cell.append(np.nonzero(own)[0]); quad_axis.append(np.full(own.sum(), axis))
            quad_lat.append(Q[own])
            fi = np.nonzero(finer)[0]
            if len(fi):
                P = Q[fi]
                m01 = (P[:, 0] + P[:, 1]) // 2; m12 = (P[:, 1] + P[:, 2]) // 2
                m23 = (P[:, 2] + P[:, 3]) // 2; m30 = (P[:, 3] + P[:, 0]) // 2
                c = (P[:, 0] + P[:, 2]) // 2
                subs = [np.stack(s, 1) for s in ([P[:, 0], m01, c, m30], [m01, P[:, 1], m12, c],
                                                 [c, m12, P[:, 2], m23], [m30, c, m23, P[:, 3]])]
                for sq in subs:
                    quad_cell.append(fi); quad_axis.append(np.full(len(fi), axis)); quad_lat.append(sq)
        qc = np.concatenate(quad_cell); del quad_cell
        qa = np.concatenate(quad_axis); del quad_axis
        ql = np.concatenate(quad_lat); del quad_lat                     # (F,4,3)
        # identity key of the square: (min corner, axis, size) packed in one int64
        idk = np.empty(len(ql), np.int64)
        for s0 in range(0, len(ql), 500000):
            q = ql[s0:s0 + 500000]
            mn = q.min(1)
            sz = (q.max(1) - mn).max(1)
            idk[s0:s0 + 500000] = (vkey(mn) * 3 + qa[s0:s0 + 500000]) * ((1 << Lmax) + 1) + sz
        del qa
        _, fkey = np.unique(idk, return_inverse=True)
        fkey = fkey.ravel(); del idk
        F = len(ql)
        cv = np.empty((F, 4), np.int64); mv = np.empty((F, 4), np.int64)
        for s0 in range(0, F, 500000):                                  # chunked: memory
            q = ql[s0:s0 + 500000]
            cv[s0:s0 + 500000] = vid_of(vkey(q))
            nxt = np.roll(q, -1, axis=1)
            elen = np.abs(nxt - q).sum(2)
            mv[s0:s0 + 500000] = np.where(elen >= 2, vid_of(vkey((q + nxt) // 2)), -1)
        assert (cv >= 0).all()
        del ql
        # polygons as CSR, grouped by cell: corners interleaved with existing mid-edge nodes
        order = np.argsort(qc, kind="stable")
        qc = qc[order]; fkey = fkey[order]; cv = cv[order]; mv = mv[order]
        inter = np.empty((len(qc), 8), np.int64)
        inter[:, 0::2] = cv
        inter[:, 1::2] = mv
        valid = inter >= 0
        fn = valid.sum(1)
        fverts = inter[valid]
        fptr = np.concatenate([[0], np.cumsum(fn)])
        cptr = np.searchsorted(qc, np.arange(nleaf + 1))
        return dict(points=points, hanging=hanging, hang_par=hang_par, hang_size=hang_size, face_cell=qc, face_key=fkey,
                    fptr=fptr, fverts=fverts, cell_fptr=cptr,
                    nface_keys=int(fkey.max()) + 1, cell_size=self.h0 / (1 << self.lev),
                    cell_level=self.lev.copy(),
                    corner_vid=vid_of(vkey(corners)))
