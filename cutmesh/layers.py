"""Prism boundary layers on wall patches of a polyMesh (post-processing step).

The mesh is shrunk away from the walls and the gap is filled with prism layers:

1. Wall points move inwards along the (area weighted) point normal by the total
   layer thickness T.  The interior points within nRings edges of the wall follow by
   Laplacian smoothing of the displacement (zero beyond the band), so the cells next
   to the wall are compressed rather than inverted.  Being based on the mesh edges,
   the displacement does not leak through thin solid walls, and gaps between a wall
   and a constraint surface are compressed evenly.
2. Points on patches without layers (inlet, outlet, ...) and on face zones (the
   rotating-zone interface) only slide along those surfaces; so do the wall points on
   their rims.
3. Between the original wall position and the shrunk wall, nLayers prism cells are
   inserted per wall face (geometric growth by expansionRatio towards the interior).
   The original wall face becomes the internal face to the first original cell.
4. Where the point normal is undefined (sharp convex edges) or a cell becomes
   invalid or too non-orthogonal, the thickness at the points involved is halved
   (nIter times) and then set to zero until the mesh is valid.
   At points with zero thickness the layer points collapse into one, so the layers
   terminate there with wedge cells (as in snappyHexMesh).

Thicknesses are relative to the local cell size (max extent of the cell at the wall)
unless relativeSizes is false.
"""
import time
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from . import quality
from .geometry import paused_gc

DEFAULTS = {
    "patches": None,             # patch names with layers (default: all wall patches)
    "nLayers": 3,
    "expansionRatio": 1.2,
    "finalLayerThickness": 0.3,  # thickness of the layer next to the interior mesh
    "relativeSizes": True,       # thicknesses relative to the local cell size
    "minThickness": 0.05,        # total thickness below this collapses the layers
    "minNormalCos": 0.25,        # collapse where cos(point normal, face normal) is below
    "nRings": 6,                 # interior points within this many edges of the wall are displaced
    "nSmooth": 60,               # Laplacian smoothing sweeps of the interior displacement
    "nIter": 8,                  # validity iterations halving the thickness at bad cells
    "nCollapseIter": 60,         # then iterations collapsing the layers there
    "maxThicknessSlope": 0.5,    # thickness may change by this much per unit length along the wall
    "maxNonOrtho": 65.0,         # [deg] cells at faces above this (original faces: above their value + 2) are bad
    "wallNonOrthMargin": None,   # [deg] wall faces may get this much worse than their value as boundary faces
    "cornerStretch": True,       # move corner points by T / cos so the layers keep thickness T
    "maxGapFraction": 0.3,       # total thickness at most this fraction of the distance to a facing wall
}


def _ptr(n):
    return np.concatenate([[0], np.cumsum(n)])


def _nonorth(C, Cf, Sf, own, nei):
    nI = len(nei)
    d = C[nei] - C[own[:nI]]
    c = np.einsum("ij,ij->i", d, Sf[:nI]) / np.maximum(
        np.linalg.norm(d, axis=1) * np.linalg.norm(Sf[:nI], axis=1), 1e-300)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _cells_bad(P, nn, vv, own, nei, nC, max_no=None):
    """cells with non-positive volume or a non-positive face pyramid (checkMesh), or a
    face more non-orthogonal than max_no (scalar or per internal face)"""
    ff = quality.FlatFaces((nn, vv))
    Cf, Sf = quality.face_geometry(P, ff)
    C, V = quality.cell_geometry(nC, own, nei, Cf, Sf)
    nI = len(nei)
    bad = V <= 0
    po = np.einsum("ij,ij->i", Sf, Cf - C[own]) <= 0
    pn = np.einsum("ij,ij->i", Sf[:nI], C[nei] - Cf[:nI]) <= 0
    bad[own[po]] = True
    bad[nei[pn]] = True
    if max_no is not None:
        hi = _nonorth(C, Cf, Sf, own, nei) > max_no
        bad[own[:nI][hi]] = True
        bad[nei[hi]] = True
    return bad


def _plane_groups(normals, tol=0.99):
    """distinct planes among unit normals (averaged within a group)"""
    groups = []
    for m in normals:
        for g in groups:
            if abs(np.dot(g[0], m)) > tol:
                g.append(m if np.dot(g[0], m) > 0 else -m)
                break
        else:
            groups.append([m])
    out = []
    for g in groups:
        a = np.sum(g, 0)
        out.append(a / np.linalg.norm(a))
    return out


def _project(d, planes):
    """remove the components of d normal to the constraint planes"""
    if not planes:
        return d
    if len(planes) == 1:
        m = planes[0]
        return d - np.dot(d, m) * m
    if len(planes) == 2:
        t = np.cross(planes[0], planes[1])
        nt = np.linalg.norm(t)
        if nt < 1e-6:
            return d - np.dot(d, planes[0]) * planes[0]
        t /= nt
        return np.dot(d, t) * t
    return np.zeros(3)


def _projectors(cplanes, ids):
    """3x3 matrices removing the components normal to the constraint planes"""
    M = np.repeat(np.eye(3)[None], len(ids), 0)
    for k, p in enumerate(ids.tolist()):
        pl = cplanes[p]
        if len(pl) == 1:
            M[k] -= np.outer(pl[0], pl[0])
        elif len(pl) == 2:
            t = np.cross(pl[0], pl[1])
            nt = np.linalg.norm(t)
            M[k] = np.outer(t, t) / nt ** 2 if nt > 1e-6 else M[k] - np.outer(pl[0], pl[0])
        elif len(pl) > 2:
            M[k] = 0.0
    return M


def _constraint_planes(pid, nrm):
    """{point: [plane normals]} from (point id, unit face normal) pairs"""
    o = np.argsort(pid, kind="stable")
    pid, nrm = pid[o], nrm[o]
    u, st = np.unique(pid, return_index=True)
    en = np.append(st[1:], len(pid))
    return {int(p): _plane_groups(list(nrm[a:b])) for p, a, b in zip(u, st, en)}


@paused_gc
def add_layers(mesh, spec=None, log=print):
    """mesh: dict(points, faces=(sizes, flat), owner, neighbour, patches,
    cell_zones={name: cells}, face_zones={name: (faces, flip)}).  Returns a new dict."""
    t0 = time.time()
    s = dict(DEFAULTS)
    s.update(spec or {})
    N = int(s["nLayers"])
    r = float(s["expansionRatio"])
    P0 = np.asarray(mesh["points"], float)
    nn = np.asarray(mesh["faces"][0], np.int64)
    vv = np.asarray(mesh["faces"][1], np.int64)
    own = np.asarray(mesh["owner"], np.int64)
    nei = np.asarray(mesh["neighbour"], np.int64)
    patches = mesh["patches"]
    cell_zones = mesh.get("cell_zones") or {}
    face_zones = mesh.get("face_zones") or {}
    nI, nF, nP0 = len(nei), len(nn), len(P0)
    nC = int(max(own.max(), nei.max() if nI else 0)) + 1
    ptr = _ptr(nn)

    names = s["patches"] or [p["name"] for p in patches if p["type"] == "wall"]
    bpatch = np.full(nF, -1, np.int64)
    for k, p in enumerate(patches):
        bpatch[p["startFace"]:p["startFace"] + p["nFaces"]] = k
    lp = np.array([p["name"] in names for p in patches])
    is_lf = np.zeros(nF, bool)
    is_lf[nI:] = lp[bpatch[nI:]]
    Lf = np.nonzero(is_lf)[0]
    if not len(Lf):
        log("addLayers: no faces on the layer patches")
        return mesh
    ff = quality.FlatFaces((nn, vv))
    Cf, Sf = quality.face_geometry(P0, ff)
    nf_unit = Sf / np.maximum(np.linalg.norm(Sf, axis=1), 1e-300)[:, None]

    # ---------------------------------------------------------------- wall points
    lfn = nn[Lf]
    lfv = vv[_gather_idx(ptr, Lf)]
    lpos = np.full(nF, -1, np.int64); lpos[Lf] = np.arange(len(Lf))
    lff = np.repeat(np.arange(len(Lf)), lfn)              # layer face index per vertex
    W = np.unique(lfv)
    widx = np.full(nP0, -1, np.int64)
    widx[W] = np.arange(len(W))
    nW = len(W)
    lfw = widx[lfv]
    npn = np.zeros((nW, 3))
    np.add.at(npn, lfw, Sf[Lf][lff])                      # area weighted
    npn /= np.maximum(np.linalg.norm(npn, axis=1), 1e-300)[:, None]
    cmin = np.ones(nW)
    np.minimum.at(cmin, lfw, np.einsum("ij,ij->i", npn[lfw], nf_unit[Lf][lff]))

    # constraint surfaces: boundary faces without layers, face zones
    cfaces = [np.nonzero((bpatch >= 0) & ~is_lf)[0]]
    for fl, _ in face_zones.values():
        cfaces.append(np.asarray(fl, np.int64))
    cfaces = np.unique(np.concatenate(cfaces)) if cfaces else np.zeros(0, np.int64)
    cidx = _gather_idx(ptr, cfaces)
    cplanes = _constraint_planes(vv[cidx], nf_unit[ff.fid[cidx]])

    # extrusion direction (inwards), projected onto the constraint planes
    dirn = -npn.copy()
    ok_dir = cmin >= float(s["minNormalCos"])
    for w in np.nonzero(np.isin(W, list(cplanes.keys())))[0].tolist():
        d = _project(dirn[w], cplanes[int(W[w])])
        nd = np.linalg.norm(d)
        if nd < 0.3:
            ok_dir[w] = False
        else:
            dirn[w] = d / nd

    # at corners the point moves along the bisector: T / cos keeps the layer thickness T
    # normal to every face around it (the offset surface of the corner)
    cdir = np.ones(nW)
    np.minimum.at(cdir, lfw, -np.einsum("ij,ij->i", dirn[lfw], nf_unit[Lf][lff]))
    stretch = 1.0 / np.clip(cdir, 0.5, 1.0) if s["cornerStretch"] else np.ones(nW)

    # local cell size: max extent of the cells at the wall (min over the point's faces)
    oc = own[Lf]
    uc, inv = np.unique(oc, return_inverse=True)
    cmap = np.full(nC, -1, np.int64); cmap[uc] = np.arange(len(uc))
    fsel = np.nonzero((cmap[own] >= 0) | (np.concatenate([cmap[nei], np.full(nF - nI, -1)]) >= 0))[0]
    lo = np.full((len(uc), 3), np.inf); hi = np.full((len(uc), 3), -np.inf)
    for side in (own, np.concatenate([nei, np.full(nF - nI, -1)])):
        fs_ = fsel[cmap[np.maximum(side[fsel], 0)] >= 0]
        fs_ = fs_[side[fs_] >= 0]
        k = np.repeat(cmap[side[fs_]], nn[fs_])
        pv = P0[vv[_gather_idx(ptr, fs_)]]
        np.minimum.at(lo, k, pv); np.maximum.at(hi, k, pv)
    hcell = (hi - lo).max(1)
    hw = np.full(nW, np.inf)
    np.minimum.at(hw, lfw, hcell[inv][lff])
    scale = hw if s["relativeSizes"] else np.ones(nW)

    gsum = float(sum(r ** -(N - 1 - j) for j in range(N)))
    tfin = float(s["finalLayerThickness"])
    T = np.where(ok_dir, tfin * gsum * scale, 0.0)
    T_target = T.copy()
    gf = s["maxGapFraction"]
    if gf:
        # narrow gaps: the layers from facing walls would meet; limit each side to a
        # fraction of the distance to the opposite wall (measured along the extrusion
        # direction to the wall points ahead whose own normal faces back)
        PW = P0[W]
        kq = min(48, nW)
        R = float(np.max(T)) / float(gf) if np.any(T > 0) else 0.0
        dist, idx = cKDTree(PW).query(PW, k=kq, distance_upper_bound=R)
        valid = idx < nW
        idc = np.where(valid, idx, 0)
        v = PW[idc] - PW[:, None, :]
        along = np.einsum("ikj,ij->ik", v, dirn)
        lat = np.linalg.norm(v - along[..., None] * dirn[:, None, :], axis=2)
        facing = np.einsum("ikj,ij->ik", npn[idc], dirn) > 0.5
        ahead = valid & facing & (along > 0) & (lat < np.maximum(0.6 * along, 0.7 * hw[:, None]))
        width = np.where(ahead, along, np.inf).min(1)
        lim = float(gf) * width
        ng = int(((lim < T) & (T > 0)).sum())
        T = np.minimum(T, lim)
        T_target = T.copy()
        log(f"addLayers: {ng} wall points in narrow gaps, thickness limited to {gf:g} x the gap")
    tmin = float(s["minThickness"]) * scale
    frac = np.concatenate([[0.0], np.cumsum([r ** -(N - 1 - j) for j in range(N)])]) / gsum

    # interior points: band of nRings edges around the wall points (point graph)
    ea = np.minimum(vv, vv[ff.nxt]); eb = np.maximum(vv, vv[ff.nxt])
    ek = np.unique(ea * nP0 + eb)
    ea, eb = ek // nP0, ek % nP0
    # conductance 1 / edge length: the displacement decays linearly with the distance,
    # and thin cells at the wall (short edges) move almost rigidly with it
    elen = np.linalg.norm(P0[ea] - P0[eb], axis=1)
    ew = 1.0 / np.maximum(elen, 1e-12 * float(np.max(elen)))
    adj = sparse.csr_matrix((np.concatenate([ew, ew]), (np.concatenate([ea, eb]), np.concatenate([eb, ea]))),
                            shape=(nP0, nP0))
    deg = np.asarray(adj.sum(1)).ravel()
    isw = widx >= 0

    def rings(mask, k):
        m = mask.astype(float)
        for _ in range(k):
            m = np.maximum(m, (adj @ m > 0).astype(float))
        return m > 0

    band = np.nonzero(rings(isw, int(s["nRings"])) & ~isw)[0]
    adjB = adj[band]
    degB = deg[band]
    bpos = np.full(nP0, -1, np.int64); bpos[band] = np.arange(len(band))
    cb = np.array([p for p in band.tolist() if p in cplanes], np.int64)
    projB = _projectors(cplanes, cb)
    cbB = bpos[cb]
    log(f"addLayers: {len(Lf)} wall faces on {names}, {nW} wall points,"
        f" {int((~ok_dir).sum())} collapsed (sharp edges), {len(band)} interior points displaced")

    def interior(Dw):
        """Laplacian smoothing of the displacement in the band (Jacobi), wall fixed"""
        Dall = np.zeros((nP0, 3))
        Dall[W] = Dw
        for _ in range(int(s["nSmooth"])):
            DB = (adjB @ Dall) / degB[:, None]
            if len(cb):
                DB[cbB] = np.einsum("nij,nj->ni", projB, DB[cbB])
            Dall[band] = DB
        return Dall[band]

    C0, _ = quality.cell_geometry(nC, own, nei, Cf, Sf)
    # non-orthogonality limit: maxNonOrtho, or for faces of the original mesh their own
    # value + 2 deg (only what the layers make worse counts)
    no0 = _nonorth(C0, Cf, Sf, own, nei) if nI else np.zeros(0)
    lim0 = np.full(nF, float(s["maxNonOrtho"]))
    lim0[:nI] = np.maximum(lim0[:nI], no0 + 2.0)
    if s["wallNonOrthMargin"] is not None:
        # wall faces become internal faces to the layers; their angle stays close to the
        # angle between the face normal and (face centre - cell centre) of the cut cell
        dw = Cf[Lf] - C0[own[Lf]]
        cw = np.einsum("ij,ij->i", dw, nf_unit[Lf]) / np.maximum(np.linalg.norm(dw, axis=1), 1e-300)
        lim0[Lf] = np.maximum(lim0[Lf], np.degrees(np.arccos(np.clip(cw, -1, 1)))
                              + float(s["wallNonOrthMargin"]))
    del C0

    zone_names = list(cell_zones.keys())
    czone = np.full(nC, -1, np.int64)
    for zi, zn in enumerate(zone_names):
        czone[np.asarray(cell_zones[zn], np.int64)] = zi
    fz_names = {zn: zn + "_interface" for zn in zone_names}

    # wall edges for the thickness taper
    wa_ = lfw
    lptr = _ptr(lfn)
    nx_ = np.arange(len(lfw)) + 1
    nx_[lptr[1:] - 1] = lptr[:-1]
    wb_ = lfw[nx_]
    we = np.unique(np.sort(np.stack([wa_, wb_], 1), 1), axis=0)
    wlen = np.linalg.norm(P0[W[we[:, 0]]] - P0[W[we[:, 1]]], axis=1)
    slope = float(s["maxThicknessSlope"])

    def taper(T):
        """limit T[p] <= T[q] + slope * |pq| along the wall (layers end gradually)"""
        for _ in range(50):
            T0 = T.copy()
            np.minimum.at(T, we[:, 0], T[we[:, 1]] + slope * wlen)
            np.minimum.at(T, we[:, 1], T[we[:, 0]] + slope * wlen)
            if np.array_equal(T, T0):
                break
        return T

    n_half = int(s["nIter"])
    n_tot = n_half + int(s["nCollapseIter"])
    for it in range(n_tot + 1):
        T[T < tmin] = 0.0
        T = taper(T)
        T[T < tmin] = 0.0
        ext = T > 0
        # ------------------------------------------------------------ positions
        P = P0.copy()
        D = dirn * (T * stretch)[:, None]
        P[band] += interior(D)
        P[W] = P0[W] + D
        ne = int(ext.sum())
        eid = np.full(nW, -1, np.int64); eid[ext] = np.arange(ne)
        lev = np.repeat(W[:, None], N + 1, 1)               # point id per level (N = moved wall point)
        lev[ext, :N] = nP0 + eid[ext][:, None] * N + np.arange(N)[None, :]
        newpos = (P0[W][ext][:, None, :] + frac[None, :N, None] * D[ext][:, None, :]).reshape(-1, 3)
        P = np.concatenate([P, newpos])

        # ------------------------------------------------------------ faces
        act = np.zeros(len(Lf), bool)
        np.logical_or.at(act, lff, ext[lfw])
        A = np.nonzero(act)[0]
        nA = len(A)
        aidx = np.full(len(Lf), -1, np.int64); aidx[A] = np.arange(nA)

        def lcell(a, k):          # layer k (0 at the wall) of active face a
            return nC + a * N + (N - 1 - k)

        F_n, F_v, F_own, F_nei, F_pat, F_zone, F_old = [], [], [], [], [], [], []

        def add(n_, v_, o_, ne_, pa_, z_=None, old=None):
            F_n.append(np.asarray(n_, np.int64)); F_v.append(np.asarray(v_, np.int64))
            F_own.append(np.asarray(o_, np.int64)); F_nei.append(np.asarray(ne_, np.int64))
            F_pat.append(np.asarray(pa_, np.int64))
            F_zone.append(np.full(len(F_own[-1]), -1, np.int64) if z_ is None else np.asarray(z_, np.int64))
            F_old.append(np.full(len(F_own[-1]), -1, np.int64) if old is None else np.asarray(old, np.int64))

        # original faces (layer faces that are active become internal)
        oldf = np.arange(nF)
        o_nei = np.concatenate([nei, np.full(nF - nI, -1)])
        o_pat = bpatch.copy()
        fa = Lf[A]
        o_nei[fa] = lcell(np.arange(nA), N - 1)
        o_pat[fa] = -1
        add(nn, vv, own, o_nei, o_pat, old=oldf)
        # level faces of the active layer faces
        am = act[lff]
        av = lfw[am]                                   # W index per vertex of active faces
        an = lfn[A]
        aa = np.repeat(np.arange(nA), an)
        for k in range(N):
            if k == 0:
                add(an, lev[av, 0], lcell(np.arange(nA), 0), np.full(nA, -1), bpatch[fa])
            else:
                add(an, lev[av, k], lcell(np.arange(nA), k), lcell(np.arange(nA), k - 1),
                    np.full(nA, -1))
        # side faces along the edges of the active faces
        aptr = _ptr(an)
        nxt = np.arange(len(av)) + 1
        nxt[aptr[1:] - 1] = aptr[:-1]
        wa, wb = av, av[nxt]
        pa_, pb_ = W[wa], W[wb]
        # face across each edge (any boundary face)
        bidx = np.arange(ptr[nI], ptr[nF])
        bnxt = ff.nxt[bidx]
        bkey = vv[bidx] * (nP0 + 1) + vv[bnxt]
        bf_ = ff.fid[bidx]
        o_ = np.argsort(bkey, kind="stable"); bkey_s = bkey[o_]
        rk = pb_ * (nP0 + 1) + pa_
        pos_ = np.clip(np.searchsorted(bkey_s, rk), 0, len(bkey_s) - 1)
        g = np.where(bkey_s[pos_] == rk, bf_[o_[pos_]], -1)
        gl = np.where(g >= 0, lpos[np.maximum(g, 0)], -1)
        ga = np.where(gl >= 0, aidx[np.maximum(gl, 0)], -1)      # active layer face across
        fa_e = aa
        both = ga >= 0
        keep = ~both | (fa_e < ga)                      # each internal side once
        for k in range(N):
            q = np.stack([lev[wa, k], lev[wa, k + 1], lev[wb, k + 1], lev[wb, k]], 1)[keep]
            c1 = lcell(fa_e[keep], k)
            c2 = np.where(both[keep], lcell(np.maximum(ga[keep], 0), k), -1)
            pat = np.where(both[keep], -1, np.where(g[keep] >= 0, bpatch[np.maximum(g[keep], 0)], -1))
            # orientation: q points out of c1; internal faces are owned by the lower cell
            flip = (c2 >= 0) & (c2 < c1)
            q[flip] = q[flip][:, ::-1]
            o2 = np.where(flip, c2, c1); n2 = np.where(flip, c1, c2)
            # collapsed ends: drop repeated points (quad -> triangle), drop degenerate
            ca = q[:, 0] == q[:, 1]
            cb = q[:, 2] == q[:, 3]
            # side faces between layer cells of different zones extend the interface
            z2 = czone[own[Lf[A[fa_e[keep]]]]]
            zg = np.where(both[keep], czone[own[Lf[A[np.maximum(ga[keep], 0)]]]], -1)
            zone_side = np.where(both[keep] & (z2 != zg), np.maximum(z2, zg), -1)
            for msk, cols in ((~ca & ~cb, [0, 1, 2, 3]), (ca & ~cb, [0, 2, 3]), (~ca & cb, [0, 1, 2])):
                if not msk.any():
                    continue
                qq = q[msk][:, cols]
                add(np.full(int(msk.sum()), len(cols)), qq.ravel(), o2[msk], n2[msk], pat[msk],
                    z_=zone_side[msk])
        n_all = np.concatenate(F_n); v_all = np.concatenate(F_v)
        o_all = np.concatenate(F_own); ne_all = np.concatenate(F_nei)
        p_all = np.concatenate(F_pat); z_all = np.concatenate(F_zone); old_all = np.concatenate(F_old)
        # cells: original + layer cells of active faces
        nCn = nC + nA * N
        lz = np.repeat(czone[own[fa]], N)
        cz = np.concatenate([czone, lz])

        # ------------------------------------------------------------ ordering
        intl = ne_all >= 0
        ii = np.nonzero(intl)[0]
        ii = ii[np.lexsort((ne_all[ii], o_all[ii]))]
        bb = np.nonzero(~intl)[0]
        bb = bb[np.lexsort((o_all[bb], p_all[bb]))]
        order = np.concatenate([ii, bb])
        fptr = _ptr(n_all)
        new_n = n_all[order]
        new_v = v_all[_gather_idx(fptr, order)]
        new_own = o_all[order]
        new_nei = ne_all[ii]
        oi = old_all[ii]
        max_no = np.where(oi >= 0, lim0[np.maximum(oi, 0)], float(s["maxNonOrtho"]))
        bad = _cells_bad(P, new_n, new_v, new_own, new_nei, nCn, max_no)
        nbad = int(bad.sum())
        log(f"  iteration {it}: {ne}/{nW} points extruded, {nA} faces with layers,"
            f" {nbad} invalid cells")
        if nbad == 0 or it == n_tot:
            break
        # ------------------------------------------------------------ shrink at bad cells
        hit = np.zeros(nW, bool)
        bl = np.nonzero(bad[nC:])[0] // N              # active face index of bad layer cells
        if len(bl):
            hit[np.unique(lfw[np.isin(lff, A[bl])])] = True
        bo = np.nonzero(bad[:nC])[0]
        if len(bo):
            fo = np.nonzero(np.isin(own, bo) | np.isin(np.concatenate([nei, np.full(nF - nI, -1)]), bo))[0]
            pv = np.zeros(nP0, bool)
            pv[vv[_gather_idx(ptr, fo)]] = True
            # wall points that displace these cells: the nearest ones along the edges
            reach = pv.copy()
            live = np.zeros(nP0, bool); live[W[T > 0]] = True
            for _ in range(int(s["nRings"]) + 1):
                if (reach & live).any():
                    break
                reach = rings(reach, 1)
            hit[widx[reach & live]] = True
        if it < n_half:
            T[hit] *= 0.5
        else:
            T[hit] = 0.0

    # ------------------------------------------------------------ result
    nb_new = np.bincount(p_all[bb], minlength=len(patches))
    out_p, start = [], len(ii)
    for k, p in enumerate(patches):
        out_p.append(dict(p, startFace=start, nFaces=int(nb_new[k])))
        start += int(nb_new[k])
    newid = np.empty(len(order), np.int64); newid[order] = np.arange(len(order))
    out_cz = {zn: np.nonzero(cz == zi)[0] for zi, zn in enumerate(zone_names)}
    out_fz = {}
    oldmap = np.full(nF, -1, np.int64)
    has_old = old_all >= 0
    oldmap[old_all[has_old]] = newid[np.nonzero(has_old)[0]]
    for name, (fl, flip) in face_zones.items():
        fl = oldmap[np.asarray(fl, np.int64)]
        flip = np.asarray(flip, bool)
        zi = next((i for i, zn in enumerate(zone_names) if fz_names[zn] == name), -1)
        add_f = np.nonzero(z_all == zi)[0] if zi >= 0 else np.zeros(0, np.int64)
        nf_ = newid[add_f]
        nflip = cz[o_all[add_f]] != zi                 # normal points into the zone
        out_fz[name] = (np.concatenate([fl, nf_]), np.concatenate([flip, nflip]))
    full = T >= 0.99 * T_target
    log(f"addLayers: {nA * N} layer cells, total {nCn} cells; wall points with full thickness"
        f" {100 * full.mean():.1f}%, reduced {100 * ((T > 0) & ~full).mean():.1f}%,"
        f" none {100 * (T <= 0).mean():.1f}% ({time.time() - t0:.1f}s)")
    if nbad:
        log(f"addLayers: WARNING {nbad} invalid cells remain")
    return dict(mesh, points=P, faces=(new_n, new_v), owner=new_own, neighbour=new_nei,
                patches=out_p, cell_zones=out_cz, face_zones=out_fz,
                layer_thickness=dict(points=W, T=T))


def _gather_idx(ptr, faces):
    faces = np.asarray(faces, np.int64)
    n = ptr[faces + 1] - ptr[faces]
    off = np.repeat(ptr[faces] - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    return np.arange(int(n.sum())) + off


def main(argv=None):
    """python -m cutmesh.layers CASE [layerDict.json] [-out CASE2] [-noCheck]

    Adds layers to CASE/constant/polyMesh (written by cutmesh or any ASCII polyMesh)
    and writes the result to CASE2 (default: CASE, overwriting the mesh)."""
    import json
    import sys
    from .readfoam import read_mesh
    from .polymesh import write_polymesh
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(main.__doc__); return 0
    out = None
    if "-out" in argv:
        i = argv.index("-out"); out = argv[i + 1]; del argv[i:i + 2]
    check = "-noCheck" not in argv
    argv = [a for a in argv if a != "-noCheck"]
    case = argv[0]
    spec = {}
    if len(argv) > 1:
        with open(argv[1]) as f:
            spec = json.load(f)
        spec = spec.get("addLayers", spec)
    m = read_mesh(case)
    L = add_layers(m, spec)
    write_polymesh(out or case, L["points"], L["faces"], L["owner"], L["neighbour"], L["patches"],
                   L["cell_zones"], L["face_zones"])
    if check:
        quality.check(L["points"], L["faces"], L["owner"], L["neighbour"])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
