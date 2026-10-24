"""checkMesh-style geometry and topology checks (same formulas as OpenFOAM's
primitiveMeshTools where practical), vectorised over flattened face lists."""
import numpy as np

from .geometry import _core      # Rust extension or None


class FlatFaces:
    """faces: list of vertex lists, or a (sizes, flat_vertices) tuple (CSR)."""

    def __init__(self, faces):
        if isinstance(faces, tuple):
            self.n = np.asarray(faces[0], np.int64)
            self.v = np.asarray(faces[1], np.int64)
            self.ptr = np.concatenate([[0], np.cumsum(self.n)])
        else:
            self.n = np.array([len(f) for f in faces], np.int64)
            self.ptr = np.concatenate([[0], np.cumsum(self.n)])
            self.v = np.fromiter((x for f in faces for x in f), np.int64, count=int(self.ptr[-1]))
        self.fid = np.repeat(np.arange(len(self.n)), self.n)
        nxt = np.arange(len(self.v)) + 1
        nxt[self.ptr[1:] - 1] = self.ptr[:-1]
        self.nxt = nxt


def face_geometry(points, ff):
    """Face centres and area vectors from a fan of triangles around the vertex average.
    The triangles are weighted by their area projected on the face normal (signed, as in
    current OpenFOAM): for a non-convex face, triangles folding back over the fan have
    negative weight, so overlapping parts are not counted twice."""
    if _core is not None:
        return _core.face_geometry(np.ascontiguousarray(points, float), ff.n, ff.v)
    p = points[ff.v]
    est = np.add.reduceat(p, ff.ptr[:-1], axis=0) / ff.n[:, None]
    e = est[ff.fid]
    a, b = p, points[ff.v[ff.nxt]]
    c = a + b + e
    nvec = np.cross(b - a, e - a)
    sumN = np.add.reduceat(nvec, ff.ptr[:-1], axis=0)
    magN = np.linalg.norm(sumN, axis=1)
    nhat = sumN / np.maximum(magN, 1e-300)[:, None]
    w = np.einsum("ij,ij->i", nvec, nhat[ff.fid])
    sumA = np.add.reduceat(w, ff.ptr[:-1])
    sumAc = np.add.reduceat(w[:, None] * c, ff.ptr[:-1], axis=0)
    ok = (magN > 1e-300) & (sumA > 1e-300)
    ctr = np.where(ok[:, None], sumAc / (3 * np.where(ok, sumA, 1.0)[:, None]), est)
    return ctr, 0.5 * sumN


def cell_geometry(nC, owner, neighbour, Cf, Sf):
    if _core is not None:
        return _core.cell_geometry(int(nC), np.asarray(owner, np.int64), np.asarray(neighbour, np.int64),
                                   np.ascontiguousarray(Cf, float), np.ascontiguousarray(Sf, float))
    nI = len(neighbour)
    cnt = np.bincount(owner, minlength=nC) + np.bincount(neighbour, minlength=nC)
    est = np.zeros((nC, 3))
    np.add.at(est, owner, Cf)
    np.add.at(est, neighbour, Cf[:nI])
    est /= np.maximum(cnt, 1)[:, None]
    po = np.einsum("ij,ij->i", Sf, Cf - est[owner])
    pn = -np.einsum("ij,ij->i", Sf[:nI], Cf[:nI] - est[neighbour])
    vol = np.bincount(owner, po, nC) + np.bincount(neighbour, pn, nC)
    cc = np.zeros((nC, 3))
    np.add.at(cc, owner, po[:, None] * (0.75 * Cf + 0.25 * est[owner]))
    np.add.at(cc, neighbour, pn[:, None] * (0.75 * Cf[:nI] + 0.25 * est[neighbour]))
    cc = cc / np.where(np.abs(vol) > 1e-300, vol, 1)[:, None]
    return cc, vol / 3.0


def face_skewness(points, ff, Cf, Sf, C, owner, neighbour, faces=None, limit=None):
    """skewness per face as in checkMesh: distance of the face centre from the line
    between the cell centres (boundary faces: from the normal through the owner centre),
    divided by the extent of the face in that direction, at least 0.2 x the centre
    distance (boundary faces: 0.4 x the distance owner centre - face).
    faces: evaluate only these (face ids), returned in that order.
    limit (scalar or per face): faces whose upper bound |sv| / (0.2 or 0.4 |d|) is at
    most this get the bound instead of the exact value (enough to compare with limit)"""
    nI = len(neighbour)
    f = np.arange(len(Sf)) if faces is None else np.asarray(faces, np.int64)
    if not len(f):
        return np.zeros(0)
    inner = f < nI
    Cpf = Cf[f] - C[owner[f]]
    d = np.empty((len(f), 3))
    d[inner] = C[neighbour[f[inner]]] - C[owner[f[inner]]]
    nb = Sf[f[~inner]] / np.maximum(np.linalg.norm(Sf[f[~inner]], axis=1), 1e-300)[:, None]
    d[~inner] = nb * np.einsum("ij,ij->i", nb, Cpf[~inner])[:, None]
    S = Sf[f]
    sv = Cpf - (np.einsum("ij,ij->i", S, Cpf) / (np.einsum("ij,ij->i", S, d) + 1e-300))[:, None] * d
    msv = np.linalg.norm(sv, axis=1)
    fd = np.where(inner, 0.2, 0.4) * np.linalg.norm(d, axis=1)
    sk = msv / (fd + 1e-300)
    e = np.arange(len(f)) if limit is None else np.nonzero(sk > limit)[0]
    if len(e):
        # the extent of the face in the direction of sv
        svh = sv[e] / (msv[e] + 1e-300)[:, None]
        nv = ff.n[f[e]]
        st = np.concatenate([[0], np.cumsum(nv)[:-1]])
        k = np.repeat(np.arange(len(e)), nv)
        vi = np.arange(int(nv.sum())) + np.repeat(ff.ptr[f[e]] - st, nv)
        proj = np.abs(np.einsum("ij,ij->i", svh[k], points[ff.v[vi]] - Cf[f[e]][k]))
        sk[e] = msv[e] / (np.maximum(fd[e], np.maximum.reduceat(proj, st)) + 1e-300)
    return sk


def check(points, faces, owner, neighbour, log=print):
    nI = len(neighbour)
    nC = int(max(owner.max(), neighbour.max() if len(neighbour) else 0)) + 1
    ff = faces if isinstance(faces, FlatFaces) else FlatFaces(faces)
    faces = ff.n                      # only len(faces) is used below
    Cf, Sf = face_geometry(points, ff)
    C, V = cell_geometry(nC, owner, neighbour, Cf, Sf)
    magSf = np.linalg.norm(Sf, axis=1)
    res = {}
    # --- topology: upper triangular ordering
    ok_ut = bool(np.all(owner[:nI] < neighbour))
    key = owner[:nI] * (nC + 1) + neighbour
    ok_order = bool(np.all(np.diff(key) >= 0))
    # --- closedness
    s = np.zeros((nC, 3)); np.add.at(s, owner, Sf); np.add.at(s, neighbour, -Sf[:nI])
    sa = np.bincount(owner, magSf, nC) + np.bincount(neighbour, magSf[:nI], nC)
    closed = np.linalg.norm(s, axis=1) / np.maximum(sa, 1e-300)
    # --- cell edge closedness (each directed edge has its reverse within the cell)
    cell_of_v = np.concatenate([owner[ff.fid], neighbour[ff.fid[ff.fid < nI]]])
    a = np.concatenate([ff.v, ff.v[ff.nxt][ff.fid < nI]])
    b = np.concatenate([ff.v[ff.nxt], ff.v[ff.fid < nI]])
    # neighbour side sees the face reversed: swap a/b for those
    nown = len(ff.v)
    a2 = a.copy(); b2 = b.copy()
    a2[nown:] = ff.v[ff.nxt][ff.fid < nI]
    b2[nown:] = ff.v[ff.fid < nI]
    NP = len(points) + 1
    fw = (cell_of_v * NP + a2) * NP + b2
    bw = (cell_of_v * NP + b2) * NP + a2
    fs = np.sort(fw)
    pos = np.clip(np.searchsorted(fs, bw), 0, len(fs) - 1)
    open_edges = int(np.sum(fs[pos] != bw))
    dup = int(len(fs) - len(np.unique(fs)))
    # --- non-orthogonality / skewness
    d = C[neighbour] - C[owner[:nI]]
    cosang = np.einsum("ij,ij->i", d, Sf[:nI]) / np.maximum(np.linalg.norm(d, axis=1) * magSf[:nI], 1e-300)
    nonorth = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    sk = face_skewness(points, ff, Cf, Sf, C, owner, neighbour)
    skew, skew_b = sk[:nI], sk[nI:]
    nB = len(faces) - nI
    # --- face pyramids (tets from face triangles to owner / neighbour centres)
    e = Cf[ff.fid]
    pa, pb = points[ff.v], points[ff.v[ff.nxt]]
    tri_n = np.cross(pb - pa, e - pa) * 0.5
    vo = np.einsum("ij,ij->i", tri_n, e - C[owner[ff.fid]]) / 3.0
    neg_own = np.minimum.reduceat(vo, ff.ptr[:-1]) <= 0
    iv = ff.fid < nI
    vn = np.einsum("ij,ij->i", tri_n[iv], e[iv] - C[neighbour[ff.fid[iv]]]) / 3.0
    neg_nei = np.zeros(len(faces), bool)
    if nI:
        neg_nei[:nI] = np.maximum.reduceat(vn, ff.ptr[:nI]) >= 0
    # checkMesh's face-pyramid test uses the whole face (area vector + centre)
    pyr_o = np.einsum("ij,ij->i", Sf, Cf - C[owner]) / 3.0
    pyr_n = np.einsum("ij,ij->i", Sf[:nI], C[neighbour] - Cf[:nI]) / 3.0
    nPyr = int(np.sum(pyr_o <= 0) + np.sum(pyr_n <= 0))
    res.update(
        nPoints=len(points), nCells=nC, nFaces=len(faces), nInternalFaces=nI,
        upperTriangular=ok_ut and ok_order,
        maxClosedness=float(closed.max()),
        openCellEdges=open_edges, duplicateCellEdges=dup,
        minVol=float(V.min()), maxVol=float(V.max()), totalVol=float(V.sum()),
        nNegVol=int(np.sum(V <= 0)),
        maxNonOrth=float(nonorth.max()) if nI else 0.0,
        avgNonOrth=float(nonorth.mean()) if nI else 0.0,
        nNonOrth70=int(np.sum(nonorth > 70)),
        maxSkewness=float(max(skew.max() if nI else 0, skew_b.max() if nB else 0)),
        nSkew4=int(np.sum(skew > 4) + np.sum(skew_b > 4)),
        nBadPyramids=nPyr,
        nBadTetDecomp=int(neg_own.sum() + neg_nei.sum()),
        minFaceArea=float(magSf.min()),
    )
    res["_C"] = C; res["_V"] = V; res["_nonorth"] = nonorth; res["_skew"] = skew
    res["_ff"] = ff; res["_Cf"] = Cf; res["_Sf"] = Sf
    if log:
        log("  checkMesh-style report")
        log(f"    points {res['nPoints']}, cells {nC}, faces {len(faces)} (internal {nI})")
        log(f"    upper-triangular order: {'OK' if res['upperTriangular'] else 'FAILED'}")
        log(f"    cell closedness max {res['maxClosedness']:.2e}, open cell edges {open_edges}")
        log(f"    volume min {res['minVol']:.3e} max {res['maxVol']:.3e} total {res['totalVol']:.6e}"
            f"  (non-positive: {res['nNegVol']})")
        log(f"    non-orthogonality max {res['maxNonOrth']:.1f} avg {res['avgNonOrth']:.2f}"
            f"  (>70: {res['nNonOrth70']})")
        log(f"    skewness max {res['maxSkewness']:.2f}  (>4: {res['nSkew4']})")
        log(f"    face pyramids (checkMesh test): {res['nBadPyramids']} non-positive")
        log(f"    face-triangle tets (stricter, tet decomposition): {res['nBadTetDecomp']} non-positive")
    return res
