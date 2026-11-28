"""Sliding-mesh interfaces: a rotating zone's interface faceZone split into two coupled patches.

The interface of a conforming rotating zone is a set of internal faces (faceZone
<name>_interface).  For a sliding mesh each of these faces becomes two boundary faces, one
owned by the cell inside the zone and one by the cell outside, and the points of the
interface are duplicated so that the zone can rotate without dragging the outside cells
(what createBaffles + mergeOrSplitBaffles -split / splitBaffles do in OpenFOAM).

Two OpenFOAM families write this differently (checked against their tutorials):
    "com"   openfoam.com (ESI, v2xxx): patches <name>_AMI1 (zone side) / <name>_AMI2 of
            type cyclicAMI (tutorials/multiphase/interFoam/RAS/mixerVesselAMI)
    "org10", "org11", "org12"   openfoam.org (Foundation): patches <name>_nonCouple1 (zone
            side) / <name>_nonCouple2 of type patch; couple them after reading the mesh with
            createNonConformalCouples -overwrite <name>_nonCouple1 <name>_nonCouple2
            (tutorials/incompressibleFluid/mixerVessel2D)
constant/dynamicMeshDict (solid-body rotation of the zones' cells) follows the same choice.

    python -m cutmesh.ami CASE [-zone NAME ...] [-openfoam com|org10|org11|org12]
                               [-out CASE2] [-noDynamicMeshDict] [-ascii | -binary]
converts an existing mesh with the cellZone NAME and the faceZone NAME_interface (written
by cutmesh, or any mesh with that pair).  The rotation for dynamicMeshDict is read from
constant/MRFProperties (MRF_NAME) when there is one.
"""
import os
import re

import numpy as np

STYLES = ("com", "org10", "org11", "org12")


def patch_names(name, style):
    if style == "com":
        return f"{name}_AMI1", f"{name}_AMI2"
    return f"{name}_nonCouple1", f"{name}_nonCouple2"


def patch_entries(name, style):
    """the two new patches (zone side first) as boundary dicts, without faces"""
    a, b = patch_names(name, style)
    if style == "com":
        extra = lambda nb: {"matchTolerance": "0.0001", "neighbourPatch": nb, "transform": "noOrdering"}
        return [dict(name=a, type="cyclicAMI", extra=extra(b)),
                dict(name=b, type="cyclicAMI", extra=extra(a))]
    return [dict(name=a, type="patch"), dict(name=b, type="patch")]


def _gather(sizes, flat, ptr, ids, reverse):
    """CSR of the faces ids (vertex order reversed where reverse)"""
    ids = np.asarray(ids, np.int64)
    n = sizes[ids]
    start = ptr[ids]
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    k = np.arange(int(n.sum())) - np.repeat(off, n)                 # position within the face
    k = np.where(np.repeat(reverse, n), np.repeat(n, n) - 1 - k, k)
    return n, flat[np.repeat(start, n) + k]


def split_interface(mesh, name, style="com"):
    """Split the faceZone <name>_interface of the cellZone <name> into two patches and
    duplicate its points on the zone side.  mesh: dict(points, faces (sizes, flat), owner,
    neighbour, patches, cell_zones, face_zones); returns a new dict (the faceZone dropped)."""
    if style not in STYLES:
        raise ValueError(f"openfoam must be one of {STYLES}, got {style!r}")
    P = np.asarray(mesh["points"], float)
    sizes, flat = (np.asarray(a, np.int64) for a in mesh["faces"])
    own = np.asarray(mesh["owner"], np.int64)
    nei = np.asarray(mesh["neighbour"], np.int64)
    nI, nF = len(nei), len(sizes)
    nC = int(max(own.max(), nei.max() if nI else 0)) + 1
    fzname = name + "_interface"
    if name not in mesh["cell_zones"] or fzname not in mesh["face_zones"]:
        raise ValueError(f"zone {name!r}: needs the cellZone {name!r} and the faceZone {fzname!r}")
    inz = np.zeros(nC, bool)
    inz[np.asarray(mesh["cell_zones"][name], np.int64)] = True
    fz = np.unique(np.asarray(mesh["face_zones"][fzname][0], np.int64))
    if len(fz) == 0:
        raise ValueError(f"zone {name!r}: the faceZone {fzname!r} is empty")
    if fz.max() >= nI:
        raise ValueError(f"zone {name!r}: the faceZone {fzname!r} holds boundary faces"
                         " (already split?)")
    if np.any(inz[own[fz]] == inz[nei[fz]]):
        raise ValueError(f"zone {name!r}: faces of {fzname!r} do not separate the cellZone"
                         " from the rest")
    is_if = np.zeros(nF, bool); is_if[fz] = True
    other = np.nonzero(~is_if[:nI] & (inz[own[:nI]] != inz[nei]))[0]
    if len(other):
        raise ValueError(f"zone {name!r}: {len(other)} internal faces between the zone and the rest"
                         f" are not in {fzname!r}")
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    # points of the interface: the zone side gets copies
    ip = np.unique(flat[np.repeat(is_if, sizes)])
    copy = np.full(len(P), -1, np.int64)
    copy[ip] = len(P) + np.arange(len(ip))
    zone_side = inz[own].copy()                    # faces of zone cells (internal: both cells)
    zone_side[fz] = False                          # interface faces: handled below
    fid = np.repeat(np.arange(nF), sizes)
    flat2 = flat.copy()
    m = zone_side[fid] & (copy[flat] >= 0)
    flat2[m] = copy[flat[m]]
    # new face order: kept internal faces, old patches, zone-side copies, outside copies
    keep_i = np.nonzero(~is_if[:nI])[0]
    o_in = inz[own[fz]]                            # interface face owned by the zone cell
    zcell = np.where(o_in, own[fz], nei[fz])
    ocell = np.where(o_in, nei[fz], own[fz])
    parts = [
        (keep_i, np.zeros(len(keep_i), bool), flat2),
        (np.arange(nI, nF), np.zeros(nF - nI, bool), flat2),
    ]
    n1, f1 = _gather(sizes, flat, ptr, fz, ~o_in)  # zone side: normal out of the zone cell
    m1 = copy[f1] >= 0
    f1 = np.where(m1, copy[f1], f1)
    n2, f2 = _gather(sizes, flat, ptr, fz, o_in)   # outside: normal out of the outside cell
    S, F = [], []
    for ids, rev, fl in parts:
        n_, f_ = _gather(sizes, fl, ptr, ids, rev)
        S.append(n_); F.append(f_)
    S += [n1, n2]; F += [f1, f2]
    owner = np.concatenate([own[keep_i], own[nI:], zcell, ocell])
    patches = []
    start = len(keep_i)
    for p in mesh["patches"]:
        q = dict(p); q["startFace"] = start; start += int(p["nFaces"])
        patches.append(q)
    for pe in patch_entries(name, style):
        pe.update(startFace=start, nFaces=len(fz)); start += len(fz)
        patches.append(pe)
    names = [p["name"] for p in patches]
    if len(set(names)) != len(names):
        raise ValueError(f"zone {name!r}: patch names {patch_names(name, style)} are taken")
    fzs = {k: v for k, v in mesh["face_zones"].items() if k != fzname}
    # faces of the other faceZones: internal ids move (deleted interface faces before them),
    # boundary ids keep their order
    new_id = np.full(nF, -1, np.int64)
    new_id[keep_i] = np.arange(len(keep_i))
    new_id[nI:] = len(keep_i) + np.arange(nF - nI)
    for k, (fl, flip) in list(fzs.items()):
        fl = np.asarray(fl, np.int64)
        ok = new_id[fl] >= 0
        fzs[k] = (new_id[fl[ok]], np.asarray(flip, bool)[ok])
    out = dict(mesh)
    out.update(points=np.concatenate([P, P[ip]]), faces=(np.concatenate(S), np.concatenate(F)),
               owner=owner, neighbour=nei[keep_i], patches=patches, face_zones=fzs)
    out["ami_split"] = dict(out.get("ami_split", {}), **{name: dict(faces=len(fz), points=len(ip))})
    return out


def _surface_normals(geom, X):
    """outward-ish normal of the zone surface at the points X (on the surface): analytic for
    a cylinder (side, bottom or top, whichever is nearest), the closest triangle's for an STL"""
    if hasattr(geom, "R") and hasattr(geom, "_local"):            # Cylinder
        t, r, dv = geom._local(X)
        k = np.argmin(np.stack([np.abs(r - geom.R), np.abs(t), np.abs(t - geom.H)], 1), 1)
        return np.where((k == 0)[:, None], dv, np.where((k == 1)[:, None], -geom.a, geom.a))
    _, _, sid = geom.closest(X)
    T = geom.s.tris[np.asarray(sid, np.int64)]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def _rim_point(geom, feat, A, B, nrm):
    """where the zone's sharp edge (rim) crosses the plane through the edge A-B with normal
    nrm, nearest to the edge; None when it does not cross close to the edge"""
    u = B - A
    L = float(np.linalg.norm(u))
    if L == 0:
        return None
    cands = []
    if hasattr(geom, "R") and hasattr(geom, "_local"):            # the two circles, exactly
        for c in (geom.p1, geom.p2):
            # n.(c - A) + R (cos th n.e1 + sin th n.e2) = 0
            p, q, s = float(nrm @ geom.e1) * geom.R, float(nrm @ geom.e2) * geom.R, float(nrm @ (c - A))
            rr = np.hypot(p, q)
            if rr == 0 or abs(s) > rr:
                continue
            ph = np.arctan2(q, p)
            dth = np.arccos(np.clip(-s / rr, -1.0, 1.0))
            for th in (ph + dth, ph - dth):
                cands.append(c + geom.R * (np.cos(th) * geom.e1 + np.sin(th) * geom.e2))
    elif feat is not None and feat._kd is not None:
        idx = feat._kd.query_ball_point(0.5 * (A + B), L + feat._half)
        if idx:
            S = feat.seg[idx]
            sA = (S[:, 0] - A) @ nrm
            sB = (S[:, 1] - A) @ nrm
            ok = (sA * sB <= 0) & (sA != sB)
            if ok.any():
                t = sA[ok] / (sA[ok] - sB[ok])
                cands = list(S[ok, 0] + t[:, None] * (S[ok, 1] - S[ok, 0]))
    best, bd = None, np.inf
    for X in cands:
        t = float((X - A) @ u) / (L * L)
        if not 0.02 < t < 0.98:
            continue
        d = float(np.linalg.norm(A + t * u - X))
        if d < 0.75 * L and d < bd:              # a right-angle corner: up to L / 2
            best, bd = X, d
    return best


def _newell(Q):
    c = Q.mean(0)
    return np.cross(Q - c, np.roll(Q, -1, 0) - c).sum(0)


def _plane_split(pts, polys, x0, n, tol):
    """Split a cell (its faces polys, oriented outward) by the plane through x0 with normal
    n.  New points on cut edges are tokens ("e", a, b) with coordinates in the returned
    dict.  Returns (lower, upper, loop, coords): lower / upper [(face index, polygon)], loop
    the new face oriented out of the lower piece; None when the plane does not split the
    cell into two closed pieces."""
    s = {}
    for p in polys:
        for v in p:
            if v not in s:
                x = float((pts(v) - x0) @ n)
                s[v] = 0.0 if abs(x) < tol else x
    if min(s.values()) >= 0 or max(s.values()) <= 0:
        return None
    lower, upper, coords = [], [], {}

    def xyz(v):
        return coords[v] if isinstance(v, tuple) else pts(v)

    def on(v):
        return isinstance(v, tuple) or s[v] == 0.0

    def despike(q):
        """drop in-plane vertices where the outline turns back along the plane (the part of
        a non-convex face, e.g. next to a rim point, keeps a vertex of the other part)"""
        changed = True
        while changed and len(q) > 3:
            changed = False
            for k in range(len(q)):
                a, v, b = q[k - 1], q[k], q[(k + 1) % len(q)]
                if on(a) and on(v) and on(b) and float((xyz(v) - xyz(a)) @ (xyz(b) - xyz(v))) < 0:
                    q = q[:k] + q[k + 1:]
                    changed = True
                    break
        return q

    for i, p in enumerate(polys):
        sp = [s[v] for v in p]
        if min(sp) >= 0 and max(sp) > 0:
            upper.append((i, list(p)))
        elif max(sp) <= 0 and min(sp) < 0:
            lower.append((i, list(p)))
        elif max(sp) == 0 and min(sp) == 0:      # in the plane: bounds the piece behind it
            nf = _newell(np.array([pts(v) for v in p]))
            (lower if nf @ n > 0 else upper).append((i, list(p)))
        else:
            lo, up = [], []
            m = len(p)
            for k in range(m):
                a, b = p[k], p[(k + 1) % m]
                if s[a] <= 0:
                    lo.append(a)
                if s[a] >= 0:
                    up.append(a)
                if s[a] * s[b] < 0:
                    tok = ("e", min(a, b), max(a, b))
                    if tok not in coords:
                        t = s[a] / (s[a] - s[b])
                        coords[tok] = pts(a) + t * (pts(b) - pts(a))
                    lo.append(tok); up.append(tok)
            lo, up = despike(lo), despike(up)
            if len(lo) < 3:
                upper.append((i, list(p)))
            elif len(up) < 3:
                lower.append((i, list(p)))
            else:
                lower.append((i, lo)); upper.append((i, up))

    def open_edges(piece):
        de = set()
        for _, p in piece:
            for k in range(len(p)):
                de.add((p[k], p[(k + 1) % len(p)]))
        return {(a, b) for a, b in de if (b, a) not in de}

    ol = open_edges(lower)
    if len(ol) < 3:
        return None
    nxt = {}
    for a, b in ol:                              # the new face runs the open edges backwards
        if b in nxt:
            return None
        nxt[b] = a
    # the loop starts at its smallest vertex (original vertices before the cut-edge tokens,
    # which hold a string: the order of a set of them changes from run to run)
    v0 = min(nxt, key=lambda v: (1, v[1], v[2]) if isinstance(v, tuple) else (0, v, 0))
    loop, v = [v0], nxt[v0]
    while v != v0 and v in nxt and len(loop) <= len(ol):
        loop.append(v); v = nxt[v]
    if v != v0 or len(loop) != len(ol):
        return None
    if open_edges(upper) != {(loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))}:
        return None
    return lower, upper, loop, coords


def _cell_ok(pts, polys, vol_min):
    """(every face pyramid clearly positive and the volume above vol_min, volume)"""
    Cf, Sf = [], []
    for p in polys:
        Q = np.array([pts(v) for v in p])
        Cf.append(Q.mean(0)); Sf.append(0.5 * _newell(Q))
    Cf = np.array(Cf); Sf = np.array(Sf)
    est = Cf.mean(0)
    pv = np.einsum("ij,ij->i", Sf, Cf - est)
    vol = pv.sum() / 3.0
    if vol <= vol_min:
        return False, vol
    cc = (pv[:, None] * (0.75 * Cf + 0.25 * est)).sum(0) / pv.sum()
    h = vol ** (1.0 / 3.0)
    sp = np.einsum("ij,ij->i", Sf, Cf - cc)
    return bool(np.all(sp > 0.02 * np.linalg.norm(Sf, axis=1) * h)), vol


def exact_rims(mesh, name, geom, feature_angle=30.0, log=None):
    """Make the interface of the zone <name> follow the sharp edges (rims) of the zone
    surface.  A cut cell outside a convex rim wraps around it, and its interface face is
    folded over the rim (some vertices on the cap, some on the side).  OpenFOAM takes a
    face by its centre, which then lies inside the corner: the interface is chamfered
    there, differently at every angle, so the turned zone does not fit itself (cyclicAMI
    weights far from 1 at the rims).

    1. On every interface edge from one smooth part of the surface to the other, the rim
       point is inserted (where the rim crosses the plane of the cell faces through that
       edge), in every face using the edge.
    2. The outside cell of a folded face is cut in two by the plane of one part of the
       face (either part, whichever works), which splits the folded face into its two
       parts.  A cut is kept only when both cells keep clearly positive face pyramids and
       neither is a sliver; otherwise the cell stays as it is.

    Run before split_interface.  Returns (mesh, n_cells_cut, n_folded_left)."""
    from .geometry import FeatureSet
    P0 = np.asarray(mesh["points"], float)
    sizes, flat = (np.asarray(a, np.int64) for a in mesh["faces"])
    own0 = np.asarray(mesh["owner"], np.int64)
    nei0 = np.asarray(mesh["neighbour"], np.int64)
    nI, nF = len(nei0), len(sizes)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    fz0 = np.unique(np.asarray(mesh["face_zones"][name + "_interface"][0], np.int64))
    if len(fz0) == 0:
        return mesh, 0, 0
    nC = int(max(own0.max(), nei0.max() if nI else 0)) + 1
    inz = np.zeros(nC, bool)
    inz[np.asarray(mesh["cell_zones"][name], np.int64)] = True

    # ---- the interface vertices: on a rim, or on a smooth part (with its normal)
    fpoly = {int(f): flat[ptr[f]:ptr[f + 1]] for f in fz0}
    V = np.unique(np.concatenate(list(fpoly.values())))
    h = float(np.median(np.concatenate([np.linalg.norm(P0[np.roll(p, -1)] - P0[p], axis=1)
                                        for p in fpoly.values()])))
    tol = 1e-3 * h                               # snapped vertices sit within ~1e-4 h of the rim
    if hasattr(geom, "R") and hasattr(geom, "_local"):
        feat = None
        t, r, _ = geom._local(P0[V])
        on_f = (np.abs(r - geom.R) < tol) & ((np.abs(t) < tol) | (np.abs(t - geom.H) < tol))
    else:
        segs = geom.s.feature_edges(feature_angle)
        if len(segs) == 0:
            return mesh, 0, 0
        feat = FeatureSet(segs, L=geom.L)
        _, d = feat.closest(P0[V])
        on_f = d < tol
    if on_f.all():
        return mesh, 0, 0
    nrm = np.zeros((len(P0), 3))
    nrm[V[~on_f]] = _surface_normals(geom, P0[V[~on_f]])
    feat_v = np.zeros(len(P0), bool)
    feat_v[V[on_f]] = True
    cmin = np.cos(np.radians(feature_angle))

    def same(a, b):
        return float(nrm[a] @ nrm[b]) >= cmin

    # quick exit: no interface face has both parts of the surface or two rim vertices
    cand = [f for f, p in fpoly.items()
            if feat_v[p].sum() >= 2 or any(not feat_v[a] and not feat_v[b] and not same(a, b)
                                           for a, b in zip(p, np.roll(p, -1)))]
    if not cand:
        return mesh, 0, 0

    # ---- mutable mesh
    faces = [flat[ptr[f]:ptr[f + 1]].tolist() for f in range(nF)]
    own = own0.tolist()
    nei = nei0.tolist() + [-1] * (nF - nI)
    patch_of = [-1] * nF
    for k, pa in enumerate(mesh["patches"]):
        s0 = int(pa["startFace"])
        for f in range(s0, s0 + int(pa["nFaces"])):
            patch_of[f] = k
    origin = list(range(nF))
    rev = [False] * nF
    alive = [True] * nF
    new_pts = []

    def pts(v):
        return P0[v] if v < len(P0) else new_pts[v - len(P0)]

    vfaces = {}

    def touch(f):
        for v in faces[f]:
            vfaces.setdefault(v, set()).add(f)

    order0 = np.argsort(flat, kind="stable")
    fids = np.repeat(np.arange(nF), sizes)[order0]
    bounds = np.searchsorted(flat[order0], np.arange(len(P0) + 1))

    def faces_at(v):
        if v not in vfaces:
            vfaces[v] = set(fids[bounds[v]:bounds[v + 1]].tolist()) if v < len(P0) else set()
        return vfaces[v]

    def add_face(p, o, nb, pa, org, rv):
        f = len(faces)
        faces.append(p); own.append(o); nei.append(nb); patch_of.append(pa)
        origin.append(org); rev.append(rv); alive.append(True)
        for v in p:
            faces_at(v).add(f)
        return f

    def kill(f):
        alive[f] = False
        for v in faces[f]:
            faces_at(v).discard(f)

    def insert_on_edge(a, b, x):
        for g in list(faces_at(a) & faces_at(b)):
            p = faces[g]
            m = len(p)
            for k in range(m):
                if (p[k] == a and p[(k + 1) % m] == b) or (p[k] == b and p[(k + 1) % m] == a):
                    faces[g] = p[:k + 1] + [x] + p[k + 1:]
                    faces_at(x).add(g)
                    break

    is_if = np.zeros(nF, bool)
    is_if[fz0] = True

    # ---- 1) rim points on the interface edges crossing a rim
    nrim = 0
    done_e = set()
    rim_ids = set()
    for f in cand:
        p = list(fpoly[f])
        for a, b in zip(p, p[1:] + p[:1]):
            if feat_v[a] or feat_v[b] or same(a, b) or (min(a, b), max(a, b)) in done_e:
                continue
            done_e.add((min(a, b), max(a, b)))
            X = None
            for g in faces_at(a) & faces_at(b):
                if g < nF and is_if[g]:
                    continue
                nn_ = _newell(np.array([pts(v) for v in faces[g]]))
                ln = np.linalg.norm(nn_)
                if ln > 0:
                    X = _rim_point(geom, feat, P0[a], P0[b], nn_ / ln)
                    if X is not None:
                        break
            if X is None:
                continue
            x = len(P0) + len(new_pts)
            new_pts.append(np.asarray(X, float))
            rim_ids.add(x)
            insert_on_edge(a, b, x)
            nrim += 1

    def is_feat(v):
        return v in rim_ids if v >= len(P0) else bool(feat_v[v])

    def fold_parts(q):
        """the two parts (A, B) of a polygon folded over a rim (on different smooth parts
        of the surface, between two rim vertices), None when it is not folded, False when
        it is folded in a way not handled (more than one fold)"""
        nq = len(q)
        folds = [k for k in range(nq) if is_feat(q[k])]
        if len(folds) < 2:
            return None
        runs = []
        for i, s0 in enumerate(folds):
            e0 = folds[(i + 1) % len(folds)]
            runs.append([q[(s0 + j) % nq] for j in range(1, (e0 - s0) % nq or nq)
                         if q[(s0 + j) % nq] < len(P0)])      # original vertices (with a normal)
        keep = [i for i in range(len(folds))
                if runs[i - 1] and runs[i] and not same(runs[i - 1][0], runs[i][0])]
        if not keep:
            return None
        if len(keep) != 2:
            return False
        s0, e0 = folds[keep[0]], folds[keep[1]]
        if (e0 - s0) % nq in (1, nq - 1):
            return None                          # the rim runs along an edge: not folded
        return ([q[(s0 + j) % nq] for j in range((e0 - s0) % nq + 1)],
                [q[(e0 + j) % nq] for j in range((s0 - e0) % nq + 1)])

    # ---- folded interface faces, by the cell outside the zone
    folded = {}
    nleft = 0
    for f in cand:
        fp = fold_parts(faces[f])
        if fp is False:
            nleft += 1
        elif fp:
            folded.setdefault(nei[f] if inz[own[f]] else own[f], []).append(f)

    # ---- 2) cut each outside cell by the plane of a part of its folded face
    cell_faces = {c: [] for c in folded}
    for g in range(len(faces)):
        if alive[g]:
            if own[g] in cell_faces:
                cell_faces[own[g]].append(g)
            if nei[g] in cell_faces:
                cell_faces[nei[g]].append(g)
    new_src = []                                 # source cell of every added cell
    ncut = 0
    for c, fl in folded.items():
        gl = [g for g in cell_faces[c] if alive[g]]
        # the cell's faces, out of the cell; its folded faces as their two parts
        pc, keys, planes = [], [], []
        for g in gl:
            p = faces[g] if own[g] == c else faces[g][::-1]
            fp = fold_parts(faces[g]) if g in fl else None
            if fp:
                for part in fp:
                    pc.append(part if own[g] == c else part[::-1]); keys.append(g)
                    if g == fl[0]:
                        Q = np.array([pts(v) for v in part])
                        nn_ = _newell(Q)
                        ln = np.linalg.norm(nn_)
                        if ln > 0:
                            planes.append((float(np.abs((Q - Q.mean(0)) @ (nn_ / ln)).max()),
                                           Q.mean(0), nn_ / ln))
            else:
                pc.append(p); keys.append(g)
        _, vol_c = _cell_ok(pts, pc, -np.inf)
        done = False
        for _, x0, n in sorted(planes, key=lambda t: t[0]):      # the flatter part first
            res = _plane_split(pts, pc, x0, n, 0.01 * h)
            if res is None:
                continue
            lower, upper, loop, coords = res
            tp = lambda v: coords[v] if isinstance(v, tuple) else pts(v)
            ok1, _ = _cell_ok(tp, [p for _, p in lower] + [loop], 0.02 * vol_c)
            ok2, _ = _cell_ok(tp, [p for _, p in upper] + [loop[::-1]], 0.02 * vol_c)
            if not (ok1 and ok2):
                continue
            # commit
            real = {}
            for tok, X in coords.items():
                real[tok] = len(P0) + len(new_pts)
                new_pts.append(X)
            Rl = lambda p: [real[v] if isinstance(v, tuple) else v for v in p]
            u = nC + len(new_src)
            new_src.append(c)
            pieces = {}                          # face -> [(cell, polygon out of c)]
            for side, lst in ((c, lower), (u, upper)):
                for i, p in lst:
                    pieces.setdefault(keys[i], []).append((side, Rl(p)))
            mine = {c: [], u: []}
            for g in gl:
                pl = pieces[g]
                o_is_c = own[g] == c
                if len(pl) == 1 and pl[0][1] == (faces[g] if o_is_c else faces[g][::-1]):
                    cell = pl[0][0]
                    if cell == u:
                        if o_is_c:
                            own[g] = u
                        else:
                            nei[g] = u
                    mine[cell].append(g)
                    continue
                nb = nei[g] if o_is_c else own[g]
                kill(g)
                for cell, part_ in pl:
                    if o_is_c:
                        fn = add_face(part_, cell, nb, patch_of[g], origin[g], rev[g])
                    else:
                        fn = add_face(part_[::-1], nb, cell, patch_of[g], origin[g], rev[g])
                    mine[cell].append(fn)
                    if nb in cell_faces:
                        cell_faces[nb].append(fn)
            fn = add_face(Rl(loop), c, u, -1, -1, False)     # out of the lower piece (c)
            mine[c].append(fn); mine[u].append(fn)
            cell_faces[c] = mine[c]; cell_faces[u] = mine[u]
            for tok, x in real.items():
                insert_on_edge(tok[1], tok[2], x)
            ncut += 1
            done = True
            break
        if not done:
            nleft += len(fl)

    if nrim == 0 and ncut == 0:
        return mesh, 0, nleft

    # ---- rebuild: internal faces in upper-triangular order, then the patches
    nC2 = nC + len(new_src)
    internal, bnd = [], {}
    for f in range(len(faces)):
        if not alive[f]:
            continue
        if nei[f] >= 0:
            if own[f] > nei[f]:
                own[f], nei[f] = nei[f], own[f]
                faces[f] = faces[f][::-1]
                rev[f] = not rev[f]
            internal.append(f)
        else:
            bnd.setdefault(patch_of[f], []).append(f)
    internal.sort(key=lambda f: (own[f], nei[f], f))
    order = list(internal)
    patches = []
    for k, pa in enumerate(mesh["patches"]):
        q = dict(pa)
        q["startFace"] = len(order)
        q["nFaces"] = len(bnd.get(k, []))
        order += bnd.get(k, [])
        patches.append(q)
    newid = {f: i for i, f in enumerate(order)}
    by_origin = {}
    for f in order:
        if origin[f] >= 0:
            by_origin.setdefault(origin[f], []).append(f)
    src = np.array(new_src, np.int64)
    czs, inz_all = {}, {}
    for zn, cells in mesh["cell_zones"].items():
        m = np.zeros(nC2, bool)
        m[np.asarray(cells, np.int64)] = True
        if len(src):
            m[nC:] = m[src]
        inz_all[zn] = m
        czs[zn] = np.nonzero(m)[0]
    fzs = {}
    for k, (fl, fp) in mesh["face_zones"].items():
        zn = k[:-len("_interface")] if k.endswith("_interface") else None
        ids, flips = [], []
        for f0, fp0 in zip(np.asarray(fl, np.int64).tolist(), np.asarray(fp, bool).tolist()):
            for f in by_origin.get(f0, []):
                ids.append(newid[f])
                if zn in inz_all and nei[f] >= 0:
                    flips.append(not inz_all[zn][own[f]])    # normal points into the zone
                else:
                    flips.append(fp0 != rev[f])
        o = np.argsort(ids, kind="stable")
        fzs[k] = (np.asarray(ids, np.int64)[o], np.asarray(flips, bool)[o])
    out = dict(mesh)
    out.update(points=np.concatenate([P0, np.array(new_pts, float).reshape(-1, 3)]),
               faces=(np.array([len(faces[f]) for f in order], np.int64),
                      np.array([v for f in order for v in faces[f]], np.int64)),
               owner=np.array([own[f] for f in order], np.int64),
               neighbour=np.array([nei[f] for f in internal], np.int64),
               patches=patches, face_zones=fzs, cell_zones=czs)
    cz = mesh.get("cell_zone")
    if cz is not None and len(cz) == nC and len(src):
        out["cell_zone"] = np.concatenate([np.asarray(cz), np.asarray(cz)[src]])
    if log:
        log(f"zone '{name}': interface made to follow the rims: {nrim} rim points inserted,"
            f" {ncut} cells outside a rim cut in two"
            + (f", {nleft} folded faces left (the cut would make a bad cell)" if nleft else ""))
    return out, ncut, nleft


def axisymmetry_error(mesh, name, geom, origin, axis, angle_deg=7.0, n_max=5000):
    """Largest change of the distance from the zone surface of the interface face centres
    when turned by angle_deg about the axis: about 0 for a surface of revolution about that
    axis (the zone can slide on itself), large otherwise (a wrong axis, a non-round zone).
    The change, not the distance: faces where the cut chamfers the zone's rim are off the
    surface before turning too."""
    sizes, flat = mesh["faces"]
    fl = np.asarray(mesh["face_zones"][name + "_interface"][0], np.int64)
    if len(fl) > n_max:
        fl = fl[np.linspace(0, len(fl) - 1, n_max).astype(np.int64)]
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    P = np.asarray(mesh["points"], float)
    C = np.array([P[flat[ptr[f]:ptr[f + 1]]].mean(0) for f in fl.tolist()])
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    t = np.radians(angle_deg)
    X = C - np.asarray(origin, float)
    Xr = X * np.cos(t) + np.cross(a, X) * np.sin(t) + np.outer(X @ a, a) * (1 - np.cos(t))
    d0 = np.abs(geom.sdf(C))
    d1 = np.abs(geom.sdf(Xr + np.asarray(origin, float)))
    return float(np.abs(d1 - d0).max())


def zone_points(mesh, name):
    """points used by the cells of the cellZone name"""
    sizes, flat = mesh["faces"]
    own, nei = np.asarray(mesh["owner"]), np.asarray(mesh["neighbour"])
    nC = int(max(own.max(), nei.max() if len(nei) else 0)) + 1
    inz = np.zeros(nC, bool); inz[np.asarray(mesh["cell_zones"][name], np.int64)] = True
    fsel = inz[own].copy()
    fsel[:len(nei)] |= inz[nei]
    return np.unique(np.asarray(flat)[np.repeat(fsel, sizes)])


# ------------------------------------------------------------------ dynamicMeshDict
_HEADER = """FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      dynamicMeshDict;
}}
// generated by cutmesh for {what} - check omega (rad/s), origin and axis

"""


def _v(x):
    return "(" + " ".join(f"{float(c):.9g}" for c in x) + ")"


def dynamic_mesh_dict(zones, style):
    """text of constant/dynamicMeshDict rotating the cellZones of zones
    ([dict(name, origin, axis, omega)]) as solid bodies"""
    if style not in STYLES:
        raise ValueError(f"openfoam must be one of {STYLES}, got {style!r}")
    if style == "com":
        head = _HEADER.format(what="openfoam.com (ESI)")
        head += "dynamicFvMesh   dynamicMotionSolverFvMesh;\n\nmotionSolverLibs (fvMotionSolvers);\n\n"
        if len(zones) == 1:
            z = zones[0]
            return head + (f"motionSolver    solidBody;\n\ncellZone        {z['name']};\n\n"
                           f"solidBodyMotionFunction  rotatingMotion;\n\n"
                           f"origin      {_v(z['origin'])};\naxis        {_v(z['axis'])};\n"
                           f"omega       {float(z['omega']):.9g}; // [rad/s]\n")
        body = "".join(f"""    {z['name']}
    {{
        solidBodyMotionFunction  rotatingMotion;
        rotatingMotionCoeffs
        {{
            origin      {_v(z['origin'])};
            axis        {_v(z['axis'])};
            omega       {float(z['omega']):.9g}; // [rad/s]
        }}
    }}
""" for z in zones)
        return head + "motionSolver    multiSolidBodyMotionSolver;\n\n" \
            f"multiSolidBodyMotionSolverCoeffs\n{{\n{body}}}\n"
    libs = '("libfvMeshMovers.so" "libfvMotionSolvers.so")' if style in ("org10", "org11") \
        else '("libfvMotionSolvers.so")'
    head = _HEADER.format(what=f"openfoam.org (OpenFOAM-{style[3:]})")

    def solid(z, ind):
        s = " " * ind
        return (f"{s}motionSolver    solidBody;\n{s}cellZone        {z['name']};\n"
                f"{s}solidBodyMotionFunction  rotatingMotion;\n"
                f"{s}origin      {_v(z['origin'])};\n{s}axis        {_v(z['axis'])};\n"
                f"{s}omega       {float(z['omega']):.9g}; // [rad/s]\n")
    if len(zones) == 1:
        return head + f"mover\n{{\n    type            motionSolver;\n    libs            {libs};\n\n" \
            + solid(zones[0], 4) + "}\n"
    body = "".join(f"        {z['name']}\n        {{\n" + solid(z, 12) + "        }\n" for z in zones)
    return head + f"mover\n{{\n    type            motionSolver;\n    libs            {libs};\n\n" \
        f"    motionSolver    motionSolverList;\n\n    solvers\n    {{\n{body}    }}\n}}\n"


def write_dynamic_mesh_dict(case_dir, zones, style):
    d = os.path.join(case_dir, "constant")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "dynamicMeshDict")
    with open(p, "w") as f:
        f.write(dynamic_mesh_dict(zones, style))
    return p


def couple_hint(names, style):
    """what to run in OpenFOAM after reading the mesh (openfoam.org), or None"""
    if style == "com":
        return None
    return "; ".join(f"createNonConformalCouples -overwrite {a} {b}"
                     for a, b in (patch_names(n, style) for n in names))


def read_mrf(case_dir):
    """{zone: dict(origin, axis, omega)} from constant/MRFProperties (as cutmesh writes it)"""
    p = os.path.join(case_dir, "constant", "MRFProperties")
    out = {}
    if not os.path.isfile(p):
        return out
    t = re.sub(r"//[^\n]*", "", open(p).read())
    for m in re.finditer(r"(\w+)\s*\{([^{}]*)\}", t):
        b = m.group(2)
        z = re.search(r"cellZone\s+(\S+);", b)
        if not z:
            continue
        g = lambda k: re.search(rf"\b{k}\s+([^;]+);", b)
        vec = lambda k: [float(x) for x in g(k).group(1).strip("() ").split()] if g(k) else None
        om = g("omega")
        out[z.group(1)] = dict(origin=vec("origin") or [0, 0, 0], axis=vec("axis") or [0, 0, 1],
                               omega=float(om.group(1).split()[0]) if om else 0.0)
    return out


GENERATED = "generated by cutmesh"      # in the header comment of the files cutmesh writes


def _block_names(t, key):
    """names of the sub-dictionaries "name { ... }" directly inside the dictionary key"""
    m = re.search(rf"\b{key}\s*\{{", t)
    if not m:
        return set()
    out, depth, i = set(), 1, m.end()
    while i < len(t) and depth:
        c = t[i]
        if c == "{":
            if depth == 1:
                w = re.search(r"([^\s{};]+)\s*$", t[m.end():i])
                if w:
                    out.add(w.group(1))
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return out


def _zones_in(path):
    """(the cellZones a dictionary refers to, written by cutmesh?) or None without the file.
    Zones are named by cellZone entries, or (multi-zone openfoam.com dynamicMeshDict) by the
    sub-dictionaries of multiSolidBodyMotionSolverCoeffs."""
    try:
        with open(path) as f:
            raw = f.read()
    except OSError:
        return None
    ours = GENERATED in raw[:2000]
    t = re.sub(r"//[^\n]*", "", raw)
    refs = set(re.findall(r"\bcellZone\s+([^\s;]+)\s*;", t)) | _block_names(t, "multiSolidBodyMotionSolverCoeffs")
    return refs, ours


def tidy_extra_files(case_dir, mrf_names, ami_names, wrote_mrf, wrote_dynamic, log=print):
    """constant/MRFProperties and constant/dynamicMeshDict left in the case from an earlier
    run, for zones this mesh does not turn that way (e.g. an earlier sliding-mesh run, now
    meshed with MRF zones: the zone would turn with the mesh and in a rotating frame).
    A file cutmesh wrote is removed when this mesh has no zones of its kind; otherwise the
    zones it should not name are reported (a file the user wrote is never changed).
    wrote_*: the file was written by this run (it is up to date).  Returns the removed paths."""
    removed = []
    for fname, names, wrote, kind in (("MRFProperties", set(mrf_names), wrote_mrf, "MRF"),
                                      ("dynamicMeshDict", set(ami_names), wrote_dynamic, "sliding-mesh")):
        if wrote:
            continue
        p = os.path.join(case_dir, "constant", fname)
        found = _zones_in(p)
        if found is None:
            continue
        refs, ours = found
        bad = sorted(refs - names)
        if not bad and not (ours and not names and fname == "dynamicMeshDict"):
            continue
        if ours and not names:
            os.remove(p)
            removed.append(p)
            log(f"constant/{fname}: removed (left from an earlier run; this mesh has no {kind} zones)")
        else:
            log(f"WARNING: constant/{fname} (not written by this run) names cellZone "
                + ", ".join(repr(b) for b in bad) + f", not a {kind} zone of this mesh: edit or remove it")
    return removed


def drop_mrf_entries(case_dir, names):
    """Remove the entries of the cellZones names from constant/MRFProperties (the zones now
    turn with the mesh; an MRF entry left there would add rotating-frame terms on top).
    Other entries and the rest of the file stay as they are.  Returns the removed count."""
    p = os.path.join(case_dir, "constant", "MRFProperties")
    if not names or not os.path.isfile(p):
        return 0
    t = open(p).read()
    names = set(names)
    removed = 0

    def drop(m):
        nonlocal removed
        z = re.search(r"\bcellZone\s+([^\s;]+)\s*;", m.group(0))
        if z and z.group(1) in names:
            removed += 1
            return ""
        return m.group(0)
    # top-level entries "name { ... }" (MRF entries hold no nested braces), after the header
    h = re.search(r"FoamFile\s*\{[^{}]*\}", t)
    start = h.end() if h else 0
    body = re.sub(r"(?m)^[ \t]*\w+\s*\{[^{}]*\}[ \t]*\n?", drop, t[start:])
    if removed:
        with open(p, "w") as f:
            f.write(t[:start] + body)
    return removed


def main(argv=None):
    import sys
    from .readfoam import read_mesh
    from .polymesh import write_polymesh
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    zones, style, out, dyn, fmt, case = [], "com", None, True, None, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-zone", "-openfoam", "-out"):
            if i + 1 >= len(argv):
                print(f"cutmesh.ami: {a} needs a value", file=sys.stderr); return 2
            v = argv[i + 1]
            if a == "-zone":
                zones.append(v)
            elif a == "-openfoam":
                style = v
            else:
                out = v
            i += 2; continue
        if a == "-noDynamicMeshDict":
            dyn = False
        elif a in ("-ascii", "-binary"):
            fmt = a[1:]
        elif a.startswith("-"):
            print(f"cutmesh.ami: unknown option {a!r}", file=sys.stderr); return 2
        elif case is None:
            case = a
        else:
            print("cutmesh.ami: one CASE expected", file=sys.stderr); return 2
        i += 1
    if case is None:
        print("cutmesh.ami: CASE required", file=sys.stderr); return 2
    if style not in STYLES:
        print(f"cutmesh.ami: -openfoam must be one of {', '.join(STYLES)}", file=sys.stderr); return 2
    m = read_mesh(case)
    if not zones:
        zones = [k for k in m["cell_zones"] if k + "_interface" in m["face_zones"]]
        if not zones:
            print("ERROR: no cellZone NAME with a faceZone NAME_interface", file=sys.stderr); return 1
    try:
        for z in zones:
            m = split_interface(m, z, style)
            s = m["ami_split"][z]
            print(f"zone {z!r}: {s['faces']} interface faces -> patches"
                  f" {' / '.join(patch_names(z, style))}, {s['points']} points duplicated")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1
    dst = out or case
    write_polymesh(dst, m["points"], m["faces"], m["owner"], m["neighbour"], m["patches"],
                   m["cell_zones"], m["face_zones"], fmt=fmt or m.get("format", "binary"))
    mrf = read_mrf(case)
    nd = drop_mrf_entries(dst, zones)
    if nd:
        print(f"removed {nd} converted zone(s) from {os.path.join(dst, 'constant', 'MRFProperties')}")
    if dyn:
        rz = [dict(name=z, **mrf[z]) for z in zones if z in mrf]
        if len(rz) == len(zones):
            print(f"wrote {write_dynamic_mesh_dict(dst, rz, style)}")
        else:
            print("no constant/MRFProperties entry for every zone: dynamicMeshDict not written"
                  " (write it by hand, or run cutmesh with \"ami\": true)")
    hint = couple_hint(zones, style)
    if hint:
        print(f"then in OpenFOAM: {hint}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
