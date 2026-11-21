"""cutmesh pipeline: STL interior -> octree -> cut cells (STL walls and rotating-zone
walls are both represented exactly by mesh faces) -> OpenFOAM polyMesh."""
import itertools
import json
import os
import time
import numpy as np
from scipy.spatial import cKDTree

from .geometry import (Surface, Cylinder, SurfaceZone, FeatureSet, read_stl,
                       sample_segments, cylinder_features, paused_gc, _core, LevelSet, unique, STLError,
                       subdivide_triangles)
from .octree import Octree
from .cylbase import CylindricalBase
from .cutcell import CutContext, split_components
from .polymesh import write_polymesh
from .featclip import PlaneClip, clip_cell, clipped_face, make_alloc
from .config import validate_config, check_names, ConfigError  # noqa: F401
from . import quality
from . import diagnose


DEFAULTS = {
    "bufferLayers": 2,          # cells per level in refinement transitions
    "padding": 1,               # base cells around the STL bounding box
    "featureAngle": 30.0,
    "featureLevel": None,       # None -> max surface level
    "snapTolerance": 0.3,       # snap vertices within this fraction of h onto surfaces
    "featureSnapTolerance": 0.5,  # snap vertices within this fraction of h onto feature edges
    "featureSnapMax": 0.5,      # experimental: up to this fraction where the octree cells stay valid
    "concaveSnapMax": 0.75,     # concave (re-entrant) feature edges: up to this fraction, where valid
    "mergeRatio": 0.25,         # merge cut cells smaller than this fraction of their hex
    "hangingSnap": "free",      # hanging nodes that cannot slide onto a surface: "free" projection or "constrained" (treated as on-surface)
    "fillThinGaps": 2,          # passes splitting cells with fluid inside but no fluid corner (Cartesian base; 0 = off)
    "checkLeaks": True,         # check the final mesh for fluid joined through a solid wall
    "strictInput": False,       # stop on open / non-manifold STL edges instead of warning
    "volumeCheckTolerance": 0.05,  # warn when the mesh volume differs more from the STL's
    "minRegionVolume": 1e-3,    # remove fluid regions cut off from the rest below this volume fraction
    "fillThinWalls": 3,         # passes splitting cells with solid inside but no solid corner (Cartesian base; 0 = off)
    "thinCutCells": 2,          # passes splitting cut cells keeping < thinCutFraction fluid next to finer cells (Cartesian base; 0 = off)
    "thinCutFraction": 0.35,
    "mergeMaxNonOrtho": 65.0,   # [deg] a merge may not make a face more non-orthogonal than this
    "mergeWallFaces": True,     # join coplanar wall pieces of one cell into one face
    "featureCaps": "domain",    # bend the cut along convex feature edges: "domain", "zone", "both" or false
    "exactFeatureCut": True,    # cut the cells at convex feature edges by clipping with the STL planes (featclip)
    "surfaceRefinement": {},
    "refinementRegions": [],
    "patchTypes": {},
    "rotatingZones": [],
    "writeMRFProperties": True,
    "writeFormat": "binary",    # polyMesh points/faces/owner/neighbour: "binary" or "ascii"
    "openfoam": "com",          # sliding-mesh patches and dynamicMeshDict: "com" (ESI) or "org10"/"org11"/"org12"
    "writeDynamicMeshDict": True,   # with "ami" zones: constant/dynamicMeshDict (solid-body rotation)
    "amiExactRims": True,       # with "ami" zones: the interface follows the zone's rims (cells outside a rim cut in two)
}


class Timer:
    def __init__(self, log):
        self.t0 = time.time(); self.t = self.t0; self.log = log; self.steps = []

    def __call__(self, name):
        now = time.time()
        self.steps.append((name, now - self.t))
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
            mem = f", peak {rss:.2f} GB"
        except Exception:
            mem = ""
        self.log(f"[{now - self.t0:7.2f}s] {name} ({now - self.t:.2f}s{mem})")
        self.t = now


def load_config(path):
    with open(path, encoding="utf-8-sig") as f:      # editors on Windows may add a BOM
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: not valid JSON: {e}") from None
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: a meshDict is a JSON object {{...}}, got {type(cfg).__name__}")
    base = os.path.dirname(os.path.abspath(path))
    full = dict(DEFAULTS); full.update(cfg)
    full["_base"] = base
    return full


def _rel(cfg, p):
    return p if os.path.isabs(p) else os.path.join(cfg["_base"], p)


def build_zones(cfg, log=None):
    zones = []
    for z in cfg["rotatingZones"]:
        if "cylinder" in z:
            c = z["cylinder"]
            g = Cylinder(c["point1"], c["point2"], c["radius"])
            origin = c.get("origin", c["point1"])
            axis = (np.asarray(c["point2"], float) - np.asarray(c["point1"], float))
            axis = axis / np.linalg.norm(axis)
        else:
            s = Surface(read_stl(_rel(cfg, z["stl"])), log=log, what=f"rotating zone {z['name']!r} STL")
            g = SurfaceZone(s)
            origin = z.get("origin", [0, 0, 0])
            axis = z.get("axis", [0, 0, 1])
        zones.append(dict(name=z["name"], geom=g, surfaceLevel=z.get("surfaceLevel", 0),
                          volumeLevel=z.get("volumeLevel", 0), origin=np.asarray(origin, float),
                          axis=np.asarray(axis, float), omega=z.get("omega", 0.0),
                          nonRotatingPatches=z.get("nonRotatingPatches", []),
                          wallProximity=z.get("wallProximity"),
                          conforming=bool(z.get("conforming", True)),
                          ami=bool(z.get("ami", False))))
    return zones


def check_zone_overlap(zones, spacing):
    """ConfigError when two rotating zones overlap (one's surface passes through the
    other's inside, one contains the other, or they coincide).  Surfaces that only
    touch are allowed.  spacing: distance of the surface samples tested."""
    probs = []
    for i, a in enumerate(zones):
        for b in zones[i + 1:]:
            tol = 1e-6 * max(a["geom"].L, b["geom"].L)
            for x, y in ((a, b), (b, a)):
                p, _ = x["geom"].samples(spacing)
                # also points inside x (a coarse grid over its box, off its surface):
                # zones with the same surface have every surface sample on the other's
                # surface, but share their inside
                lo, hi = p.min(0), p.max(0)
                g = np.linspace(0.0, 1.0, 12)[1:-1]
                G = lo + (hi - lo) * np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
                G = G[x["geom"].inside(G)]
                if len(G):
                    _, dx, _ = x["geom"].closest(G)
                    G = G[dx > tol]
                p = np.concatenate([p, G])
                p = p[y["geom"].inside(p)]
                if len(p):
                    _, d, _ = y["geom"].closest(p)
                    p = p[d > tol]
                if len(p):
                    q = p[0]
                    probs.append(f"rotating zones {x['name']!r} and {y['name']!r} overlap"
                                 f" (e.g. at ({q[0]:.6g} {q[1]:.6g} {q[2]:.6g}))")
                    break
    if probs:
        raise ConfigError("meshDict: " + "; ".join(probs) + " - rotating zones must not overlap")


def _newell(P):
    return np.cross(P, np.roll(P, -1, 0)).sum(0)


def make_chord_hook(feat):
    """Face refinement: where a sharp feature pierces a face that the level set
    splits, return the piercing point so the face's cut chord bends through it.
    Only convex corners of the kept (phi < 0) side: bending the chord into the kept
    side would make the cell non-convex."""
    if feat is None or feat._kd is None:
        return None
    if getattr(feat, "_rs", None) is not None:
        def chord_rs(ctx, poly, a, b):
            pos, phi = ctx.pos, ctx.phi
            x = feat._rs.pierce([pos[v] for v in poly], pos[a], pos[b],
                                [pos[v] for v in poly if phi[v] < 0])
            return None if x is None else list(x)
        chord_rs.feat = feat                 # the Rust cut kernel pierces itself
        return chord_rs

    def chord(ctx, poly, a, b):
        pos = ctx.pos
        V = np.array([pos[v] for v in poly])
        A = np.array(pos[a]); B = np.array(pos[b])
        mid = 0.5 * (A + B)
        diam = float(np.ptp(V, 0).max())
        idx = feat._kd.query_ball_point(mid, diam + feat._half)
        if not idx:
            return None
        n = _newell(V)
        nn = np.linalg.norm(n)
        if nn == 0:
            return None
        n /= nn
        c0 = V.mean(0)
        S = feat.seg[idx]
        sA = (S[:, 0] - c0) @ n
        sB = (S[:, 1] - c0) @ n
        ok = (sA * sB <= 0) & (sA != sB)
        if not ok.any():
            return None
        t = sA[ok] / (sA[ok] - sB[ok])
        X = S[ok, 0] + t[:, None] * (S[ok, 1] - S[ok, 0])
        X -= ((X - c0) @ n)[:, None] * n
        E = np.roll(V, -1, 0) - V
        side = np.einsum("kj,j->k", np.cross(E[None], X[:, None] - V[None]).reshape(-1, 3), n).reshape(len(X), -1)
        inside = (side >= -1e-9 * diam * diam).all(1)
        u = B - A
        L = float(np.linalg.norm(u))
        if L == 0:
            return None
        dl = np.linalg.norm(np.cross(X - A, u), axis=1) / L
        good = inside & (dl > 1e-3 * diam) & (dl < L)
        if not good.any():
            return None
        X = X[good]
        # side of the chord: the kept vertices on one side, a convex corner on the other
        neg = [k for k, v in enumerate(poly) if ctx.phi[v] < 0]
        sn = np.cross(u, V[neg] - A) @ n
        sx = np.cross(u, X - A) @ n
        X = X[(sx * sn.mean() < 0)]
        if not len(X):
            return None
        return X[np.argmin(np.linalg.norm(X - mid, axis=1))].tolist()

    return chord


def make_pinch_hook(dom, ccv, h_of_vertex):
    """A face cut by a straight chord a-b next to a concave edge: a surface vertex of the
    kept part beyond the chord (the tip of the wedge of solid, a vertex on the edge)
    with solid between it and the chord.  The chord bends through it: the kept part of
    the face splits in two there (the cell splits along the edge, cutcell.cut)."""
    def hook(ctx, poly, a, b, cand):
        pos = ctx.pos
        A = np.array(pos[a]); B = np.array(pos[b])
        V = np.array([pos[v] for v in poly])
        n = _newell(V - V.mean(0))
        nn = np.linalg.norm(n)
        u = B - A
        L = np.linalg.norm(u)
        if nn == 0 or L == 0:
            return None
        n /= nn
        side = np.cross(u, n)                   # in the face plane, normal to the chord
        neg = [pos[v] for v in poly if ctx.phi[v] < 0]
        if not neg:
            return None
        sk = np.sign(np.dot(np.mean(neg, 0) - A, side))   # the kept side
        X = np.array([pos[v] for v in cand])
        dist = sk * (X - A) @ side / L
        h = float(min(h_of_vertex(a), h_of_vertex(b)))
        ok = dist > 0.02 * h
        if not ok.any():
            return None
        # beyond the chord, within the face span along it
        t = (X - A) @ u / (L * L)
        ok &= (t > 0) & (t < 1)
        if not ok.any():
            return None
        k = np.nonzero(ok)[0]
        # on (or next to) a concave edge
        _, de = ccv.closest(X[k])
        k = k[de < 0.25 * h]
        k = k[np.argsort(-dist[k])]
        for j in k[:2].tolist():
            T = X[j]
            # solid between the chord and the vertex, fluid beyond the vertex
            q = np.array([(A + B + T) / 3, T + 0.1 * (T - 0.5 * (A + B))])
            sd = dom.sdf(q)
            if sd[0] > 0.01 * h and sd[1] < 0:
                return cand[j]
        return None
    return hook


def _split_on_features(ctx, loop, feat, h, tol=0.05, reach=0.75, depth=4):
    """Split a cap loop along the feature edges through the cell: from loop vertex to
    loop vertex on the feature (entry/exit points), or spokes to a feature corner inside
    the cell.  Curved features are followed by recursive midpoint projection.
    Returns the list of polygons or None."""
    m = len(loop)
    pos = ctx.pos
    P = np.array([pos[v] for v in loop])
    _, dv = feat.closest(P)
    F = [k for k, v in enumerate(loop) if v in ctx.onfeat or dv[k] < 1e-6 * h]
    if len(F) < 2:
        return None
    amin = 1e-3 * h * h

    def path(A, B, lev=0):
        """points strictly between A and B along the feature, or None"""
        M = 0.5 * (A + B)
        cp, d = feat.closest(M[None])
        if d[0] < tol * h:
            return []
        if lev >= depth or d[0] > 0.5 * np.linalg.norm(B - A):
            return None
        l = path(A, cp[0], lev + 1)
        r = path(cp[0], B, lev + 1) if l is not None else None
        if r is None:
            return None
        return l + [cp[0]] + r

    def area_ok(polys):
        """non-degenerate pieces that fold convexly (outward normals, kept side inside)"""
        N, Cc = [], []
        for Q in polys:
            Q = np.asarray(Q)
            nq = _newell(Q - Q.mean(0))
            if len(Q) < 3 or np.linalg.norm(nq) < 2 * amin:
                return False
            N.append(nq / np.linalg.norm(nq)); Cc.append(Q.mean(0))
        for i in range(len(polys)):
            for j in range(len(polys)):
                if i != j and np.dot(Cc[j] - Cc[i], N[i]) > 1e-3 * h:
                    return False
        return True

    def add(X):
        vid = len(pos)
        pos.append(list(X)); ctx.phi.append(0.0)
        ctx.new_vertices.append(vid); ctx.new_parents.append((-1, -1))
        ctx.onfeat.add(vid)
        return vid

    # feature corner inside the cell: spokes from the corner to the entry points
    if len(F) >= 3 and len(feat.corners):
        dc = np.linalg.norm(feat.corners - P.mean(0), axis=1)
        j = int(np.argmin(dc))
        X = feat.corners[j]
        if dc[j] < reach * h and np.min(np.linalg.norm(P - X, axis=1)) > 1e-6 * h:
            spk, sp = [], {}
            for k in F:
                t = path(P[k], X)
                if t is not None:
                    spk.append(k); sp[k] = t
            if len(spk) >= 2:
                sect = []
                for i in range(len(spk)):
                    s0, s1 = spk[i], spk[(i + 1) % len(spk)]
                    q = [(s0 + k) % m for k in range(((s1 - s0) % m or m) + 1)]
                    sect.append((q, sp[s1], sp[s0]))
                geo = [[P[k] for k in q] + t1 + [X] + t0[::-1] for q, t1, t0 in sect]
                if area_ok(geo):
                    vx = add(X)
                    ids = {k: [add(x) for x in sp[k]] for k in spk}
                    return [[loop[k] for k in q] + ids[s1] + [vx] + ids[s0][::-1]
                            for (q, _, _), s0, s1 in zip(sect, spk, spk[1:] + spk[:1])]
    # entry point -> exit point along the feature; accept non-crossing splits
    polys = [list(loop)]
    used = 0
    lpos = {v: P[k] for k, v in enumerate(loop)}
    for a_ in range(len(F)):
        for b_ in range(a_ + 1, len(F)):
            vi, vj = loop[F[a_]], loop[F[b_]]
            for pi, q in enumerate(polys):
                if vi not in q or vj not in q:
                    continue
                i, j = sorted((q.index(vi), q.index(vj)))
                if j - i == 1 or (i == 0 and j == len(q) - 1):
                    break
                t = path(lpos[vi], lpos[vj])
                if t is None:
                    break
                g = lambda ids: [lpos[v] if v in lpos else np.asarray(pos[v]) for v in ids]
                g1 = g(q[i:j + 1]) + t[::-1]
                g2 = g(q[j:] + q[:i + 1]) + t
                if area_ok([g1, g2]):
                    tid = [add(x) for x in t]
                    polys[pi:pi + 1] = [q[i:j + 1] + tid[::-1], q[j:] + q[:i + 1] + tid]
                    used += 1
                break
    if not used:
        return None
    return polys


def make_feature_hook(feat, h_of_cell, min_bend=0.03, reach=0.75):
    """Cap refinement: split the cap along the sharp feature edges running through the
    cut cell (entry/exit points come from the face chords).  Otherwise, if a feature
    runs through the cell, insert the feature point into the cap and fan-triangulate,
    so the corner is not chamfered."""
    if feat is None or feat._kd is None:
        return None
    state = {"cell": 0, "n": 0, "split": 0}

    def hook(ctx, loop):
        if len(loop) < 3:
            return [loop]
        h = h_of_cell[state["cell"]]
        if state.get("forbid"):
            return [loop]
        sp = _split_on_features(ctx, loop, feat, h)
        if sp is not None:
            state["did"] = True
            return sp
        P = np.array([ctx.pos[v] for v in loop])
        c = P.mean(0)
        fp, d = feat.closest(c[None])
        fp = fp[0]
        if d[0] > reach * h:
            return [loop]
        # plane of the cap (Newell normal)
        nrm = np.cross(P - c, np.roll(P, -1, 0) - c).sum(0)
        nn = np.linalg.norm(nrm)
        if nn == 0:
            return [loop]
        bend = np.dot(fp - c, nrm / nn)     # > 0: feature bulges out of the kept cell
        if bend < min_bend * h:
            return [loop]                    # flat, or concave corner (keep cell convex)
        # the fan must stay consistently oriented
        tri_n = np.cross(P - fp, np.roll(P, -1, 0) - fp)
        # every fan triangle must keep the orientation and a non-negligible area
        if np.any(tri_n @ (nrm / nn) <= 0.1 * nn / len(loop)):
            return [loop]
        vid = len(ctx.pos)
        ctx.pos.append(fp.tolist()); ctx.phi.append(0.0)
        ctx.new_vertices.append(vid); ctx.new_parents.append((-1, -1))
        state["n"] += 1
        state["did"] = True
        m = len(loop)
        return [[loop[i], loop[(i + 1) % m], vid] for i in range(m)]

    hook.feat, hook.reach = feat, reach      # for the Rust cut kernel's quick test
    return hook, state


def _leaf_bad(P, leaves, fptr, fverts, cptr, csize, flat_at=None, flat_angle=2.0):
    """octree leaves (with moved vertices) where some face does not clearly see the
    cell centre - non-convex or inverted.  flat_at: vertices (mask) that must also keep
    a face corner (not pushed straight, at flat_angle degrees or less)"""
    if _core is not None and hasattr(_core, "leaf_bad"):
        return _core.leaf_bad(np.ascontiguousarray(P, float), np.asarray(leaves, np.int64),
                              np.asarray(fptr, np.int64), np.asarray(fverts, np.int64),
                              np.asarray(cptr, np.int64), np.ascontiguousarray(csize, float),
                              None if flat_at is None else np.ascontiguousarray(flat_at, bool),
                              float(np.sin(np.radians(flat_angle))))
    return _leaf_bad_py(P, leaves, fptr, fverts, cptr, csize, flat_at, flat_angle)


def _leaf_bad_py(P, leaves, fptr, fverts, cptr, csize, flat_at=None, flat_angle=2.0):
    occs = np.concatenate([np.arange(cptr[c], cptr[c + 1]) for c in leaves.tolist()])
    lens = cptr[leaves + 1] - cptr[leaves]
    owner_ = np.repeat(np.arange(len(leaves)), lens)
    nn_ = fptr[occs + 1] - fptr[occs]
    off = np.repeat(fptr[occs] - np.concatenate([[0], np.cumsum(nn_)[:-1]]), nn_)
    ffl = quality.FlatFaces((nn_, fverts[np.arange(int(nn_.sum())) + off]))
    cf_, sf_ = quality.face_geometry(P, ffl)
    est = np.zeros((len(leaves), 3)); np.add.at(est, owner_, cf_); est /= lens[:, None]
    pv = np.einsum("ij,ij->i", sf_, cf_ - est[owner_])
    vv = np.bincount(owner_, pv, len(leaves))
    cc = np.zeros((len(leaves), 3)); np.add.at(cc, owner_, pv[:, None] * (0.75 * cf_ + 0.25 * est[owner_]))
    cc /= np.where(np.abs(vv) > 0, vv, 1.0)[:, None]
    sp = np.einsum("ij,ij->i", sf_, cf_ - cc[owner_])
    hl = csize[leaves][owner_]
    af = np.linalg.norm(sf_, axis=1)
    badf = (sp < 0.05 * af * hl) | (af < 0.05 * hl * hl)    # collapsed edges give empty faces
    if flat_at is not None:
        # a face corner pushed straight (or past it): the corner vertex no longer bounds
        # the face and the cut breaks the cell (hanging vertices are straight by nature)
        prv = np.empty_like(ffl.nxt); prv[ffl.nxt] = np.arange(len(ffl.nxt))
        Q = P[ffl.v]; e_in = Q - P[ffl.v[prv]]; e_out = P[ffl.v[ffl.nxt]] - Q
        nh = sf_[ffl.fid] / np.maximum(af[ffl.fid], 1e-300)[:, None]
        s_ = np.einsum("ij,ij->i", np.cross(e_in, e_out), nh)
        lin = np.linalg.norm(e_in, axis=1) * np.linalg.norm(e_out, axis=1)
        flat = (s_ < np.sin(np.radians(flat_angle)) * lin) & flat_at[ffl.v]
        badf |= np.bincount(ffl.fid, flat.astype(int), len(af)) > 0
    return (np.bincount(owner_, badf.astype(int), len(leaves)) > 0) | (vv <= 0)


def _tets_ok(P, faces, cf, cc, vmin):
    """every (face centre, edge, cell centre) tet has a volume above vmin"""
    for (_, p), c in zip(faces, cf):
        Q = P[p]
        t = np.cross(np.roll(Q, -1, 0) - Q, c - Q) @ (c - cc) / 6.0
        if np.any(t <= vmin):
            return False
    return True


def _merged_face_ok(P, loop, parts, nrms, cmin, flat_tol):
    """the joined face keeps the geometry of its parts: every part normal is within the
    angle acos(cmin) of the joined normal (the pairwise test is transitive across a curved
    wall), all part vertices lie within flat_tol * sqrt(area) of its plane, and the fan
    triangles from both the vertex mean and the area-weighted centre are positive (a
    U-shaped loop has its centre in the notch)"""
    Q = P[loop]
    c0 = Q.mean(0)
    tri = np.cross(Q - c0, np.roll(Q, -1, 0) - c0)
    S = 0.5 * tri.sum(0)
    a = float(np.linalg.norm(S))
    if a <= 0:
        return False
    n = S / a
    if any(float(np.dot(m, n)) <= cmin for m in nrms):
        return False
    vs = np.unique(np.concatenate([np.asarray(q, np.int64) for q in parts]))
    if np.abs((P[vs] - c0) @ n).max() > flat_tol * np.sqrt(a):
        return False
    at = tri @ n
    if np.any(at <= 1e-9 * a):
        return False
    cf = ((at[:, None]) * (Q + np.roll(Q, -1, 0) + c0)).sum(0) / (3.0 * at.sum())
    at = np.cross(Q - cf, np.roll(Q, -1, 0) - cf) @ n
    return bool(np.all(at > 1e-9 * a))


def _join_cell_faces(P, polys, cmin, cpart, flat_tol):
    """merge_coplanar_wall_faces for the boundary faces of one cell in one patch:
    (faces, number of faces joined away)"""
    joined = 0
    nrm = []
    for q in polys:
        Q = P[q]
        s = np.cross(Q - Q.mean(0), np.roll(Q, -1, 0) - Q.mean(0)).sum(0)
        nrm.append(s / max(np.linalg.norm(s), 1e-300))
    par = list(range(len(polys)))

    def fnd(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    emap = {}
    for i, q in enumerate(polys):
        for k in range(len(q)):
            emap[(q[k], q[(k + 1) % len(q)])] = i
    for (a, b), i in emap.items():
        j = emap.get((b, a))
        if j is not None and j != i and float(np.dot(nrm[i], nrm[j])) > cmin:
            par[fnd(i)] = fnd(j)
    groups = {}
    for i in range(len(polys)):
        groups.setdefault(fnd(i), []).append(i)
    res = []
    for ids in groups.values():
        if len(ids) == 1:
            res.append(polys[ids[0]]); continue
        de = {}
        for i in ids:
            q = polys[i]
            for k in range(len(q)):
                e = (q[k], q[(k + 1) % len(q)])
                de[e] = de.get(e, 0) + 1
        segs = [e for e in de if (e[1], e[0]) not in de]
        nxt = {}
        ok = True
        for a, b in segs:
            if a in nxt:
                ok = False; break
            nxt[a] = b
        loop = []
        a = a0 = None
        if ok and segs:
            a0 = segs[0][0]; a = a0
            for _ in range(len(segs) + 1):
                loop.append(a); a = nxt.get(a)
                if a is None or a == a0:
                    break
        if (ok and a0 is not None and a == a0 and len(loop) == len(segs) and len(loop) >= 3
                and _merged_face_ok(P, loop, [polys[i] for i in ids], [nrm[i] for i in ids],
                                    cpart, flat_tol)):
            res.append(loop); joined += len(ids) - 1
        else:
            res.extend(polys[i] for i in ids)
    return res, joined


def merge_coplanar_wall_faces(P, nn, vv, owner, nI, patches, max_angle=5.0):
    """Join the boundary faces of one cell that lie in one plane (see
    _merge_coplanar_wall_faces_py; the Rust kernel gives the same faces)."""
    if _core is None or not hasattr(_core, "merge_wall_faces"):
        return _merge_coplanar_wall_faces_py(P, nn, vv, owner, nI, patches, max_angle)
    cmin = np.cos(np.radians(max_angle))
    cpart = np.cos(np.radians(0.5 * max_angle))
    flat_tol = 0.25 * np.sin(np.radians(max_angle))
    nn2, vv2, ow2, counts, joined = _core.merge_wall_faces(
        np.ascontiguousarray(P, float), np.ascontiguousarray(nn, np.int64), np.ascontiguousarray(vv, np.int64),
        np.ascontiguousarray(owner, np.int64), int(nI), [int(p["startFace"]) for p in patches],
        [int(p["nFaces"]) for p in patches], float(cmin), float(cpart), float(flat_tol))
    new_p, start = [], nI
    for p, nf in zip(patches, counts):
        new_p.append(dict(p, startFace=start, nFaces=nf))
        start += nf
    return nn2, vv2, ow2, new_p, joined


def _merge_coplanar_wall_faces_py(P, nn, vv, owner, nI, patches, max_angle=5.0):
    """Join the boundary faces of one cell that lie in one plane, belong to one patch and
    share edges into a single face.  Cut cells can carry a flat wall split into several
    faces; a piece off to the side of a thin cell has its centre far off the cell
    centre's normal, which becomes a large non-orthogonality once layers are added on top.
    Neighbouring pieces join when their normals are within max_angle; the joined face is
    accepted when every piece is within max_angle / 2 of its normal.  So two pieces of a
    curved wall folded by up to max_angle join (they are never exactly coplanar), while a
    strip that keeps bending is kept apart.  The out-of-plane bound matches: a fold by
    max_angle puts the vertices about 0.24 * max_angle [rad] * sqrt(area) off the plane.
    Returns (nn, vv, owner, patches, n_joined)."""
    ptr = np.concatenate([[0], np.cumsum(nn)])
    cmin = np.cos(np.radians(max_angle))
    cpart = np.cos(np.radians(0.5 * max_angle))
    flat_tol = 0.25 * np.sin(np.radians(max_angle))
    out_n = [nn[:nI]]; out_v = [vv[:ptr[nI]]]; out_o = [owner[:nI]]
    new_p, start, joined = [], nI, 0
    for p in patches:
        f0, nf = p["startFace"], p["nFaces"]
        # runs of faces of one cell: only those with several faces can join; the others
        # are copied as they are, a range of faces at a time
        o = np.asarray(owner[f0:f0 + nf])
        rs = np.flatnonzero(np.concatenate([[True], o[1:] != o[:-1]])) + f0
        re_ = np.concatenate([rs[1:], [f0 + nf]])
        multi = np.flatnonzero(re_ - rs > 1)
        pn, pv, po = [], [], []
        nfo = 0
        prev = f0
        for m in multi.tolist():
            f, g = int(rs[m]), int(re_[m])
            if f > prev:
                pn.append(nn[prev:f]); pv.append(vv[ptr[prev]:ptr[f]]); po.append(owner[prev:f])
                nfo += f - prev
            polys = [vv[ptr[k]:ptr[k + 1]].tolist() for k in range(f, g)]
            polys, nj = _join_cell_faces(P, polys, cmin, cpart, flat_tol)
            joined += nj
            pn.append(np.array([len(q) for q in polys], np.int64))
            pv.append(np.array([v for q in polys for v in q], np.int64))
            po.append(np.full(len(polys), owner[f], np.int64))
            nfo += len(polys)
            prev = g
        if f0 + nf > prev:
            pn.append(nn[prev:f0 + nf]); pv.append(vv[ptr[prev]:ptr[f0 + nf]]); po.append(owner[prev:f0 + nf])
            nfo += f0 + nf - prev
        out_n += [np.asarray(x, np.int64) for x in pn]
        out_v += [np.asarray(x, np.int64) for x in pv]
        out_o += [np.asarray(x, np.int64) for x in po]
        new_p.append(dict(p, startFace=start, nFaces=nfo))
        start += nfo
    return (np.concatenate(out_n), np.concatenate(out_v), np.concatenate(out_o), new_p, joined)


def _piece_volume(pos, comp):
    """volume enclosed by the faces of a piece (outward oriented)"""
    v = 0.0
    for _, p in comp:
        Q = np.array([pos[u] for u in p])
        c = Q.mean(0)
        v += float(np.dot(c, np.cross(Q - c, np.roll(Q, -1, 0) - c).sum(0))) / 6.0
    return v


def _piece_bad(pos, comp, h):
    """some face does not clearly see the centroid (same test as the merging stage)"""
    Cf, Sf = [], []
    for _, p in comp:
        Q = np.array([pos[v] for v in p])
        c = Q.mean(0)
        Cf.append(c); Sf.append(0.5 * np.cross(Q - c, np.roll(Q, -1, 0) - c).sum(0))
    Cf = np.array(Cf); Sf = np.array(Sf)
    est = Cf.mean(0)
    pv = np.einsum("ij,ij->i", Sf, Cf - est)
    if pv.sum() <= 0:
        return True
    cc = (pv[:, None] * (0.75 * Cf + 0.25 * est)).sum(0) / pv.sum()
    return bool(np.any(np.einsum("ij,ij->i", Sf, Cf - cc) < 0.05 * np.linalg.norm(Sf, axis=1) * h))


def surface_contact(dom, pts, phi, cells, faces_of, csize, push=0.2, face_ok=None, csr=None):
    """Faces touching the surface along more than a vertex (vertices with phi == 0,
    typically snapped onto a wall next to a concave edge).  Interpolating phi there
    chamfers the concave edge by up to a cell size, differently at every cell, which
    makes the wall bumpy along the edge.  The side of the surface is sampled just
    inside the face (per face, so both cells agree) or just inside the cell:

    - edges between two phi == 0 vertices (both really on the surface) of a face that is
      cut: +1 if the face is in the solid there, -1 if in the fluid -> CutContext.zero_edge
    - faces lying on the surface (all vertices with phi == 0, centre on the surface):
      dropped from the cells whose side of the face is solid (the cell on the fluid
      side keeps them as walls) -> {cell: {face id}}.  Sampled a little off the face
      into the cell, at the centre and towards every vertex; all must be solid.

    face_ok(ci): flags of the cell's faces that can matter (the others are skipped).
    csr: (cptr, fptr, fverts, f_key, flags per face) of the cells: sampled in Rust."""
    ekeys, esamp, etol, eends = [], [], [], []
    dcell, dfid, dctr, dsamp, dgrp, dtol = [], [], [], [], [], []
    seen = set()
    if csr is not None and _core is not None and hasattr(_core, "surface_contact_samples"):
        i64 = lambda a: np.ascontiguousarray(a, np.int64)            # noqa: E731
        cptr, fptr, fverts, f_key, fok = csr
        dcell, dfid, dctr, dtol, dsamp, dgrp, ekeys, esamp, etol, eends = _core.surface_contact_samples(
            np.ascontiguousarray(pts, float), np.ascontiguousarray(phi, float), [int(c) for c in cells],
            i64(cptr), i64(fptr), i64(fverts), i64(f_key), np.ascontiguousarray(csize, float),
            np.ascontiguousarray(fok, bool), float(push))
        cells = ()
    for ci in cells:
        fl = faces_of(ci)
        cc = None
        h = csize[ci]
        ok = face_ok(ci) if face_ok is not None else None
        for j, (fid, p) in enumerate(fl):
            if ok is not None and not ok[j]:
                continue
            ph = phi[p]
            z = ph == 0
            if z.all():
                Q = pts[p]
                cf = Q.mean(0)
                nrm = _newell(Q - cf)
                nn_ = np.linalg.norm(nrm)
                if nn_ == 0:
                    continue
                nrm /= nn_
                if cc is None:
                    cc = pts[[v for _, q in fl for v in q]].mean(0)
                if np.dot(cc - cf, nrm) < 0:
                    nrm = -nrm
                g = len(dcell)
                dcell.append(ci); dfid.append(fid); dctr.append(cf); dtol.append(h)
                off = 0.1 * h * nrm
                for x in [cf] + list(0.5 * (Q + cf)):
                    dsamp.append(x + off); dgrp.append(g)
                continue
            if z.sum() < 2 or not (ph < 0).any() or not (ph > 0).any():
                continue
            cf = None
            m = len(p)
            for k in range(m):
                a, b = p[k], p[(k + 1) % m]
                if not (z[k] and z[(k + 1) % m]):
                    continue
                key = (fid, a, b) if a < b else (fid, b, a)
                if key in seen:
                    continue
                seen.add(key)
                if cf is None:
                    cf = pts[p].mean(0)
                mid = 0.5 * (pts[a] + pts[b])
                ekeys.append(key); esamp.append(mid + push * (cf - mid)); etol.append(h)
                eends.append((a, b))
    zero_edge, drop = {}, {}
    nd, ns = len(dctr), len(dsamp)
    if nd + len(esamp):
        sd = dom.sdf(np.concatenate([np.reshape(x, (-1, 3)) for x in (dctr, dsamp, esamp)]))
        if nd:
            ht = np.array(dtol)
            on = np.abs(sd[:nd]) < 0.02 * ht
            solid = np.ones(nd, bool)
            np.logical_and.at(solid, np.array(dgrp), sd[nd:nd + ns] > 1e-3 * ht[np.array(dgrp)])
            # a face between surface vertices through the solid (the chord across a
            # wedge at a concave edge) is dropped from every cell
            inner = sd[:nd] > 0.02 * ht
            for k in np.nonzero((on & solid) | inner)[0].tolist():
                drop.setdefault(dcell[k], set()).add(dfid[k])
        se = sd[nd + ns:]
        eps = 1e-3 * np.array(etol)
        # only edges between vertices really on the surface (hanging vertices that could
        # not reach the surface also carry phi == 0)
        if eends:
            ev, inv = np.unique(np.array(eends), return_inverse=True)
            _, dv, _ = dom.closest(pts[ev])
            real = (dv[inv.reshape(-1, 2)] < 1e-3 * np.array(etol)[:, None]).all(1)
        for k, key in enumerate(ekeys):
            if real[k] and abs(se[k]) > eps[k]:
                zero_edge[key] = 1 if se[k] > 0 else -1
    return zero_edge, drop


def contact_splits(dom, pts, phi, cells, faces_of, csize, csr=None):
    """Faces partly in the fluid and partly lying on the wall: vertices with phi < 0 and
    a contiguous run of >= 3 surface vertices (phi == 0), e.g. the face on the wall
    plane at a hole rim, between the rim vertices and the wall.  Kept whole, the cell
    below keeps a solid wedge (the rim is chamfered).  They are split along the chord
    between the ends of the run (existing vertices, so both cells agree) into a fluid
    part and a contact part.  Returns {face id: (poly, fluid part, contact part, h)}.
    csr: (cptr, fptr, fverts, f_key) of the cells: the candidates are found in Rust."""
    if csr is not None and _core is not None and hasattr(_core, "contact_candidates"):
        cptr, fptr, fverts, f_key = csr
        keys, polys, rest, run, hs, ca, cb, wa, wb = _core.contact_candidates(
            np.ascontiguousarray(pts, float), np.ascontiguousarray(phi, float), [int(c) for c in cells],
            np.ascontiguousarray(cptr, np.int64), np.ascontiguousarray(fptr, np.int64),
            np.ascontiguousarray(fverts, np.int64), np.ascontiguousarray(f_key, np.int64),
            np.ascontiguousarray(csize, float))
        if not keys:
            return {}
        h = np.array(hs)
        ends = np.array([[r[0], r[-1]] for r in run])
        sd = dom.sdf(np.concatenate([ca, cb]))
        _, de, _ = dom.closest(pts[ends.ravel()])
        ok = _contact_ok(sd, de, h, np.array(wa), np.array(wb))
        return {f: (p, a, b, x) for f, p, a, b, x, o in zip(keys, polys, rest, run, hs, ok.tolist()) if o}
    cand = {}
    for ci in cells:
        for fid, p in faces_of(ci):
            if fid in cand:
                continue
            ph = phi[p]
            z = ph == 0
            if (ph > 0).any() or not (ph < 0).any() or z.sum() < 3:
                continue
            m = len(p)
            starts = [i for i in range(m) if z[i] and not z[i - 1]]
            if len(starts) != 1:
                continue
            i0 = starts[0]
            k = 0
            while z[(i0 + k) % m]:
                k += 1
            run = [p[(i0 + t) % m] for t in range(k)]
            rest = [p[(i0 + k - 1 + t) % m] for t in range(m - k + 2)]
            cand[fid] = (list(p), rest, run, float(csize[ci]))
    if not cand:
        return {}
    keys = list(cand)
    h = np.array([cand[f][3] for f in keys])
    ca = np.array([pts[cand[f][1]].mean(0) for f in keys])
    cb = np.array([pts[cand[f][2]].mean(0) for f in keys])
    ends = np.array([[cand[f][2][0], cand[f][2][-1]] for f in keys])
    sd = dom.sdf(np.concatenate([ca, cb]))
    _, de, _ = dom.closest(pts[ends.ravel()])
    def width(q):
        """2 x area / longest edge: small for a sliver along the edge"""
        Q = pts[q]
        a = np.linalg.norm(_newell(Q - Q.mean(0))) * 0.5
        return 2.0 * a / max(np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1).max(), 1e-300)

    wa = np.array([width(cand[f][1]) for f in keys])
    wb = np.array([width(cand[f][2]) for f in keys])
    ok = _contact_ok(sd, de, h, wa, wb)
    return {f: cand[f] for f, o in zip(keys, ok) if o}


def _contact_ok(sd, de, h, wa, wb):
    """contact_splits: the fluid part in the fluid, the contact part on the wall or in the
    solid (the wedge at a concave edge, cut off by the chord through the vertex on the
    edge), both run ends on the wall, and a real contact area (not a sliver along the edge)"""
    n = len(h)
    return (sd[:n] < -1e-3 * h) & ((np.abs(sd[n:]) < 0.02 * h) | (sd[n:] > 0.02 * h)) \
        & (de.reshape(-1, 2) < 1e-3 * h[:, None]).all(1) \
        & (wa > 0.05 * h) & (wb > 0.1 * h)


def _same_orientation(p, q):
    """q is p (cyclically) in the same direction"""
    m = len(p)
    if len(q) != m or p[0] not in q:
        return True
    i = q.index(p[0])
    return q[(i + 1) % m] == p[1]


def _unique_points(p):
    """p with (most) repeated points removed.  The STL samples repeat every subdivision
    vertex once per sub-triangle (about 2.7x the points); a KD-tree over fewer points
    is faster to build and to search, and gives the same nearest distances as long as
    every distinct point is kept.  Rows are sorted by a 64-bit hash of their bits and a
    row equal to the one before it is dropped (a hash collision can only leave a
    repeat in, never drop a distinct point)."""
    p = np.ascontiguousarray(p, float)
    if len(p) < 2:
        return p
    b = p.view(np.uint64)

    def mix(x):                                  # splitmix64 finaliser
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))
    o = np.argsort(mix(mix(mix(b[:, 0]) ^ b[:, 1]) ^ b[:, 2]))
    bs = b[o]
    keep = np.ones(len(p), bool)
    keep[1:] = (bs[1:] != bs[:-1]).any(1)
    return p[o[keep]]


def _nearest_in(pts, keep, q):
    """cKDTree(pts).query(q) over the points of the mask keep only: (distance, index into
    pts).  A tree over all the vertices costs more than the few queries it answers; the
    callers keep every vertex that can be near enough to matter (inf, 0 when none)."""
    idx = np.flatnonzero(keep)
    if not len(idx):
        return np.full(len(q), np.inf), np.zeros(len(q), np.int64)
    d, j = cKDTree(pts[idx]).query(q)
    return d, idx[j]


def _points_kd(p):
    """KD-tree for nearest-distance queries of the sample points p (distances only).
    Sliding-midpoint splits without shrinking the node boxes build ~2.5x faster than
    the default median splits; the nearest distances are exact either way."""
    return cKDTree(_unique_points(p), balanced_tree=False, compact_nodes=False)


class _PosList:
    """CutContext.pos (a list of [x, y, z] that grows) over an array of the first
    positions: a row becomes a list when it is read, instead of the whole array at once
    (pts.tolist() of ten million points takes seconds).  Writes to the first rows go
    into the array, which the caller shares on purpose."""

    def __init__(self, a):
        self.a, self.n0, self.added = a, len(a), []

    def __len__(self):
        return self.n0 + len(self.added)

    def __getitem__(self, v):
        if isinstance(v, slice):
            return [self[i] for i in range(*v.indices(len(self)))]
        if v < 0:
            v += len(self)
        return self.a[v].tolist() if v < self.n0 else self.added[v - self.n0]

    def __setitem__(self, v, x):
        if v < 0:
            v += len(self)
        if v < self.n0:
            self.a[v] = x
        else:
            self.added[v - self.n0] = x

    def append(self, x):
        self.added.append(x)

    def extend(self, xs):
        self.added.extend(xs)

    def array(self):
        """np.array(positions)"""
        return np.concatenate([self.a, np.array(self.added, float).reshape(-1, 3)])


class _PosView:
    """positions for a Python hook called from the Rust cut kernel: the vertices it was
    given, and the ones it adds (ids from base on)"""

    def __init__(self, base, known):
        self.base, self.known, self.added = base, known, []

    def __getitem__(self, v):
        return self.known[v] if v < self.base else self.added[v - self.base]

    def __len__(self):
        return self.base + len(self.added)

    def append(self, x):
        self.added.append([float(c) for c in x])


class _OnFeat:
    def __init__(self, s):
        self.s, self.new = set(s), []

    def __contains__(self, v):
        return v in self.s

    def add(self, v):
        self.s.add(v)
        self.new.append(v)


class _HookCtx:
    """what the hooks read from / add to the CutContext"""

    def __init__(self, pos, phi, onfeat=()):
        self.pos, self.phi, self.onfeat = pos, phi, onfeat
        self.new_vertices, self.new_parents = [], []


class _FaceOcc:
    """{face id: [pieces holding it]} (pieces in order, a piece once per face it has)
    looked up in the sorted face ids instead of a dict built over every face"""

    def __init__(self, key, cell):
        o = _argsort_stable(key)
        self.k, self.c = key[o], cell[o]

    def get(self, fid, default=()):
        a = int(np.searchsorted(self.k, fid))
        b = int(np.searchsorted(self.k, fid, side="right"))
        return self.c[a:b].tolist() if b > a else default


def _argsort_stable(a):
    """np.argsort(a, kind="stable") of an int64 array, sorted on all cores in Rust when
    it is large (a stable sort has one answer)"""
    a = np.asarray(a)
    if len(a) > 100000 and a.dtype == np.int64 and _core is not None and hasattr(_core, "argsort_stable"):
        return _core.argsort_stable(np.ascontiguousarray(a))
    return np.argsort(a, kind="stable")


def _rust_merge_ok():
    return _core is not None and hasattr(_core, "merge_cells_csr")


def _rust_cut_ok(ctx, hk):
    if _core is None or not hasattr(_core, "CutKernel"):
        return False
    ch = ctx.chord_hook
    if ch is not None and getattr(getattr(ch, "feat", None), "_rs", None) is None:
        return False
    return not hk or getattr(getattr(hk[0], "feat", None), "_rs", None) is not None


def _cut_with_fallback_rs(ctx, items, shared_cap, hk, csize, P0, max_iter=4, leaf_csr=None):
    """cut_with_fallback with the cuts in the Rust kernel (the same results and ids);
    P0: array of the first len(P0) positions of the context.  The context's state is
    brought up to date afterwards.  leaf_csr: (cptr, fptr, fverts, f_key) for items
    whose faces are None (the leaf's own); the results are then the kernel's arrays
    (item, side, face ptr, face ids, vertex ptr, vertices), not lists."""
    hook = hk[0] if hk else None
    st = hk[1] if hk else {}
    n0 = len(P0)
    Pa = np.concatenate([np.asarray(P0, float), np.array(ctx.pos[n0:], float).reshape(-1, 3)])
    pa = np.array(ctx.phi, float)
    pinch_cb = cap_cb = None
    if ctx.pinch_hook is not None:
        ph = ctx.pinch_hook

        def pinch_cb(poly, a, b, cand, pos, phi):
            return ph(_HookCtx(pos, phi), poly, a, b, cand)
    if hook is not None:
        def cap_cb(loop, P, flags, base, leaf, forbid):
            st["cell"] = leaf; st["forbid"] = forbid; st["did"] = False
            c = _HookCtx(_PosView(base, dict(zip(loop, P))), [],
                         _OnFeat(v for v, f in zip(loop, flags) if f))
            polys = hook(c, loop)
            return [list(q) for q in polys], c.pos.added, c.onfeat.new, bool(st["did"])
    ch = ctx.chord_hook
    K = _core.CutKernel(Pa, pa, int(ctx.next_id), bool(shared_cap), ctx.zero_edge, ctx.plain_faces,
                        ctx.onfeat, ctx.cache, ctx.subids, ctx.chord_cache, ctx.pinch_cache,
                        ch.feat._rs if ch is not None else None,
                        hook.feat._rs if hook is not None else None,
                        float(hook.reach) if hook is not None else 0.0,
                        np.ascontiguousarray(csize, float), pinch_cb, cap_cb, float(ctx.pinch_min_area))
    del Pa, pa
    if leaf_csr is not None:
        K.set_items_csr(*_items_csr(items, *leaf_csr))
    else:
        K.set_items(items)
    n = len(items)
    # the first pass in parallel where no Python hook is needed (the same results)
    used = K.cut_all() if hasattr(K, "cut_all") else K.cut(list(range(n)), [False] * n)
    forbid = set()
    nfb = 0
    for _ in range(max_iter):
        bad = K.bad_items([k for k in range(n) if used[k][0] or used[k][1]])
        if not bad:
            break
        redo = set(bad)
        for k in bad:
            forbid.add(k)
            for key in used[k][0]:
                K.forbid_chord(key)
                redo.update(K.items_of_face(key[0]))
        nfb += len(bad)
        ks = sorted(redo)
        for k, u in zip(ks, K.cut(ks, [k in forbid for k in ks])):
            used[k] = u
    nsharp = sum(1 for ch_, did in used if ch_ or did)
    res = K.results_csr(not shared_cap) if leaf_csr is not None else K.results()
    newp, newv, newpar, nid, cd, sd, clog, nf = K.export()
    del K
    ctx.pos.extend(newp.tolist())
    ctx.phi.extend([0.0] * len(newp))
    ctx.new_vertices.extend(newv)
    ctx.new_parents.extend(newpar)
    ctx.next_id = nid
    ctx.cache.update(cd)
    ctx.subids.update(sd)
    for which, key, val in clog:
        (ctx.chord_cache if which == 0 else ctx.pinch_cache)[key] = val
    ctx.onfeat.update(nf)
    return res, nsharp, nfb


def cut_with_fallback(ctx, items, shared_cap, hk, csize, max_iter=4, P0=None, faces_of=None, leaf_csr=None):
    """Cut items [(faces, parent_leaf)].  Cells where the feature refinement (bent face
    chords, split caps) produced a non-convex piece are re-cut without it, together
    with the cells sharing those faces.  Returns ([(neg_comps, pos_comps)], n_sharp,
    n_fallback).  P0 (array of the context's first positions): cut in Rust.
    faces None in an item: the leaf's own faces (faces_of(leaf)); with leaf_csr and the
    Rust kernel the results are flat arrays (_cut_with_fallback_rs)."""
    if P0 is not None and _rust_cut_ok(ctx, hk):
        if leaf_csr is None and faces_of is not None:
            items = [(fl if fl is not None else faces_of(ci), ci) for fl, ci in items]
        return _cut_with_fallback_rs(ctx, items, shared_cap, hk, csize, P0, max_iter, leaf_csr)
    if faces_of is not None:
        items = [(fl if fl is not None else faces_of(ci), ci) for fl, ci in items]
    hook = hk[0] if hk else None
    st = hk[1] if hk else {}
    res = [None] * len(items)
    used = [None] * len(items)
    forbid = set()
    face_items = {}

    def do(k):
        faces, leaf = items[k]
        st["cell"] = leaf; st["forbid"] = k in forbid; st["did"] = False
        neg, pos, _ = ctx.cut(faces, shared_cap=shared_cap, cap_hook=hook)
        res[k] = (split_components(neg, ctx.last_pinch), split_components(pos) if shared_cap else [])
        used[k] = (list(ctx.used_chords), bool(st["did"]))

    for k, (faces, _) in enumerate(items):
        for fid, _ in faces:
            face_items.setdefault(fid, []).append(k)
        do(k)
    nfb = 0
    for _ in range(max_iter):
        bad = []
        for k in range(len(items)):
            ch, did = used[k]
            if not (ch or did):
                continue
            h = csize[items[k][1]]
            if any(_piece_bad(ctx.pos, c, h) for c in res[k][0] + res[k][1]):
                bad.append(k)
        if not bad:
            break
        redo = set(bad)
        for k in bad:
            forbid.add(k)
            for key in used[k][0]:
                ctx.forbid_chord(key)
                redo.update(face_items.get(key[0], ()))
        nfb += len(bad)
        for k in sorted(redo):
            do(k)
    nsharp = sum(1 for ch, did in used if ch or did)
    return res, nsharp, nfb


def _cat_ranges(lo, hi):
    """np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) without the loop"""
    lo = np.asarray(lo, np.int64); n = np.asarray(hi, np.int64) - lo
    if not len(n):
        return np.zeros(0, np.int64)
    return np.repeat(lo - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(int(n.sum()))


def _ptr(counts):
    """offsets of consecutive runs of the given lengths: [0, c0, c0 + c1, ...]"""
    c = np.asarray(counts, np.int64)
    out = np.zeros(len(c) + 1, np.int64)
    np.cumsum(c, out=out[1:])
    return out


class _Pieces:
    """Cut pieces [faces, parent leaf, zone] as flat arrays: piece i has the faces
    fptr[i]..fptr[i+1], face f the id key[f] and the vertices verts[vptr[f]..vptr[f+1]].
    pieces[i] is the [faces, leaf, zone] list of the Python code (built when read), so
    the cut results go from the Rust kernels to the merging and the assembly without a
    Python list per face."""

    def __init__(self, leaf, zone, fptr, key, vptr, verts):
        self.leaf = np.ascontiguousarray(leaf, np.int64)
        self.zone = np.array(np.broadcast_to(np.asarray(zone, np.int64), self.leaf.shape))
        self.fptr, self.key, self.vptr, self.verts = (np.ascontiguousarray(x, np.int64)
                                                      for x in (fptr, key, vptr, verts))

    @classmethod
    def from_lists(cls, lst):
        """from [[faces, leaf, zone], ...]"""
        return cls([pc[1] for pc in lst], [pc[2] for pc in lst], _ptr([len(pc[0]) for pc in lst]),
                   [fid for pc in lst for fid, _ in pc[0]],
                   _ptr([len(p) for pc in lst for _, p in pc[0]]),
                   [v for pc in lst for _, p in pc[0] for v in p])

    @classmethod
    def from_leaves(cls, leaves, zone, cptr, fptr, fverts, f_key):
        """uncut octree leaves (their faces as leaf_faces gives them)"""
        leaves = np.asarray(leaves, np.int64)
        occ = _cat_ranges(cptr[leaves], cptr[leaves + 1])
        return cls(leaves, zone, _ptr(cptr[leaves + 1] - cptr[leaves]), f_key[occ],
                   _ptr(fptr[occ + 1] - fptr[occ]), fverts[_cat_ranges(fptr[occ], fptr[occ + 1])])

    @classmethod
    def from_csr(cls, leaf, zone, csr):
        """from a kernel's (face ptr, face ids, vertex ptr, vertices)"""
        return cls(leaf, zone, *csr)

    def __len__(self):
        return len(self.leaf)

    def faces(self, i):
        a, b = int(self.fptr[i]), int(self.fptr[i + 1])
        vp = self.vptr[a:b + 1].tolist()
        o = vp[0]
        vs = self.verts[o:vp[-1]].tolist()
        return [(k, vs[vp[j] - o:vp[j + 1] - o]) for j, k in enumerate(self.key[a:b].tolist())]

    def __getitem__(self, i):
        return [self.faces(i), int(self.leaf[i]), int(self.zone[i])]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def csr(self):
        return self.fptr, self.key, self.vptr, self.verts

    def take(self, idx):
        """the pieces idx, in that order"""
        idx = np.asarray(idx, np.int64)
        occ = _cat_ranges(self.fptr[idx], self.fptr[idx + 1])
        return _Pieces(self.leaf[idx], self.zone[idx], _ptr(self.fptr[idx + 1] - self.fptr[idx]),
                       self.key[occ], _ptr(self.vptr[occ + 1] - self.vptr[occ]),
                       self.verts[_cat_ranges(self.vptr[occ], self.vptr[occ + 1])])

    @staticmethod
    def concat(parts):
        parts = [p if isinstance(p, _Pieces) else _Pieces.from_lists(p) for p in parts]
        fo = np.cumsum([0] + [len(p.key) for p in parts])
        vo = np.cumsum([0] + [len(p.verts) for p in parts])
        return _Pieces(np.concatenate([p.leaf for p in parts] + [np.zeros(0, np.int64)]),
                       np.concatenate([p.zone for p in parts] + [np.zeros(0, np.int64)]),
                       np.concatenate([[0]] + [p.fptr[1:] + o for p, o in zip(parts, fo)]),
                       np.concatenate([p.key for p in parts] + [np.zeros(0, np.int64)]),
                       np.concatenate([[0]] + [p.vptr[1:] + o for p, o in zip(parts, vo)]),
                       np.concatenate([p.verts for p in parts] + [np.zeros(0, np.int64)]))

    def per_vertex(self):
        """piece of each entry of verts"""
        nv = np.diff(self.vptr)
        face_piece = np.repeat(np.arange(len(self), dtype=np.int64), np.diff(self.fptr))
        return np.repeat(face_piece, nv)


def _items_csr(items, cptr, fptr, fverts, f_key):
    """cut items [(faces or None for the leaf's own faces, leaf)] as flat arrays:
    (leaves, face ptr, face ids, vertex ptr, vertices)"""
    leaf = np.fromiter((ci for _, ci in items), np.int64, count=len(items))
    own = np.fromiter((fl is None for fl, _ in items), bool, count=len(items))
    lv = _Pieces.from_leaves(leaf[own], -1, cptr, fptr, fverts, f_key)
    if own.all():
        return leaf, lv.fptr, lv.key, lv.vptr, lv.verts
    ex = _Pieces.from_lists([[fl, ci, -1] for fl, ci in items if fl is not None])
    # interleave the two in item order
    k_own = np.flatnonzero(own); k_ex = np.flatnonzero(~own)
    order = np.argsort(np.concatenate([k_own, k_ex]), kind="stable")
    return leaf, *_Pieces.concat([lv, ex]).take(order).csr()


def select_clip_cells(pc, cand, faces_of, cells_of_face, special, pts):
    """Cells cut by plane clipping (featclip): the candidates whose surroundings are
    convex, grown so that the faces they share with level-set cells come out the same
    both ways (the neighbour joins where its surroundings are convex too).  Returns
    (cells, faces): the faces where that is not so (the neighbour is not convex there)
    are cut by the level-set rule in the clipped cell as well."""
    elig = {}

    def cell_pts(c):
        P = pts[unique(np.concatenate([np.asarray(p, np.int64) for _, p in faces_of(c)]))]
        return P, 1.5 * float(np.linalg.norm(P - P.mean(0), axis=1).max())

    def eligible(c):
        e = elig.get(c)
        if e is None:
            if c in special:
                e = False
            else:
                e, _ = pc.convex_at(*cell_pts(c))
            elig[c] = e
        return e

    # the candidates in one batch (the neighbours reached below go one by one)
    cs = [c for c in dict.fromkeys(cand) if c not in special]
    for c in cand:
        if c in special:
            elig[c] = False
    if cs:
        Pr = [cell_pts(c) for c in cs]
        for c, e in zip(cs, pc.convex_many([P for P, _ in Pr], [r for _, r in Pr])):
            elig[c] = e

    compat, around = {}, {}

    def cells_at(fid):
        r = around.get(fid)
        if r is None:
            r = around[fid] = cells_of_face(fid)
        return r

    def mismatch(fid, poly):
        ok = compat.get(fid)
        if ok is None:
            ok = compat[fid] = pc.ls_compatible(fid, poly)
        return not ok

    E = set(c for c in cand if eligible(c))
    todo = sorted(E)
    while todo:
        c = todo.pop()
        for fid, poly in faces_of(c):
            others = [o for o in cells_at(fid) if o != c and o not in E]
            if others and mismatch(fid, poly) and all(eligible(o) for o in others):
                E.update(others)
                todo.extend(others)
    # Where a clipped cell shares a face with a cell that cannot be clipped and the two
    # cuts differ on it, the level-set cell takes the clipped face (below).  That works
    # for the cells kept out for their own handling (special: wall contact, splits); in a
    # cell whose surroundings are not convex its cut then makes pieces of no or negative
    # volume.  So next to such a cell the clipped cells go back to the level-set cut (the
    # edge is chamfered there as before), until no such face is left.
    dropped = 0
    polys = {}

    def poly_of(c, fid):
        d = polys.get(c)
        if d is None:
            d = polys[c] = dict(faces_of(c))
        return d[fid]

    def is_hard(fid):
        cs = cells_at(fid)
        if not any(o not in E and o not in special for o in cs):
            return False
        return any(c in E and any(o not in E for o in cs if o != c) and mismatch(fid, poly_of(c, fid))
                   for c in cs)

    # A face's state only depends on which of its cells are in E: after a removal only
    # the faces of the removed cells can change (every hard face is one of them).
    hard = set(fid for c in E for fid, _ in faces_of(c) if is_hard(fid))
    while hard:
        bad = set(c for fid in hard for c in cells_at(fid) if c in E)
        E -= bad
        dropped += len(bad)
        hard = set(fid for c in bad for fid, _ in faces_of(c) if is_hard(fid))
    mixed = set(fid for c in E for fid, poly in faces_of(c)
                if any(o not in E for o in cells_at(fid) if o != c) and mismatch(fid, poly))
    return E, mixed, dropped


def _fluid_fraction(dom, c, h, n=6):
    """Fluid fraction of cells (centres c, sizes h) from the trilinear interpolation of
    the signed distance at their corners, and whether the wall cuts them."""
    corners = np.array(list(itertools.product((-1, 1), repeat=3)), float)
    sd = dom.sdf((c[:, None, :] + 0.5 * h[:, None, None] * corners[None]).reshape(-1, 3)).reshape(len(c), 8)
    t = (np.arange(n) + 0.5) / n
    T = np.array(list(itertools.product(t, t, t)))                       # sample points in [0,1]^3
    # trilinear weights of the 8 corners (corner k at (-1|1)^3 -> (0|1)^3)
    cw = (corners + 1) / 2
    W = np.prod(np.where(cw[None, :, :] > 0, T[:, None, :], 1 - T[:, None, :]), axis=2)   # (n^3, 8)
    phi = sd @ W.T
    return (phi < 0).mean(1), (sd.min(1) < 0) & (sd.max(1) > 0)


def _thin_wall_cells(dom, c, h):
    """Cells (centres c, sizes h) with no corner in the solid that contain a solid wall:
    on one of 13 lines through the cell the surface is entered and left again with
    opposite normals (the two faces of a wall; a convex edge of the solid, poking into
    the cell, has its face normals at 90 degrees or less and does not count)."""
    D = np.array([d for d in itertools.product((-1, 0, 1), repeat=3)
                  if any(d) and next(x for x in d if x) > 0], float)            # 13
    corners = np.array(list(itertools.product((-1, 1), repeat=3)), float)
    sd = dom.sdf((c[:, None, :] + 0.5 * h[:, None, None] * corners[None]).reshape(-1, 3))
    no_solid = ~(sd.reshape(len(c), 8) > 1e-3 * h[:, None]).any(1)
    idx = np.nonzero(no_solid)[0]
    out = np.zeros(len(c), bool)
    if len(idx) and dom._rs is not None:
        # the 13 lines stay within 0.5 sqrt(3) h of the centre: no crossing farther out
        _, dc, _ = dom.closest(c[idx])
        idx = idx[dc <= (0.5 * np.sqrt(3) + 1e-6) * h[idx]]
    if not len(idx):
        return out
    A = (c[idx][:, None, :] - 0.5 * h[idx][:, None, None] * D[None]).reshape(-1, 3)
    B = (c[idx][:, None, :] + 0.5 * h[idx][:, None, None] * D[None]).reshape(-1, 3)
    T = dom.tris
    tn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    tn /= np.maximum(np.linalg.norm(tn, axis=1), 1e-300)[:, None]
    seg, t, tri = dom.segment_hits(A, B)
    if not len(seg):
        return out
    n = tn[tri]
    enter = np.einsum("ij,ij->i", n, (B - A)[seg]) > 0       # outward normal along the line
    # drop repeated crossings (a line through a shared edge or vertex hits two triangles)
    keep = np.ones(len(seg), bool)
    keep[1:] = ~((seg[1:] == seg[:-1]) & (np.abs(t[1:] - t[:-1]) < 1e-9) & (enter[1:] == enter[:-1]))
    seg, n, enter = seg[keep], n[keep], enter[keep]
    pair = (seg[1:] == seg[:-1]) & enter[:-1] & ~enter[1:] & (np.einsum("ij,ij->i", n[:-1], n[1:]) < -0.5)
    hit = unique(seg[:-1][pair]) // len(D)
    out[idx[hit]] = True
    return out


def _auto_gap_spec(cfg, sref):
    """(cells across a gap, highest level) of autoGapRefinement (true = defaults)"""
    ag = cfg["autoGapRefinement"]
    ag = {} if ag is True else ag
    lev = [int(v) for v in sref.values()] or [0]
    return float(ag.get("cellsAcross", 3)), int(ag.get("maxLevel", max(lev) + 2))


def _fluid_side(dom, p, nrm, eps):
    """Unit normals at the surface points p turned towards the fluid.  inside() shifts its
    query points by dom.inside_shift, so the probes go at least a few times that far out;
    where both or neither probe is in the fluid (a wall thinner than that, a corner) the
    side is taken from the probe behind the triangle as before."""
    e = max(eps, 4.0 * dom.inside_shift)
    back = dom.inside(p - e * nrm)
    front = dom.inside(p + e * nrm)
    flip = np.where(back != front, back, dom.inside(p - eps * nrm))
    return np.where(flip[:, None], -nrm, nrm)


def _interior_samples(tris, spacing):
    """Centroids of a longest-edge bisection of the triangles (every edge <= spacing): one
    point per piece and none on an edge (a point on an edge shared with another wall
    measures a zero-width gap along that wall).  Returns (pts, tri_index)."""
    T, par = subdivide_triangles(tris, spacing)
    return T.mean(1), par


def _gap_widths(dom, p, ti, lw, h0, nacross, eps):
    """Gap width w at wall samples p (on triangles ti, surface level lw): from each sample a
    line goes into the fluid along the normal, and the first wall it hits within nacross
    cells of level lw gives w (inf: no wall that close)."""
    T = dom.tris[ti]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    into = _fluid_side(dom, p, nrm, eps)
    wmax = nacross * h0 / 2.0 ** lw                         # wider gaps are resolved anyway
    a = p + eps * into
    b = p + wmax[:, None] * into
    seg, t, _ = dom.segment_hits(a, b)
    w = np.full(len(p), np.inf)
    if len(seg):
        first = np.ones(len(seg), bool)
        first[1:] = seg[1:] != seg[:-1]
        w[seg[first]] = eps + t[first] * (wmax[seg[first]] - eps)
    return w


def _tri_neighbours(dom, tri):
    """tri plus the triangles sharing a vertex with one of them."""
    V = dom.tris.reshape(-1, 3)
    tol = 1e-9 * max(dom.L, 1e-300)
    _, vid = np.unique(np.round(V / tol), axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    mark = np.zeros(vid.max() + 1, bool)
    mark[vid[tri].ravel()] = True
    return np.nonzero(mark[vid].any(1))[0]


def _auto_gap_samples(dom, h0, region_level, nacross, lmax):
    """Wall samples of gaps too narrow for nacross cells of the wall's surface level:
    the gap width w at a sample is measured along the wall normal into the fluid (first
    wall hit), and the level whose cells fit nacross times into w is the target (capped
    at lmax).  Returns ([(level, samples, n_unresolved)], n_unresolved): n_unresolved
    samples would need more than lmax.

    Two passes, so that only the walls at a narrow gap are sampled finely: every wall at
    half its own cell size first (a gap narrower than nacross of those cells marks its
    triangle), then the marked triangles and their neighbours at the finest spacing
    h0 / 2^lmax.  Sampling the whole STL that finely took 110 million samples (7 GB) on a
    0.3 m tank with lmax 5."""
    eps = 1e-3 * h0 / 2 ** lmax
    tri_level = region_level[dom.region]
    cand = []
    for lv in np.unique(tri_level):
        if lv >= lmax:
            continue
        sel = np.nonzero(tri_level == lv)[0]
        p, ids = _interior_samples(dom.tris[sel], h0 / 2.0 ** lv / 2)
        ti = sel[ids]
        w = _gap_widths(dom, p, ti, tri_level[ti], h0, nacross, eps)
        cand.append(np.unique(ti[w < nacross * h0 / 2.0 ** lv]))
    cand = np.concatenate(cand) if cand else np.zeros(0, np.int64)
    if not len(cand):
        return [], 0
    cand = _tri_neighbours(dom, np.unique(cand))
    p, ids = _interior_samples(dom.tris[cand], h0 / 2 ** lmax)
    ti = cand[ids]
    lw = tri_level[ti]
    w = _gap_widths(dom, p, ti, lw, h0, nacross, eps)
    with np.errstate(divide="ignore"):
        need = np.ceil(np.log2(nacross * h0 / w)).astype(float)
    need = np.where(np.isfinite(w), need, -1)
    out = []
    nun = int(((need > lmax) & (need > lw)).sum())
    lev = np.minimum(need, lmax)
    for L in range(int(lw.min()) + 1, lmax + 1):
        m = (lev == L) & (L > lw)
        if m.any():
            out.append((L, p[m], nun))
    return out, nun


def _thin_gap(dom, p, into, tg, eps, max_steps=512):
    """wall samples p whose fluid along `into` ends (leaves the domain) within tg.
    Marches by the distance to the STL - no wall can be closer, so none is skipped; a step
    falls back to eps only where the walk grazes another wall, and there the inside test
    still catches any wall thicker than eps.  Returns (mask, n_unresolved)."""
    if dom._rs is not None and hasattr(dom._rs, "thin_gap"):     # per sample, in parallel
        return dom._rs.thin_gap(np.ascontiguousarray(p, float), np.ascontiguousarray(into, float),
                                float(tg), float(eps), int(max_steps))
    return _thin_gap_py(dom, p, into, tg, eps, max_steps)


def _thin_gap_py(dom, p, into, tg, eps, max_steps=512):
    thin = np.zeros(len(p), bool)
    t = np.full(len(p), eps)
    act = np.nonzero(dom.inside(p + eps * into))[0]
    for k in range(1, 9):
        # cheap pass first: a fixed step that lands outside has found a wall for sure
        out = ~dom.inside(p[act] + (k * tg / 8) * into[act])
        thin[act[out]] = True
        act = act[~out]
    # one query settles most of the rest: no wall within the ball on the probe segment
    r = 0.5 * (tg - eps)
    _, d, _ = dom.closest(p[act] + (eps + r) * into[act])
    act = act[d < r]
    for _ in range(max_steps):
        if not len(act):
            break
        q = p[act] + t[act, None] * into[act]
        out = ~dom.inside(q)
        thin[act[out]] = True
        act = act[~out]; q = q[~out]
        _, d, _ = dom.closest(q)
        t[act] += np.maximum(d, eps)
        act = act[t[act] < tg]
    return thin, len(act)


@paused_gc
def generate(cfg, log=print):
    T = Timer(log)
    validate_config(cfg, log)
    dom = Surface(read_stl(_rel(cfg, cfg["domain"])), log=log, what="domain STL")
    check_names(cfg, dom.names, log)
    rp = dom.repair
    if cfg["strictInput"] and (rp["openEdges"] or rp["nonManifoldEdges"]):
        raise STLError(f"strictInput: the domain STL has {rp['openEdges']} open and "
                         f"{rp['nonManifoldEdges']} non-manifold edges (see the log)")
    zones = build_zones(cfg, log)
    h0 = float(cfg["baseCellSize"])
    sref = cfg["surfaceRefinement"]
    default_level = int(sref.get("default", 0))
    region_level = np.array([int(sref.get(n, default_level)) for n in dom.names])
    flevel = cfg["featureLevel"]
    if flevel is None:
        flevel = int(region_level.max())
    regions = []
    for r in cfg["refinementRegions"]:
        c = r["cylinder"]
        regions.append((Cylinder(c["point1"], c["point2"], c["radius"]), int(r["level"])))
    Lmax = int(max([region_level.max(), flevel] + [max(z["surfaceLevel"], z["volumeLevel"]) for z in zones]
                   + [int(z["wallProximity"]["level"]) for z in zones if z["wallProximity"]]
                   + ([int(cfg["gapRefinement"]["level"])] if cfg.get("gapRefinement") else [])
                   + ([_auto_gap_spec(cfg, sref)[1]] if cfg.get("autoGapRefinement") else [])
                   + [lv for _, lv in regions]))
    hmin = h0 / 2 ** Lmax
    log(f"domain: {len(dom.tris)} triangles, regions {dom.names}")
    log(f"base cell {h0:g}, max level {Lmax} (smallest cell {hmin:g})")
    bm = cfg.get("baseMesh") or {}
    cyl = None
    if bm.get("type", "cartesian") == "cylindrical":
        cyls = [(c.p1, c.a, c.R) for c, _ in regions] + \
               [(z["geom"].p1, z["geom"].a, z["geom"].R) for z in zones if isinstance(z["geom"], Cylinder)]
        surfs = [z["geom"].s.tris for z in zones if isinstance(z["geom"], SurfaceZone)]
        cyl = CylindricalBase(bm, dom, h0, hmin, Lmax, log=log, cylinders=cyls, surfaces=surfs)
        if cyl.min_base_size < hmin:
            hmin = cyl.min_base_size
            log(f"smallest cell of the cylindrical base {hmin:g}")
    elif bm.get("type", "cartesian") != "cartesian":
        raise ValueError(f"baseMesh type {bm.get('type')!r}: 'cartesian' or 'cylindrical'")

    # ------------------------------------------------------------ search structures
    dom.build_search(2 * hmin)
    for z in zones:
        if isinstance(z["geom"], SurfaceZone):
            z["geom"].s.build_search(2 * hmin)
    check_zone_overlap(zones, 2 * hmin)
    nbuf = int(cfg["bufferLayers"])
    ssrc, vsrc = [], []
    for r, name in enumerate(dom.names):
        if region_level[r] > 0:
            p, _ = dom.samples(h0 / 2 ** region_level[r] / 2, regions=[r])
            ssrc.append((_points_kd(p), int(region_level[r]), nbuf))
    if cfg.get("autoGapRefinement"):
        nacross, lag = _auto_gap_spec(cfg, sref)
        groups, nun = _auto_gap_samples(dom, h0, region_level, nacross, lag)
        for lv, pts_, _ in groups:
            ssrc.append((_points_kd(pts_), lv, nbuf))
            log(f"auto gap refinement: {len(pts_)} wall samples -> level {lv}")
        if nun:
            log(f"auto gap refinement: {nun} wall samples in gaps narrower than {nacross:g} cells"
                f" of level {lag} (raise maxLevel to resolve them)")
    gr = cfg.get("gapRefinement")
    if gr:
        # fluid thinner than `thickness` (slot tips, narrow channels): refine its walls to `level`
        lg = int(gr["level"])
        tg = float(gr["thickness"])
        p, ti = dom.samples(h0 / 2 ** lg)
        T_ = dom.tris[ti]
        nrm = np.cross(T_[:, 1] - T_[:, 0], T_[:, 2] - T_[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
        eps = 0.02 * h0 / 2 ** lg
        into = np.where(dom.inside(p - eps * nrm)[:, None], -nrm, nrm)    # towards the fluid
        thin, nun = _thin_gap(dom, p, into, tg, eps)
        if nun:
            log(f"gap refinement: {nun} wall samples not resolved (walk along a wall)")
        p = p[thin]
        if len(p):
            ssrc.append((_points_kd(p), lg, nbuf))
        log(f"gap refinement: {len(p)} wall samples where the fluid is thinner than {tg:g} -> level {lg}")
    seg = dom.feature_edges(cfg["featureAngle"])
    dom_feat = FeatureSet(seg, L=dom.L)
    # Concave edges (the solid juts into the fluid) are chamfered by the cut, unlike convex
    # ones (the chord / cap hooks bend the cut through them, which would make the cell
    # non-convex here).  A vertex snapped onto the edge keeps it: those vertices may come
    # from farther, as long as the octree cells stay valid.
    dom_ccv = FeatureSet(dom.feature_edges(cfg["featureAngle"], concave_only=True), L=dom.L) \
        if float(cfg["concaveSnapMax"]) > float(cfg["featureSnapMax"]) else None
    if flevel > 0:
        if len(seg):
            fp = sample_segments(seg, h0 / 2 ** flevel / 2)
            ssrc.append((_points_kd(fp), int(flevel), nbuf))
            log(f"feature edges: {len(seg)} segments -> level {flevel}")
    for z in zones:
        if z["surfaceLevel"] > 0:
            p, _ = z["geom"].samples(h0 / 2 ** z["surfaceLevel"] / 2)
            ssrc.append((_points_kd(p), z["surfaceLevel"], nbuf))
        wp = z["wallProximity"]
        if wp:
            # the zone surface closer than `distance` to the STL: refine to `level` there only
            # (an interface a fraction of a cell off a wall needs cells finer than the gap)
            lw = int(wp["level"])
            p, _ = z["geom"].samples(h0 / 2 ** lw / 2)
            _, d, _ = dom.closest(p)
            p = p[d < float(wp["distance"])]
            if len(p):
                ssrc.append((_points_kd(p), lw, int(wp.get("bufferLayers", nbuf))))
            log(f"zone '{z['name']}': {len(p)} surface samples within {float(wp['distance']):g} of the STL"
                f" -> level {lw}")
        if z["volumeLevel"] > 0:
            g_ = z["geom"]
            vsrc.append((LevelSet(g_.refine_sdf, g_.refine_below) if isinstance(g_, SurfaceZone) else g_.sdf,
                         z["volumeLevel"], nbuf))
    for g, lv in regions:
        vsrc.append((g.sdf, lv, nbuf))
    T("geometry / search structures")

    # ------------------------------------------------------------ octree
    def keep_fn(c, hd):
        out = ~dom.inside(c)
        far = ~dom.approx_below(c, np.nextafter(hd + dom._rmax, np.inf))     # approx > hd + rmax
        return ~(out & far)
    if cyl is not None:
        oc = cyl
        oc.refine(ssrc, vsrc, log=lambda s: log(s), keep_fn=keep_fn)
    else:
        pad = int(cfg["padding"])
        lo = dom.bbmin - pad * h0
        n = np.ceil((dom.bbmax - dom.bbmin) / h0 + 2 * pad).astype(int)
        nfill_g = int(cfg["fillThinGaps"])
        nfill_w = int(cfg["fillThinWalls"])
        oc = Octree(lo, h0, n, Lmax + max(1 if nfill_g else 0, nfill_w))
        log(f"root grid {n.tolist()}")
        oc.refine(ssrc, vsrc, log=lambda s: log(s), keep_fn=keep_fn)
        sgn = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)   # 27 points
        corner = np.all(sgn != 0, axis=1)
        for it in range(max(nfill_g, nfill_w)):
            c = oc.centers(); hl = oc.leaf_sizes()
            near = oc.lev < oc.Lmax
            near[near] = dom.approx_below(c[near], 0.9 * hl[near] + dom._rmax)
            mark = np.zeros(len(oc.lev), bool)
            nh = nw = 0
            if it < nfill_g:
                # fluid thinner than the cell between its corners (a gap passing between
                # the vertex planes): no corner is in the fluid, so the cut would drop the
                # cell and close the gap there.  One level beyond Lmax.
                cand = np.nonzero(near & (oc.lev <= Lmax))[0]
                q = (c[cand][:, None, :] + 0.5 * hl[cand][:, None, None] * sgn[None]).reshape(-1, 3)
                ins = dom.inside(q).reshape(len(cand), 27)
                hole = ~ins[:, corner].any(1) & ins[:, ~corner].any(1)
                mark[cand[hole]] = True
                nh = int(hole.sum())
            if it < nfill_w:
                # a solid wall inside the cell with no corner in the solid vanishes: the
                # fluid on both sides joins (a bypass).  Split until a corner lands in it.
                cand = np.nonzero(near)[0]
                wall = _thin_wall_cells(dom, c[cand], hl[cand])
                mark[cand[wall]] = True
                nw = int(wall.sum())
            if not mark.any():
                break
            oc.split(mark)
            oc.balance(lambda s: None)
            log(f"thin gaps / walls: split {nh} cells with fluid inside but no fluid corner,"
                f" {nw} with a solid wall inside but no solid corner")
        n_tc = int(cfg["thinCutCells"])
        thr = float(cfg["thinCutFraction"])
        nsplit = 0
        for it in range(n_tc):
            # A cut cell keeping little fluid next to finer cells: its centre sits on the
            # wall, far from the plane of the finer cells' centres, and the faces between
            # them are strongly non-orthogonal.  Split it: the level transition moves off
            # the wall, into whole cells.
            c = oc.centers(); hl = oc.leaf_sizes()
            near = oc.lev < Lmax
            near[near] = dom.approx_below(c[near], 0.9 * hl[near] + dom._rmax)
            cand = np.nonzero(near)[0]
            fr, cut = _fluid_fraction(dom, c[cand], hl[cand])
            cand = cand[cut & (fr > 0) & (fr < thr)]
            fine = np.zeros(len(cand), bool)
            for lv in np.unique(oc.lev[cand]).tolist():
                m_ = oc.lev[cand] == lv
                fl = np.nonzero(oc.lev > lv)[0]
                if not len(fl):
                    continue
                hc = float(hl[cand[m_]][0])
                dd, _ = cKDTree(c[fl]).query(c[cand[m_]], distance_upper_bound=0.95 * hc)
                fine[m_] = dd < 0.95 * hc
            mark = np.zeros(len(oc.lev), bool)
            mark[cand[fine]] = True
            if not mark.any():
                break
            nsplit += int(mark.sum())
            oc.split(mark)
            oc.balance(lambda s: None)
        if n_tc:
            log(f"thin cut cells at level transitions: split {nsplit}")
    T(f"octree ({len(oc.lev)} leaves)")
    M = oc.build_cells()
    nleaf = len(M["cell_size"])
    leaves_total = nleaf
    pts = M["points"]
    hanging = M["hanging"]
    hang_par = M["hang_par"]
    hang_size = M["hang_size"]
    csize = M["cell_size"]
    cvid = M["corner_vid"]
    f_cell, f_key, fptr, fverts, cptr = (M["face_cell"], M["face_key"], M["fptr"],
                                         M["fverts"], M["cell_fptr"])
    fn = np.diff(fptr)
    cvol = M["cell_volume"]
    vh_min = np.full(len(pts), np.inf); vh_max = np.zeros(len(pts))
    np.minimum.at(vh_min, cvid.ravel(), np.repeat(csize, 8))
    np.maximum.at(vh_max, cvid.ravel(), np.repeat(csize, 8))
    lone = ~np.isfinite(vh_min)       # not a cell corner (vertices added at block interfaces)
    if lone.any():
        fc_ = np.repeat(csize[f_cell], fn)
        m_ = lone[fverts]
        np.minimum.at(vh_min, fverts[m_], fc_[m_])
        np.maximum.at(vh_max, fverts[m_], fc_[m_])
    del cvid, M
    T(f"polyhedral cells ({len(pts)} points)")

    def leaf_faces(ci):
        a, b = int(cptr[ci]), int(cptr[ci + 1])
        fp = fptr[a:b + 1].tolist()              # one conversion per cell, not per face
        o = fp[0]
        vs = fverts[o:fp[-1]].tolist()
        return [(k, vs[fp[j] - o:fp[j + 1] - o]) for j, k in enumerate(f_key[a:b].tolist())]

    def face_signs(phi):
        """per face: the number of vertices with phi < 0, > 0 and == 0"""
        if _core is not None and hasattr(_core, "face_sign_counts"):
            return _core.face_sign_counts(np.ascontiguousarray(phi, float), fptr, fverts)
        pv = phi[fverts]
        return tuple(np.add.reduceat(m.astype(np.int32), fptr[:-1]) for m in (pv < 0, pv > 0, pv == 0))

    def classify_leaves(phi, leaves, signs=None):
        """-1 all vertices <= 0, +1 all >= 0 (and some > 0), 0 straddling.  signs:
        face_signs(phi) if known."""
        nneg, npos, _ = face_signs(phi) if signs is None else signs
        cn = np.logical_or.reduceat(nneg > 0, cptr[:-1])[leaves]
        cp_ = np.logical_or.reduceat(npos > 0, cptr[:-1])[leaves]
        return np.where(~cp_, -1, np.where(~cn, 1, 0))

    # ------------------------------------------------------------ level sets + snapping
    alpha = float(cfg["snapTolerance"])
    free_hang = cfg["hangingSnap"] == "free"
    near = dom.approx_below(pts, np.sqrt(3) * vh_max + dom._rmax + hmin)
    phi_d = dom.sdf(pts, exact_mask=near, far_value=dom.L, exact_within=alpha * h0 + 2 * hmin)
    movable = ~hanging
    tol_v = alpha * vh_min
    tol_f = float(cfg["featureSnapTolerance"]) * vh_min
    tol_f2 = np.maximum(float(cfg["featureSnapMax"]) * vh_min, tol_f)
    tol_cc = np.maximum(float(cfg["concaveSnapMax"]) * vh_min, tol_f2)
    # not for the parents of hanging vertices: those follow, off the wall, into thin wedges
    hp_ = hang_par[hanging].ravel()
    tol_cc[hp_[hp_ >= 0]] = tol_f2[hp_[hp_ >= 0]]
    snapped = np.zeros(len(pts), bool)
    far_snap = np.zeros(len(pts), bool)    # snapped onto a feature from beyond tol_f
    ccv_far = np.zeros(len(pts), bool)     # of those, onto a concave feature edge
    slide_far = np.zeros(len(pts), bool)   # wall vertices slid onto a concave edge from beyond tol_f
    no_far = np.zeros(len(pts), bool)      # far snaps that broke a cell: not allowed
    free_snap = np.zeros(len(pts), bool)   # hanging vertices projected freely onto a surface
    no_free = np.zeros(len(pts), bool)     # free projections that broke a cell: not allowed
    msgs = []
    occ_leaf = np.repeat(f_cell, fn)
    leaf_vptr = fptr[cptr[:-1]]                 # vertex occurrences by leaf (each has faces)

    def leaf_any(m):
        """per leaf: some vertex of its faces in the mask m"""
        return np.logical_or.reduceat(m[fverts], leaf_vptr)

    def side_width(kv, side):
        """Thickness of the fluid (side -1) / solid (side 1) at the vertices kv, from the
        nearest wall through the vertex to the opposite wall if within 1.5 cells (else inf).
        Only for the vertices off the wall (mask off); returns (off, nearest wall point,
        direction away from it, width) for those."""
        cpk, dk, _ = dom.closest(pts[kv])
        # vertices already on the wall (phi a rounding error off zero) carry nothing and
        # give no direction
        off = dk > 0.01 * vh_min[kv]
        kv, cpk, dk = kv[off], cpk[off], dk[off]
        dirk = (pts[kv] - cpk) / np.maximum(dk, 1e-300)[:, None]
        width = np.full(len(kv), np.inf)
        for t in np.linspace(0.05, 1.5, 30):
            q = cpk + (t * vh_min[kv])[:, None] * dirk
            hit = np.isinf(width) & (dom.inside(q) != (side < 0))
            width[hit] = t * vh_min[kv][hit]
        return off, cpk, dirk, width

    def snap_features(feat, phi, label, concave=None):
        """corners first, then feature edges, then (farther) the concave feature edges;
        moved vertices get phi = 0.  Returns the ids of the vertices now on the features."""
        nc = 0
        if feat is None or feat._kd is None:
            return np.zeros(0, np.int64)
        phi0 = phi.copy()
        on = []
        lim = np.where(no_far, tol_f, tol_f2)
        if len(feat.corners):
            # a vertex within lim of a corner (on the surface) is within lim of the surface
            d, j = _nearest_in(pts, np.abs(phi) < 2 * float(lim.max()) + hmin, feat.corners)
            for ci_, (dd, jj) in enumerate(zip(d.tolist(), j.tolist())):
                if movable[jj] and not snapped[jj] and dd < lim[jj]:
                    far_snap[jj] = dd >= tol_f[jj]
                    pts[jj] = feat.corners[ci_]; phi[jj] = 0.0; snapped[jj] = True; nc += 1
                    on.append(jj)
        sel = np.nonzero(movable & ~snapped & (np.abs(phi) < 2 * tol_f2 + hmin))[0]
        cp, d = feat.closest(pts[sel])
        ok = d < lim[sel]
        sel, cp, d = sel[ok], cp[ok], d[ok]
        far_snap[sel] = d >= tol_f[sel]
        pts[sel] = cp; phi[sel] = 0.0; snapped[sel] = True
        msgs.append(f"snapped {nc} vertices to {label} corners, {len(sel)} to {label} feature edges")
        res = [np.array(on, np.int64), sel]
        if concave is not None and concave._kd is not None:
            # the vertex in the solid wedge goes onto its edge (a fluid vertex there would
            # flatten the cell holding the wedge, and the cut would still chamfer it); of
            # the solid vertices only the nearest to its foot point, and none where a
            # vertex lies already, so no two vertices land on one point
            lim = np.where(no_far, tol_f, tol_cc)
            cand = movable & ~snapped & (phi > 0) & (phi < 2 * tol_cc + hmin)
            c_sel, c_X, c_d = [], [], []
            if len(concave.corners):
                d, j = _nearest_in(pts, np.abs(phi) < 2 * float(lim.max()) + hmin, concave.corners)
                for X, dd, jj in zip(concave.corners, d.tolist(), j.tolist()):
                    if cand[jj] and dd < lim[jj]:
                        c_sel.append(jj); c_X.append(X); c_d.append(dd); cand[jj] = False
            c_X = np.reshape(c_X, (-1, 3))
            sel = np.nonzero(cand)[0]
            cp, d = concave.closest(pts[sel])
            ok = d < lim[sel]
            if ok.any():
                dn, _ = cKDTree(pts[sel[ok]]).query(cp[ok])       # nearest solid candidate
                d0, _ = _nearest_in(np.concatenate([pts, c_X]),               # a vertex there already
                                    np.concatenate([np.abs(phi) < hmin, np.ones(len(c_X), bool)]), cp[ok])
                ok[ok] = (d[ok] <= dn * (1 + 1e-9)) & (d0 > 1e-6 * hmin)
            sel, cp, d = sel[ok], cp[ok], d[ok]
            _, first = np.unique(np.round(cp / (1e-6 * hmin)).astype(np.int64), axis=0, return_index=True)
            sel = np.concatenate([np.array(c_sel, np.int64), sel[first]])
            cp = np.concatenate([c_X, cp[first]])
            d = np.concatenate([np.array(c_d, float), d[first]])
            # A cell that loses its last solid vertex through these snaps (the earlier
            # feature snaps of this pass included, by the signs before them) keeps no solid:
            # right at a wedge of a thick solid, but a solid thinner than the reach (a post
            # or fin narrower than ~1.5 cells) would vanish there.  Those are not snapped.
            gone = (phi0 > 0) & (phi <= 0)
            gone[sel] = True
            lost = leaf_any(gone) & ~leaf_any((phi0 > 0) & ~gone) & leaf_any(phi0 < 0)
            nthin = 0
            if lost.any():
                occ = lost[occ_leaf]
                at = np.isin(sel, fverts[occ])
                off, _, _, width = side_width(sel[at], 1.0)
                thin = np.zeros(len(sel), bool)
                thin[np.nonzero(at)[0][off]] = np.isfinite(width)
                sel, cp, d = sel[~thin], cp[~thin], d[~thin]
                nthin = int(thin.sum())
            far_snap[sel] = ccv_far[sel] = d >= tol_f[sel]
            pts[sel] = cp; phi[sel] = 0.0; snapped[sel] = True
            n0 = int(np.isin(sel, c_sel).sum())
            msgs.append(f"snapped {n0} more vertices to the ends and {len(sel) - n0} to the concave"
                        f" {label} feature edges ({nthin} not: the solid there is thinner than 1.5 cells)")
            res.append(sel)
        return np.concatenate(res)

    # Far feature snaps (tol_f .. tol_f2) and free projections of hanging vertices are
    # kept only where the octree cells stay valid after all vertices (including the
    # hanging ones) have moved; otherwise they are forbidden for the vertices of the
    # broken cells and the stage is repeated.
    pts0 = pts.copy(); phi_d0 = phi_d.copy()
    hp2 = np.where(hanging[:, None], hang_par, -1)
    for attempt in range(5):
        if attempt:
            pts[:] = pts0; phi_d = phi_d0.copy()
            snapped[:] = False; far_snap[:] = False; free_snap[:] = False; ccv_far[:] = False; slide_far[:] = False
            msgs.clear()
        # ---- 1) regular (non-hanging) vertices: features, then surfaces
        on_feat_d = snap_features(dom_feat, phi_d, "STL", dom_ccv)
        snap = movable & ~snapped & (np.abs(phi_d) < tol_v) & (phi_d != 0)
        # A cut cell must keep a vertex on either side.  In a fluid gap (a solid wall)
        # about one cell thick the cell's vertices on that side can all lie near the two
        # walls; snapping them all would leave the cell without fluid (the gap closes) or
        # without solid (the wall vanishes: a bypass).  There they go to the middle instead.
        for side, label in ((-1.0, "fluid vertices to the middle of thin gaps"),
                            (1.0, "solid vertices to the middle of thin walls")):
            on = side * phi_d > 0
            lost = leaf_any(on) & ~leaf_any(on & ~snap) & leaf_any(-side * phi_d > 0)
            if not lost.any():
                continue
            occ = lost[occ_leaf]
            kv = unique(fverts[occ][on[fverts[occ]]])
            # thickness of this side; where it is not within 1.5 cells, the snap removes a
            # sliver as intended (vertices on the wall: dropping that side is right)
            off, cpk, dirk, width = side_width(kv, side)
            kv = kv[off]
            thin = np.isfinite(width)
            # vertices that would have to move too far are snapped as usual (left close to
            # a wall unsnapped, they cut skewed slivers)
            mid = cpk + 0.5 * np.where(thin, width, 0.0)[:, None] * dirk
            mv = thin & (np.linalg.norm(mid - pts[kv], axis=1) < 0.4 * vh_min[kv])
            j = kv[mv]
            snap[j] = False
            pts[j] = mid[mv]
            phi_d[j] = dom.sdf(pts[j])
            snapped[j] = True
            msgs.append(f"thin gaps / walls: moved {int(mv.sum())} {label}"
                        f" (their cells would lose the last vertex on that side when snapped)")
        cp, _, _ = dom.closest(pts[snap])
        pts[snap] = cp
        phi_d[snap] = 0.0
        snapped |= snap
        msgs.append(f"snapped {int(snap.sum())} vertices onto the domain surface")
        if dom_ccv is not None and dom_ccv._kd is not None:
            # A concave edge with no solid vertex near it (e.g. at a level transition, where
            # its wall vertices belong to finer cells) leaves the wedge of solid between
            # them in a cell without a solid vertex: the cut keeps it, the edge is
            # chamfered.  The nearest wall vertex slides along its wall onto the edge.
            lim = np.where(no_far, tol_f, tol_cc)
            onf = np.zeros(len(pts), bool)
            onf[on_feat_d] = True
            sel = np.nonzero(movable & (phi_d == 0) & ~onf)[0]
            cp, d = dom_ccv.closest(pts[sel])
            ok = (d < lim[sel]) & (d > 1e-6 * hmin)
            sel, cp, d = sel[ok], cp[ok], d[ok]
            if len(sel) and len(dom_ccv.corners):
                # not onto an end or a corner of the edges: from a third wall (e.g. the one
                # the edge ends at) a vertex there would twist the cells along the edge
                dc, _ = cKDTree(dom_ccv.corners).query(cp)
                ok = dc > 1e-6 * hmin
                sel, cp, d = sel[ok], cp[ok], d[ok]
            if len(sel):
                # on a wall of the edge, not across a gap: the way there stays on the wall
                ok = np.abs(dom.sdf(0.5 * (pts[sel] + cp))) < 0.02 * vh_min[sel]
                sel, cp, d = sel[ok], cp[ok], d[ok]
            if len(sel):
                dn, _ = cKDTree(pts[sel]).query(cp)               # nearest candidate
                d0, _ = _nearest_in(pts, np.abs(phi_d) < hmin, cp)   # a vertex there already
                ok = (d <= dn * (1 + 1e-9)) & (d0 > 1e-6 * hmin)
                sel, cp, d = sel[ok], cp[ok], d[ok]
                _, first = np.unique(np.round(cp / (1e-6 * hmin)).astype(np.int64), axis=0, return_index=True)
                sel, cp, d = sel[first], cp[first], d[first]
                far_snap[sel] = slide_far[sel] = d >= tol_f[sel]
                pts[sel] = cp
                on_feat_d = np.concatenate([on_feat_d, sel])
            msgs.append(f"slid {len(sel)} wall vertices along their wall onto the concave STL feature edges")
        phi_z = []
        zone_feats = []
        on_feat_z = []
        for z in zones:
            g = z["geom"]
            if isinstance(g, SurfaceZone):
                nearz = g.s.approx_below(pts, np.sqrt(3) * vh_max + g.s._rmax + hmin)
                pz = g.sdf(pts, exact_mask=nearz, far_value=g.L)
            else:
                pz = g.sdf(pts)
            if not z["conforming"]:           # cells are assigned by centre later: no snapping
                zone_feats.append(None); on_feat_z.append(np.zeros(0, np.int64)); phi_z.append(pz)
                continue
            if isinstance(g, Cylinder):
                zf = FeatureSet(cylinder_features(g, hmin / 4), L=g.L)
            else:
                zf = FeatureSet(g.s.feature_edges(cfg["featureAngle"]), L=g.L)
            zone_feats.append(zf)
            before = snapped.copy()           # vertices already on the STL are not moved
            on_feat_z.append(snap_features(zf, pz, f"zone '{z['name']}'"))
            sz = movable & ~snapped & (np.abs(pz) < tol_v) & (pz != 0)
            cp, _, _ = g.closest(pts[sz])
            pts[sz] = cp
            pz[sz] = 0.0
            snapped |= sz
            moved = snapped & ~before
            if moved.any():   # re-evaluate the domain level set where vertices moved
                phi_d[moved] = dom.sdf(pts[moved])
            msgs.append(f"snapped {int(sz.sum())} vertices onto zone '{z['name']}'")
            phi_z.append(pz)

        # ---- 2) hanging vertices, coarsest constraint first: follow the (moved) parents,
        # then slide along the coarse edge / within the coarse face onto a surface.  If the
        # constraint is nearly tangent to the surface, the node is projected freely
        # (hangingSnap "free") or treated as on-surface ("constrained").
        geoms = [(dom, phi_d, None)] + [(z["geom"], phi_z[i], i) for i, z in enumerate(zones)
                                         if z["conforming"]]
        nslide = nvirt = nfree = 0
        for hsz in sorted(set(hang_size[hanging].tolist()), reverse=True):
            idx = np.nonzero(hanging & (hang_size == hsz))[0]
            par = hang_par[idx]
            isface = par[:, 2] != par[:, 0]
            newx = np.where(isface[:, None], pts[par].mean(1), 0.5 * (pts[par[:, 0]] + pts[par[:, 1]]))
            ch = np.linalg.norm(newx - pts[idx], axis=1) > 0
            pts[idx] = newx
            ci = idx[ch]
            if len(ci):
                phi_d[ci] = dom.sdf(pts[ci])
                for i, z in enumerate(zones):
                    phi_z[i][ci] = z["geom"].sdf(pts[ci])
            # constraint basis
            A = pts[par[:, 1]] - pts[par[:, 0]]
            B = np.where(isface[:, None], pts[par[:, 2]] - pts[par[:, 1]], 0.0)  # other direction
            e1 = A / np.maximum(np.linalg.norm(A, axis=1), 1e-300)[:, None]
            B = B - (B * e1).sum(1)[:, None] * e1
            nb = np.linalg.norm(B, axis=1)
            e2 = np.where((nb > 1e-12 * h0)[:, None], B / np.maximum(nb, 1e-300)[:, None], 0.0)
            for geom, phi, zi in geoms:
                excl = snapped[idx] | ((phi_d[idx] == 0) if zi is not None else False)
                sel = np.nonzero(~excl & (np.abs(phi[idx]) < tol_v[idx]) & (phi[idx] != 0))[0]
                if len(sel) == 0:
                    continue
                j = idx[sel]
                x = pts[j].copy(); ph = phi[j].copy()
                E1, E2 = e1[sel], e2[sel]
                ok = np.ones(len(j), bool)
                for _ in range(4):
                    cp, d, _ = geom.closest(x)
                    g = (x - cp) / np.maximum(d, 1e-300)[:, None] * np.sign(ph)[:, None]
                    ph = np.sign(ph) * d
                    gf = (g * E1).sum(1)[:, None] * E1 + (g * E2).sum(1)[:, None] * E2
                    gn = (gf ** 2).sum(1)
                    ok &= gn > 0.25
                    x = np.where(ok[:, None], x - (ph / np.maximum(gn, 1e-300))[:, None] * gf, x)
                    ph = geom.sdf(x)
                mv = np.linalg.norm(x - pts[j], axis=1)
                ok &= (np.abs(ph) < 1e-6 * vh_min[j]) & (mv < 0.45 * vh_min[j])
                if free_hang and not ok.all():
                    # the constraint cannot reach the surface (nearly tangent): project
                    # freely, like a regular vertex (the coarse face gets slightly warped
                    # instead of leaving a dent of up to snapTolerance x h in the wall)
                    nf = np.nonzero(~ok & ~no_free[j])[0]
                    cp, d, _ = geom.closest(pts[j[nf]])
                    good = d < tol_v[j[nf]]
                    x[nf[good]] = cp[good]
                    ok[nf[good]] = True
                    free_snap[j[nf[good]]] = True
                    nfree += int(good.sum())
                pts[j[ok]] = x[ok]
                phi[j] = 0.0
                snapped[j[ok]] = True
                if zi is not None and ok.any():
                    phi_d[j[ok]] = dom.sdf(pts[j[ok]])
                if zi is None:
                    for i2, z in enumerate(zones):
                        if ok.any():
                            phi_z[i2][j[ok]] = z["geom"].sdf(pts[j[ok]])
                nslide += int(ok.sum()); nvirt += int((~ok).sum())
        msgs.append(f"hanging nodes: {nslide - nfree} slid onto a surface along their edge/face,"
                    f" {nfree} projected freely, {nvirt} treated as on-surface")
        if not (far_snap.any() or free_snap.any()):
            break
        mv_ = (pts != pts0).any(1)           # hanging vertices move with their parents too
        touch = np.maximum.reduceat(mv_[fverts].astype(np.int8), fptr[:-1])
        tl = np.nonzero(np.maximum.reduceat(touch, cptr[:-1]))[0]
        badl = tl[_leaf_bad(pts, tl, fptr, fverts, cptr, csize)] if len(tl) else tl
        if ccv_far.any():
            # a vertex pushed onto a concave edge must not straighten a face corner either
            tc = np.maximum.reduceat(np.maximum.reduceat(ccv_far[fverts].astype(np.int8), fptr[:-1]), cptr[:-1])
            tc = np.nonzero(tc)[0]
            badl = np.union1d(badl, tc[_leaf_bad(pts, tc, fptr, fverts, cptr, csize, flat_at=ccv_far)])
        if slide_far.any():
            # a slid vertex often lands on the line of the wall vertices next to it (the
            # cell becomes a wedge with a straight face corner, still convex and uncut):
            # only a corner pushed clearly past straight breaks the cell
            tc = np.maximum.reduceat(np.maximum.reduceat(slide_far[fverts].astype(np.int8), fptr[:-1]), cptr[:-1])
            tc = np.nonzero(tc)[0]
            badl = np.union1d(badl, tc[_leaf_bad(pts, tc, fptr, fverts, cptr, csize, flat_at=slide_far,
                                                 flat_angle=-1.0)])
        if not len(badl):
            break
        fo = np.concatenate([np.arange(cptr[c], cptr[c + 1]) for c in badl.tolist()])
        vb = np.concatenate([fverts[fptr[f]:fptr[f + 1]] for f in fo.tolist()])
        for _ in range(3):          # hanging vertices move with their parents
            p_ = hp2[vb].ravel()
            vb = unique(np.concatenate([vb, p_[p_ >= 0]]))
        resp = vb[far_snap[vb] & ~no_far[vb]]
        respf = vb[free_snap[vb] & ~no_free[vb]]
        if not (len(resp) or len(respf)):
            break
        no_far[resp] = True
        no_free[respf] = True
    for m_ in msgs:
        log(m_)
    if (tol_cc > tol_f).any():
        log(f"far feature snaps: {int(far_snap.sum())} kept, {int(no_far.sum())} not allowed (cells would break)")
    phi_d[(phi_d != 0) & (np.abs(phi_d) < 1e-9 * vh_min)] = 0.0
    for pz in phi_z:
        pz[(pz != 0) & (np.abs(pz) < 1e-9 * vh_min)] = 0.0
    moved_any = snapped | (pts != pts0).any(1)
    del pts0, phi_d0
    del vh_min, vh_max, near, movable, tol_v, tol_f, tol_f2, tol_cc
    T("level sets and vertex snapping")

    # ------------------------------------------------------------ pass 1: domain cut
    # Uncut leaves stay in flat arrays ("bulk"); only cut cells become python pieces.
    signs_d = face_signs(phi_d)
    cls = classify_leaves(phi_d, np.arange(nleaf), signs_d)
    bulk_alive = cls < 0
    bulk_zone = np.full(nleaf, -1, np.int64)
    cut_leaves = np.nonzero(cls == 0)[0]
    positions = _PosList(pts)          # shares pts (both are updated together below)
    ctx = CutContext(positions, phi_d.tolist(), M_nkeys := int(f_key.max()) + 1)
    fc = cfg["featureCaps"]
    hk = make_feature_hook(dom_feat, csize) if fc in ("both", "domain", True) else None
    if hk:
        ctx.chord_hook = make_chord_hook(dom_feat)
        ctx.onfeat = set(on_feat_d.tolist())
    if dom_ccv is not None and dom_ccv._kd is not None:
        if _core is not None and hasattr(_core, "vertex_min"):
            vh_ = _core.vertex_min(len(pts), fptr, fverts, np.ascontiguousarray(csize[f_cell], float))
        else:
            vh_ = np.full(len(pts), np.inf)
            np.minimum.at(vh_, fverts, np.repeat(csize[f_cell], fn))
        ctx.pinch_hook = make_pinch_hook(dom, dom_ccv, lambda v: vh_[v] if v < len(vh_) else hmin)
    pieces = []        # [faces, parent_leaf, zone]
    cl = cut_leaves.tolist()
    cb = []
    if dom_ccv is not None and dom_ccv._kd is not None:
        # Uncut cells at a concave edge: a vertex on the edge and the wall vertices next
        # to it can enclose the wedge of solid (all phi <= 0).  Those faces are split
        # along the chord through the edge vertex and the cell drops the wedge (below).
        zc = signs_d[2]
        fz = (zc >= 3) | (zc == fn)
        cz = np.maximum.reduceat(fz.astype(np.int8), cptr[:-1]) > 0
        cand_b = np.nonzero(cz & bulk_alive)[0]
        if len(cand_b):
            occ_b = _cat_ranges(fptr[cptr[cand_b]], fptr[cptr[cand_b + 1]])
            own_b = np.repeat(np.arange(len(cand_b)), fptr[cptr[cand_b + 1]] - fptr[cptr[cand_b]])
            # np.add.at, column by column (both add in index order)
            xb = pts[fverts[occ_b]]
            cen = np.stack([np.bincount(own_b, xb[:, k], len(cand_b)) for k in range(3)], 1)
            del xb
            cen /= np.bincount(own_b, minlength=len(cand_b))[:, None]
            _, dce = dom_ccv.closest(cen)
            cb = cand_b[dce < csize[cand_b]].tolist()
    # only cells with a face that can matter there (several surface vertices) are visited:
    # the same faces in the same order, without a Python pass over every cut cell
    fz = signs_d[2]
    fneg = signs_d[0] > 0
    fpos = signs_d[1] > 0
    f_sc = (fz == fn) | ((fz >= 2) & fneg & fpos)
    c_sc = np.logical_or.reduceat(f_sc, cptr[:-1])
    c_cs = np.logical_or.reduceat(~fpos & fneg & (fz >= 3), cptr[:-1])
    del fz, fneg, fpos, signs_d
    ctx.zero_edge, drop = surface_contact(dom, pts, phi_d, [c for c in cl + cb if c_sc[c]], leaf_faces, csize,
                                          face_ok=lambda ci: f_sc[cptr[ci]:cptr[ci + 1]].tolist(),
                                          csr=(cptr, fptr, fverts, f_key, f_sc))
    del f_sc
    log(f"surface contact: {sum(v == 1 for v in ctx.zero_edge.values())} face edges in the solid,"
        f" {sum(len(s) for s in drop.values())} wall faces dropped from solid-side cells")
    # faces partly on the wall: split; the contact part is kept by the fluid-side cell
    splits = contact_splits(dom, pts, phi_d, [c for c in cl + cb if c_cs[c]], leaf_faces, csize,
                            csr=(cptr, fptr, fverts, f_key))
    split_keep = {}
    if splits:
        sk = np.array(list(splits), np.int64)
        mk = np.zeros(M_nkeys, bool)          # a lookup, not np.isin over every face
        mk[sk] = True
        occ_ = np.nonzero(mk[f_key])[0]
        del mk
        sc, sf = f_cell[occ_], f_key[occ_]
        live = (cls[sc] <= 0)
        sc, sf = sc[live], sf[live]
        samp, hh = [], []
        if _core is not None and hasattr(_core, "split_samples"):
            hh = [splits[fid][3] for fid in sf.tolist()]
            samp = _core.split_samples(np.ascontiguousarray(pts, float), sc.tolist(),
                                       [splits[fid][2] for fid in sf.tolist()], hh,
                                       np.ascontiguousarray(cptr, np.int64), np.ascontiguousarray(fptr, np.int64),
                                       np.ascontiguousarray(fverts, np.int64))
        else:
            for ci, fid in zip(sc.tolist(), sf.tolist()):
                _, _, run, hf = splits[fid]
                Q = pts[run]
                c0 = Q.mean(0)
                nrm = _newell(Q - c0)
                nrm /= max(np.linalg.norm(nrm), 1e-300)
                cc = pts[[v for _, p_ in leaf_faces(ci) for v in p_]].mean(0)
                if np.dot(cc - c0, nrm) < 0:
                    nrm = -nrm
                samp.append(c0 + 0.1 * hf * nrm); hh.append(hf)
        sd = dom.sdf(np.array(samp))
        for ci, fid, d_, hf in zip(sc.tolist(), sf.tolist(), sd.tolist(), hh):
            # the cell drops the contact part where it is on its solid side (an uncut
            # cell is then cut too, the cut closes it)
            split_keep[(ci, fid)] = not (d_ > 1e-3 * hf)
        split_ids = {fid: (ctx.new_face_id(), ctx.new_face_id()) for fid in splits}

    def apply_splits(ci, fl):
        out = []
        for fid, poly in fl:
            if fid not in splits:
                out.append((fid, poly)); continue
            p0, qa, qb, _ = splits[fid]
            if not _same_orientation(p0, poly):
                qa, qb = qa[::-1], qb[::-1]
            ia, ib = split_ids[fid]
            out.append((ia, qa))
            if split_keep.get((ci, fid), True):
                out.append((ib, qb))
        return out

    split_cells = set(ci for ci, _ in split_keep)
    log(f"surface contact: {len(splits)} faces split into fluid and wall parts")
    # uncut cells dropping a face or a part of one are cut (closed by the cut)
    opened = sorted(set(ci for ci in drop if bulk_alive[ci])
                    | set(ci for (ci, _), k in split_keep.items() if not k and bulk_alive[ci]))
    for ci in opened:
        bulk_alive[ci] = False
    cl = cl + opened
    if opened:
        log(f"surface contact: {len(opened)} uncut cells at concave edges drop a wedge of solid")
    items = []                         # (faces, leaf); None: the leaf's own faces
    for ci in cl:
        if ci not in drop and ci not in split_cells:
            items.append((None, ci))
            continue
        fl = leaf_faces(ci)
        if ci in drop:
            fl = [f for f in fl if f[0] not in drop[ci]]
        if ci in split_cells:
            fl = apply_splits(ci, fl)
        items.append((fl, ci))
    # uncut cells sharing a split face become pieces holding both parts
    for ci in sorted(split_cells):
        if bulk_alive[ci]:
            pieces.append([apply_splits(ci, leaf_faces(ci)), ci, -1])
            bulk_alive[ci] = False
    clip, mixed, nback = set(), set(), 0
    if hk and cfg["exactFeatureCut"] and dom_feat._kd is not None:
        # cut and solid-side cells next to a convex feature edge: clipped by the STL planes
        # (the concave edges whether or not vertices are snapped onto them)
        ccv = dom_ccv if dom_ccv is not None else             FeatureSet(dom.feature_edges(cfg["featureAngle"], concave_only=True), L=dom.L)
        pc = PlaneClip(dom, pts, phi_d, hmin, dom_feat, ccv, cfg["featureAngle"])
        pc.zero_edge = ctx.zero_edge
        special = set(drop) | split_cells | set(opened) | set(cb)
        mphi = np.minimum.reduceat(np.minimum.reduceat(phi_d[fverts], fptr[:-1]), cptr[:-1])
        cnd = np.nonzero((cls >= 0) & (mphi < 0.9 * csize))[0]
        if len(cnd):
            occ_c = _cat_ranges(fptr[cptr[cnd]], fptr[cptr[cnd + 1]])
            own_c = np.repeat(np.arange(len(cnd)), fptr[cptr[cnd + 1]] - fptr[cptr[cnd]])
            cen = np.zeros((len(cnd), 3)); np.add.at(cen, own_c, pts[fverts[occ_c]])
            cen /= np.bincount(own_c, minlength=len(cnd))[:, None]
            _, df = dom_feat.closest(cen)
            ok = df < 0.9 * csize[cnd]
            if ccv._kd is not None:
                _, dcv = ccv.closest(cen)
                ok &= dcv > 1.5 * csize[cnd]
            # wall vertices of these cells a hair off a crease go onto it (as the clip's own
            # points do, make_alloc): left there, the clip keeps a strip of the other wall
            # as narrow as the gap
            cv = unique(fverts[_cat_ranges(fptr[cptr[cnd[ok]]], fptr[cptr[cnd[ok] + 1]])]) \
                if ok.any() else np.zeros(0, np.int64)
            cv = cv[phi_d[cv] == 0]
            if len(cv):
                cpv, dv = dom_feat.closest(pts[cv])
                w = (dv > 0) & (dv < 0.05 * hmin)
                for v, x in zip(cv[w].tolist(), cpv[w]):
                    pts[v] = x                # positions shares pts
                # the zone cuts read the zones' level sets of these vertices later: refresh
                # them (a vertex on a zone surface stays on it, 0.05 hmin is below the snap)
                mv = cv[w]
                for i, z in enumerate(zones):
                    if len(mv) and z["conforming"]:
                        m_ = mv[phi_z[i][mv] != 0]
                        if len(m_):
                            phi_z[i][m_] = z["geom"].sdf(pts[m_])
            ko = _argsort_stable(f_key)
            ks = f_key[ko]

            def cells_of_face(fid):
                return f_cell[ko[np.searchsorted(ks, fid):np.searchsorted(ks, fid, side="right")]].tolist()
            clip, mixed, nback = select_clip_cells(pc, cnd[ok].tolist(), leaf_faces, cells_of_face, special, pts)
            # the faces they share with level-set cells: cut by the plain rule there (no
            # chord or pinch points), which select_clip_cells checked the clip agrees with
            ctx.plain_faces = set(fid for c in clip for fid, _ in leaf_faces(c)
                                  if any(o not in clip for o in cells_of_face(fid)))
        keep = [k for k, (_, ci) in enumerate(items) if ci not in clip]
        items = [items[k] for k in keep]
        cl = [cl[k] for k in keep]
        if clip:
            def feat_closest(x):
                cp, d = dom_feat.closest(np.asarray(x)[None])
                return cp[0], d[0]
            alloc = make_alloc(ctx, phi_d, feat_closest, 1e-6 * hmin, 0.05 * hmin)
        if mixed:
            # where the clip differs and the neighbour cannot be clipped (its
            # surroundings are not convex), the neighbour takes the clipped face (on the
            # fluid side or on the wall, it keeps it whole and closes its cut along it)
            pre = {}
            for fid in sorted(mixed):
                c = next(o for o in cells_of_face(fid) if o in clip)
                poly = next(p_ for f_, p_ in leaf_faces(c) if f_ == fid)
                pre[fid] = (poly, clipped_face(pc, ctx, fid, poly, alloc))

            def take_clipped(fl):
                out = []
                for f_, p_ in fl:
                    if f_ not in pre:
                        out.append((f_, p_)); continue
                    poly, kept = pre[f_]
                    if kept is not None:
                        f2, q = kept
                        out.append((f2, list(q) if _same_orientation(poly, p_) else list(q[::-1])))
                return out
            at = {ci: k for k, (_, ci) in enumerate(items)}
            for ci in sorted(set(o for fid in mixed for o in cells_of_face(fid)) - clip):
                if ci in at:
                    k = at[ci]
                    items[k] = (take_clipped(items[k][0] if items[k][0] is not None else leaf_faces(ci)), ci)
                elif bulk_alive[ci]:
                    # an uncut fluid cell: cut, closed along the clipped face
                    bulk_alive[ci] = False
                    items.append((take_clipped(leaf_faces(ci)), ci))
                    cl.append(ci)
    res, nsh, nfb = cut_with_fallback(ctx, items, False, hk, csize, P0=pts, faces_of=leaf_faces,
                                      leaf_csr=(cptr, fptr, fverts, f_key))
    del items
    if isinstance(res, tuple):         # the kernel's arrays (negative components)
        kpieces = _Pieces.from_csr(np.asarray(cl, np.int64)[res[0]], -1, res[2:])
    else:
        kpieces = _Pieces.from_lists([[comp, ci, -1] for ci, (negc, _) in zip(cl, res) for comp in negc])
    del res
    pieces = [pieces, kpieces, []]     # split cells, cut cells, clipped cells (below)
    del kpieces
    if clip:
        hook, st = hk
        nbad = nsl = nfac = ngap = 0
        for ci in sorted(clip):
            st["cell"] = ci; st["forbid"] = False
            creases, walls = set(), set()
            neg = clip_cell(pc, ctx, leaf_faces(ci), alloc, hook, creases, walls)
            if not neg:
                continue
            bulk_alive[ci] = False
            kept = False
            for comp in split_components(neg):
                # faces lying on a wall plane alone (vertices snapped onto it): no volume
                if _piece_volume(ctx.pos, comp) < 1e-6 * csize[ci] ** 3:
                    continue
                # a sliver in a solid-side cell is kept where its caps lie on the walls
                # (bent over a crease, or in one wall plane); one between a chord and a
                # faceted curved wall only resolves the facets: dropped, the chord is the
                # wall there (as the level-set cut has it; the neighbours keep the shared
                # faces as walls)
                if cls[ci] > 0 and not any(fid in creases for fid, _ in comp):
                    nfac += 1
                    continue
                # one between two opposite walls that do not meet in the cell (wall faces
                # facing away from each other, no vertex in common - a wedge's meet along
                # the crease) is a thin gap: left to the thin-gap handling (fillThinGaps)
                if cls[ci] > 0:
                    cf = [p for fid, p in comp if fid in walls]
                    nw = [_newell(np.array([ctx.pos[v] for v in p])) for p in cf]
                    nw = [x / max(np.linalg.norm(x), 1e-300) for x in nw]
                    if any(nw[i] @ nw[j] < -0.5 and not set(cf[i]) & set(cf[j])
                           for i in range(len(cf)) for j in range(i + 1, len(cf))):
                        ngap += 1
                        continue
                kept = True
                nbad += _piece_bad(ctx.pos, comp, csize[ci])
                pieces[2].append([comp, ci, -1])
            nsl += kept and cls[ci] > 0
        log(f"exact feature cut: {len(clip)} cells clipped by the STL planes ({nsl} solid-side cells"
            f" keeping a sliver, {nfac} slivers along faceted walls and {ngap} in thin gaps dropped,"
            f" {nbad} pieces not convex; {nback} cells left to the level-set cut where a neighbour"
            f" could not be clipped)")
    pieces = _Pieces.concat(pieces)
    next_id = ctx.next_id
    log(f"domain cut: {len(cut_leaves)} cut cells, {int(bulk_alive.sum()) + len(pieces)} cells kept"
        + (f", {nsh} cells follow feature edges ({nfb} fell back to chamfers to stay convex)"
           if hk else ""))
    newv = np.array(ctx.new_vertices, np.int64)
    P = positions.array()
    del positions, ctx
    if len(newv):
        cp, d, _ = dom.closest(P[newv])
        ok = d < 0.5 * h0            # safety: never move a crossing absurdly far
        P[newv[ok]] = cp[ok]
    on_dom = np.zeros(len(P), bool)
    on_dom[:len(phi_d)] = phi_d == 0
    on_dom[len(phi_d):] = True
    n_orig = len(phi_d)
    T("domain cut")

    # ------------------------------------------------------------ pass 2..: zone cuts
    for zi, z in enumerate(zones):
        if not z["conforming"]:
            continue
        g = z["geom"]
        pz = np.zeros(len(P))
        pz[:n_orig] = phi_z[zi]
        extra = np.arange(n_orig, len(P))
        if len(extra):
            pz[extra] = g.sdf(P[extra], exact_mask=None)
            pz[extra[np.abs(pz[extra]) < 1e-9 * hmin]] = 0.0
        # bulk leaves
        al = np.nonzero(bulk_alive)[0]
        czl = classify_leaves(pz[:n_orig], al)
        bulk_zone[al[czl < 0]] = zi
        newcut = al[czl == 0]
        bulk_alive[newcut] = False
        pieces = _Pieces.concat([pieces, _Pieces.from_leaves(newcut, -1, cptr, fptr, fverts, f_key)])
        if isinstance(g, Cylinder):
            zfeat = FeatureSet(cylinder_features(g, hmin / 4), L=g.L)
        else:
            zfeat = FeatureSet(g.s.feature_edges(cfg["featureAngle"]), L=g.L)
        zh = make_feature_hook(zfeat, csize) if fc in ("both", "zone", True) else None
        rust_cut = _core is not None and hasattr(_core, "cut_batch_csr") and not zh   # no hooks: whole batch in Rust
        # pieces entirely inside go into the zone, those straddling are cut (CutContext.classify)
        pv_ = pz[pieces.verts]
        pp_ = pieces.per_vertex()
        hasN = np.bincount(pp_, pv_ < 0, len(pieces)) > 0
        hasP = np.bincount(pp_, pv_ > 0, len(pieces)) > 0
        del pv_, pp_
        straddle = hasN & hasP
        pieces.zone[~straddle & ~hasP] = zi
        tocut = pieces.take(np.flatnonzero(straddle))
        kept = pieces.take(np.flatnonzero(~straddle))
        nc = len(tocut)
        if rust_cut:
            (it, side, *rcsr), newpts, newpar, next_id = _core.cut_batch_csr(P, pz, next_id, *tocut.csr(), True, {})
            zsh = zfb = 0
            newv = np.arange(len(P), len(P) + len(newpts), dtype=np.int64)
            P = np.concatenate([P, newpts])
            par = np.array(newpar, np.int64).reshape(-1, 2)
            cutp = _Pieces.from_csr(tocut.leaf[it], np.where(side == 0, zi, tocut.zone[it]), rcsr)
            del it, side, rcsr
        else:
            positions = _PosList(P)
            ctx = CutContext(positions, pz.tolist(), next_id)
            if zh:
                ctx.chord_hook = make_chord_hook(zfeat)
                ctx.onfeat = set(on_feat_z[zi].tolist())
            tl = list(tocut)
            res, zsh, zfb = cut_with_fallback(ctx, [(pc[0], pc[1]) for pc in tl], True, zh, csize, P0=P)
            next_id = ctx.next_id
            P = positions.array()
            newv = np.array(ctx.new_vertices, np.int64)
            par = np.array(ctx.new_parents, np.int64).reshape(-1, 2)
            cutp = _Pieces.from_lists([x for pc, (negc, posc) in zip(tl, res)
                                       for x in [[comp, pc[1], zi] for comp in negc]
                                       + [[comp, pc[1], pc[2]] for comp in posc]])
            del res, tl, positions, ctx
        pieces = _Pieces.concat([kept, cutp])
        del tocut, kept, cutp
        if len(newv):
            isfeat = par[:, 0] < 0            # feature points: already exact
            on_curve = (~isfeat) & on_dom[np.maximum(par[:, 0], 0)] & on_dom[np.maximum(par[:, 1], 0)]
            cp, _, _ = g.closest(P[newv[~isfeat]])
            P[newv[~isfeat]] = cp
            cv = newv[on_curve]               # on STL and zone surface: alternate projections
            for _ in range(8):
                if len(cv) == 0:
                    break
                c1, _, _ = dom.closest(P[cv])
                c2, _, _ = g.closest(c1)
                P[cv] = c2
            if len(cv):
                c1, _, _ = dom.closest(P[cv]); P[cv] = c1
            on_dom = np.concatenate([on_dom, np.zeros(len(P) - len(on_dom), bool)])
            on_dom[cv] = True
        log(f"zone '{z['name']}': {nc} cut cells, total {int(bulk_alive.sum()) + len(pieces)}"
            + (f", {zsh} cells follow the rims ({zfb} fell back to chamfers)" if zh else ""))
        T(f"zone cut '{z['name']}'")

    # ------------------------------------------------------------ distorted uncut cells
    # uncut leaves whose corners were snapped can become non-convex: hand them to the
    # merging stage like cut cells
    touch = np.maximum.reduceat(moved_any[fverts].astype(np.int8), fptr[:-1])
    touch = np.maximum.reduceat(touch, cptr[:-1]).astype(bool) & bulk_alive
    tl = np.nonzero(touch)[0]
    nbad_bulk = 0
    if len(tl):
        badc = tl[_leaf_bad(P, tl, fptr, fverts, cptr, csize)]
        pieces = _Pieces.concat([pieces, _Pieces.from_leaves(badc, bulk_zone[badc], cptr, fptr, fverts, f_key)])
        bulk_alive[badc] = False
        nbad_bulk = len(badc)
    if nbad_bulk:
        log(f"{nbad_bulk} snapped uncut cells are non-convex -> merge candidates")

    # ------------------------------------------------------------ small-cell merging
    nP = len(pieces)
    # bulk neighbour of an original octree face key
    bsel = np.nonzero(bulk_alive[f_cell])[0]
    key2bulk = np.full(M_nkeys, -1, np.int64)
    key2bulk[f_key[bsel]] = f_cell[bsel]
    py_key = pieces.key
    py_cell = np.repeat(np.arange(nP, dtype=np.int64), np.diff(pieces.fptr))
    occ = _FaceOcc(py_key, py_cell)       # face id -> pieces holding it, in piece order
    vol = np.zeros(nP)
    farea = {}
    p_n = np.zeros(0, np.int64); p_verts = np.zeros(0, np.int64)
    if nP:
        ffp = quality.FlatFaces((np.diff(pieces.vptr), pieces.verts))
        p_n, p_verts = ffp.n, ffp.v
        Cf, Sf = quality.face_geometry(P, ffp)
        ref = np.zeros((nP, 3)); ref[py_cell] = Cf
        cnt_ = np.bincount(py_cell, minlength=nP)
        # np.add.at column by column (bincount adds in the same index order)
        est = np.stack([np.bincount(py_cell, Cf[:, k], nP) for k in range(3)], 1) if nP else np.zeros((0, 3))
        est /= np.maximum(cnt_, 1)[:, None]
        pv = np.einsum("ij,ij->i", Sf, Cf - est[py_cell])
        vol = np.bincount(py_cell, pv, nP) / 3.0
        w_ = pv[:, None] * (0.75 * Cf + 0.25 * est[py_cell])
        cc = np.stack([np.bincount(py_cell, w_[:, k], nP) for k in range(3)], 1)
        del w_
        cc /= np.where(np.abs(vol) > 0, 3.0 * vol, 1.0)[:, None]
        # "bad" pieces: some face does not clearly see the centroid (non-convex cut cell)
        hp = csize[pieces.leaf]
        selfpyr = np.einsum("ij,ij->i", Sf, Cf - cc[py_cell])
        badf = selfpyr < 0.05 * np.linalg.norm(Sf, axis=1) * hp[py_cell]
        bad_piece = np.bincount(py_cell, badf.astype(int), nP) > 0
        farea = dict(zip(py_key.tolist(), np.linalg.norm(Sf, axis=1).tolist()))
        del ffp, Cf, Sf, ref, est, pv, cc, selfpyr, badf
    else:
        bad_piece = np.zeros(0, bool)
    hexvol = cvol[pieces.leaf].astype(float)
    ratio = vol / np.maximum(hexvol, 1e-300)
    zone_of = pieces.zone.tolist()
    parent = list(range(nP))           # piece index -> piece index, or nP + bulk leaf

    def find(i):
        while i < nP and parent[i] != i:
            parent[i] = parent[parent[i]] if parent[i] < nP else parent[i]
            i = parent[i]
        return i

    def zone_of_node(r):
        return zone_of[r] if r < nP else int(bulk_zone[r - nP])

    beta = float(cfg["mergeRatio"])
    small = np.nonzero(ratio < beta)[0]
    small = small[np.argsort(ratio[small])]
    nsmall = len(small)
    small = np.concatenate([small, np.nonzero(bad_piece & (ratio >= beta))[0]])
    gvol = vol.copy()
    members = {}                       # root piece -> list of pieces merged into it
    bulk_members = {}                  # bulk leaf -> list of pieces merged into it

    def node_faces(r):
        if r >= nP:
            fl = leaf_faces(r - nP)
            for q in bulk_members.get(r - nP, []):
                fl = fl + pieces[q][0]
            return fl
        fl = list(pieces[r][0])
        for q in members.get(r, []):
            fl = fl + pieces[q][0]
        return fl

    # bulk cells on either side of an original octree face key
    kb1 = np.full(M_nkeys, -1, np.int64); kb2 = np.full(M_nkeys, -1, np.int64)
    kk = f_key[bsel]; kc = f_cell[bsel]
    o_ = _argsort_stable(kk); kk = kk[o_]; kc = kc[o_]
    first = np.ones(len(kk), bool); first[1:] = kk[1:] != kk[:-1]
    kb1[kk[first]] = kc[first]
    kb2[kk[~first]] = kc[~first]
    del kk, kc, o_, first

    def face_neighbour(fid, me_set):
        """node on the other side of face fid (or -1 for boundary)"""
        o = occ.get(fid, ())
        for q in o:
            r = find(q)
            if r not in me_set:
                return r
        if fid < M_nkeys:
            for b_ in (kb1[fid], kb2[fid]):
                if b_ >= 0:
                    r = find_bulk(int(b_))
                    if r not in me_set:
                        return r
        return -1

    def find_bulk(leaf):
        return nP + leaf

    def poly_geom(fl):
        cnt = {}
        for fid, _ in fl:
            cnt[fid] = cnt.get(fid, 0) + 1
        keep = [(fid, p) for fid, p in fl if cnt[fid] == 1]
        if _core is not None:
            cf, sf, V, cc = _core.poly_cell(P, [p for _, p in keep])
            return keep, cf, sf, V, cc
        ffm = quality.FlatFaces([p for _, p in keep])
        cf, sf = quality.face_geometry(P, ffm)
        est = cf.mean(0)
        pv = np.einsum("ij,ij->i", sf, cf - est)
        V = pv.sum() / 3.0
        cc = (pv[:, None] * (0.75 * cf + 0.25 * est)).sum(0) / pv.sum() if V > 0 else est
        return keep, cf, sf, V, cc

    gcache = {}

    def node_centre(r):
        c = gcache.get(r)
        if c is None:
            _, _, _, V, cc = poly_geom(node_faces(r))
            c = gcache[r] = cc
        return c

    max_no = float(cfg["mergeMaxNonOrtho"])
    cos_lim = np.cos(np.radians(max_no))

    def merge_score(fa, r_small, r_target, hcell):
        """worst cos(non-orthogonality) of the union, or None if invalid"""
        keep, cf, sf, V, cc = poly_geom(fa + node_faces(r_target))
        if V <= 0:
            return None
        pyr = np.einsum("ij,ij->i", sf, cf - cc)
        if np.any(pyr <= 0.02 * np.linalg.norm(sf, axis=1) * hcell):
            return None
        me = {r_small, r_target}
        sel, cen = [], []
        for i, (fid, _) in enumerate(keep):
            nb = face_neighbour(fid, me)
            if nb >= 0:
                sel.append(i); cen.append(node_centre(nb))
        if not sel:
            return 1.0
        d = (np.array(cen) - cc)[:, None, :]
        s = sf[sel][:, None, :]
        dT, sT = d.transpose(0, 2, 1), s.transpose(0, 2, 1)
        # row-wise dot products through matmul: bit for bit what np.dot / np.linalg.norm
        # of each row give (the same BLAS kernel; a sum over the axis rounds differently
        # and changes borderline merges)
        c = (d @ sT)[:, 0, 0] / (np.sqrt((d @ dT)[:, 0, 0]) * np.sqrt((s @ sT)[:, 0, 0]) + 1e-300)
        return min(1.0, float(c.min()))

    merged = rejected = 0
    rust_merge = _rust_merge_ok()
    if rust_merge:
        # the same greedy loop in Rust (candidates in this set order, same arithmetic)
        # the pieces as flat arrays (flattened above): no conversion of the Python lists
        i64 = lambda a: np.ascontiguousarray(a, np.int64)            # noqa: E731
        p_leaf = pieces.leaf
        par_, mem_, bmem_, gvol_, merged, rejected = _core.merge_cells_csr(
            np.ascontiguousarray(P, float),
            i64(np.concatenate([[0], np.cumsum(np.bincount(py_cell, minlength=nP))])), i64(py_key),
            i64(np.concatenate([[0], np.cumsum(p_n)])), i64(p_verts), i64(zone_of),
            i64(bulk_zone), i64(cptr), i64(fptr), i64(fverts), i64(f_key), kb1, kb2, i64(small), int(nsmall),
            np.ascontiguousarray(gvol, float), np.ascontiguousarray(hexvol, float), beta,
            np.ascontiguousarray(csize[p_leaf], float), float(cos_lim))
        del p_leaf
        parent[:] = par_
        members.update(mem_)
        bulk_members.update(bmem_)
        gvol[:] = gvol_
    for qi, pi in enumerate([] if rust_merge else small.tolist()):
        if find(pi) != pi or (qi < nsmall and gvol[pi] >= beta * hexvol[pi]):
            continue
        fa = node_faces(pi)
        me = {pi}
        cands = set()
        for fid, _ in fa:
            nb = face_neighbour(fid, me)
            if nb >= 0 and zone_of_node(nb) == zone_of[pi]:
                cands.add(nb)
        hcell = csize[pieces[pi][1]]
        best, bscore = -1, cos_lim
        for ro in cands:
            sc = merge_score(fa, pi, ro, hcell)
            if sc is not None and sc > bscore:
                best, bscore = ro, sc
        if best >= 0:
            parent[pi] = best
            mem = [pi] + members.pop(pi, [])
            if best < nP:
                gvol[best] += gvol[pi]
                members.setdefault(best, []).extend(mem)
            else:
                bulk_members.setdefault(best - nP, []).extend(mem)
            gcache.pop(best, None); gcache.pop(pi, None)
            merged += 1
        elif cands:
            rejected += 1
    log(f"merged {merged} cut cells ({nsmall} small < {beta:g} of hex volume,"
        f" {len(small) - nsmall} non-convex); {rejected} left as they are"
        f" (no merge partner keeps the cell valid)")
    proot = np.array([find(i) for i in range(nP)], np.int64)

    # ------------------------------------------------------------ invalid wall slivers
    # Small cells left unmerged whose face triangles do not all see the centre (e.g. a
    # lens between a curved wall and an octree face lying on it) fail checkMesh or its
    # tet decomposition.  If they touch the wall they are removed: their faces become
    # walls of the neighbours.
    # A cell failing only the tet test is removed if the faces it exposes stay close to
    # the STL (no dent in the wall).
    removed = np.zeros(nP, bool)
    tet_cand = []
    for r in np.nonzero(proot == np.arange(nP))[0].tolist():
        if gvol[r] >= beta * hexvol[r] and not bad_piece[r]:
            continue
        keep_, cf_, sf_, V_, cc_ = poly_geom(node_faces(r))
        if V_ > 0 and _tets_ok(P, keep_, cf_, cc_, 1e-3 * V_):
            continue
        nb_ = [face_neighbour(fid, {r}) for fid, _ in keep_]
        if min(nb_) >= 0:
            continue
        hc = csize[pieces[r][1]]
        if V_ <= 0 or np.any(np.einsum("ij,ij->i", sf_, cf_ - cc_) <= 0.01 * np.linalg.norm(sf_, axis=1) * hc):
            removed[r] = True
        else:
            exposed = {v for (_, p), b in zip(keep_, nb_) if b >= 0 for v in p}
            tet_cand.append((r, sorted(exposed), hc))
    if tet_cand:
        ev = np.array(sorted({v for _, ex, _ in tet_cand for v in ex}), np.int64)
        _, dv, _ = dom.closest(P[ev])
        dmap = dict(zip(ev.tolist(), dv.tolist()))
        for r, ex, hc in tet_cand:
            # no internal face at all (a sliver enclosed by walls): removing it exposes
            # nothing, so it goes as well (max() of the empty set used to crash here)
            if max((dmap[v] for v in ex), default=0.0) < 0.4 * hc:
                removed[r] = True
    if removed.any():
        log(f"removed {int(removed.sum())} invalid wall slivers"
            f" ({float(np.sum(np.maximum(gvol[removed], 0))) * 1e9:.3g} mm^3)")

    # ------------------------------------------------------------ final cell numbering
    # nodes: bulk leaves (alive) and root pieces; ordered by parent leaf
    alive = np.nonzero(bulk_alive)[0]
    rp = np.nonzero((proot == np.arange(nP)) & ~removed)[0]
    node_leaf = np.concatenate([alive, pieces.leaf[rp]])
    order = _argsort_stable(node_leaf)
    ids = np.empty(len(order), np.int64); ids[order] = np.arange(len(order))
    nC = len(order)
    leaf_cid = np.full(nleaf, -1, np.int64); leaf_cid[alive] = ids[:len(alive)]
    piece_cid = np.full(nP, -1, np.int64); piece_cid[rp] = ids[len(alive):]
    to_bulk = proot >= nP
    piece_cid[to_bulk] = leaf_cid[proot[to_bulk] - nP]
    piece_cid[~to_bulk] = piece_cid[proot[~to_bulk]]    # -1 for removed cells
    cell_zone = np.empty(nC, np.int64)
    cell_zone[leaf_cid[alive]] = bulk_zone[alive]
    cell_zone[piece_cid[rp]] = pieces.zone[rp]

    # ------------------------------------------------------------ assembly (all numpy)
    bn = fn[bsel]
    b_off = np.repeat(fptr[bsel] - np.concatenate([[0], np.cumsum(bn)[:-1]]), bn)
    b_verts = fverts[np.arange(int(bn.sum())) + b_off]
    pn = p_n                                  # the pieces' faces, flattened above
    o_n = np.concatenate([bn, pn])
    o_ptr = np.concatenate([[0], np.cumsum(o_n)])
    o_verts = np.concatenate([b_verts, p_verts])
    o_cell = np.concatenate([leaf_cid[f_cell[bsel]], piece_cid[py_cell]])
    o_key = np.concatenate([f_key[bsel], py_key])
    del b_verts, p_verts
    srt = np.nonzero(o_cell >= 0)[0]
    srt = srt[_argsort_stable(o_key[srt])]
    # runs of equal keys in the sorted order (np.unique would sort them again)
    ks_ = o_key[srt]
    st = np.flatnonzero(np.concatenate([[True], ks_[1:] != ks_[:-1]])) if len(ks_) else np.zeros(0, np.int64)
    cnt = np.diff(np.concatenate([st, [len(ks_)]]))
    del ks_
    if (cnt > 2).any():
        log(f"WARNING: {int((cnt > 2).sum())} faces shared by more than two cells")
    two = st[cnt == 2]
    a1, a2 = srt[two], srt[two + 1]
    c1, c2 = o_cell[a1], o_cell[a2]
    keep = c1 != c2
    a1, a2, c1, c2 = a1[keep], a2[keep], c1[keep], c2[keep]
    i_occ = np.where(c1 < c2, a1, a2)
    i_own = np.minimum(c1, c2); i_nei = np.maximum(c1, c2)
    o = np.lexsort((i_nei, i_own))
    i_occ, i_own, i_nei = i_occ[o], i_own[o], i_nei[o]
    b_occ = srt[st[cnt == 1]]
    b_own = o_cell[b_occ]
    T("assembly and small-cell merging")

    def gather(occs):
        n_ = o_n[occs]
        off = np.repeat(o_ptr[occs] - np.concatenate([[0], np.cumsum(n_)[:-1]]), n_)
        return n_, o_verts[np.arange(int(n_.sum())) + off]

    # ------------------------------------------------------------ isolated fluid regions
    # fragments cut off from the main fluid (e.g. the tip of an acute slot narrower than
    # the cells) have no inflow and break the solution: small ones are removed
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ncomp, lab = connected_components(coo_matrix((np.ones(len(i_own)), (i_own, i_nei)), shape=(nC, nC)),
                                      directed=False)
    if ncomp > 1:
        ann, avv = gather(np.concatenate([i_occ, b_occ]))
        ffa = quality.FlatFaces((ann, avv))
        Cf_, Sf_ = quality.face_geometry(P, ffa)
        _, Vc = quality.cell_geometry(nC, np.concatenate([i_own, b_own]), i_nei, Cf_, Sf_)
        del ann, avv, ffa, Cf_, Sf_
        vol = np.bincount(lab, np.abs(Vc), ncomp)
        drop = vol < cfg["minRegionVolume"] * vol.sum()
        drop[np.argmax(vol)] = False            # never the main fluid region
        if drop.any():
            kc = ~drop[lab]
            new = np.cumsum(kc) - 1
            ki = kc[i_own]
            i_occ, i_own, i_nei = i_occ[ki], new[i_own[ki]], new[i_nei[ki]]
            kb = kc[b_own]
            b_occ, b_own = b_occ[kb], new[b_own[kb]]
            cell_zone = cell_zone[kc]
            log(f"isolated fluid regions: {int(drop.sum())} removed ({int((~kc).sum())} cells,"
                f" volume {vol[drop].sum():.3g} = {vol[drop].sum() / vol.sum():.2e} of the total)")
            nC = int(kc.sum())
        if (~drop).sum() > 1:
            big = np.sort(vol[~drop])[::-1]
            log(f"WARNING: the fluid has {len(big)} separate regions (volumes {', '.join(f'{v:.3g}' for v in big[:6])})"
                " - check for gaps closed by the mesh")

    # ------------------------------------------------------------ patches
    bnn, bverts = gather(b_occ)
    ffb = quality.FlatFaces((bnn, bverts))
    bC, bS = quality.face_geometry(P, ffb)
    # volume check: the mesh volume (divergence theorem over the boundary) against the STL's
    # signed volume.  A large mismatch means an open or self-intersecting STL (parity gives
    # the wrong inside) or fluid lost to features thinner than the cells
    vmesh = float(np.einsum("ij,ij->", bC, bS)) / 3.0
    vstl = abs(dom.repair["volume"])
    if dom.repair["inconsistentEdges"] or vstl <= 0:
        log(f"volume check: skipped (STL orientation inconsistent); mesh volume {vmesh:.6g}")
    else:
        dv = vmesh / vstl - 1.0
        log(f"volume check: mesh {vmesh:.6g}, STL {vstl:.6g} ({dv:+.2%})")
        if abs(dv) > float(cfg["volumeCheckTolerance"]):
            log(f"WARNING: mesh volume differs from the STL volume by {dv:+.1%} - the STL may be open or"
                f" self-intersecting, or fluid thinner than the cells was cut off")
    _, dd, tid = dom.closest(bC)
    reg = dom.region[tid]
    log(f"boundary faces: max distance of face centre from STL {dd.max():.3g}"
        f" ({dd.max() / hmin:.2f} x smallest cell)")
    if dd.max() > 0.75 * h0:
        log("WARNING: boundary faces far from the STL - check that the STL is watertight")
    ptypes = cfg["patchTypes"]
    patches, b_order = [], []
    start = len(i_occ)
    for r, name in enumerate(dom.names):
        sel = np.nonzero(reg == r)[0]
        sel = sel[np.argsort(b_own[sel], kind="stable")]
        patches.append(dict(name=name, type=ptypes.get(name, "wall"), startFace=start, nFaces=len(sel)))
        b_order.append(sel)
        start += len(sel)
    b_order = np.concatenate(b_order)
    all_occ = np.concatenate([i_occ, b_occ[b_order]])
    owner = np.concatenate([i_own, b_own[b_order]])
    neighbour = i_nei
    nn_, vv = gather(all_occ)
    del o_verts
    if cfg["mergeWallFaces"]:
        nn_, vv, owner, patches, nj = merge_coplanar_wall_faces(P, nn_, vv, owner, len(neighbour), patches)
        log(f"wall faces: {nj} coplanar pieces joined to their neighbour in the same cell")

    # ------------------------------------------------------------ zones
    if any(not z["conforming"] for z in zones):
        # non-conforming zones (MRF only needs the cellZone): cells whose centre is inside
        ffa = quality.FlatFaces((nn_, vv))
        Cf_, Sf_ = quality.face_geometry(P, ffa)
        Cc, _ = quality.cell_geometry(nC, owner, neighbour, Cf_, Sf_)
        del ffa, Cf_, Sf_
        for zi, z in enumerate(zones):
            if not z["conforming"]:
                ins = (cell_zone < 0) & (z["geom"].sdf(Cc) < 0)
                cell_zone[ins] = zi
                log(f"zone '{z['name']}' (non-conforming): {int(ins.sum())} cells with the centre inside")
        del Cc
    cell_zones, face_zones = {}, {}
    for zi, z in enumerate(zones):
        cell_zones[z["name"]] = np.nonzero(cell_zone == zi)[0]
        zo = cell_zone[i_own] == zi
        zn = cell_zone[i_nei] == zi
        fz = np.nonzero(zo != zn)[0]
        face_zones[z["name"] + "_interface"] = (fz, ~zo[fz])   # flip: normal points into zone

    # ------------------------------------------------------------ compact points
    used = np.zeros(len(P), bool)
    used[vv] = True
    remap = np.cumsum(used) - 1
    Pout = P[used]
    faces = (nn_, remap[vv])
    T("patches, zones, renumbering")
    if cfg["checkLeaks"]:
        # fluid joined through a solid wall (thinner than the cells there) breaks the
        # simulation: report it loudly
        nw, Xw = diagnose.through_wall(dom, Pout, faces, owner, neighbour)
        if nw:
            log(f"WARNING: leak check: {nw} cell-to-face lines cross a solid wall (fluid on both"
                f" sides joined; refine there), e.g. at " + ", ".join(
                    "(" + " ".join(f"{v:.6g}" for v in x) + ")" for x in Xw[:5]))
        else:
            log("leak check: no fluid connection through a solid wall")
        T("leak check")
    return dict(points=Pout, faces=faces, owner=owner, neighbour=neighbour, patches=patches,
                cell_zones=cell_zones, face_zones=face_zones, zones=zones, timer=T,
                cell_zone=cell_zone, leaves=leaves_total)


def _n_cells(R):
    nei = R["neighbour"]
    return int(max(R["owner"].max(), nei.max() if len(nei) else 0)) + 1


def run(config_path, case_dir=None, log=print, check=True):
    cfg = load_config(config_path)
    case_dir = case_dir or _rel(cfg, cfg.get("case", "."))
    R = generate(cfg, log)
    T = R["timer"]
    if cfg.get("addLayers") is not None:             # {} = layers with the default settings
        from .layers import add_layers
        # the interface of a non-conforming zone is a staircase between cells, not a
        # surface: it must not constrain the layers; its faceZone is rebuilt afterwards
        nc = [z["name"] for z in R["zones"] if not z["conforming"]]
        fz = {k: v for k, v in R["face_zones"].items() if k[:-len("_interface")] not in nc}
        R = add_layers(dict(R, face_zones=fz), cfg["addLayers"], log=log)
        if nc:
            nC = _n_cells(R)
            nI = len(R["neighbour"])
            fz = dict(R["face_zones"])
            for name in nc:
                inz = np.zeros(nC, bool); inz[R["cell_zones"][name]] = True
                zo, zn = inz[R["owner"][:nI]], inz[R["neighbour"]]
                f = np.nonzero(zo != zn)[0]
                fz[name + "_interface"] = (f, ~zo[f])        # flip: normal points into zone
            R["face_zones"] = fz
        T("boundary layers")
    ami_z = [z for z in R["zones"] if z["ami"]]
    if ami_z:
        from . import ami
        for z in ami_z:
            if cfg["amiExactRims"]:
                R, _, _ = ami.exact_rims(R, z["name"], z["geom"], cfg["featureAngle"], log=log)
            err = ami.axisymmetry_error(R, z["name"], z["geom"], z["origin"], z["axis"])
            hz = float(cfg["baseCellSize"]) / 2 ** max(int(z["surfaceLevel"]), 0)
            if err > 0.05 * hz:
                log(f"WARNING: zone '{z['name']}': its interface is not a surface of revolution about"
                    f" the axis {list(np.round(z['axis'], 6))} through {list(np.round(z['origin'], 6))}"
                    f" (turned by 7 degrees it is {err:.3g} off itself): the zone cannot slide."
                    " Check origin / axis")
            R = ami.split_interface(R, z["name"], cfg["openfoam"])
            st = R["ami_split"][z["name"]]
            log(f"zone '{z['name']}': {st['faces']} interface faces -> sliding-mesh patches "
                + " / ".join(ami.patch_names(z["name"], cfg["openfoam"]))
                + f", {st['points']} points duplicated on the zone side")
        T("sliding-mesh interfaces")
    # one flattening of the face lists serves the binary writer and the quality check
    ff = quality.FlatFaces(R["faces"]) if (check or cfg["writeFormat"] == "binary") else None
    write_polymesh(case_dir, R["points"], (ff.n, ff.v) if cfg["writeFormat"] == "binary" else R["faces"],
                   R["owner"], R["neighbour"], R["patches"],
                   R["cell_zones"], R["face_zones"], fmt=cfg["writeFormat"])
    mrf_z = [z for z in R["zones"] if not z["ami"]]
    if cfg["writeMRFProperties"] and mrf_z:
        write_mrf(case_dir, mrf_z)
    if ami_z:
        # an MRFProperties left from an earlier run must not keep the zones that now turn
        # with the mesh (rotating-frame terms on top of the motion)
        nd = ami.drop_mrf_entries(case_dir, [z["name"] for z in ami_z])
        if nd:
            log(f"MRFProperties: removed {nd} entr{'y' if nd == 1 else 'ies'} of sliding-mesh zones")
    if ami_z and cfg["writeDynamicMeshDict"]:
        ami.write_dynamic_mesh_dict(case_dir, ami_z, cfg["openfoam"])
    if ami_z:
        hint = ami.couple_hint([z["name"] for z in ami_z], cfg["openfoam"])
        if hint:
            log(f"sliding mesh (openfoam.org): after reading the mesh run {hint}")
    T("write polyMesh")
    if check:
        R["quality"] = quality.check(R["points"], ff, R["owner"], R["neighbour"], log=log)
        T("quality check")
    nC = _n_cells(R)
    log(f"total {time.time() - T.t0:.1f}s, {nC} cells")
    return R


def write_mrf(case_dir, zones):
    d = os.path.join(case_dir, "constant")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "MRFProperties"), "w") as f:
        f.write("""FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      MRFProperties;
}
// generated by cutmesh - set omega and nonRotatingPatches for your case

""")
        for z in zones:
            o, a = z["origin"], z["axis"]
            nr = " ".join(z["nonRotatingPatches"])
            f.write(f"""MRF_{z['name']}
{{
    cellZone    {z['name']};
    active      yes;
    nonRotatingPatches ({nr});
    origin      ({o[0]:.9g} {o[1]:.9g} {o[2]:.9g});
    axis        ({a[0]:.9g} {a[1]:.9g} {a[2]:.9g});
    omega       {z['omega']:.9g}; // [rad/s]
}}

""")
