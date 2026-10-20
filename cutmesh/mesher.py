"""cutmesh pipeline: STL interior -> octree -> cut cells (STL walls and rotating-zone
walls are both represented exactly by mesh faces) -> OpenFOAM polyMesh."""
import json
import os
import time
import numpy as np
from scipy.spatial import cKDTree

from .geometry import (Surface, Cylinder, SurfaceZone, FeatureSet, read_stl,
                       sample_segments, cylinder_features)
from .octree import Octree
from .cutcell import CutContext, split_components
from .polymesh import write_polymesh
from . import quality


DEFAULTS = {
    "bufferLayers": 2,          # cells per level in refinement transitions
    "padding": 1,               # base cells around the STL bounding box
    "featureAngle": 30.0,
    "featureLevel": None,       # None -> max surface level
    "snapTolerance": 0.3,       # snap vertices within this fraction of h onto surfaces
    "mergeRatio": 0.25,         # merge cut cells smaller than this fraction of their hex
    "patchTypes": {},
    "rotatingZones": [],
    "writeMRFProperties": True,
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
    with open(path) as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    full = dict(DEFAULTS); full.update(cfg)
    full["_base"] = base
    return full


def _rel(cfg, p):
    return p if os.path.isabs(p) else os.path.join(cfg["_base"], p)


def build_zones(cfg):
    zones = []
    for z in cfg["rotatingZones"]:
        if "cylinder" in z:
            c = z["cylinder"]
            g = Cylinder(c["point1"], c["point2"], c["radius"])
            origin = c.get("origin", c["point1"])
            axis = (np.asarray(c["point2"], float) - np.asarray(c["point1"], float))
            axis = axis / np.linalg.norm(axis)
        else:
            s = Surface(read_stl(_rel(cfg, z["stl"])))
            g = SurfaceZone(s)
            origin = z.get("origin", [0, 0, 0])
            axis = z.get("axis", [0, 0, 1])
        zones.append(dict(name=z["name"], geom=g, surfaceLevel=z.get("surfaceLevel", 0),
                          volumeLevel=z.get("volumeLevel", 0), origin=np.asarray(origin, float),
                          axis=np.asarray(axis, float), omega=z.get("omega", 0.0),
                          nonRotatingPatches=z.get("nonRotatingPatches", [])))
    return zones


def make_feature_hook(feat, h_of_cell, min_bend=0.03, reach=0.75):
    """Cap refinement: if a sharp feature runs through the cut cell, insert the
    feature point into the cap and fan-triangulate, so the corner is not chamfered."""
    if feat is None or feat._kd is None:
        return None
    state = {"cell": 0, "n": 0}

    def hook(ctx, loop):
        if len(loop) < 3:
            return [loop]
        P = np.array([ctx.pos[v] for v in loop])
        c = P.mean(0)
        h = h_of_cell[state["cell"]]
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
        m = len(loop)
        return [[loop[i], loop[(i + 1) % m], vid] for i in range(m)]

    return hook, state


def generate(cfg, log=print):
    T = Timer(log)
    dom = Surface(read_stl(_rel(cfg, cfg["domain"])))
    zones = build_zones(cfg)
    h0 = float(cfg["baseCellSize"])
    sref = cfg.get("surfaceRefinement", {})
    default_level = int(sref.get("default", 0))
    region_level = np.array([int(sref.get(n, default_level)) for n in dom.names])
    flevel = cfg["featureLevel"]
    if flevel is None:
        flevel = int(region_level.max())
    regions = []
    for r in cfg.get("refinementRegions", []):
        c = r["cylinder"]
        regions.append((Cylinder(c["point1"], c["point2"], c["radius"]), int(r["level"])))
    Lmax = int(max([region_level.max(), flevel] + [max(z["surfaceLevel"], z["volumeLevel"]) for z in zones]
                   + [lv for _, lv in regions]))
    hmin = h0 / 2 ** Lmax
    log(f"domain: {len(dom.tris)} triangles, regions {dom.names}")
    log(f"base cell {h0:g}, max level {Lmax} (smallest cell {hmin:g})")

    # ------------------------------------------------------------ search structures
    dom.build_search(2 * hmin)
    for z in zones:
        if isinstance(z["geom"], SurfaceZone):
            z["geom"].s.build_search(2 * hmin)
    nbuf = int(cfg["bufferLayers"])
    ssrc, vsrc = [], []
    for r, name in enumerate(dom.names):
        if region_level[r] > 0:
            p, _ = dom.samples(h0 / 2 ** region_level[r] / 2, regions=[r])
            ssrc.append((cKDTree(p), int(region_level[r]), nbuf))
    seg = dom.feature_edges(cfg["featureAngle"])
    dom_feat = FeatureSet(seg, L=dom.L)
    if flevel > 0:
        if len(seg):
            fp = sample_segments(seg, h0 / 2 ** flevel / 2)
            ssrc.append((cKDTree(fp), int(flevel), nbuf))
            log(f"feature edges: {len(seg)} segments -> level {flevel}")
    for z in zones:
        if z["surfaceLevel"] > 0:
            p, _ = z["geom"].samples(h0 / 2 ** z["surfaceLevel"] / 2)
            ssrc.append((cKDTree(p), z["surfaceLevel"], nbuf))
        if z["volumeLevel"] > 0:
            vsrc.append((getattr(z["geom"], "refine_sdf", z["geom"].sdf), z["volumeLevel"], nbuf))
    for g, lv in regions:
        vsrc.append((g.sdf, lv, nbuf))
    T("geometry / search structures")

    # ------------------------------------------------------------ octree
    pad = int(cfg["padding"])
    lo = dom.bbmin - pad * h0
    n = np.ceil((dom.bbmax - dom.bbmin) / h0 + 2 * pad).astype(int)
    oc = Octree(lo, h0, n, Lmax)
    log(f"root grid {n.tolist()}")
    def keep_fn(c, hd):
        out = ~dom.inside(c)
        far = dom.approx_distance(c, upper=hd + dom._rmax) > hd + dom._rmax
        return ~(out & far)
    oc.refine(ssrc, vsrc, log=lambda s: log(s), keep_fn=keep_fn)
    T(f"octree ({len(oc.lev)} leaves)")
    M = oc.build_cells()
    nleaf = len(oc.lev)
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
    vh_min = np.full(len(pts), np.inf); vh_max = np.zeros(len(pts))
    np.minimum.at(vh_min, cvid.ravel(), np.repeat(csize, 8))
    np.maximum.at(vh_max, cvid.ravel(), np.repeat(csize, 8))
    del cvid, M
    T(f"polyhedral cells ({len(pts)} points)")

    def leaf_faces(ci):
        out = []
        for f in range(cptr[ci], cptr[ci + 1]):
            out.append((int(f_key[f]), fverts[fptr[f]:fptr[f + 1]].tolist()))
        return out

    def classify_leaves(phi, leaves):
        """-1 all vertices <= 0, +1 all >= 0 (and some > 0), 0 straddling."""
        pv = phi[fverts]
        hn = np.maximum.reduceat((pv < 0).astype(np.int8), fptr[:-1])
        hp = np.maximum.reduceat((pv > 0).astype(np.int8), fptr[:-1])
        cn = np.maximum.reduceat(hn, cptr[:-1])[leaves]
        cp_ = np.maximum.reduceat(hp, cptr[:-1])[leaves]
        return np.where(cp_ == 0, -1, np.where(cn == 0, 1, 0))

    # ------------------------------------------------------------ level sets + snapping
    alpha = float(cfg["snapTolerance"])
    approx = dom.approx_distance(pts, upper=float(np.sqrt(3) * vh_max.max() + dom._rmax + hmin))
    near = approx < np.sqrt(3) * vh_max + dom._rmax + hmin
    del approx
    phi_d = dom.sdf(pts, exact_mask=near, far_value=dom.L, exact_within=alpha * h0 + 2 * hmin)
    movable = ~hanging
    tol_v = alpha * vh_min
    snapped = np.zeros(len(pts), bool)

    def snap_features(feat, phi, label):
        """corners first, then feature edges; moved vertices get phi = 0"""
        nc = 0
        if feat is None or feat._kd is None:
            return
        if len(feat.corners):
            kd = cKDTree(pts)
            d, j = kd.query(feat.corners)
            del kd
            for ci_, (dd, jj) in enumerate(zip(d.tolist(), j.tolist())):
                if movable[jj] and not snapped[jj] and dd < tol_v[jj]:
                    pts[jj] = feat.corners[ci_]; phi[jj] = 0.0; snapped[jj] = True; nc += 1
        sel = np.nonzero(movable & ~snapped & (np.abs(phi) < 2 * tol_v + hmin))[0]
        cp, d = feat.closest(pts[sel])
        ok = d < tol_v[sel]
        pts[sel[ok]] = cp[ok]; phi[sel[ok]] = 0.0; snapped[sel[ok]] = True
        log(f"snapped {nc} vertices to {label} corners, {int(ok.sum())} to {label} feature edges")

    # ---- 1) regular (non-hanging) vertices: features, then surfaces
    snap_features(dom_feat, phi_d, "STL")
    snap = movable & ~snapped & (np.abs(phi_d) < tol_v) & (phi_d != 0)
    cp, _, _ = dom.closest(pts[snap])
    pts[snap] = cp
    phi_d[snap] = 0.0
    snapped |= snap
    log(f"snapped {int(snap.sum())} vertices onto the domain surface")
    phi_z = []
    zone_feats = []
    for z in zones:
        g = z["geom"]
        if isinstance(g, SurfaceZone):
            nearz = g.s.approx_distance(pts, upper=float(np.sqrt(3) * vh_max.max() + g.s._rmax + hmin)) \
                < np.sqrt(3) * vh_max + g.s._rmax + hmin
            pz = g.sdf(pts, exact_mask=nearz, far_value=g.L)
        else:
            pz = g.sdf(pts)
        if isinstance(g, Cylinder):
            zf = FeatureSet(cylinder_features(g, hmin / 4), L=g.L)
        else:
            zf = FeatureSet(g.s.feature_edges(cfg["featureAngle"]), L=g.L)
        zone_feats.append(zf)
        before = snapped.copy()           # vertices already on the STL are not moved
        snap_features(zf, pz, f"zone '{z['name']}'")
        sz = movable & ~snapped & (np.abs(pz) < tol_v) & (pz != 0)
        cp, _, _ = g.closest(pts[sz])
        pts[sz] = cp
        pz[sz] = 0.0
        snapped |= sz
        moved = snapped & ~before
        if moved.any():   # re-evaluate the domain level set where vertices moved
            phi_d[moved] = dom.sdf(pts[moved])
        log(f"snapped {int(sz.sum())} vertices onto zone '{z['name']}'")
        phi_z.append(pz)

    # ---- 2) hanging vertices, coarsest constraint first: follow the (moved) parents,
    # then slide along the coarse edge / within the coarse face onto a surface.  If the
    # constraint is nearly tangent to the surface, the node is treated as on-surface.
    geoms = [(dom, phi_d, None)] + [(z["geom"], phi_z[i], i) for i, z in enumerate(zones)]
    nslide = nvirt = 0
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
    log(f"hanging nodes: {nslide} slid onto a surface along their edge/face,"
        f" {nvirt} treated as on-surface")
    phi_d[(phi_d != 0) & (np.abs(phi_d) < 1e-9 * vh_min)] = 0.0
    for pz in phi_z:
        pz[(pz != 0) & (np.abs(pz) < 1e-9 * vh_min)] = 0.0
    moved_any = snapped.copy()
    del vh_min, vh_max, near, movable, tol_v
    T("level sets and vertex snapping")

    # ------------------------------------------------------------ pass 1: domain cut
    # Uncut leaves stay in flat arrays ("bulk"); only cut cells become python pieces.
    cls = classify_leaves(phi_d, np.arange(nleaf))
    bulk_alive = cls < 0
    bulk_zone = np.full(nleaf, -1, np.int64)
    cut_leaves = np.nonzero(cls == 0)[0]
    positions = pts.tolist()
    ctx = CutContext(positions, phi_d.tolist(), M_nkeys := int(f_key.max()) + 1)
    fc = cfg.get("featureCaps", "domain")
    hk = make_feature_hook(dom_feat, csize) if fc in ("both", "domain", True) else None
    pieces = []        # [faces, parent_leaf, zone]
    for ci in cut_leaves.tolist():
        if hk:
            hk[1]["cell"] = ci
        neg, _, _ = ctx.cut(leaf_faces(ci), shared_cap=False, cap_hook=hk[0] if hk else None)
        for comp in split_components(neg):
            pieces.append([comp, ci, -1])
    next_id = ctx.next_id
    log(f"domain cut: {len(cut_leaves)} cut cells, {int(bulk_alive.sum()) + len(pieces)} cells kept"
        + (f", {hk[1]['n']} feature points inserted" if hk else ""))
    newv = np.array(ctx.new_vertices, np.int64)
    P = np.array(positions)
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
        for ci in newcut.tolist():
            pieces.append([leaf_faces(ci), ci, -1])
        positions = P.tolist()
        ctx = CutContext(positions, pz.tolist(), next_id)
        if isinstance(g, Cylinder):
            zfeat = FeatureSet(cylinder_features(g, hmin / 4), L=g.L)
        else:
            zfeat = FeatureSet(g.s.feature_edges(cfg["featureAngle"]), L=g.L)
        zh = make_feature_hook(zfeat, csize) if fc in ("both", "zone", True) else None
        newp = []
        nc = 0
        for pc in pieces:
            c = ctx.classify(pc[0])
            if c < 0:
                pc[2] = zi
                newp.append(pc)
            elif c > 0:
                newp.append(pc)
            else:
                nc += 1
                if zh:
                    zh[1]["cell"] = pc[1]
                neg, pos, _ = ctx.cut(pc[0], shared_cap=True, cap_hook=zh[0] if zh else None)
                for comp in split_components(neg):
                    newp.append([comp, pc[1], zi])
                for comp in split_components(pos):
                    newp.append([comp, pc[1], pc[2]])
        pieces = newp
        next_id = ctx.next_id
        P = np.array(positions)
        newv = np.array(ctx.new_vertices, np.int64)
        par = np.array(ctx.new_parents, np.int64).reshape(-1, 2)
        del positions, ctx
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
        log(f"zone '{z['name']}': {nc} cut cells, total {int(bulk_alive.sum()) + len(pieces)}")
        T(f"zone cut '{z['name']}'")

    # ------------------------------------------------------------ distorted uncut cells
    # uncut leaves whose corners were snapped can become non-convex: hand them to the
    # merging stage like cut cells
    touch = np.maximum.reduceat(moved_any[fverts].astype(np.int8), fptr[:-1])
    touch = np.maximum.reduceat(touch, cptr[:-1]).astype(bool) & bulk_alive
    tl = np.nonzero(touch)[0]
    nbad_bulk = 0
    if len(tl):
        occs = np.concatenate([np.arange(cptr[c], cptr[c + 1]) for c in tl])
        lens = cptr[tl + 1] - cptr[tl]
        owner_ = np.repeat(np.arange(len(tl)), lens)
        nn_ = fn[occs]
        off = np.repeat(fptr[occs] - np.concatenate([[0], np.cumsum(nn_)[:-1]]), nn_)
        ffl = quality.FlatFaces((nn_, fverts[np.arange(int(nn_.sum())) + off]))
        cf_, sf_ = quality.face_geometry(P, ffl)
        est = np.zeros((len(tl), 3)); np.add.at(est, owner_, cf_); est /= lens[:, None]
        pv = np.einsum("ij,ij->i", sf_, cf_ - est[owner_])
        vv = np.bincount(owner_, pv, len(tl))
        cc = np.zeros((len(tl), 3)); np.add.at(cc, owner_, pv[:, None] * (0.75 * cf_ + 0.25 * est[owner_]))
        cc /= np.where(np.abs(vv) > 0, vv, 1.0)[:, None]
        sp = np.einsum("ij,ij->i", sf_, cf_ - cc[owner_])
        badf = sp < 0.05 * np.linalg.norm(sf_, axis=1) * csize[tl][owner_]
        badc = tl[np.bincount(owner_, badf.astype(int), len(tl)) > 0]
        for ci in badc.tolist():
            pieces.append([leaf_faces(ci), ci, int(bulk_zone[ci])])
        bulk_alive[badc] = False
        nbad_bulk = len(badc)
        del occs, owner_, ffl, cf_, sf_, est, pv, cc, sp, badf
    if nbad_bulk:
        log(f"{nbad_bulk} snapped uncut cells are non-convex -> merge candidates")

    # ------------------------------------------------------------ small-cell merging
    nP = len(pieces)
    occ = {}
    for pi, pc in enumerate(pieces):
        for k, (fid, poly) in enumerate(pc[0]):
            occ.setdefault(fid, []).append(pi)
    # bulk neighbour of an original octree face key
    bsel = np.nonzero(bulk_alive[f_cell])[0]
    key2bulk = np.full(M_nkeys, -1, np.int64)
    key2bulk[f_key[bsel]] = f_cell[bsel]
    py_polys, py_cell, py_key = [], [], []
    for pi, pc in enumerate(pieces):
        for fid, poly in pc[0]:
            py_polys.append(poly); py_cell.append(pi); py_key.append(fid)
    py_cell = np.array(py_cell, np.int64); py_key = np.array(py_key, np.int64)
    vol = np.zeros(nP)
    farea = {}
    if nP:
        ffp = quality.FlatFaces(py_polys)
        Cf, Sf = quality.face_geometry(P, ffp)
        ref = np.zeros((nP, 3)); ref[py_cell] = Cf
        cnt_ = np.bincount(py_cell, minlength=nP)
        est = np.zeros((nP, 3)); np.add.at(est, py_cell, Cf); est /= np.maximum(cnt_, 1)[:, None]
        pv = np.einsum("ij,ij->i", Sf, Cf - est[py_cell])
        vol = np.bincount(py_cell, pv, nP) / 3.0
        cc = np.zeros((nP, 3))
        np.add.at(cc, py_cell, pv[:, None] * (0.75 * Cf + 0.25 * est[py_cell]))
        cc /= np.where(np.abs(vol) > 0, 3.0 * vol, 1.0)[:, None]
        # "bad" pieces: some face does not clearly see the centroid (non-convex cut cell)
        hp = np.array([csize[pc[1]] for pc in pieces])
        selfpyr = np.einsum("ij,ij->i", Sf, Cf - cc[py_cell])
        badf = selfpyr < 0.05 * np.linalg.norm(Sf, axis=1) * hp[py_cell]
        bad_piece = np.bincount(py_cell, badf.astype(int), nP) > 0
        farea = dict(zip(py_key.tolist(), np.linalg.norm(Sf, axis=1).tolist()))
        del ffp, Cf, Sf, ref, est, pv, cc, selfpyr, badf
    else:
        bad_piece = np.zeros(0, bool)
    hexvol = np.array([csize[pc[1]] ** 3 for pc in pieces])
    ratio = vol / np.maximum(hexvol, 1e-300)
    zone_of = [pc[2] for pc in pieces]
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
    o_ = np.argsort(kk, kind="stable"); kk = kk[o_]; kc = kc[o_]
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

    max_no = float(cfg.get("mergeMaxNonOrtho", 65.0))
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
        worst = 1.0
        for (fid, _), s in zip(keep, sf):
            nb = face_neighbour(fid, me)
            if nb < 0:
                continue
            d = node_centre(nb) - cc
            c = float(np.dot(d, s) / (np.linalg.norm(d) * np.linalg.norm(s) + 1e-300))
            worst = min(worst, c)
        return worst

    merged = rejected = 0
    for qi, pi in enumerate(small.tolist()):
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
    proot = np.array([find(i) for i in range(nP)], np.int64)
    log(f"merged {merged} cut cells ({nsmall} small < {beta:g} of hex volume,"
        f" {len(small) - nsmall} non-convex); {rejected} left as they are"
        f" (no merge partner keeps the cell valid)")

    # ------------------------------------------------------------ final cell numbering
    # nodes: bulk leaves (alive) and root pieces; ordered by parent leaf
    alive = np.nonzero(bulk_alive)[0]
    rp = np.nonzero(proot == np.arange(nP))[0]
    node_leaf = np.concatenate([alive, np.array([pieces[r][1] for r in rp], np.int64)])
    order = np.argsort(node_leaf, kind="stable")
    ids = np.empty(len(order), np.int64); ids[order] = np.arange(len(order))
    nC = len(order)
    leaf_cid = np.full(nleaf, -1, np.int64); leaf_cid[alive] = ids[:len(alive)]
    piece_cid = np.full(nP, -1, np.int64); piece_cid[rp] = ids[len(alive):]
    to_bulk = proot >= nP
    piece_cid[to_bulk] = leaf_cid[proot[to_bulk] - nP]
    piece_cid[~to_bulk] = piece_cid[proot[~to_bulk]]
    cell_zone = np.empty(nC, np.int64)
    cell_zone[leaf_cid[alive]] = bulk_zone[alive]
    cell_zone[piece_cid[rp]] = np.array([zone_of[r] for r in rp], np.int64)

    # ------------------------------------------------------------ assembly (all numpy)
    bn = fn[bsel]
    b_off = np.repeat(fptr[bsel] - np.concatenate([[0], np.cumsum(bn)[:-1]]), bn)
    b_verts = fverts[np.arange(int(bn.sum())) + b_off]
    pn = np.array([len(p) for p in py_polys], np.int64)
    p_verts = np.fromiter((v for p in py_polys for v in p), np.int64, count=int(pn.sum()))
    del py_polys
    o_n = np.concatenate([bn, pn])
    o_ptr = np.concatenate([[0], np.cumsum(o_n)])
    o_verts = np.concatenate([b_verts, p_verts])
    o_cell = np.concatenate([leaf_cid[f_cell[bsel]], piece_cid[py_cell]])
    o_key = np.concatenate([f_key[bsel], py_key])
    del b_verts, p_verts
    srt = np.argsort(o_key, kind="stable")
    uk, st, cnt = np.unique(o_key[srt], return_index=True, return_counts=True)
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

    # ------------------------------------------------------------ patches
    bnn, bverts = gather(b_occ)
    ffb = quality.FlatFaces((bnn, bverts))
    bC, _ = quality.face_geometry(P, ffb)
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

    # ------------------------------------------------------------ zones
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
    return dict(points=Pout, faces=faces, owner=owner, neighbour=neighbour, patches=patches,
                cell_zones=cell_zones, face_zones=face_zones, zones=zones, timer=T,
                cell_zone=cell_zone, leaves=leaves_total)


def run(config_path, case_dir=None, log=print, check=True):
    cfg = load_config(config_path)
    case_dir = case_dir or _rel(cfg, cfg.get("case", "."))
    R = generate(cfg, log)
    T = R["timer"]
    write_polymesh(case_dir, R["points"], R["faces"], R["owner"], R["neighbour"], R["patches"],
                   R["cell_zones"], R["face_zones"])
    if cfg["writeMRFProperties"] and R["zones"]:
        write_mrf(case_dir, R["zones"])
    T("write polyMesh")
    if check:
        R["quality"] = quality.check(R["points"], R["faces"], R["owner"], R["neighbour"], log=log)
        T("quality check")
    nC = int(max(R["owner"].max(), R["neighbour"].max())) + 1
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
