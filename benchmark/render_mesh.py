"""Pictures of a generated mesh: an outside overview and a cut through the inside.

    python benchmark/render_mesh.py CASE_DIR [-o OUT_DIR] [--title NAME]

CASE_DIR holds constant/polyMesh (ASCII, as cutmesh writes it).  Writes into OUT_DIR
(default: CASE_DIR's parent):
    overview.png  the wall faces seen from two sides (front/above and back/below), coloured
                  by patch, every cell face outlined
    cut.png       left: the mesh with the half x > x_mid of the cells removed (whole cells,
                  no clipping: the cut face shows the cell layout); right: plane sections
                  through the middle of the bounding box along x, y and z
Cells failing the checkMesh limits (non-positive volume, non-orthogonality > 70, skewness
> 4) are drawn red in every view.  run_bench.py calls render() after each case.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from cutmesh import quality  # noqa: E402
from cutmesh.viz import slice_segments  # noqa: E402

PATCH_COLOURS = ["#9fb7d4", "#e3b778", "#9ccf9c", "#d39ccf", "#a8d8d8", "#d8c59c", "#c0c0c0"]
CUT_COLOUR = "#f3e2c0"
BAD_COLOUR = "#e0302a"


# ------------------------------------------------------------------ geometry helpers
def _mesh_geometry(R):
    q = R.get("quality") or {}
    if "_ff" not in q:
        q = quality.check(R["points"], R["faces"], R["owner"], R["neighbour"], log=None)
    ff, Cf, Sf, C, V = q["_ff"], q["_Cf"], q["_Sf"], q["_C"], q["_V"]
    nI = len(R["neighbour"])
    nC = len(C)
    bad = V <= 0
    badf = np.zeros(nI, bool)
    if nI:
        badf |= q["_nonorth"] > 70
        badf |= q["_skew"] > 4
    own, nei = R["owner"], R["neighbour"]
    bad[own[:nI][badf]] = True
    bad[nei[badf]] = True
    return dict(ff=ff, Cf=Cf, Sf=Sf, C=C, bad=bad, nI=nI, nC=nC)


def _view_basis(az, el):
    az, el = np.radians(az), np.radians(el)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # towards camera
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(-d, up)
    right /= np.linalg.norm(right)
    sup = np.cross(right, -d)
    return d, right, sup


def _draw_faces(ax, P, ff, fids, normals, colours, d, right, sup, lw):
    """Painter's algorithm: back faces culled, the rest drawn far to near, Lambert shading."""
    if len(fids) == 0:
        return
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
    front = n @ d > 1e-9
    fids, n, colours = fids[front], n[front], colours[front]
    if len(fids) == 0:
        return
    light = d + 0.6 * sup + 0.3 * right
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
    rgb = colours * shade[:, None]
    ptr = ff.ptr
    cen = np.add.reduceat(P[ff.v], ptr[:-1], axis=0)[fids] / ff.n[fids, None]
    order = np.argsort(cen @ d)
    polys = []
    proj = np.stack([P @ right, P @ sup], 1)
    for f in fids[order]:
        polys.append(proj[ff.v[ptr[f]:ptr[f + 1]]])
    ax.add_collection(PolyCollection(polys, facecolors=rgb[order], edgecolors="#20242a",
                                     linewidths=lw, antialiased=True))


def _rgb(hexs):
    hexs = np.atleast_1d(hexs)
    return np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in hexs])


def _finish(ax, title):
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def _edge_width(nfaces):
    return 0.25 if nfaces < 5000 else 0.12 if nfaces < 30000 else 0.06


def _patch_colours(R, geo):
    nF = geo["ff"].n.shape[0]
    col = np.tile(_rgb("#c0c0c0"), (nF, 1))
    for k, p in enumerate(R["patches"]):
        s = int(p["startFace"])
        col[s:s + int(p["nFaces"])] = _rgb(PATCH_COLOURS[k % len(PATCH_COLOURS)])
    return col


# ------------------------------------------------------------------ figures
def overview(R, geo, path, title=""):
    P, ff, nI = R["points"], geo["ff"], geo["nI"]
    nF = ff.n.shape[0]
    bf = np.arange(nI, nF)
    col = _patch_colours(R, geo)[bf]
    bad = geo["bad"][R["owner"][bf]]
    col[bad] = _rgb(BAD_COLOUR)
    lw = _edge_width(len(bf) / 2)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (az, el, lab) in zip(axs, [(-55, 28, "front / above"), (125, -22, "back / below")]):
        d, right, sup = _view_basis(az, el)
        _draw_faces(ax, P, ff, bf, geo["Sf"][bf], col, d, right, sup, lw)
        _finish(ax, lab)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PATCH_COLOURS[k % len(PATCH_COLOURS)], ec="none")
               for k in range(len(R["patches"]))]
    labels = [p["name"] for p in R["patches"]]
    if bad.any():
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=BAD_COLOUR, ec="none"))
        labels.append(f"bad cells ({int(geo['bad'].sum())})")
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False, fontsize=9)
    fig.suptitle(f"{title}   {geo['nC']} cells, {len(bf)} wall faces", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _slice_panel(ax, R, geo, axis, value):
    P = R["points"]
    seg, fid = slice_segments(P, R["faces"], axis, value)
    keep = [i for i in range(3) if i != axis]
    s2 = seg[:, :, keep]
    nI = geo["nI"]
    own = R["owner"][fid]
    nei = own.copy()
    m = fid < nI
    nei[m] = R["neighbour"][fid[m]]
    badseg = geo["bad"][own] | geo["bad"][nei]
    bnd = fid >= nI
    lw = 0.35 if len(seg) < 20000 else 0.18
    ax.add_collection(LineCollection(s2[~bnd & ~badseg], colors="#6b7280", linewidths=lw))
    ax.add_collection(LineCollection(s2[bnd & ~badseg], colors="#111827", linewidths=1.0))
    ax.add_collection(LineCollection(s2[badseg], colors=BAD_COLOUR, linewidths=1.2))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(f"{'xyz'[axis]} = {value:.4g}", fontsize=10)
    ax.set_xlabel("xyz"[keep[0]]); ax.set_ylabel("xyz"[keep[1]])
    ax.tick_params(labelsize=7)


def cut(R, geo, path, title=""):
    P, ff, nI, C = R["points"], geo["ff"], geo["nI"], geo["C"]
    lo, hi = P.min(0), P.max(0)
    mid = (lo + hi) / 2 + 0.0123 * (hi - lo)       # off any grid plane through the centre
    keep = C[:, 0] < mid[0]
    own, nei = R["owner"], R["neighbour"]
    nF = ff.n.shape[0]
    # wall faces of kept cells
    bf = np.arange(nI, nF)
    bf = bf[keep[own[bf]]]
    # internal faces between kept and removed cells, oriented out of the kept part
    ko, kn = keep[own[:nI]], keep[nei]
    cf = np.nonzero(ko != kn)[0]
    cn = geo["Sf"][cf] * np.where(ko[cf], 1.0, -1.0)[:, None]
    cell_of_cf = np.where(ko[cf], own[cf], nei[cf])
    fids = np.concatenate([bf, cf])
    normals = np.concatenate([geo["Sf"][bf], cn])
    col = np.concatenate([_patch_colours(R, geo)[bf], np.tile(_rgb(CUT_COLOUR), (len(cf), 1))])
    badc = np.concatenate([geo["bad"][own[bf]], geo["bad"][cell_of_cf]])
    col[badc] = _rgb(BAD_COLOUR)
    fig = plt.figure(figsize=(16, 9))
    ax3 = fig.add_axes([0.0, 0.0, 0.52, 0.93])
    d, right, sup = _view_basis(25, 24)              # camera on the +x side, facing the cut
    _draw_faces(ax3, P, ff, fids, normals, col, d, right, sup, _edge_width(len(fids) / 2))
    _finish(ax3, f"cells with centre x < {mid[0]:.4g} (cut face in beige)")
    for k, axis in enumerate((0, 1, 2)):
        ax = fig.add_axes([0.58, 0.675 - k * 0.305, 0.40, 0.225])
        _slice_panel(ax, R, geo, axis, mid[axis])
    fig.suptitle(f"{title}   cut through the middle" + (f"   ({int(geo['bad'].sum())} bad cells in red)"
                                                         if geo["bad"].any() else ""), fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render(R, outdir, title=""):
    """Write overview.png and cut.png for the mesh dict R (points, faces, owner, neighbour,
    patches[, quality]) into outdir; return the paths."""
    geo = _mesh_geometry(R)
    p1, p2 = os.path.join(outdir, "overview.png"), os.path.join(outdir, "cut.png")
    overview(R, geo, p1, title)
    cut(R, geo, p2, title)
    return [p1, p2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title")
    a = ap.parse_args(argv)
    from cutmesh.readfoam import read_mesh
    R = read_mesh(a.case)
    out = a.out or os.path.dirname(os.path.abspath(a.case))
    title = a.title or os.path.basename(out)
    for p in render(R, out, title):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
