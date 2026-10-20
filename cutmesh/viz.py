"""Plane slices of a generated mesh (matplotlib), for quick visual inspection."""
import numpy as np
from .quality import FlatFaces


def slice_segments(points, faces, axis, value):
    ff = FlatFaces(faces)
    s = points[:, axis] - value
    # nudge vertices lying exactly on the plane
    s = np.where(np.abs(s) < 1e-12, 1e-12, s)
    a, b = ff.v, ff.v[ff.nxt]
    sa, sb = s[a], s[b]
    cr = (sa > 0) != (sb > 0)
    t = sa[cr] / (sa[cr] - sb[cr])
    p = points[a[cr]] + t[:, None] * (points[b[cr]] - points[a[cr]])
    fid = ff.fid[cr]
    o = np.argsort(fid, kind="stable")
    fid, p = fid[o], p[o]
    u, st, cnt = np.unique(fid, return_index=True, return_counts=True)
    segs, sfid = [], []
    for k in range(0, cnt.max() if len(cnt) else 0, 2):
        m = cnt > k + 1
        segs.append(np.stack([p[st[m] + k], p[st[m] + k + 1]], 1)); sfid.append(u[m])
    if not segs:
        return np.zeros((0, 2, 3)), np.zeros(0, int)
    return np.concatenate(segs), np.concatenate(sfid)


def plot_slice(ax, R, axis, value, title="", interface_names=None, lw=0.25):
    from matplotlib.collections import LineCollection
    pts, faces = R["points"], R["faces"]
    nI = len(R["neighbour"])
    seg, fid = slice_segments(pts, faces, axis, value)
    keep = [i for i in range(3) if i != axis]
    s2 = seg[:, :, keep]
    nF = len(faces[0]) if isinstance(faces, tuple) else len(faces)
    iface = np.zeros(nF, bool)
    for name, (fl, _) in R["face_zones"].items():
        iface[fl] = True
    bnd = fid >= nI
    itf = iface[fid]
    normal = ~bnd & ~itf
    ax.add_collection(LineCollection(s2[normal], colors="#6b7280", linewidths=lw))
    ax.add_collection(LineCollection(s2[bnd], colors="#111827", linewidths=1.2))
    ax.add_collection(LineCollection(s2[itf], colors="#dc2626", linewidths=1.2))
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("xyz"[keep[0]] + " [m]"); ax.set_ylabel("xyz"[keep[1]] + " [m]")
