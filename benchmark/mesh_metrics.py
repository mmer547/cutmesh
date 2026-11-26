"""Mesh measures for the quality gate (run_bench.py --gate) that checkMesh does not give.

feature_edge_deviation: how far the STL feature edges are from the mesh's boundary edges
    (the same measure as tools/edge_check.py: samples every hmin / 4 along the feature
    edges, distance to the nearest boundary edge, in units of the smallest cell).
layer_coverage: the share of wall points with full / reduced / no layer thickness, from the
    addLayers line of the log.
"""
import re

import numpy as np
from scipy.spatial import cKDTree

LAYER_RE = re.compile(r"wall points with full thickness ([\d.]+)%, reduced ([\d.]+)%, none ([\d.]+)%")
# "smallest cell 0.001" and, where a cylindrical base lowers it, "smallest cell of the cylindrical base 0.0008"
HMIN_RE = re.compile(r"smallest cell (?:of the cylindrical base )?([0-9.eE+-]+)")


def smallest_cell(lines):
    """the smallest cell size the log reports (the last one: a cylindrical base may lower it)"""
    h = None
    for s in lines:
        m = HMIN_RE.search(s)
        if m:
            h = float(m.group(1))
    return h


def layer_coverage(lines):
    """{"full", "reduced", "none"} in percent of the wall points, or None without layers"""
    out = None
    for s in lines:
        m = LAYER_RE.search(s)
        if m:
            out = {"full": float(m.group(1)), "reduced": float(m.group(2)), "none": float(m.group(3))}
    return out


def boundary_edges(points, faces, nI):
    """unique edges (pairs of point ids) of the boundary faces (faces nI..)"""
    sizes, flat = faces
    sizes = np.asarray(sizes, np.int64)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    s0 = int(ptr[nI])
    v = np.asarray(flat[s0:], np.int64)
    n = sizes[nI:]
    p = ptr[nI:] - s0
    nxt = np.arange(len(v)) + 1
    nxt[p[1:] - 1] = p[:-1]
    a, b = v, v[nxt]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(lo * (len(points) + 1) + hi)
    return np.stack([key // (len(points) + 1), key % (len(points) + 1)], 1)


def _segment_distance(p, a, b):
    ab = b - a
    t = np.einsum("...k,...k->...", p - a, ab) / np.maximum(np.einsum("...k,...k->...", ab, ab), 1e-300)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def feature_edge_deviation(points, faces, nI, segments, hmin, k=24, chunk=200000):
    """distances of samples on the feature segments (every hmin / 4) to the nearest boundary
    edge, divided by hmin.  The nearest edge is searched among the k edges with the nearest
    midpoints.  Returns {"samples", "mean", "p95", "max", "over02"} (over02: the share of
    samples farther than 0.2 hmin), or None without feature edges."""
    from cutmesh.geometry import sample_segments
    if segments is None or len(segments) == 0:
        return None
    fs = sample_segments(np.asarray(segments, float), hmin / 4)
    if len(fs) == 0:
        return None
    if isinstance(faces, tuple):
        sizes, flat = faces
    else:
        sizes = np.array([len(f) for f in faces], np.int64)
        flat = np.fromiter((v for f in faces for v in f), np.int64, count=int(sizes.sum()))
    P = np.asarray(points, float)
    e = boundary_edges(P, (sizes, flat), nI)
    A, B = P[e[:, 0]], P[e[:, 1]]
    tree = cKDTree(0.5 * (A + B))
    k = min(k, len(e))
    d = np.empty(len(fs))
    for s in range(0, len(fs), chunk):
        q = fs[s:s + chunk]
        _, j = tree.query(q, k=k, workers=-1)
        j = j.reshape(len(q), -1)
        d[s:s + chunk] = _segment_distance(q[:, None, :], A[j], B[j]).min(1)
    r = d / hmin
    return {"samples": int(len(r)), "mean": float(r.mean()), "p95": float(np.percentile(r, 95)),
            "max": float(r.max()), "over02": float((r > 0.2).mean())}


def domain_feature_segments(meshdict_path):
    """the domain STL's feature edges with the meshDict's featureAngle"""
    from cutmesh.mesher import load_config, _rel
    from cutmesh.geometry import Surface, read_stl
    cfg = load_config(meshdict_path)
    dom = Surface(read_stl(_rel(cfg, cfg["domain"])), log=lambda s: None, what="domain STL")
    return dom.feature_edges(cfg["featureAngle"])
