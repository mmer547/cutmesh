"""Measure how well sharp edges are captured by the mesh.

For points sampled along the STL feature edges and along the rims of cylindrical
rotating zones, report the distance to the nearest mesh edge on the boundary (STL)
or on the zone interface faceZone.  A captured edge gives ~0; a chamfered one
gives a fraction of the local cell size.

python tools/edge_check.py meshDict.json [-case DIR]
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cutmesh.mesher import load_config, generate, _rel                      # noqa: E402
from cutmesh.geometry import Surface, Cylinder, read_stl, sample_segments, cylinder_features  # noqa: E402


def edge_samples(P, nn, vv, faces, spacing):
    ptr = np.concatenate([[0], np.cumsum(nn)])
    a, b = [], []
    for f in faces:
        p = vv[ptr[f]:ptr[f + 1]]
        a.append(p); b.append(np.roll(p, -1))
    a = np.concatenate(a); b = np.concatenate(b)
    e = np.unique(np.sort(np.stack([a, b], 1), 1), axis=0)
    return sample_segments(P[e], spacing)


def report(label, d, h, pts=None, worst=5):
    r = d / h
    print(f"{label}: {len(d)} samples, distance / hmin: mean {r.mean():.3f}  p95 {np.percentile(r, 95):.3f}"
          f"  max {r.max():.3f}   >0.05: {100 * (r > 0.05).mean():.1f}%   >0.2: {100 * (r > 0.2).mean():.1f}%")
    if pts is not None and worst:
        for i in np.argsort(-d)[:worst]:
            print(f"    {d[i] * 1e3:7.3f} mm at ({pts[i][0]:+.4f} {pts[i][1]:+.4f} {pts[i][2]:+.4f})")
    return r


def main(argv):
    case = None
    if "-case" in argv:
        i = argv.index("-case"); case = argv[i + 1]; del argv[i:i + 2]
    if not argv:
        print(__doc__); return
    cfgp = argv[0]
    cfg = load_config(cfgp)
    R = generate(cfg, log=lambda s: None)
    if case:
        from cutmesh.polymesh import write_polymesh
        write_polymesh(case, R["points"], R["faces"], R["owner"], R["neighbour"], R["patches"],
                       R["cell_zones"], R["face_zones"])
    P = R["points"]; nn, vv = R["faces"]
    nI = len(R["neighbour"])
    nF = len(nn)
    h0 = float(cfg["baseCellSize"])
    lv = [int(v) for v in cfg.get("surfaceRefinement", {}).values()]
    lv += [int(r["level"]) for r in cfg.get("refinementRegions", [])]
    lv += [max(z.get("surfaceLevel", 0), z.get("volumeLevel", 0)) for z in cfg["rotatingZones"]]
    if cfg.get("featureLevel") is not None:
        lv.append(int(cfg["featureLevel"]))
    hmin = h0 / 2 ** max(lv + [0])
    nC = int(max(R["owner"].max(), R["neighbour"].max())) + 1
    print(f"{nC} cells, hmin {hmin * 1e3:g} mm")

    dom = Surface(read_stl(_rel(cfg, cfg["domain"])))
    dom.build_search(2 * hmin)
    seg = dom.feature_edges(cfg.get("featureAngle", 30.0))
    fs = sample_segments(seg, hmin / 4)
    be = edge_samples(P, nn, vv, range(nI, nF), hmin / 16)
    d, _ = cKDTree(be).query(fs)
    report("STL feature edges", d, hmin, fs)

    for z in cfg["rotatingZones"]:
        if "cylinder" not in z:
            continue
        c = z["cylinder"]
        cyl = Cylinder(c["point1"], c["point2"], c["radius"])
        rs = sample_segments(cylinder_features(cyl, hmin / 4), hmin / 4)
        # rim points inside the fluid, away from the STL (where the rim is a real edge)
        sd = dom.sdf(rs)
        rs = rs[sd < -0.5 * hmin]
        fz, _ = R["face_zones"][z["name"] + "_interface"]
        ze = edge_samples(P, nn, vv, fz, hmin / 16)
        d, _ = cKDTree(ze).query(rs)
        report(f"zone '{z['name']}' rims", d, hmin, rs)


if __name__ == "__main__":
    main(sys.argv[1:])
