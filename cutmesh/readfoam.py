"""Minimal ASCII polyMesh reader (for meshes written by cutmesh, or ASCII OpenFOAM meshes)."""
import os
import re
import numpy as np


def _body(path):
    t = open(path).read()
    i = t.find("// * *")
    t = t[t.index("\n", i) + 1:] if i >= 0 else t
    m = re.search(r"^\s*(\d+)\s*\n\s*\(", t, re.M)
    return int(m.group(1)), t[m.end():t.rindex(")")]


def _strip_header(path):
    t = open(path).read()
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    t = re.sub(r"FoamFile\s*\{.*?\}", "", t, count=1, flags=re.S)
    return t


def read_polymesh(case):
    d = os.path.join(case, "constant", "polyMesh")
    n, b = _body(os.path.join(d, "points"))
    pts = np.array(b.replace("(", " ").replace(")", " ").split(), float).reshape(n, 3)
    n, b = _body(os.path.join(d, "faces"))
    tok = b.replace("(", " ").replace(")", " ").split()
    arr = np.array(tok, np.int64)
    sizes = []
    verts = []
    i = 0
    for _ in range(n):
        k = arr[i]; sizes.append(k); verts.append(arr[i + 1:i + 1 + k]); i += 1 + k
    faces = (np.array(sizes), np.concatenate(verts))
    _, b = _body(os.path.join(d, "owner")); owner = np.array(b.split(), np.int64)
    _, b = _body(os.path.join(d, "neighbour")); nei = np.array(b.split(), np.int64)
    return pts, faces, owner, nei


def read_boundary(case):
    """[dict(name, type, nFaces, startFace)]"""
    t = _strip_header(os.path.join(case, "constant", "polyMesh", "boundary"))
    out = []
    for m in re.finditer(r"(\w[\w.:-]*)\s*\{([^{}]*)\}", t):
        body = m.group(2)
        g = lambda k: re.search(rf"\b{k}\s+([^;]+);", body).group(1).strip()
        out.append(dict(name=m.group(1), type=g("type"), nFaces=int(g("nFaces")),
                        startFace=int(g("startFace"))))
    return out


def _lists(body, key, conv):
    m = re.search(rf"{key}\s+List<\w+>\s*(\d+)\s*\(([^)]*)\)", body)
    if not m:
        m2 = re.search(rf"{key}\s+List<\w+>\s*(\d+)\s*\{{\s*(\w+)\s*\}}", body)   # uniform list
        if m2:
            return np.full(int(m2.group(1)), conv(m2.group(2)))
        return np.zeros(0, np.int64)
    return np.array([conv(x) for x in m.group(2).split()])


def read_zones(case):
    """(cell_zones {name: cells}, face_zones {name: (faces, flip)}) - empty if absent"""
    d = os.path.join(case, "constant", "polyMesh")
    cz, fz = {}, {}
    b = lambda x: x in ("1", "true", "on", "yes")
    p = os.path.join(d, "cellZones")
    if os.path.isfile(p):
        t = _strip_header(p)
        for m in re.finditer(r"(\w[\w.:-]*)\s*\{([^{}]*)\}", t):
            cz[m.group(1)] = _lists(m.group(2), "cellLabels", int).astype(np.int64)
    p = os.path.join(d, "faceZones")
    if os.path.isfile(p):
        t = _strip_header(p)
        for m in re.finditer(r"(\w[\w.:-]*)\s*\{([^{}]*)\}", t):
            fl = _lists(m.group(2), "faceLabels", int).astype(np.int64)
            fp = _lists(m.group(2), "flipMap", b).astype(bool)
            if len(fp) != len(fl):
                fp = np.zeros(len(fl), bool)
            fz[m.group(1)] = (fl, fp)
    return cz, fz


def read_mesh(case):
    """everything as the dict used by cutmesh (points, faces, owner, neighbour, patches,
    cell_zones, face_zones)"""
    pts, faces, owner, nei = read_polymesh(case)
    cz, fz = read_zones(case)
    return dict(points=pts, faces=faces, owner=owner, neighbour=nei, patches=read_boundary(case),
                cell_zones=cz, face_zones=fz)
