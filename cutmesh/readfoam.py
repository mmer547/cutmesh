"""Minimal ASCII polyMesh reader (for checking meshes written by cutmesh)."""
import os
import re
import numpy as np


def _body(path):
    t = open(path).read()
    i = t.find("// * *")
    t = t[t.index("\n", i) + 1:] if i >= 0 else t
    m = re.search(r"^\s*(\d+)\s*\n\s*\(", t, re.M)
    return int(m.group(1)), t[m.end():t.rindex(")")]


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
