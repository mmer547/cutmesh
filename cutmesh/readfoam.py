"""Minimal polyMesh reader: meshes written by cutmesh (binary or ASCII) and OpenFOAM meshes
(ASCII, or binary with the usual faceCompactList faces and contiguous label lists)."""
import os
import re
import numpy as np


class FoamFileError(ValueError):
    pass


def _skip(b, i):
    """Index of the next token after whitespace and comments."""
    n = len(b)
    while i < n:
        c = b[i:i + 1]
        if c.isspace():
            i += 1
        elif b.startswith(b"//", i):
            j = b.find(b"\n", i)
            i = n if j < 0 else j + 1
        elif b.startswith(b"/*", i):
            j = b.find(b"*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            break
    return i


def _read_file(path):
    """(header dict, raw bytes, index just after the FoamFile header)"""
    with open(path, "rb") as f:
        b = f.read()
    i = _skip(b, 0)
    hdr = {}
    if b.startswith(b"FoamFile", i):
        j = b.index(b"{", i)
        k = b.index(b"}", j)
        for m in re.finditer(rb"(\w+)\s+([^;]*);", b[j + 1:k]):
            hdr[m.group(1).decode()] = m.group(2).decode().strip().strip('"')
        i = k + 1
    return hdr, b, i


def _dtypes(hdr):
    """(label dtype, scalar dtype) of a binary file from its arch entry."""
    arch = hdr.get("arch", "LSB;label=32;scalar=64")
    end = ">" if "MSB" in arch else "<"
    m = re.search(r"label=(\d+)", arch)
    lab = f"{end}i{int(m.group(1)) // 8 if m else 4}"
    m = re.search(r"scalar=(\d+)", arch)
    sca = f"{end}f{int(m.group(1)) // 8 if m else 8}"
    return np.dtype(lab), np.dtype(sca)


def _int(b, i):
    i = _skip(b, i)
    m = re.compile(rb"[+-]?\d+").match(b, i)
    if not m:
        raise FoamFileError(f"expected a list size at byte {i}: {b[i:i + 20]!r}")
    return int(m.group(0)), m.end()


def _expect(b, i, ch, what):
    i = _skip(b, i)
    if b[i:i + 1] != ch:
        raise FoamFileError(f"expected {ch!r} ({what}) at byte {i}: {b[i:i + 20]!r}")
    return i + 1


def _raw(b, i, n, dtype, what):
    """Binary list body: `(` raw `)`; nothing at all when n == 0."""
    if n == 0:
        return np.zeros(0, dtype), i
    i = _expect(b, i, b"(", what)
    nb = n * dtype.itemsize
    if i + nb > len(b):
        raise FoamFileError(f"{what}: file ends inside a binary list of {n} items")
    a = np.frombuffer(b, dtype, n, i)
    i += nb
    if b[i:i + 1] != b")":
        raise FoamFileError(f"{what}: binary list of {n} items is not closed by ')'"
                            " (wrong label or scalar size in arch?)")
    return a, i + 1


def _list(b, i, binary, dtype, what, width=1):
    """One list of numbers (width per item, e.g. 3 for vectors) -> (array, index after it)."""
    n, i = _int(b, i)
    if binary:
        a, i = _raw(b, i, n * width, dtype, what)
    else:
        if n == 0:
            j = _skip(b, i)
            if b[j:j + 1] == b"(":
                i = b.index(b")", j) + 1
            return np.zeros(0, dtype), i
        i = _expect(b, i, b"(", what)
        # "(x y z)" items: the list ends at the last ')' of the file
        j = b.rfind(b")") if width > 1 else b.index(b")", i)
        a = np.array(b[i:j].replace(b"(", b" ").replace(b")", b" ").split(), dtype)
        i = j + 1
    a = np.asarray(a)
    if len(a) != n * width:
        raise FoamFileError(f"{what}: {len(a)} values for a list of {n} items")
    return (a.reshape(n, width) if width > 1 else a), i


def read_polymesh(case, with_format=False):
    d = os.path.join(case, "constant", "polyMesh")
    hdr, b, i = _read_file(os.path.join(d, "points"))
    binary = hdr.get("format") == "binary"
    lab, sca = _dtypes(hdr)
    pts, _ = _list(b, i, binary, sca if binary else np.dtype(float), "points", 3)
    pts = pts.astype(float)

    hdr, b, i = _read_file(os.path.join(d, "faces"))
    fbin = hdr.get("format") == "binary"
    lab, _ = _dtypes(hdr)
    if hdr.get("class") == "faceCompactList":
        off, i = _list(b, i, fbin, lab if fbin else np.dtype(np.int64), "faces offsets")
        flat, i = _list(b, i, fbin, lab if fbin else np.dtype(np.int64), "faces vertices")
        off = off.astype(np.int64)
        faces = (np.diff(off), flat.astype(np.int64))
    elif fbin:
        raise FoamFileError(f"{d}/faces: binary faceList is not supported"
                            " (write the mesh in ascii, or as faceCompactList)")
    else:
        n, i = _int(b, i)
        i = _expect(b, i, b"(", "faces")
        j = b.rindex(b")")
        arr = np.array(b[i:j].replace(b"(", b" ").replace(b")", b" ").split(), np.int64)
        sizes = np.empty(n, np.int64)
        starts = np.empty(n, np.int64)
        k = 0
        for f in range(n):                 # "<n>(v0 v1 ...)": sizes are interleaved
            sizes[f] = arr[k]; starts[f] = k + 1; k += 1 + arr[k]
        if k != len(arr):
            raise FoamFileError(f"{d}/faces: {len(arr)} numbers do not make {n} faces")
        idx = np.repeat(starts - np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes) + \
            np.arange(int(sizes.sum()))
        faces = (sizes, arr[idx])

    out = []
    for obj in ("owner", "neighbour"):
        hdr, b, i = _read_file(os.path.join(d, obj))
        obin = hdr.get("format") == "binary"
        lab, _ = _dtypes(hdr)
        a, _ = _list(b, i, obin, lab if obin else np.dtype(np.int64), obj)
        out.append(a.astype(np.int64))
    if with_format:
        return pts, faces, out[0], out[1], ("binary" if binary else "ascii")
    return pts, faces, out[0], out[1]


def _strip_header(path):
    t = open(path).read()
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    t = re.sub(r"FoamFile\s*\{.*?\}", "", t, count=1, flags=re.S)
    return t


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


_WORD = re.compile(rb"[^\s{}();]+")
_TYPES = {b"label": "label", b"bool": "bool", b"scalar": "scalar"}


def _zone_entries(path):
    """[(zone name, {key: value})] of a cellZones / faceZones file, binary or ASCII.
    List<label>/List<bool> values are arrays; other values are the text up to ';'."""
    hdr, b, i = _read_file(path)
    binary = hdr.get("format") == "binary"
    lab, sca = _dtypes(hdr)
    n, i = _int(b, i)
    i = _expect(b, i, b"(", "zone list")
    zones = []
    for _ in range(n):
        i = _skip(b, i)
        m = _WORD.match(b, i)
        name = m.group(0).decode(); i = _expect(b, m.end(), b"{", f"zone {name}")
        ent = {}
        while True:
            i = _skip(b, i)
            if b[i:i + 1] == b"}":
                i += 1; break
            m = _WORD.match(b, i)
            key = m.group(0).decode(); i = _skip(b, m.end())
            m = re.compile(rb"List<(\w+)>").match(b, i)
            if m and m.group(1) in _TYPES:
                t = _TYPES[m.group(1)]
                i = _skip(b, m.end())
                cnt, i = _int(b, i)
                j = _skip(b, i)
                if b[j:j + 1] == b"{":                         # uniform list: N{v}
                    k = b.index(b"}", j)
                    v = b[j + 1:k].strip()
                    val = np.full(cnt, v in (b"1", b"true", b"on", b"yes")) if t == "bool" \
                        else np.full(cnt, float(v) if t == "scalar" else int(v))
                    i = k + 1
                elif binary:
                    dt = {"label": lab, "bool": np.dtype(np.uint8), "scalar": sca}[t]
                    val, i = _raw(b, i, cnt, dt, f"{key} of zone {name}")
                else:
                    if cnt == 0:
                        val = np.zeros(0)
                        if b[j:j + 1] == b"(":
                            i = b.index(b")", j) + 1
                    else:
                        i = _expect(b, i, b"(", key)
                        k = b.index(b")", i)
                        tok = b[i:k].split()
                        val = np.array([x in (b"1", b"true", b"on", b"yes") for x in tok]) \
                            if t == "bool" else np.array(tok, float if t == "scalar" else np.int64)
                        i = k + 1
                ent[key] = np.asarray(val)
                i = _expect(b, i, b";", key)
            else:
                k = b.index(b";", i)
                ent[key] = b[i:k].decode(errors="replace").strip()
                i = k + 1
        zones.append((name, ent))
    return zones


def read_zones(case):
    """(cell_zones {name: cells}, face_zones {name: (faces, flip)}) - empty if absent"""
    d = os.path.join(case, "constant", "polyMesh")
    cz, fz = {}, {}
    p = os.path.join(d, "cellZones")
    if os.path.isfile(p):
        for name, e in _zone_entries(p):
            cz[name] = np.asarray(e.get("cellLabels", np.zeros(0)), np.int64)
    p = os.path.join(d, "faceZones")
    if os.path.isfile(p):
        for name, e in _zone_entries(p):
            fl = np.asarray(e.get("faceLabels", np.zeros(0)), np.int64)
            fp = np.asarray(e.get("flipMap", np.zeros(0)), bool)
            if len(fp) != len(fl):
                fp = np.zeros(len(fl), bool)
            fz[name] = (fl, fp)
    return cz, fz


def read_mesh(case):
    """everything as the dict used by cutmesh (points, faces, owner, neighbour, patches,
    cell_zones, face_zones, format)"""
    pts, faces, owner, nei, fmt = read_polymesh(case, with_format=True)
    cz, fz = read_zones(case)
    return dict(points=pts, faces=faces, owner=owner, neighbour=nei, patches=read_boundary(case),
                cell_zones=cz, face_zones=fz, format=fmt)
