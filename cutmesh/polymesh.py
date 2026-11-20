"""Write an OpenFOAM polyMesh (binary or ASCII).

Binary follows OpenFOAM's own layout: the FoamFile header says `format binary;` and
`arch "LSB;label=32;scalar=64";`, a list is `N` then `(` + raw little-endian data + `)`
(an empty list is just `0`), and faces are a `faceCompactList` (the N+1 offsets, then
the vertex ids).  points, faces, owner and neighbour are binary; boundary, zones and
MRFProperties are short text and stay ASCII (every file says its own format, so OpenFOAM
reads the mix).
"""
import os
import numpy as np

FORMATS = ("binary", "ascii")

# a neutral comment instead of the OpenFOAM banner: the files are not written by OpenFOAM
# (OPENFOAM is a trademark of OpenCFD Ltd); the FoamFile dictionary is all a reader needs
HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  written by cutmesh
\\*---------------------------------------------------------------------------*/
FoamFile
{{
{version}    format      {fmt};
{arch}    class       {cls};
{note}    location    "constant/polyMesh";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def _header(obj, cls, note="", fmt="ascii", label_bits=32):
    binary = fmt == "binary"
    return HEADER.format(
        cls=cls, obj=obj, fmt=fmt,
        version="    version     2.0;\n" if binary else "",
        arch=f'    arch        "LSB;label={label_bits};scalar=64";\n' if binary else "",
        note=(f'    note        "{note}";\n' if note else ""))


def _open(d, obj, cls, note=""):
    f = open(os.path.join(d, obj), "w")
    f.write(_header(obj, cls, note))
    return f


def _label_list(f, arr, per_line=10):
    arr = np.asarray(arr, np.int64)
    f.write(f"{len(arr)}\n(\n")
    for i in range(0, len(arr), per_line):
        f.write(" ".join(map(str, arr[i:i + per_line].tolist())) + "\n")
    f.write(")\n")


def _raw_list(f, arr, dtype):
    """One binary list: `N\\n(` raw `)\\n` (OpenFOAM writes no brackets for an empty list)."""
    arr = np.ascontiguousarray(arr, dtype=dtype)
    n = len(arr)
    f.write(f"{n}\n".encode())
    if n:
        f.write(b"(")
        f.write(arr.tobytes())
        f.write(b")\n")


def _write_binary(d, points, sizes, flat, owner, neighbour, note, label_bits=None):
    """label_bits: 32 or 64 (None: 64 only when a label does not fit in 32 bits)"""
    if label_bits is None:
        big = max(len(flat), len(sizes) + 1, len(points)) >= 2 ** 31 - 1
        label_bits = 64 if big else 32
    bits = label_bits
    lab = "<i8" if bits == 64 else "<i4"

    def bopen(obj, cls, note=""):
        f = open(os.path.join(d, obj), "wb")
        f.write(_header(obj, cls, note, "binary", bits).encode())
        return f
    with bopen("points", "vectorField") as f:
        _raw_list(f, np.asarray(points, np.float64).reshape(-1, 3), "<f8")
    with bopen("faces", "faceCompactList") as f:
        offsets = np.zeros(len(sizes) + 1, np.int64)
        np.cumsum(sizes, out=offsets[1:])
        _raw_list(f, offsets, lab)
        f.write(b"\n")
        _raw_list(f, flat, lab)
    with bopen("owner", "labelList", note) as f:
        _raw_list(f, owner, lab)
    with bopen("neighbour", "labelList", note) as f:
        _raw_list(f, neighbour, lab)


def _write_ascii(d, points, faces, nfaces, owner, neighbour, note):
    with _open(d, "points", "vectorField") as f:
        f.write(f"{len(points)}\n(\n")
        f.write("\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in points.tolist()))
        f.write("\n)\n")
    with _open(d, "faces", "faceList") as f:
        f.write(f"{nfaces}\n(\n")
        if isinstance(faces, tuple):
            sizes = np.asarray(faces[0]); flat_s = np.asarray(faces[1]).astype(str)
            ptr = np.concatenate([[0], np.cumsum(sizes)])
            # one token per vertex: "<n>(" before a face's first vertex, then the vertex,
            # then " " or ")\n" after its last one (joined in chunks of faces)
            CH = 200000
            for s0 in range(0, nfaces, CH):
                s1 = min(nfaces, s0 + CH)
                a0, a1 = int(ptr[s0]), int(ptr[s1])
                tok = np.empty(3 * (a1 - a0), dtype=object)
                tok[0::3] = ""
                tok[1::3] = flat_s[a0:a1]
                tok[2::3] = " "
                tok[3 * (ptr[s0:s1] - a0)] = [f"{n}(" for n in sizes[s0:s1].tolist()]
                tok[3 * (ptr[s0 + 1:s1 + 1] - a0) - 1] = ")\n"
                f.write("".join(tok.tolist()))
        else:
            f.write("\n".join(f"{len(p)}(" + " ".join(map(str, p)) + ")" for p in faces))
            f.write("\n")
        f.write(")\n")
    with _open(d, "owner", "labelList", note) as f:
        _label_list(f, owner)
    with _open(d, "neighbour", "labelList", note) as f:
        _label_list(f, neighbour)


def write_polymesh(case_dir, points, faces, owner, neighbour, patches,
                   cell_zones=None, face_zones=None, fmt="binary"):
    """faces: list of vertex-id lists or CSR (sizes, flat vertex ids); internal faces first,
    then patches in order.  patches: list of dict(name, type, startFace, nFaces).
    fmt: "binary" (default) or "ascii" for points / faces / owner / neighbour."""
    if fmt not in FORMATS:
        raise ValueError(f"writeFormat must be one of {FORMATS}, got {fmt!r}")
    d = os.path.join(case_dir, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    for fn in ("cellZones", "faceZones", "pointZones", "sets"):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            os.remove(p)
    owner = np.asarray(owner); neighbour = np.asarray(neighbour)
    points = np.asarray(points, np.float64)
    nC = int(max(owner.max(), neighbour.max() if len(neighbour) else 0)) + 1
    nF = len(faces[0]) if isinstance(faces, tuple) else len(faces)
    note = f"nPoints:{len(points)}  nCells:{nC}  nFaces:{nF}  nInternalFaces:{len(neighbour)}"

    if fmt == "binary":
        if isinstance(faces, tuple):
            sizes, flat = np.asarray(faces[0]), np.asarray(faces[1])
        else:
            sizes = np.array([len(p) for p in faces], np.int64)
            flat = np.concatenate([np.asarray(p, np.int64) for p in faces]) if nF else \
                np.zeros(0, np.int64)
        _write_binary(d, points, sizes, flat, owner, neighbour, note)
    else:
        _write_ascii(d, points, faces, nF, owner, neighbour, note)
    with _open(d, "boundary", "polyBoundaryMesh") as f:
        f.write(f"{len(patches)}\n(\n")
        for p in patches:
            f.write(f"    {p['name']}\n    {{\n        type            {p['type']};\n")
            if p["type"] in ("wall", "cyclicAMI"):
                f.write(f"        inGroups        List<word> 1({p['type']});\n")
            f.write(f"        nFaces          {p['nFaces']};\n        startFace       {p['startFace']};\n")
            for k, v in (p.get("extra") or {}).items():      # e.g. neighbourPatch of a cyclicAMI
                f.write(f"        {k:<15} {v};\n")
            f.write("    }\n")
        f.write(")\n")
    if cell_zones:
        with _open(d, "cellZones", "regIOobject") as f:
            f.write(f"{len(cell_zones)}\n(\n")
            for name, cells in cell_zones.items():
                f.write(f"{name}\n{{\n    type cellZone;\ncellLabels      List<label> ")
                _label_list(f, cells)
                f.write(";\n}\n")
            f.write(")\n")
    if face_zones:
        with _open(d, "faceZones", "regIOobject") as f:
            f.write(f"{len(face_zones)}\n(\n")
            for name, (fl, flip) in face_zones.items():
                f.write(f"{name}\n{{\n    type faceZone;\nfaceLabels      List<label> ")
                _label_list(f, fl)
                f.write(";\nflipMap         List<bool> ")
                _label_list(f, np.asarray(flip, int), per_line=40)
                f.write(";\n}\n")
            f.write(")\n")
