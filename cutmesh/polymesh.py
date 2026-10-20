"""Write an OpenFOAM polyMesh (ASCII)."""
import os
import numpy as np

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           | written by cutmesh
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
{note}    location    "constant/polyMesh";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def _open(d, obj, cls, note=""):
    f = open(os.path.join(d, obj), "w")
    f.write(HEADER.format(cls=cls, obj=obj, note=(f'    note        "{note}";\n' if note else "")))
    return f


def _label_list(f, arr, per_line=10):
    arr = np.asarray(arr, np.int64)
    f.write(f"{len(arr)}\n(\n")
    for i in range(0, len(arr), per_line):
        f.write(" ".join(map(str, arr[i:i + per_line].tolist())) + "\n")
    f.write(")\n")


def write_polymesh(case_dir, points, faces, owner, neighbour, patches,
                   cell_zones=None, face_zones=None):
    """faces: list of vertex-id lists (internal faces first, then patches in order)
    patches: list of dict(name, type, startFace, nFaces)"""
    d = os.path.join(case_dir, "constant", "polyMesh")
    os.makedirs(d, exist_ok=True)
    for fn in ("cellZones", "faceZones", "pointZones", "sets"):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            os.remove(p)
    nC = int(max(owner.max(), neighbour.max() if len(neighbour) else 0)) + 1
    nF = len(faces[0]) if isinstance(faces, tuple) else len(faces)
    note = f"nPoints:{len(points)}  nCells:{nC}  nFaces:{nF}  nInternalFaces:{len(neighbour)}"

    with _open(d, "points", "vectorField") as f:
        f.write(f"{len(points)}\n(\n")
        f.write("\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in points.tolist()))
        f.write("\n)\n")
    if isinstance(faces, tuple):          # CSR (sizes, flat vertex ids)
        sizes = np.asarray(faces[0]); flat = np.asarray(faces[1])
        ptr = np.concatenate([[0], np.cumsum(sizes)])
        flat_s = flat.astype(str)
        nfaces = len(sizes)
    else:
        nfaces = len(faces)
    with _open(d, "faces", "faceList") as f:
        f.write(f"{nfaces}\n(\n")
        if isinstance(faces, tuple):
            CH = 200000
            for s0 in range(0, nfaces, CH):
                s1 = min(nfaces, s0 + CH)
                f.write("\n".join(f"{sizes[i]}(" + " ".join(flat_s[ptr[i]:ptr[i + 1]]) + ")"
                                   for i in range(s0, s1)))
                f.write("\n")
        else:
            f.write("\n".join(f"{len(p)}(" + " ".join(map(str, p)) + ")" for p in faces))
            f.write("\n")
        f.write(")\n")
    with _open(d, "owner", "labelList", note) as f:
        _label_list(f, owner)
    with _open(d, "neighbour", "labelList", note) as f:
        _label_list(f, neighbour)
    with _open(d, "boundary", "polyBoundaryMesh") as f:
        f.write(f"{len(patches)}\n(\n")
        for p in patches:
            f.write(f"    {p['name']}\n    {{\n        type            {p['type']};\n")
            if p["type"] == "wall":
                f.write("        inGroups        List<word> 1(wall);\n")
            f.write(f"        nFaces          {p['nFaces']};\n        startFace       {p['startFace']};\n    }}\n")
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
