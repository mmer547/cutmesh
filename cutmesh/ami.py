"""Sliding-mesh interfaces: a rotating zone's interface faceZone split into two coupled patches.

The interface of a conforming rotating zone is a set of internal faces (faceZone
<name>_interface).  For a sliding mesh each of these faces becomes two boundary faces, one
owned by the cell inside the zone and one by the cell outside, and the points of the
interface are duplicated so that the zone can rotate without dragging the outside cells
(what createBaffles + mergeOrSplitBaffles -split / splitBaffles do in OpenFOAM).

Two OpenFOAM families write this differently (checked against their tutorials):
    "com"   openfoam.com (ESI, v2xxx): patches <name>_AMI1 (zone side) / <name>_AMI2 of
            type cyclicAMI (tutorials/multiphase/interFoam/RAS/mixerVesselAMI)
    "org10", "org11", "org12"   openfoam.org (Foundation): patches <name>_nonCouple1 (zone
            side) / <name>_nonCouple2 of type patch; couple them after reading the mesh with
            createNonConformalCouples -overwrite <name>_nonCouple1 <name>_nonCouple2
            (tutorials/incompressibleFluid/mixerVessel2D)
constant/dynamicMeshDict (solid-body rotation of the zones' cells) follows the same choice.

    python -m cutmesh.ami CASE [-zone NAME ...] [-openfoam com|org10|org11|org12]
                               [-out CASE2] [-noDynamicMeshDict] [-ascii | -binary]
converts an existing mesh with the cellZone NAME and the faceZone NAME_interface (written
by cutmesh, or any mesh with that pair).  The rotation for dynamicMeshDict is read from
constant/MRFProperties (MRF_NAME) when there is one.
"""
import os
import re

import numpy as np

STYLES = ("com", "org10", "org11", "org12")


def patch_names(name, style):
    if style == "com":
        return f"{name}_AMI1", f"{name}_AMI2"
    return f"{name}_nonCouple1", f"{name}_nonCouple2"


def patch_entries(name, style):
    """the two new patches (zone side first) as boundary dicts, without faces"""
    a, b = patch_names(name, style)
    if style == "com":
        extra = lambda nb: {"matchTolerance": "0.0001", "neighbourPatch": nb, "transform": "noOrdering"}
        return [dict(name=a, type="cyclicAMI", extra=extra(b)),
                dict(name=b, type="cyclicAMI", extra=extra(a))]
    return [dict(name=a, type="patch"), dict(name=b, type="patch")]


def _gather(sizes, flat, ptr, ids, reverse):
    """CSR of the faces ids (vertex order reversed where reverse)"""
    ids = np.asarray(ids, np.int64)
    n = sizes[ids]
    start = ptr[ids]
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    k = np.arange(int(n.sum())) - np.repeat(off, n)                 # position within the face
    k = np.where(np.repeat(reverse, n), np.repeat(n, n) - 1 - k, k)
    return n, flat[np.repeat(start, n) + k]


def split_interface(mesh, name, style="com"):
    """Split the faceZone <name>_interface of the cellZone <name> into two patches and
    duplicate its points on the zone side.  mesh: dict(points, faces (sizes, flat), owner,
    neighbour, patches, cell_zones, face_zones); returns a new dict (the faceZone dropped)."""
    if style not in STYLES:
        raise ValueError(f"openfoam must be one of {STYLES}, got {style!r}")
    P = np.asarray(mesh["points"], float)
    sizes, flat = (np.asarray(a, np.int64) for a in mesh["faces"])
    own = np.asarray(mesh["owner"], np.int64)
    nei = np.asarray(mesh["neighbour"], np.int64)
    nI, nF = len(nei), len(sizes)
    nC = int(max(own.max(), nei.max() if nI else 0)) + 1
    fzname = name + "_interface"
    if name not in mesh["cell_zones"] or fzname not in mesh["face_zones"]:
        raise ValueError(f"zone {name!r}: needs the cellZone {name!r} and the faceZone {fzname!r}")
    inz = np.zeros(nC, bool)
    inz[np.asarray(mesh["cell_zones"][name], np.int64)] = True
    fz = np.unique(np.asarray(mesh["face_zones"][fzname][0], np.int64))
    if len(fz) == 0:
        raise ValueError(f"zone {name!r}: the faceZone {fzname!r} is empty")
    if fz.max() >= nI:
        raise ValueError(f"zone {name!r}: the faceZone {fzname!r} holds boundary faces"
                         " (already split?)")
    if np.any(inz[own[fz]] == inz[nei[fz]]):
        raise ValueError(f"zone {name!r}: faces of {fzname!r} do not separate the cellZone"
                         " from the rest")
    is_if = np.zeros(nF, bool); is_if[fz] = True
    other = np.nonzero(~is_if[:nI] & (inz[own[:nI]] != inz[nei]))[0]
    if len(other):
        raise ValueError(f"zone {name!r}: {len(other)} internal faces between the zone and the rest"
                         f" are not in {fzname!r}")
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    # points of the interface: the zone side gets copies
    ip = np.unique(flat[np.repeat(is_if, sizes)])
    copy = np.full(len(P), -1, np.int64)
    copy[ip] = len(P) + np.arange(len(ip))
    zone_side = inz[own].copy()                    # faces of zone cells (internal: both cells)
    zone_side[fz] = False                          # interface faces: handled below
    fid = np.repeat(np.arange(nF), sizes)
    flat2 = flat.copy()
    m = zone_side[fid] & (copy[flat] >= 0)
    flat2[m] = copy[flat[m]]
    # new face order: kept internal faces, old patches, zone-side copies, outside copies
    keep_i = np.nonzero(~is_if[:nI])[0]
    o_in = inz[own[fz]]                            # interface face owned by the zone cell
    zcell = np.where(o_in, own[fz], nei[fz])
    ocell = np.where(o_in, nei[fz], own[fz])
    parts = [
        (keep_i, np.zeros(len(keep_i), bool), flat2),
        (np.arange(nI, nF), np.zeros(nF - nI, bool), flat2),
    ]
    n1, f1 = _gather(sizes, flat, ptr, fz, ~o_in)  # zone side: normal out of the zone cell
    m1 = copy[f1] >= 0
    f1 = np.where(m1, copy[f1], f1)
    n2, f2 = _gather(sizes, flat, ptr, fz, o_in)   # outside: normal out of the outside cell
    S, F = [], []
    for ids, rev, fl in parts:
        n_, f_ = _gather(sizes, fl, ptr, ids, rev)
        S.append(n_); F.append(f_)
    S += [n1, n2]; F += [f1, f2]
    owner = np.concatenate([own[keep_i], own[nI:], zcell, ocell])
    patches = []
    start = len(keep_i)
    for p in mesh["patches"]:
        q = dict(p); q["startFace"] = start; start += int(p["nFaces"])
        patches.append(q)
    for pe in patch_entries(name, style):
        pe.update(startFace=start, nFaces=len(fz)); start += len(fz)
        patches.append(pe)
    names = [p["name"] for p in patches]
    if len(set(names)) != len(names):
        raise ValueError(f"zone {name!r}: patch names {patch_names(name, style)} are taken")
    fzs = {k: v for k, v in mesh["face_zones"].items() if k != fzname}
    # faces of the other faceZones: internal ids move (deleted interface faces before them),
    # boundary ids keep their order
    new_id = np.full(nF, -1, np.int64)
    new_id[keep_i] = np.arange(len(keep_i))
    new_id[nI:] = len(keep_i) + np.arange(nF - nI)
    for k, (fl, flip) in list(fzs.items()):
        fl = np.asarray(fl, np.int64)
        ok = new_id[fl] >= 0
        fzs[k] = (new_id[fl[ok]], np.asarray(flip, bool)[ok])
    out = dict(mesh)
    out.update(points=np.concatenate([P, P[ip]]), faces=(np.concatenate(S), np.concatenate(F)),
               owner=owner, neighbour=nei[keep_i], patches=patches, face_zones=fzs)
    out["ami_split"] = dict(out.get("ami_split", {}), **{name: dict(faces=len(fz), points=len(ip))})
    return out


def axisymmetry_error(mesh, name, geom, origin, axis, angle_deg=7.0, n_max=5000):
    """Largest change of the distance from the zone surface of the interface face centres
    when turned by angle_deg about the axis: about 0 for a surface of revolution about that
    axis (the zone can slide on itself), large otherwise (a wrong axis, a non-round zone).
    The change, not the distance: faces where the cut chamfers the zone's rim are off the
    surface before turning too."""
    sizes, flat = mesh["faces"]
    fl = np.asarray(mesh["face_zones"][name + "_interface"][0], np.int64)
    if len(fl) > n_max:
        fl = fl[np.linspace(0, len(fl) - 1, n_max).astype(np.int64)]
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    P = np.asarray(mesh["points"], float)
    C = np.array([P[flat[ptr[f]:ptr[f + 1]]].mean(0) for f in fl.tolist()])
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    t = np.radians(angle_deg)
    X = C - np.asarray(origin, float)
    Xr = X * np.cos(t) + np.cross(a, X) * np.sin(t) + np.outer(X @ a, a) * (1 - np.cos(t))
    d0 = np.abs(geom.sdf(C))
    d1 = np.abs(geom.sdf(Xr + np.asarray(origin, float)))
    return float(np.abs(d1 - d0).max())


def zone_points(mesh, name):
    """points used by the cells of the cellZone name"""
    sizes, flat = mesh["faces"]
    own, nei = np.asarray(mesh["owner"]), np.asarray(mesh["neighbour"])
    nC = int(max(own.max(), nei.max() if len(nei) else 0)) + 1
    inz = np.zeros(nC, bool); inz[np.asarray(mesh["cell_zones"][name], np.int64)] = True
    fsel = inz[own].copy()
    fsel[:len(nei)] |= inz[nei]
    return np.unique(np.asarray(flat)[np.repeat(fsel, sizes)])


# ------------------------------------------------------------------ dynamicMeshDict
_HEADER = """FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      dynamicMeshDict;
}}
// generated by cutmesh for {what} - check omega (rad/s), origin and axis

"""


def _v(x):
    return "(" + " ".join(f"{float(c):.9g}" for c in x) + ")"


def dynamic_mesh_dict(zones, style):
    """text of constant/dynamicMeshDict rotating the cellZones of zones
    ([dict(name, origin, axis, omega)]) as solid bodies"""
    if style not in STYLES:
        raise ValueError(f"openfoam must be one of {STYLES}, got {style!r}")
    if style == "com":
        head = _HEADER.format(what="openfoam.com (ESI)")
        head += "dynamicFvMesh   dynamicMotionSolverFvMesh;\n\nmotionSolverLibs (fvMotionSolvers);\n\n"
        if len(zones) == 1:
            z = zones[0]
            return head + (f"motionSolver    solidBody;\n\ncellZone        {z['name']};\n\n"
                           f"solidBodyMotionFunction  rotatingMotion;\n\n"
                           f"origin      {_v(z['origin'])};\naxis        {_v(z['axis'])};\n"
                           f"omega       {float(z['omega']):.9g}; // [rad/s]\n")
        body = "".join(f"""    {z['name']}
    {{
        solidBodyMotionFunction  rotatingMotion;
        rotatingMotionCoeffs
        {{
            origin      {_v(z['origin'])};
            axis        {_v(z['axis'])};
            omega       {float(z['omega']):.9g}; // [rad/s]
        }}
    }}
""" for z in zones)
        return head + "motionSolver    multiSolidBodyMotionSolver;\n\n" \
            f"multiSolidBodyMotionSolverCoeffs\n{{\n{body}}}\n"
    libs = '("libfvMeshMovers.so" "libfvMotionSolvers.so")' if style in ("org10", "org11") \
        else '("libfvMotionSolvers.so")'
    head = _HEADER.format(what=f"openfoam.org (OpenFOAM-{style[3:]})")

    def solid(z, ind):
        s = " " * ind
        return (f"{s}motionSolver    solidBody;\n{s}cellZone        {z['name']};\n"
                f"{s}solidBodyMotionFunction  rotatingMotion;\n"
                f"{s}origin      {_v(z['origin'])};\n{s}axis        {_v(z['axis'])};\n"
                f"{s}omega       {float(z['omega']):.9g}; // [rad/s]\n")
    if len(zones) == 1:
        return head + f"mover\n{{\n    type            motionSolver;\n    libs            {libs};\n\n" \
            + solid(zones[0], 4) + "}\n"
    body = "".join(f"        {z['name']}\n        {{\n" + solid(z, 12) + "        }\n" for z in zones)
    return head + f"mover\n{{\n    type            motionSolver;\n    libs            {libs};\n\n" \
        f"    motionSolver    motionSolverList;\n\n    solvers\n    {{\n{body}    }}\n}}\n"


def write_dynamic_mesh_dict(case_dir, zones, style):
    d = os.path.join(case_dir, "constant")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "dynamicMeshDict")
    with open(p, "w") as f:
        f.write(dynamic_mesh_dict(zones, style))
    return p


def couple_hint(names, style):
    """what to run in OpenFOAM after reading the mesh (openfoam.org), or None"""
    if style == "com":
        return None
    return "; ".join(f"createNonConformalCouples -overwrite {a} {b}"
                     for a, b in (patch_names(n, style) for n in names))


def read_mrf(case_dir):
    """{zone: dict(origin, axis, omega)} from constant/MRFProperties (as cutmesh writes it)"""
    p = os.path.join(case_dir, "constant", "MRFProperties")
    out = {}
    if not os.path.isfile(p):
        return out
    t = re.sub(r"//[^\n]*", "", open(p).read())
    for m in re.finditer(r"(\w+)\s*\{([^{}]*)\}", t):
        b = m.group(2)
        z = re.search(r"cellZone\s+(\S+);", b)
        if not z:
            continue
        g = lambda k: re.search(rf"\b{k}\s+([^;]+);", b)
        vec = lambda k: [float(x) for x in g(k).group(1).strip("() ").split()] if g(k) else None
        om = g("omega")
        out[z.group(1)] = dict(origin=vec("origin") or [0, 0, 0], axis=vec("axis") or [0, 0, 1],
                               omega=float(om.group(1).split()[0]) if om else 0.0)
    return out


def main(argv=None):
    import sys
    from .readfoam import read_mesh
    from .polymesh import write_polymesh
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    zones, style, out, dyn, fmt, case = [], "com", None, True, None, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-zone", "-openfoam", "-out"):
            if i + 1 >= len(argv):
                print(f"cutmesh.ami: {a} needs a value", file=sys.stderr); return 2
            v = argv[i + 1]
            if a == "-zone":
                zones.append(v)
            elif a == "-openfoam":
                style = v
            else:
                out = v
            i += 2; continue
        if a == "-noDynamicMeshDict":
            dyn = False
        elif a in ("-ascii", "-binary"):
            fmt = a[1:]
        elif a.startswith("-"):
            print(f"cutmesh.ami: unknown option {a!r}", file=sys.stderr); return 2
        elif case is None:
            case = a
        else:
            print("cutmesh.ami: one CASE expected", file=sys.stderr); return 2
        i += 1
    if case is None:
        print("cutmesh.ami: CASE required", file=sys.stderr); return 2
    if style not in STYLES:
        print(f"cutmesh.ami: -openfoam must be one of {', '.join(STYLES)}", file=sys.stderr); return 2
    m = read_mesh(case)
    if not zones:
        zones = [k for k in m["cell_zones"] if k + "_interface" in m["face_zones"]]
        if not zones:
            print("ERROR: no cellZone NAME with a faceZone NAME_interface", file=sys.stderr); return 1
    try:
        for z in zones:
            m = split_interface(m, z, style)
            s = m["ami_split"][z]
            print(f"zone {z!r}: {s['faces']} interface faces -> patches"
                  f" {' / '.join(patch_names(z, style))}, {s['points']} points duplicated")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1
    dst = out or case
    write_polymesh(dst, m["points"], m["faces"], m["owner"], m["neighbour"], m["patches"],
                   m["cell_zones"], m["face_zones"], fmt=fmt or m.get("format", "binary"))
    if dyn:
        mrf = read_mrf(case)
        rz = [dict(name=z, **mrf[z]) for z in zones if z in mrf]
        if len(rz) == len(zones):
            print(f"wrote {write_dynamic_mesh_dict(dst, rz, style)}")
        else:
            print("no constant/MRFProperties entry for every zone: dynamicMeshDict not written"
                  " (write it by hand, or run cutmesh with \"ami\": true)")
    hint = couple_hint(zones, style)
    if hint:
        print(f"then in OpenFOAM: {hint}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
