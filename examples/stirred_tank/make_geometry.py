"""Generate a watertight test geometry: flat-bottom stirred tank with a 4-blade
paddle impeller and a shaft running up to the lid.

Solids (-> OpenFOAM patches): tankWall, bottom, top, impeller, shaft
Units: metres.  Tank D=0.30, H=0.30, impeller D=0.10 at z=0.08..0.12.

python make_geometry.py               -> tank.stl, rotor.stl
python make_geometry.py --stator 0.5  -> tank_stator_0.5mm.stl: the same tank with a fixed ring
    (solids statorInner = the face towards the impeller, stator = the rest) around the
    impeller, leaving a gap of 0.5 mm to the outermost blade corners (rotor-stator mixer)
python make_geometry.py --tilt        -> tank_tilted.stl: the tank turned by TILT (no edge along
    the octree axes, as in most real geometry; the rotor cylinder turns with it, see tilt_point)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cutmesh.geometry import write_stl_ascii

R_TANK, H_TANK = 0.15, 0.30
R_HUB, R_SHAFT, R_BLADE = 0.015, 0.008, 0.05
HALF_W = 0.003            # blade half thickness (6 mm blades)
Z0, Z1 = 0.08, 0.12       # impeller bottom / top
NBLADE = 4
NARC = 16                 # points per hub arc between blades
R_TIP = float(np.hypot(R_BLADE, HALF_W))   # outermost blade corner (50.09 mm)
STATOR_T = 0.006          # stator ring: radial thickness
STATOR_DZ = 0.003         # stator ring: reaches this far above / below the impeller


def impeller_polygon():
    """Star-shaped (w.r.t. the axis) cross-section: hub circle + radial blades, CCW."""
    pts = []
    a_h = np.arcsin(HALF_W / R_HUB)
    xh = np.sqrt(R_HUB ** 2 - HALF_W ** 2)
    for k in range(NBLADE):
        ph = 2 * np.pi * k / NBLADE
        u = np.array([np.cos(ph), np.sin(ph)])
        v = np.array([-np.sin(ph), np.cos(ph)])
        pts += [xh * u - HALF_W * v, R_BLADE * u - HALF_W * v,
                R_BLADE * u + HALF_W * v, xh * u + HALF_W * v]
        ph_next = 2 * np.pi * (k + 1) / NBLADE
        arc = np.linspace(ph + a_h, ph_next - a_h, NARC + 2)[1:-1]
        pts += [R_HUB * np.array([np.cos(t), np.sin(t)]) for t in arc]
    P = np.array(pts)
    ang = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    assert np.all(np.diff(ang) > 0), "polygon must be star-shaped"
    return P, np.arctan2(P[:, 1], P[:, 0])


def ring(r, ang, z):
    return np.stack([r * np.cos(ang), r * np.sin(ang), np.full(len(ang), z)], 1)


def strip(lo, hi, flip=False):
    """Triangles between two closed polylines with matching point counts."""
    n = len(lo)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        t1 = [lo[i], lo[j], hi[j]]
        t2 = [lo[i], hi[j], hi[i]]
        if flip:
            t1, t2 = t1[::-1], t2[::-1]
        tris += [t1, t2]
    return np.array(tris)


def fan(center, loop, flip=False):
    n = len(loop)
    tris = []
    for i in range(n):
        t = [center, loop[i], loop[(i + 1) % n]]
        tris.append(t[::-1] if flip else t)
    return np.array(tris)


def build():
    P2, ang = impeller_polygon()
    imp_bot = np.c_[P2, np.full(len(P2), Z0)]
    imp_top = np.c_[P2, np.full(len(P2), Z1)]
    sh_bot = ring(R_SHAFT, ang, Z1)
    sh_top = ring(R_SHAFT, ang, H_TANK)
    tk_bot = ring(R_TANK, ang, 0.0)
    tk_top = ring(R_TANK, ang, H_TANK)

    # Normals must point OUT of the fluid region's complement consistently; we build
    # the fluid boundary with outward normals (pointing away from fluid).
    tankWall = strip(tk_bot, tk_top)                       # outward (+r)
    bottom = fan(np.array([0, 0, 0.0]), tk_bot, flip=True)  # -z
    top = strip(sh_top, tk_top, flip=True)                  # +z annulus
    # impeller is a hole in the fluid: normals point into the impeller (away from fluid)
    imp_side = strip(imp_bot, imp_top, flip=True)
    imp_bottom = fan(np.array([0, 0, Z0]), imp_bot)         # +z (into solid)
    imp_topann = strip(sh_bot, imp_top)                     # -z (into solid)
    shaft = strip(sh_bot, sh_top, flip=True)
    impeller = np.concatenate([imp_side, imp_bottom, imp_topann])
    return [("tankWall", tankWall), ("bottom", bottom), ("top", top),
            ("impeller", impeller), ("shaft", shaft)]


def stator_ring(gap, n=360):
    """Fixed ring around the impeller: inner radius R_TIP + gap (the polygon's edges lie
    inside that circle by R(1 - cos(pi/n)) = 1.9 um at n=360, small next to the gap).
    The ring is a hole in the fluid: normals point into the ring."""
    ri, ro = R_TIP + gap, R_TIP + gap + STATOR_T
    z0, z1 = Z0 - STATOR_DZ, Z1 + STATOR_DZ
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    i0, i1, o0, o1 = ring(ri, ang, z0), ring(ri, ang, z1), ring(ro, ang, z0), ring(ro, ang, z1)
    inner = strip(i0, i1)                  # +r: away from the fluid inside the ring
    outer = strip(o0, o1, flip=True)       # -r: into the ring
    bot = strip(i0, o0, flip=True)         # +z: into the ring
    top = strip(i1, o1)                    # -z: into the ring
    return [("statorInner", inner), ("stator", np.concatenate([outer, bot, top]))]


def _rot(axis, deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


TILT = _rot("z", 17) @ _rot("x", 23) @ _rot("y", 11)


def tilt_point(p):
    """a point (e.g. the rotor cylinder's ends) turned like tank_tilted.stl"""
    return (TILT @ np.asarray(p, float)).tolist()


def rotor_cylinder(r=0.07, z0=0.06, z1=0.14, n=96):
    """Closed cylinder STL for the rotating zone (alternative to the analytic cylinder)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lo, hi = ring(r, ang, z0), ring(r, ang, z1)
    side = strip(lo, hi)
    bot = fan(np.array([0, 0, z0]), lo, flip=True)
    top = fan(np.array([0, 0, z1]), hi)
    return [("rotor", np.concatenate([side, bot, top]))]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if "--tilt" in sys.argv:
        solids = [(n, t @ TILT.T) for n, t in build()]
        write_stl_ascii(os.path.join(here, "tank_tilted.stl"), solids)
        print("wrote tank_tilted.stl; rotor cylinder ends",
              [round(x, 9) for x in tilt_point([0, 0, 0.06])], [round(x, 9) for x in tilt_point([0, 0, 0.14])])
        sys.exit(0)
    if "--stator" in sys.argv:
        gap_mm = float(sys.argv[sys.argv.index("--stator") + 1])
        solids = build() + stator_ring(gap_mm * 1e-3)
        name = f"tank_stator_{gap_mm:g}mm.stl"
        write_stl_ascii(os.path.join(here, name), solids)
        print(f"wrote {name}: stator ring {(R_TIP + gap_mm * 1e-3) * 1e3:.3f} mm inside radius,"
              f" blade corners at {R_TIP * 1e3:.3f} mm")
        sys.exit(0)
    solids = build()
    write_stl_ascii(os.path.join(here, "tank.stl"), solids)
    write_stl_ascii(os.path.join(here, "rotor.stl"), rotor_cylinder())
    print("wrote tank.stl", sum(len(s[1]) for s in solids), "triangles, and rotor.stl")
