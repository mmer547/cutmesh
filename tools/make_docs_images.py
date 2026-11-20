"""Mesh pictures for the documentation site (docs/images/).

python tools/make_docs_images.py [--only NAME ...] [--work DIR]

Meshes the examples (stirred tank: rotating zone, cylindrical base, boundary layers; the
rotor-stator with a 1 mm gap), a few benchmark shapes and the showcase suite
(pipes, ducts, a bluff body) (benchmark/make_synthetic.py is
run first if their STLs are missing), then draws 3D views and plane sections with
benchmark/render_mesh.py and cutmesh.viz.  Takes about 6 minutes with the Rust extension.
docs/images/stats.json records cells and times shown on the pages.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cutmesh.mesher import load_config, run  # noqa: E402
from cutmesh.viz import plot_slice  # noqa: E402
import render_mesh as rm  # noqa: E402

TANK = os.path.join(ROOT, "examples", "stirred_tank")
SYN = os.path.join(ROOT, "benchmark", "data", "synthetic")
OUT = os.path.join(ROOT, "docs", "images")
DPI = 110


def save(fig, name):
    """PNG with a 256-colour palette (line drawings stay sharp, files get small)."""
    from PIL import Image
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    im = Image.open(p).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(p, optimize=True)
    print(f"  {name}: {os.path.getsize(p) // 1024} KB")


def mesh(md, work, name, check=True):
    """Run cutmesh on the meshDict md (paths absolute) in work/name; return (R, seconds)."""
    case = os.path.join(work, name)
    md = dict(md, case=case)
    p = os.path.join(work, name + ".json")
    with open(p, "w") as f:
        json.dump(md, f, indent=1)
    t0 = time.time()
    R = run(p, log=lambda s: None, check=check)
    return R, time.time() - t0


def n_cells(R):
    return int(max(R["owner"].max(), R["neighbour"].max())) + 1


def tank_md(name):
    cfg = json.load(open(os.path.join(TANK, name)))
    cfg["domain"] = os.path.join(TANK, cfg["domain"])
    for z in cfg.get("rotatingZones", []):
        if "stl" in z:
            z["stl"] = os.path.join(TANK, z["stl"])
    return cfg


def slices(R, panels, name, suptitle="", size=(13, 6.2), widths=None):
    """panels: [(axis, value, title, xlim, ylim)] side by side"""
    from matplotlib.ticker import MaxNLocator
    fig, axs = plt.subplots(1, len(panels), figsize=size, gridspec_kw={"width_ratios": widths})
    axs = np.atleast_1d(axs)
    for ax, (axis, value, title, xl, yl) in zip(axs, panels):
        lw = 0.25 if xl is None else 0.5
        plot_slice(ax, R, axis, value, title, lw=lw)
        if xl is not None:
            ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    save(fig, name)


def _trim(im, pad=12):
    """crop the white margin around a picture"""
    from PIL import ImageChops, Image
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not box:
        return im
    x0, y0, x1, y1 = box
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))


def three_d(R, name, title, which="overview"):
    geo = rm._mesh_geometry(R)
    p = os.path.join(OUT, name)
    (rm.overview if which == "overview" else rm.cut)(R, geo, p, title)
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if which == "cut":
        # render_mesh.cut puts the 3D view left and the sections right with a wide gap: close it.
        # The titles above them are left out (the page says what the picture is).
        w, h = im.size
        left = _trim(im.crop((0, int(0.085 * h), int(0.55 * w), h)))
        right = _trim(im.crop((int(0.55 * w), int(0.06 * h), w, h)))
        H = max(left.height, right.height)
        out = Image.new("RGB", (left.width + right.width + 20, H), (255, 255, 255))
        out.paste(left, (0, (H - left.height) // 2))
        out.paste(right, (left.width + 20, (H - right.height) // 2))
        im = out
    else:
        im = _trim(im)
    w, h = im.size
    im = im.resize((int(w * DPI / 160), int(h * DPI / 160)), Image.LANCZOS)
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(p, optimize=True)
    print(f"  {name}: {os.path.getsize(p) // 1024} KB")


LEGEND = "black: STL walls   red: rotor interface (faceZone)   grey: cell faces"


def case_tank(work, stats):
    R, t = mesh(tank_md("meshDict.json"), work, "tank")
    stats["tank"] = dict(cells=n_cells(R), seconds=round(t, 1))
    three_d(R, "tank_overview.png", "stirred tank (meshDict.json)")
    three_d(R, "tank_cut.png", "stirred tank (meshDict.json)", "cut")
    slices(R, [(2, 0.1001, "z = 0.10 (impeller mid-plane)", None, None),
               (1, 0.0001, "y = 0 (through the blades)", None, None)], "tank_slices.png", LEGEND)


def case_cylindrical(work, stats):
    R, t = mesh(tank_md("meshDict_cylindrical.json"), work, "cyl")
    stats["cylindrical"] = dict(cells=n_cells(R), seconds=round(t, 1))
    slices(R, [(2, 0.1001, "z = 0.10: cylindrical base mesh", None, None),
               (2, 0.1001, "close-up", (-0.08, 0.08), (-0.08, 0.08))], "cylindrical_slices.png", LEGEND)


def case_layers(work, stats):
    R, t = mesh(tank_md("meshDict_layers.json"), work, "layers")
    stats["layers"] = dict(cells=n_cells(R), seconds=round(t, 1))
    slices(R, [(1, 0.0001, "y = 0: 3 boundary layers on every wall", None, None),
               (1, 0.0001, "corner of the bottom and the tank wall", (0.118, 0.152), (-0.002, 0.032))],
           "layers_slices.png", LEGEND)


def case_stator(work, stats):
    stl = os.path.join(TANK, "tank_stator_1mm.stl")
    if not os.path.exists(stl):
        subprocess.run([sys.executable, os.path.join(TANK, "make_geometry.py"), "--stator", "1"], check=True)
    rtip = math.hypot(0.05, 0.003)
    md = tank_md("meshDict_stator.json")
    md["domain"] = stl
    md["autoGapRefinement"]["maxLevel"] = 5
    z = md["rotatingZones"][0]
    z["cylinder"]["radius"] = round(rtip + 0.0005, 6)
    z["wallProximity"] = {"distance": 0.001, "level": 5}
    md.pop("_comment", None)
    R, t = mesh(md, work, "stator")
    stats["stator_1mm"] = dict(cells=n_cells(R), seconds=round(t, 1))
    slices(R, [(2, 0.1001, "z = 0.10: impeller, 1 mm gap, fixed ring", (-0.065, 0.065), (-0.065, 0.065)),
               (2, 0.1001, "z = 0.10, blade tip (2 mm wide)", (0.0494, 0.0516), (-0.0011, 0.0011)),
               (1, 0.0001, "y = 0, top corner of the blade tip", (0.0478, 0.0532), (0.1172, 0.1262))],
           "stator_gap.png", LEGEND, size=(16, 6.0), widths=[1.2, 1, 0.75])


SHAPES = [("sphere", "smooth convex surface"), ("cube_rotated", "sharp edges not aligned with the octree"),
          ("torus", "a hole through the fluid"), ("thin_channels", "channels 2 cells wide"),
          ("l_bracket", "a concave edge")]


def case_shapes(work, stats):
    if not all(os.path.exists(os.path.join(SYN, s + ".stl")) for s, _ in SHAPES):
        subprocess.run([sys.executable, os.path.join(ROOT, "benchmark", "make_synthetic.py")], check=True)
    from cutmesh.geometry import read_stl
    for s, what in SHAPES:
        p = os.path.join(SYN, s + ".stl")
        V = np.concatenate([t.reshape(-1, 3) for _, t in read_stl(p)])
        h = float((V.max(0) - V.min(0)).max()) / 24          # as benchmark/suites/synthetic.json
        R, t = mesh({"domain": p, "baseCellSize": h, "surfaceRefinement": {"default": 2}}, work, s)
        stats[s] = dict(cells=n_cells(R), seconds=round(t, 1))
        three_d(R, f"shape_{s}.png", f"{s}: {what}", "cut")


# Showcase shapes (benchmark/suites/showcase.json): the meshDict comes from the suite; the
# figure is one or two plane sections (axis, value, title, xlim, ylim) or a 3D overview.
SHOWCASE = {
    "pipe_elbow": dict(panels=[(2, 0.0001, "z = 0: through the bend", None, None),
                               (1, 0.5, "y = 0.5: cross-section of the outlet leg", None, None)],
                       size=(11, 5.6), widths=[1.6, 1]),
    "helical_coil": dict(view="overview"),
    "venturi": dict(panels=[(1, 0.0001, "y = 0: contraction, throat and 7° diffuser", None, None),
                            (1, 0.0001, "throat", (-0.21, 0.21), (0.35, 0.75))],
                    size=(11, 7.0), widths=[1, 1.1]),
    "orifice_pipe": dict(panels=[(1, 0.0001, "y = 0: orifice plate (bore 0.5 D)", None, None),
                                 (1, 0.0001, "sharp edge of the bore", (0.04, 0.16), (0.40, 0.53))],
                         size=(11, 7.0), widths=[0.6, 1.1]),
    "backward_step": dict(panels=[(2, 0.15, "z = 0.15: backward-facing step", None, None),
                                  (2, 0.15, "step corner", (-0.1, 0.4), (-0.01, 0.21))],
                          size=(15, 3.0), widths=[3.6, 1.4]),
    "t_junction": dict(panels=[(2, 0.1, "z = 0.1: T-junction of square ducts", None, None)],
                       size=(7.5, 4.4), widths=None),
    "pin_fins": dict(panels=[(2, 0.075, "z = 0.075: staggered pin bank", None, None),
                             (2, 0.075, "around a pin", (0.42, 0.73), (0.13, 0.37))],
                     size=(13, 4.8), widths=[2.2, 1]),
    "body_in_tunnel": dict(panels=[(1, 0.0001, "y = 0: body in the wind tunnel", None, None),
                                   (1, 0.0001, "rounded nose, 25° rear slant", (0.52, 1.18), (0.0, 0.28))],
                           size=(13, 3.6), widths=[1.6, 1.3]),
}


def showcase_md(name):
    suite = json.load(open(os.path.join(ROOT, "benchmark", "suites", "showcase.json")))
    c = next(c for c in suite["cases"] if c["name"] == name)
    md = json.loads(json.dumps(suite["defaults"]["meshDict"]))
    md.update(c["meshDict"])
    md["domain"] = os.path.normpath(os.path.join(ROOT, "benchmark", "suites", c["stl"]))
    return md


def case_showcase(work, stats):
    if not all(os.path.exists(os.path.join(SYN, s + ".stl")) for s in SHOWCASE):
        subprocess.run([sys.executable, os.path.join(ROOT, "benchmark", "make_synthetic.py")], check=True)
    for s, fig in SHOWCASE.items():
        R, t = mesh(showcase_md(s), work, s)
        stats[s] = dict(cells=n_cells(R), seconds=round(t, 1))
        if fig.get("view"):
            three_d(R, f"showcase_{s}.png", s, fig["view"])
        else:
            slices(R, fig["panels"], f"showcase_{s}.png", size=fig["size"], widths=fig["widths"])


CASES = {"tank": case_tank, "cylindrical": case_cylindrical, "layers": case_layers,
         "stator": case_stator, "shapes": case_shapes,
         "showcase": case_showcase}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(CASES))
    ap.add_argument("--work", help="directory for the meshes (default: a temporary one)")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(OUT, "stats.json")
    stats = json.load(open(sp)) if os.path.exists(sp) else {}
    with tempfile.TemporaryDirectory() as tmp:
        work = a.work or tmp
        os.makedirs(work, exist_ok=True)
        for name in a.only or CASES:
            print(name)
            CASES[name](work, stats)
    with open(sp, "w") as f:
        json.dump(stats, f, indent=1)
    update_page(stats)
    return 0


def update_page(stats):
    """Write the cells / seconds into docs/examples.html (<span data-stat="case.key">)."""
    import re
    p = os.path.join(ROOT, "docs", "examples.html")
    if not os.path.exists(p):
        return

    def value(m):
        case, key = m.group(1).split(".")
        if case not in stats:
            return m.group(0)
        v = stats[case][key]
        txt = f"{v:,}" if key == "cells" else (f"{v:.0f}" if v >= 20 else f"{v:.1f}")
        return f'<span data-stat="{m.group(1)}">{txt}</span>'
    with open(p, encoding="utf-8") as f:
        s = f.read()
    s = re.sub(r'<span data-stat="(\w+\.\w+)">[^<]*</span>', value, s)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


if __name__ == "__main__":
    sys.exit(main())
