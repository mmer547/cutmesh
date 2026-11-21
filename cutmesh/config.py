"""Checks of a meshDict before meshing.

validate_config(cfg, log): types and ranges of the settings, required keys, input files.
    Every problem found is collected and raised together as one ConfigError; keys that
    cutmesh does not know (usually a typo: the setting would be ignored silently) are
    logged as WARNING lines, with the closest known key.
check_names(cfg, names, log): names that should be solids of the domain STL
    (surfaceRefinement, patchTypes, nonRotatingPatches, addLayers.patches) but are not.

Keys starting with "_" or "//" are comments and are not checked.
"""
import difflib
import math
import os


class ConfigError(ValueError):
    """meshDict settings that cannot be used (all problems found, one per line)."""


# keys read by the mesher besides DEFAULTS (cutmesh/mesher.py)
OTHER_KEYS = {"domain", "baseCellSize", "gapRefinement", "autoGapRefinement", "case",
              "addLayers", "baseMesh"}
ZONE_KEYS = {"name", "cylinder", "stl", "origin", "axis", "surfaceLevel", "volumeLevel", "omega",
             "nonRotatingPatches", "wallProximity", "conforming", "ami"}
CYLINDER_KEYS = {"point1", "point2", "radius", "origin"}
BASEMESH_KEYS = {"type", "origin", "axis", "zRange", "nz", "dz", "core", "blocks", "maxSizeRatio",
                 "rOuter", "cellSize"}


def _is_comment(k):
    return isinstance(k, str) and (k.startswith("_") or k.startswith("//"))


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class _Checker:
    def __init__(self, log):
        self.errors = []
        self.log = log

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        if self.log:
            self.log(f"WARNING: meshDict: {msg}")

    def unknown(self, d, known, where):
        for k in d:
            if k in known or _is_comment(k):
                continue
            close = difflib.get_close_matches(str(k), sorted(known), n=1, cutoff=0.6)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            self.warn(f"unknown key {where}{k!r} is ignored{hint}")

    # ---- value checks: each returns True when the value is usable
    def int_(self, v, name, lo=0):
        if not _is_int(v) or v < lo:
            self.err(f"{name} must be an integer >= {lo}, got {v!r}")
            return False
        return True

    def num(self, v, name, lo=None, hi=None, lo_open=False, hi_open=False):
        if not _is_num(v):
            self.err(f"{name} must be a number, got {v!r}")
            return False
        bad = ((lo is not None and (v <= lo if lo_open else v < lo))
               or (hi is not None and (v >= hi if hi_open else v > hi)))
        if bad:
            rng = (("(" if lo_open else "[") + ("-inf" if lo is None else f"{lo:g}") + ", "
                   + ("inf" if hi is None else f"{hi:g}") + (")" if hi_open else "]"))
            self.err(f"{name} must be in {rng}, got {v!r}")
            return False
        return True

    def bool_(self, v, name):
        if not isinstance(v, bool):
            self.err(f"{name} must be true or false, got {v!r}")
            return False
        return True

    def vec3(self, v, name):
        if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(_is_num(x) for x in v)):
            self.err(f"{name} must be a list of 3 numbers, got {v!r}")
            return False
        return True

    def dict_(self, v, name):
        if not isinstance(v, dict):
            self.err(f"{name} must be an object {{...}}, got {v!r}")
            return False
        return True

    def list_(self, v, name):
        if not isinstance(v, list):
            self.err(f"{name} must be a list [...], got {v!r}")
            return False
        return True

    def file(self, cfg, p, name):
        if not isinstance(p, str) or not p:
            self.err(f"{name} must be a file name, got {p!r}")
            return False
        full = p if os.path.isabs(p) else os.path.join(cfg.get("_base", "."), p)
        if not os.path.isfile(full):
            self.err(f"{name}: file not found: {full}")
            return False
        return True

    def cylinder(self, c, name):
        if not self.dict_(c, name):
            return
        self.unknown(c, CYLINDER_KEYS, f"in {name}: ")
        for k in ("point1", "point2", "radius"):
            if k not in c:
                self.err(f"{name}: {k} is required")
        ok = all(self.vec3(c[k], f"{name}.{k}") for k in ("point1", "point2") if k in c)
        if ok and "point1" in c and "point2" in c and list(c["point1"]) == list(c["point2"]):
            self.err(f"{name}: point1 and point2 must differ (the axis has length 0)")
        if "radius" in c:
            self.num(c["radius"], f"{name}.radius", lo=0, lo_open=True)
        if "origin" in c:
            self.vec3(c["origin"], f"{name}.origin")


def validate_config(cfg, log=None, layer_keys=None):
    """Check a meshDict as returned by mesher.load_config (DEFAULTS merged in).
    Raises ConfigError with every problem; logs WARNING lines for unknown keys."""
    from .mesher import DEFAULTS                     # (mesher imports this module)
    if layer_keys is None:
        from .layers import DEFAULTS as LD
        layer_keys = set(LD)
    c = _Checker(log)
    c.unknown({k: v for k, v in cfg.items() if k != "_base"}, set(DEFAULTS) | OTHER_KEYS, "")

    for k in ("domain", "baseCellSize"):
        if k not in cfg:
            c.err(f"{k} is required")
    if "domain" in cfg:
        c.file(cfg, cfg["domain"], "domain")
    if "baseCellSize" in cfg:
        c.num(cfg["baseCellSize"], "baseCellSize", lo=0, lo_open=True)
    if not isinstance(cfg.get("case", "."), str):
        c.err(f"case must be a directory name, got {cfg['case']!r}")

    for k in ("bufferLayers", "padding", "fillThinGaps", "fillThinWalls", "thinCutCells"):
        c.int_(cfg[k], k)
    if cfg["featureLevel"] is not None:
        c.int_(cfg["featureLevel"], "featureLevel")
    c.num(cfg["featureAngle"], "featureAngle", lo=0, hi=180, lo_open=True, hi_open=True)
    for k in ("snapTolerance", "featureSnapTolerance", "featureSnapMax", "concaveSnapMax",
              "volumeCheckTolerance"):
        c.num(cfg[k], k, lo=0)
    c.num(cfg["mergeRatio"], "mergeRatio", lo=0, hi=1, hi_open=True)
    c.num(cfg["mergeMaxNonOrtho"], "mergeMaxNonOrtho", lo=0, hi=90, lo_open=True)
    c.num(cfg["thinCutFraction"], "thinCutFraction", lo=0, hi=1, lo_open=True, hi_open=True)
    c.num(cfg["minRegionVolume"], "minRegionVolume", lo=0, hi=1, hi_open=True)
    for k in ("checkLeaks", "strictInput", "writeMRFProperties", "mergeWallFaces", "writeDynamicMeshDict",
              "exactFeatureCut", "amiExactRims"):
        c.bool_(cfg[k], k)
    if cfg["hangingSnap"] not in ("free", "constrained"):
        c.err(f"hangingSnap must be \"free\" or \"constrained\", got {cfg['hangingSnap']!r}")
    if cfg["openfoam"] not in ("com", "org10", "org11", "org12"):
        c.err(f"openfoam must be \"com\" (openfoam.com) or \"org10\", \"org11\", \"org12\""
              f" (openfoam.org), got {cfg['openfoam']!r}")
    if cfg["writeFormat"] not in ("binary", "ascii"):
        c.err(f"writeFormat must be \"binary\" or \"ascii\", got {cfg['writeFormat']!r}")
    if cfg["featureCaps"] not in ("domain", "zone", "both", True, False, None):
        c.err(f"featureCaps must be \"domain\", \"zone\", \"both\" or false, got {cfg['featureCaps']!r}")

    sr = cfg["surfaceRefinement"]
    if c.dict_(sr, "surfaceRefinement"):
        for k, v in sr.items():
            if not _is_comment(k):
                c.int_(v, f"surfaceRefinement.{k}")
    pt = cfg["patchTypes"]
    if c.dict_(pt, "patchTypes"):
        for k, v in pt.items():
            if not _is_comment(k) and (not isinstance(v, str) or not v):
                c.err(f"patchTypes.{k} must be a patch type name (wall, patch, ...), got {v!r}")

    gr = cfg.get("gapRefinement")
    if gr is not None and c.dict_(gr, "gapRefinement"):
        c.unknown(gr, {"thickness", "level"}, "in gapRefinement: ")
        for k in ("thickness", "level"):
            if k not in gr:
                c.err(f"gapRefinement: {k} is required")
        if "thickness" in gr:
            c.num(gr["thickness"], "gapRefinement.thickness", lo=0, lo_open=True)
        if "level" in gr:
            c.int_(gr["level"], "gapRefinement.level")
    ag = cfg.get("autoGapRefinement")
    if ag is not None and not isinstance(ag, bool) and c.dict_(ag, "autoGapRefinement"):
        c.unknown(ag, {"cellsAcross", "maxLevel"}, "in autoGapRefinement: ")
        if "cellsAcross" in ag:
            c.num(ag["cellsAcross"], "autoGapRefinement.cellsAcross", lo=0, lo_open=True)
        if "maxLevel" in ag:
            c.int_(ag["maxLevel"], "autoGapRefinement.maxLevel")

    rr = cfg["refinementRegions"]
    if c.list_(rr, "refinementRegions"):
        for i, r in enumerate(rr):
            nm = f"refinementRegions[{i}]"
            if not c.dict_(r, nm):
                continue
            c.unknown(r, {"cylinder", "level"}, f"in {nm}: ")
            if "cylinder" not in r:
                c.err(f"{nm}: cylinder is required (the only region shape)")
            else:
                c.cylinder(r["cylinder"], f"{nm}.cylinder")
            if "level" not in r:
                c.err(f"{nm}: level is required")
            else:
                c.int_(r["level"], f"{nm}.level")

    zs = cfg["rotatingZones"]
    if c.list_(zs, "rotatingZones"):
        seen = set()
        for i, z in enumerate(zs):
            nm = f"rotatingZones[{i}]"
            if not c.dict_(z, nm):
                continue
            c.unknown(z, ZONE_KEYS, f"in {nm}: ")
            name = z.get("name")
            if not isinstance(name, str) or not name:
                c.err(f"{nm}: name is required")
            elif name in seen:
                c.err(f"{nm}: zone name {name!r} is used twice")
            else:
                seen.add(name)
                nm = f"rotatingZones[{i}] ({name!r})"
            if ("cylinder" in z) == ("stl" in z):
                c.err(f"{nm}: give either cylinder or stl")
            elif "cylinder" in z:
                c.cylinder(z["cylinder"], f"{nm}.cylinder")
            else:
                c.file(cfg, z["stl"], f"{nm}.stl")
            for k in ("surfaceLevel", "volumeLevel"):
                if k in z:
                    c.int_(z[k], f"{nm}.{k}")
            for k in ("origin", "axis"):
                if k in z and c.vec3(z[k], f"{nm}.{k}") and k == "axis" and not any(z[k]):
                    c.err(f"{nm}.axis must not be zero")
            if "omega" in z:
                c.num(z["omega"], f"{nm}.omega")
            if "conforming" in z:
                c.bool_(z["conforming"], f"{nm}.conforming")
            if "ami" in z:
                c.bool_(z["ami"], f"{nm}.ami")
                if z["ami"] is True and z.get("conforming", True) is False:
                    c.err(f"{nm}: \"ami\": true needs a conforming interface (\"conforming\": false"
                          " only selects cells by their centre)")
            nr = z.get("nonRotatingPatches", [])
            if c.list_(nr, f"{nm}.nonRotatingPatches") and not all(isinstance(p, str) for p in nr):
                c.err(f"{nm}.nonRotatingPatches must be a list of patch names, got {nr!r}")
            wp = z.get("wallProximity")
            if wp is not None and c.dict_(wp, f"{nm}.wallProximity"):
                c.unknown(wp, {"distance", "level", "bufferLayers"}, f"in {nm}.wallProximity: ")
                for k in ("distance", "level"):
                    if k not in wp:
                        c.err(f"{nm}.wallProximity: {k} is required")
                if "distance" in wp:
                    c.num(wp["distance"], f"{nm}.wallProximity.distance", lo=0, lo_open=True)
                if "level" in wp:
                    c.int_(wp["level"], f"{nm}.wallProximity.level")
                if "bufferLayers" in wp:
                    c.int_(wp["bufferLayers"], f"{nm}.wallProximity.bufferLayers")

    al = cfg.get("addLayers")
    if al is not None and c.dict_(al, "addLayers"):
        c.unknown(al, layer_keys, "in addLayers: ")
        p = al.get("patches")
        if p is not None and (not isinstance(p, list) or not all(isinstance(x, str) for x in p)):
            c.err(f"addLayers.patches must be a list of patch names, got {p!r}")
        if "nLayers" in al:
            c.int_(al["nLayers"], "addLayers.nLayers", lo=1)

    bm = cfg.get("baseMesh")
    if bm is not None and c.dict_(bm, "baseMesh"):
        c.unknown(bm, BASEMESH_KEYS, "in baseMesh: ")
        if bm.get("type", "cartesian") not in ("cartesian", "cylindrical"):
            c.err(f"baseMesh.type must be \"cartesian\" or \"cylindrical\", got {bm['type']!r}")
        if bm.get("type") == "cylindrical":
            if ("core" in bm) != ("blocks" in bm):
                c.err("baseMesh: give both core and blocks, or neither (automatic layout)")
            if "core" in bm and ("rOuter" in bm or "cellSize" in bm):
                c.err("baseMesh: rOuter and cellSize are for the automatic layout (without core and blocks)")
            for k in ("rOuter", "cellSize"):
                if k in bm:
                    c.num(bm[k], "baseMesh." + k, lo=0, lo_open=True)

    if c.errors:
        raise ConfigError("meshDict: " + str(len(c.errors)) + " problem(s):\n  "
                          + "\n  ".join(c.errors))


def check_names(cfg, names, log):
    """WARNING lines for names that should be solids (patches) of the domain STL."""
    have = set(names)

    def warn(where, k):
        close = difflib.get_close_matches(str(k), sorted(have), n=1, cutoff=0.6)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        log(f"WARNING: meshDict: {where} names {k!r}, which is not a solid of the domain STL"
            f" (solids: {', '.join(names)}); it is ignored{hint}")

    for k in cfg["surfaceRefinement"]:
        if k != "default" and k not in have and not _is_comment(k):
            warn("surfaceRefinement", k)
    for k in cfg["patchTypes"]:
        if k not in have and not _is_comment(k):
            warn("patchTypes", k)
    for z in cfg["rotatingZones"]:
        for k in z.get("nonRotatingPatches", []):
            if k not in have:
                warn(f"rotating zone {z['name']!r} nonRotatingPatches", k)
    al = cfg.get("addLayers") or {}
    for k in al.get("patches") or []:
        if k not in have:
            warn("addLayers.patches", k)
