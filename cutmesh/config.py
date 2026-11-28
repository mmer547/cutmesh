"""Checks of a meshDict before meshing.

validate_config(cfg, log): types and ranges of the settings, required keys, input files.
    Every problem found is collected and raised together as one ConfigError; keys that
    cutmesh does not know (usually a typo: the setting would be ignored silently) are
    logged as WARNING lines, with the closest known key.
check_names(cfg, names, log): names that should be solids of the domain STL
    (surfaceRefinement, patchTypes, nonRotatingPatches, addLayers.patches) but are not.
upgrade_keys(cfg): top-level keys renamed or retired (DEPRECATED_KEYS) read as their
    replacement; validate_config logs a WARNING for each.

Keys starting with "_" or "//" are comments and are not checked.
"""
import difflib
import functools
import json
import math
import os


class ConfigError(ValueError):
    """meshDict settings that cannot be used (all problems found, one per line)."""


# Top-level meshDict keys renamed or retired, with the version that did it:
#   old key -> (new key, or None when the setting is gone, version, note)
# Compatibility (the manual's "meshDict compatibility"): a key leaves the meshDict only
# through this table.  It keeps being read (as the new key) with a WARNING for at least one
# minor version, and is removed from the table at the next major version.
DEPRECATED_KEYS = {}


def upgrade_keys(cfg):
    """cfg (a meshDict as read) with the deprecated keys read as their replacement:
    (new dict, [(old, new, version, note)] for the keys found).  A replacement given in
    cfg as well wins over the old key."""
    out, found = {}, []
    for k, v in cfg.items():
        if k in DEPRECATED_KEYS:
            new, ver, note = DEPRECATED_KEYS[k]
            found.append((k, new, ver, note))
            if new is not None and new not in cfg:
                out[new] = v
        else:
            out[k] = v
    return out, found


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
    """an integer; a float with an integer value (3.0) counts, as in JSON schema"""
    if isinstance(v, float):
        return math.isfinite(v) and v.is_integer()
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


_TYPE_OK = {
    "integer": _is_int,
    "number": _is_num,
    "boolean": lambda v: isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}
_TYPE_WORD = {"integer": "an integer", "number": "a number", "boolean": "true or false", "string": "a string",
              "array": "a list [...]", "object": "an object {...}", "null": "null"}


def _type_words(types):
    w = [_TYPE_WORD[t] for t in types]
    return w[0] if len(w) == 1 else ", ".join(w[:-1]) + " or " + w[-1]


@functools.lru_cache(maxsize=1)
def _schema():
    from .schema import meshdict_schema
    return meshdict_schema("en")


def _schema_props(key):
    """the property schemas of the top-level object key (cutmesh/schema.py)"""
    return _schema()["properties"][key]["properties"]


def _ints(v, p):
    """v with the floats of integer value (3.0) turned into int where the schema p wants an
    integer (the mesher uses them as counts)"""
    if not isinstance(p, dict):
        return v
    t = p.get("type")
    types = t if isinstance(t, list) else [t]
    if isinstance(v, float) and "integer" in types and "number" not in types and v.is_integer():
        return int(v)
    if isinstance(v, list) and isinstance(p.get("items"), dict):
        return [_ints(x, p["items"]) for x in v]
    if isinstance(v, dict):
        props = p.get("properties", {})
        extra = p.get("additionalProperties")
        return {k: _ints(x, props.get(k, extra)) for k, x in v.items()}
    return v


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

    def by_schema(self, v, p, name):
        """v against the property schema p (cutmesh/schema.py): type, allowed values, bounds,
        items, nested objects (required keys, unknown keys).  True when usable."""
        t = p.get("type")
        types = t if isinstance(t, list) else ([t] if t else [])
        if types and not any(_TYPE_OK[x](v) for x in types):
            self.err(f"{name} must be {_type_words(types)}, got {v!r}")
            return False
        if v is None:
            return True
        if "enum" in p and not any(v == e and isinstance(v, bool) == isinstance(e, bool) for e in p["enum"]):
            self.err(f"{name} must be one of {', '.join(json.dumps(e) for e in p['enum'])}, got {v!r}")
            return False
        if _is_num(v):
            lo, lo_open = (p["exclusiveMinimum"], True) if "exclusiveMinimum" in p else (p.get("minimum"), False)
            hi, hi_open = (p["exclusiveMaximum"], True) if "exclusiveMaximum" in p else (p.get("maximum"), False)
            if (lo is not None or hi is not None) and not self.num(v, name, lo, hi, lo_open, hi_open):
                return False
        if isinstance(v, list):
            n = len(v)
            if n < p.get("minItems", 0) or n > p.get("maxItems", n):
                want = (f"{p['minItems']}" if p.get("minItems") == p.get("maxItems")
                        else f"{p.get('minItems', 0)} to {p.get('maxItems', 'any number of')}")
                self.err(f"{name} must have {want} items, got {v!r}")
                return False
            if isinstance(p.get("items"), dict):
                return all([self.by_schema(x, p["items"], f"{name}[{i}]") for i, x in enumerate(v)])
        if isinstance(v, dict) and "properties" in p:
            ok = True
            for k in p.get("required", []):
                if k not in v:
                    self.err(f"{name}: {k} is required")
                    ok = False
            self.unknown(v, set(p["properties"]), f"in {name}: ")
            for k, x in v.items():
                if k in p["properties"]:
                    ok = self.by_schema(x, p["properties"][k], f"{name}.{k}") and ok
            return ok
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
    for old, new, ver, note in cfg.get("_deprecated", ()):
        c.warn(f"{old!r} is deprecated since {ver}" + (f": read as {new!r}" if new else ": it is ignored")
               + (f" ({note})" if note else "") + "; it will be removed in the next major version")
    c.unknown({k: v for k, v in cfg.items() if k not in ("_base", "_deprecated")}, set(DEFAULTS) | OTHER_KEYS, "")

    for k in ("domain", "baseCellSize"):
        if k not in cfg:
            c.err(f"{k} is required")
    if "domain" in cfg:
        c.file(cfg, cfg["domain"], "domain")
    if "baseCellSize" in cfg:
        c.num(cfg["baseCellSize"], "baseCellSize", lo=0, lo_open=True)
    if not isinstance(cfg.get("case", "."), str):
        c.err(f"case must be a directory name, got {cfg['case']!r}")
    # (that the case is not a file is checked by mesher.run on the directory actually used:
    # -case overrides it)

    for k in ("bufferLayers", "padding", "fillThinGaps", "fillThinWalls", "thinCutCells"):
        c.int_(cfg[k], k)
    if cfg["featureLevel"] is not None:
        c.int_(cfg["featureLevel"], "featureLevel")
    c.num(cfg["featureAngle"], "featureAngle", lo=0, hi=180, lo_open=True, hi_open=True)
    for k in ("snapTolerance", "featureSnapTolerance", "featureSnapMax", "concaveSnapMax",
              "volumeCheckTolerance"):
        c.num(cfg[k], k, lo=0)
    st = cfg["snapTolerance"]
    if _is_num(st) and st >= 0 and not 0.2 <= st <= 0.45:
        # measured on the benchmark suites (34 cases): only 0.2 to 0.45 keep (nearly) every
        # face below checkMesh's skewness limit; the stages after snapping are tuned for 0.3
        c.warn(f"snapTolerance {st:g} is outside 0.2 to 0.45: the mesh may get faces skewed over 4,"
               " which checkMesh fails (the default 0.3 has none on the examples)")
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
    fc = cfg["featureCaps"]
    if not (fc is None or isinstance(fc, bool) or (isinstance(fc, str) and fc in ("domain", "zone", "both"))):
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
        props = _schema_props("addLayers")
        for k, v in al.items():
            if k in props and k != "patches":
                c.by_schema(v, props[k], "addLayers." + k)

    bm = cfg.get("baseMesh")
    if bm is not None and c.dict_(bm, "baseMesh"):
        c.unknown(bm, BASEMESH_KEYS, "in baseMesh: ")
        props = _schema_props("baseMesh")
        ok = {k: c.by_schema(v, props[k], "baseMesh." + k) for k, v in bm.items() if k in props}
        if ok.get("axis") and not any(bm["axis"]):
            c.err("baseMesh.axis must not be zero")
        if ok.get("maxSizeRatio") and bm["maxSizeRatio"] <= 1:
            # the cells across a block face halve in size: a ratio up to 1 is never met (it
            # refined without end)
            c.err(f"baseMesh.maxSizeRatio must be greater than 1, got {bm['maxSizeRatio']!r}")
        if ok.get("zRange") and not bm["zRange"][0] < bm["zRange"][1]:
            c.err(f"baseMesh.zRange must be [low, high] with low < high, got {bm['zRange']!r}")
        if bm.get("type") == "cylindrical":
            if ("core" in bm) != ("blocks" in bm):
                c.err("baseMesh: give both core and blocks, or neither (automatic layout)")
            if "core" in bm and ("rOuter" in bm or "cellSize" in bm):
                c.err("baseMesh: rOuter and cellSize are for the automatic layout (without core and blocks)")

    if c.errors:
        raise ConfigError("meshDict: " + str(len(c.errors)) + " problem(s):\n  "
                          + "\n  ".join(c.errors))
    # counts given as 3.0 (valid JSON schema integers) are used as int
    props = _schema()["properties"]
    for k in list(cfg):
        if k in props:
            cfg[k] = _ints(cfg[k], props[k])


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
