"""The settings of a meshDict as a JSON Schema (draft 2020-12), for editors and for the
meshDict form of the documentation site (docs/meshdict.html).

meshdict_schema() returns the schema as a dict.  Every key that config.validate_config()
knows is here, with its default (taken from mesher.DEFAULTS and layers.DEFAULTS), its type
and its range; tests/test_schema.py keeps the two in step.  The descriptions follow the
tables of docs/manual.html.

Besides the standard keywords, each property carries a few "x-" annotations for the form:
    x-group     section of the form ("basic", "refine", "snap", "thin", "check", ...)
    x-advanced  true: shown only with "詳細な設定も表示"
    x-widget    how to edit a value the standard keywords do not describe well:
                "vec3", "stl" (file name), "solidLevels" / "solidTypes" (solid name -> value),
                "solidList" (solid names)
    x-doc       anchor of the section in docs/manual.html
    x-unit      unit shown next to the field

    python -m cutmesh.schema            # print the schema
    python tools/gen_meshdict_schema.py # write docs/meshdict.schema.json and docs/meshdict-data.js
"""
import copy
import json
import sys

SCHEMA_ID = "https://mmer547.github.io/cutmesh/meshdict.schema.json"

# sections of the form, in order: (key, title, anchor in docs/manual.html)
GROUPS = [
    ("basic", "基本", "md-basic"),
    ("refine", "細分化", "md-refine"),
    ("zones", "回転ゾーン", "zones"),
    ("snap", "スナップと統合", "md-snap"),
    ("thin", "薄い形状", "md-thin"),
    ("check", "検査と出力", "md-check"),
    ("baseMesh", "ベースメッシュ", "cylbase"),
    ("layers", "境界層", "layers"),
]

PATCH_TYPES = ["wall", "patch", "symmetry", "symmetryPlane", "empty", "wedge"]


def _p(typ, desc, default=None, *, group=None, advanced=False, widget=None, unit=None,
       doc=None, has_default=True, **kw):
    """one property: typ is a JSON type name or a list of them"""
    d = {"type": typ, "description": desc}
    if has_default:
        d["default"] = copy.deepcopy(default)
    d.update(kw)
    for k, v in (("x-group", group), ("x-advanced", advanced or None), ("x-widget", widget),
                 ("x-unit", unit), ("x-doc", doc)):
        if v is not None:
            d[k] = v
    return d


def _vec3(desc, default=None, **kw):
    kw.setdefault("widget", "vec3")
    return _p("array", desc, default, items={"type": "number"}, minItems=3, maxItems=3, **kw)


def _cylinder(desc):
    return {
        "type": "object", "description": desc, "x-widget": "cylinder",
        "required": ["point1", "point2", "radius"],
        "additionalProperties": False,
        "properties": {
            "point1": _vec3("軸の一端", has_default=False, unit="m"),
            "point2": _vec3("軸のもう一端（point1 と違う点）", has_default=False, unit="m"),
            "radius": _p("number", "半径", has_default=False, exclusiveMinimum=0, unit="m"),
            "origin": _vec3("回転の中心（回転ゾーンのみ。省略時は point1）", has_default=False,
                            unit="m", advanced=True),
        },
    }


def _top(D):
    """top-level properties; D = mesher.DEFAULTS"""
    G = {}

    def add(key, prop):
        G[key] = prop

    # ---- basic
    add("domain", _p("string", "流体領域の STL（閉曲面、複数 solid 可。solid 名がパッチ名）。"
                     "meshDict.json のあるフォルダーからの相対パス", has_default=False,
                     group="basic", widget="stl", minLength=1, doc="md-basic"))
    add("baseCellSize", _p("number", "レベル 0 のセルの辺長。レベル L のセルは baseCellSize / 2^L",
                           has_default=False, group="basic", exclusiveMinimum=0, unit="m",
                           doc="md-basic"))
    add("case", _p("string", "出力先のケースディレクトリ", ".", group="basic", doc="md-basic"))
    add("patchTypes", _p("object", "パッチ名 → タイプ（wall, patch など）。書かないパッチは wall",
                         D["patchTypes"], group="basic", widget="solidTypes",
                         additionalProperties={"type": "string", "minLength": 1,
                                               "examples": PATCH_TYPES},
                         doc="md-basic"))
    add("padding", _p("integer", "STL の外接箱の周りに足すベースセルの数", D["padding"],
                      group="basic", advanced=True, minimum=0, doc="md-basic"))

    # ---- refinement
    add("surfaceRefinement", _p("object", "solid 名 → 細分化レベル。default で残りの solid の既定値",
                                D["surfaceRefinement"], group="refine", widget="solidLevels",
                                additionalProperties={"type": "integer", "minimum": 0},
                                doc="md-refine"))
    add("featureLevel", _p(["integer", "null"], "特徴線（鋭角エッジ）の近くの細分化レベル"
                           "（null = 表面の最大レベル）", D["featureLevel"], group="refine",
                           minimum=0, doc="md-refine"))
    add("featureAngle", _p("number", "特徴線を抽出する角度", D["featureAngle"], group="refine",
                           exclusiveMinimum=0, exclusiveMaximum=180, unit="deg", doc="md-refine"))
    add("refinementRegions", _p("array", "体積の細分化領域（円筒の中をレベル level まで細かくする）",
                                D["refinementRegions"], group="refine", doc="md-refine",
                                items={"type": "object", "required": ["cylinder", "level"],
                                       "additionalProperties": False,
                                       "properties": {
                                           "cylinder": _cylinder("細分化する円筒"),
                                           "level": _p("integer", "細分化レベル", has_default=False,
                                                       minimum=0)}}))
    add("bufferLayers", _p("integer", "レベルの遷移ごとのバッファセルの数。1 にするとセル数が減る",
                           D["bufferLayers"], group="refine", minimum=0, doc="md-refine"))
    add("gapRefinement", _p("object", "流体の厚みが thickness 未満の所（スロットの先端、狭い流路）の壁を"
                            " level まで細分化する", has_default=False, group="refine",
                            advanced=True, doc="md-refine",
                            required=["thickness", "level"], additionalProperties=False,
                            properties={
                                "thickness": _p("number", "流体の厚み", has_default=False,
                                                exclusiveMinimum=0, unit="m"),
                                "level": _p("integer", "細分化レベル", has_default=False, minimum=0)}))
    add("autoGapRefinement", _p(["boolean", "object"],
                                "壁から向かいの壁までの隙間に cellsAcross 個のセルが入るレベル"
                                "（上限 maxLevel）まで壁を細分化する。true で既定値",
                                has_default=False, group="refine", doc="md-refine",
                                additionalProperties=False,
                                properties={
                                    "cellsAcross": _p("number", "隙間に入れるセルの数", 3,
                                                      exclusiveMinimum=0),
                                    "maxLevel": _p("integer", "レベルの上限（既定: 表面レベルの最大 + 2）",
                                                   has_default=False, minimum=0)}))

    # ---- rotating zones
    zone = {
        "type": "object", "required": ["name"], "additionalProperties": False,
        "oneOf": [{"required": ["cylinder"]}, {"required": ["stl"]}],
        "properties": {
            "name": _p("string", "ゾーン名（cellZone 名。faceZone は <name>_interface）",
                       has_default=False, minLength=1),
            "cylinder": _cylinder("回転ゾーンの円筒（stl とどちらか一方）"),
            "stl": _p("string", "円筒の代わりに閉じた STL で与える（cylinder とどちらか一方）",
                      has_default=False, widget="stl", minLength=1),
            "origin": _vec3("STL ゾーンの回転の中心", [0, 0, 0], unit="m"),
            "axis": _vec3("STL ゾーンの回転軸（0 ベクトル不可）", [0, 0, 1]),
            "surfaceLevel": _p("integer", "ゾーンの面の細分化レベル", 0, minimum=0),
            "volumeLevel": _p("integer", "ゾーンの内部の細分化レベル", 0, minimum=0),
            "omega": _p("number", "角速度（MRFProperties・dynamicMeshDict に書く）", 0.0,
                        unit="rad/s"),
            "nonRotatingPatches": _p("array", "ゾーン内でも回転させないパッチ", [],
                                     widget="solidList", items={"type": "string"}),
            "wallProximity": _p("object", "ゾーンの面のうち STL から distance 以内の部分だけを level まで"
                                "細分化する（界面を狭い隙間に置くとき）", has_default=False,
                                advanced=True, required=["distance", "level"],
                                additionalProperties=False,
                                properties={
                                    "distance": _p("number", "STL からの距離", has_default=False,
                                                   exclusiveMinimum=0, unit="m"),
                                    "level": _p("integer", "細分化レベル", has_default=False,
                                                minimum=0),
                                    "bufferLayers": _p("integer", "バッファセルの数",
                                                       has_default=False, minimum=0)}),
            "ami": _p("boolean", "true: 界面をスライディングメッシュ用の 2 つのパッチに分け、"
                      "dynamicMeshDict を書く", False),
            "conforming": _p("boolean", "false: 界面をメッシュに沿わせず、セル中心が中にあるセルを"
                             " cellZone にする（MRF 用。ami と併用不可）", True, advanced=True),
        },
    }
    add("rotatingZones", _p("array", "回転ゾーン（MRF・スライディングメッシュ）のリスト。"
                            "ゾーンどうしは重ならないこと", D["rotatingZones"], group="zones",
                            items=zone, doc="zones"))

    # ---- snapping and merging
    for key, desc, unit, adv, rng in (
            ("snapTolerance", "節点を表面へ寄せる距離（セル辺長比）。スライバーセルを防ぐ", None, False,
             dict(minimum=0)),
            ("featureSnapTolerance", "特徴線（STL の鋭角エッジ・円筒の縁）へ寄せる距離（セル辺長比）",
             None, False, dict(minimum=0)),
            ("featureSnapMax", "実験的: この距離まで特徴線へ寄せ、八分木セルが壊れる所だけ取り消す",
             None, True, dict(minimum=0)),
            ("concaveSnapMax", "凹の鋭角エッジで、固体側の節点をこの距離までエッジ上へ寄せる", None, True,
             dict(minimum=0)),
            ("mergeRatio", "元の六面体の体積比でこれ未満のカットセルを隣のセルに統合する", None, False,
             dict(minimum=0, exclusiveMaximum=1)),
            ("mergeMaxNonOrtho", "統合後の非直交角の上限", "deg", True,
             dict(exclusiveMinimum=0, maximum=90))):
        add(key, _p("number", desc, D[key], group="snap", advanced=adv, unit=unit, doc="md-snap",
                    **rng))
    add("hangingSnap", _p("string", "表面へ届かないぶら下がり節点の扱い（free: 表面へ直接投影、"
                          "constrained: 動かさない）", D["hangingSnap"], group="snap", advanced=True,
                          enum=["free", "constrained"], doc="md-snap"))
    add("featureCaps", _p(["string", "boolean", "null"], "凸の鋭角部でカット面を特徴線に沿わせる"
                          "（domain / zone / both / false）", D["featureCaps"], group="snap",
                          advanced=True, enum=["domain", "zone", "both", True, False, None],
                          doc="md-snap"))
    add("exactFeatureCut", _p("boolean", "凸の鋭角エッジの近くのセルを STL の平面で直接切る"
                              "（featureCaps が domain か both のときだけ働く）", D["exactFeatureCut"],
                              group="snap", advanced=True, doc="md-snap"))
    add("mergeWallFaces", _p("boolean", "1 つのセルの、ほぼ同一平面の壁面の断片を 1 つの面にまとめる",
                             D["mergeWallFaces"], group="snap", advanced=True, doc="md-snap"))

    # ---- thin shapes
    for key, desc in (
            ("fillThinGaps", "中に流体があるのに角の頂点がどれも流体にないセルを分割する回数。"
             "0 で無効（直交格子のみ）"),
            ("fillThinWalls", "中に固体の薄い壁があるのに角の頂点がどれも固体にないセルを分割する回数"
             "（直交格子のみ）"),
            ("thinCutCells", "壁で切られて流体が少ししか残らず、隣に細かいセルがあるセルを分割する回数"
             "（直交格子のみ）")):
        add(key, _p("integer", desc, D[key], group="thin", advanced=True, minimum=0, doc="md-thin"))
    add("thinCutFraction", _p("number", "thinCutCells の対象にする流体の割合", D["thinCutFraction"],
                              group="thin", advanced=True, exclusiveMinimum=0, exclusiveMaximum=1,
                              doc="md-thin"))
    add("minRegionVolume", _p("number", "流体の本体とつながっていない領域のうち、体積が全体のこの割合"
                              "未満のものを削除する", D["minRegionVolume"], group="thin",
                              advanced=True, minimum=0, exclusiveMaximum=1, doc="md-thin"))

    # ---- checks and output
    add("checkLeaks", _p("boolean", "流体が固体の壁を越えてつながっていないかを検査する",
                         D["checkLeaks"], group="check", doc="md-check"))
    add("strictInput", _p("boolean", "STL に開いた辺・非多様体の辺があるとき、警告ではなくエラーで止める",
                          D["strictInput"], group="check", advanced=True, doc="md-check"))
    add("volumeCheckTolerance", _p("number", "メッシュと STL の体積の差がこの割合を超えたら警告する",
                                   D["volumeCheckTolerance"], group="check", advanced=True,
                                   minimum=0, doc="md-check"))
    add("writeFormat", _p("string", "polyMesh の形式", D["writeFormat"], group="check",
                          enum=["binary", "ascii"], doc="format"))
    add("writeMRFProperties", _p("boolean", "回転ゾーンがあるとき constant/MRFProperties の雛形を書く",
                                 D["writeMRFProperties"], group="check", doc="md-check"))
    add("openfoam", _p("string", "スライディングメッシュの書き方: com（openfoam.com / ESI）、"
                       "org10・org11・org12（openfoam.org）", D["openfoam"], group="check",
                       enum=["com", "org10", "org11", "org12"], doc="ami"))
    add("writeDynamicMeshDict", _p("boolean", "ami のゾーンがあるとき constant/dynamicMeshDict を書く",
                                   D["writeDynamicMeshDict"], group="check", advanced=True,
                                   doc="md-check"))
    add("amiExactRims", _p("boolean", "ami のゾーンの界面を、ゾーンの縁にぴったり沿わせる",
                           D["amiExactRims"], group="check", advanced=True, doc="ami"))

    # ---- base mesh
    add("baseMesh", _p("object", "ベースメッシュ。省略時は直交格子。cylindrical で回転軸まわりの"
                       "同心ブロック", has_default=False, group="baseMesh", doc="cylbase",
                       additionalProperties=False, properties={
                           "type": _p("string", "cartesian（直交格子）または cylindrical（円筒）",
                                      "cartesian", enum=["cartesian", "cylindrical"]),
                           "origin": _vec3("回転軸の原点", [0, 0, 0], unit="m"),
                           "axis": _vec3("回転軸の方向", [0, 0, 1]),
                           "zRange": _p("array", "軸方向の範囲 [z0, z1]（origin からの距離。"
                                        "既定: STL の範囲 ± baseCellSize）", has_default=False,
                                        items={"type": "number"}, minItems=2, maxItems=2,
                                        unit="m", advanced=True),
                           "nz": _p("integer", "軸方向の分割数（nz か dz）", has_default=False,
                                    minimum=1),
                           "dz": _p("number", "軸方向の幅（既定: baseCellSize）", has_default=False,
                                    exclusiveMinimum=0, unit="m"),
                           "rOuter": _p("number", "最外ブロックの外半径（自動配置。既定: STL の半径 +"
                                        " 2/3 × cellSize）", has_default=False,
                                        exclusiveMinimum=0, unit="m", advanced=True),
                           "cellSize": _p("number", "自動配置の目標寸法（既定: baseCellSize と dz の"
                                          "小さい方）", has_default=False, exclusiveMinimum=0,
                                          unit="m", advanced=True),
                           "maxSizeRatio": _p("number", "ブロック境界をまたぐセル寸法の比の上限", 2.5,
                                              exclusiveMinimum=0, advanced=True),
                           "core": _p("object", "手動配置: 中心の正方形ブロック（blocks と一緒に書く）",
                                      has_default=False, advanced=True, widget="json",
                                      required=["halfWidth", "n"],
                                      properties={
                                          "halfWidth": _p("number", "半幅", has_default=False,
                                                          exclusiveMinimum=0, unit="m"),
                                          "n": _p("integer", "一辺の分割数", has_default=False,
                                                  minimum=1)}),
                           "blocks": _p("array", "手動配置: 外側へ並ぶ同心ブロック（core と一緒に書く）",
                                        has_default=False, advanced=True, widget="json",
                                        items={"type": "object", "required": ["rOuter", "nr"],
                                               "properties": {
                                                   "rOuter": {"type": "number", "exclusiveMinimum": 0},
                                                   "nr": {"type": "integer", "minimum": 1},
                                                   "nTheta": {"type": "integer", "minimum": 4},
                                                   "radii": {"type": "array",
                                                             "items": {"type": "number"}}}}),
                       }))
    return G


def _layers(LD):
    """properties of addLayers; LD = layers.DEFAULTS"""
    def p(key, typ, desc, adv=False, **kw):
        return key, _p(typ, desc, LD[key], advanced=adv, **kw)

    return dict([
        p("patches", ["array", "null"], "層を付けるパッチ名のリスト（null: type が wall の全パッチ）",
          widget="solidList", items={"type": "string"}),
        p("nLayers", "integer", "層の数", minimum=1),
        p("expansionRatio", "number", "壁から内側への層厚の比", exclusiveMinimum=0),
        p("finalLayerThickness", "number", "最も内側の層の厚さ（relativeSizes なら局所セル寸法比）",
          exclusiveMinimum=0),
        p("relativeSizes", "boolean", "厚さを局所セル寸法に対する比で与える"),
        p("minThickness", "number", "総厚がこれ未満の所は層を付けない", minimum=0),
        p("maxNonOrtho", "number", "非直交角がこれを超えるセルは不正とみなし、そこで層を薄くする",
          unit="deg", exclusiveMinimum=0, maximum=90),
        p("maxSkewness", ["number", "null"], "歪度がこれを超える面のセルは不正とみなす（null で無効）",
          exclusiveMinimum=0),
        p("maxGapFraction", "number", "総厚を、向かい合う壁までの距離のこの割合以下に抑える（0 で無効）",
          minimum=0),
        p("maxThicknessSlope", "number", "壁に沿った総厚の変化の上限（層の端を徐々に細らせる）",
          minimum=0),
        p("minNormalCos", "number", "節点法線と面法線の cos がこれ未満（鋭い凸エッジ）の所は層を付けない",
          True),
        p("wallNonOrthMargin", ["number", "null"], "元の壁面に、境界面としての角度 + この値まで許す",
          True, unit="deg"),
        p("cornerStretch", "boolean", "角の節点を T/cos 動かし、角でも層厚 T を保つ", True),
        p("mergeWallPoints", "number", "壁上でこの値 × 周りの辺長より近い節点どうしを 1 点にまとめる"
          "（0 で無効）", True, minimum=0),
        p("concaveSlide", ["number", "null"], "凹の角で、もう一方の壁の層に食い込む節点を自分の壁に沿って"
          "滑らせる（null で無効）", True),
        p("nRings", "integer", "内部節点を押し込む範囲（壁からの辺数）", True, minimum=0),
        p("nSmooth", "integer", "内部の変位のラプラス平滑化の回数", True, minimum=0),
        p("nIter", "integer", "不正セルの所の厚さを半減する反復回数", True, minimum=0),
        p("nCollapseIter", "integer", "その後に層を止める反復回数", True, minimum=0),
    ])


def meshdict_schema():
    from .mesher import DEFAULTS
    from .layers import DEFAULTS as LD
    props = _top(DEFAULTS)
    props["addLayers"] = _p("object", "境界層（壁パッチにプリズム層を追加）。{} で既定値",
                            has_default=False, group="layers", doc="layers",
                            additionalProperties=False, properties=_layers(LD))
    order = [g for g, _, _ in GROUPS]
    keys = sorted(props, key=lambda k: order.index(props[k]["x-group"]))   # stable within a group
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "cutmesh meshDict",
        "description": "cutmesh の設定ファイル meshDict.json。_ または // で始まるキーはコメント。",
        "type": "object",
        "required": ["domain", "baseCellSize"],
        "properties": {k: props[k] for k in keys},
        "patternProperties": {"^(_|//)": {"description": "コメント（無視される）"}},
        "additionalProperties": False,
        "x-groups": [{"key": g, "title": t, "doc": a} for g, t, a in GROUPS],
        "x-patchTypes": PATCH_TYPES,
    }


def schema_json(indent=1):
    return json.dumps(meshdict_schema(), ensure_ascii=False, indent=indent) + "\n"


if __name__ == "__main__":
    sys.stdout.write(schema_json())
