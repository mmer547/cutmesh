# cutmesh — STL内部＋MRF円筒ゾーン特化のOpenFOAMメッシャー（プロトタイプ）

STL（水密な閉曲面）の**内部**に八分木ベースの多面体メッシュを生成し、OpenFOAMの
`constant/polyMesh` を直接書き出します。回転領域（円筒）は**メッシュ生成の一部として**
切断されるため、円筒壁そのものがメッシュの内部面になります（生成後にセル中心で
ゾーンを選ぶ方式ではありません）。

- 円筒は **寸法指定**（2点＋半径）でも **STL** でも指定可能
- 出力: `cellZones`（回転領域）、`faceZones`（`<name>_interface`：円筒界面、向き付き）、
  STLのsolid名ごとのパッチ、`constant/MRFProperties` の雛形
- 依存: Python 3.9+, numpy, scipy（断面図に matplotlib）。OpenFOAMは不要

> この環境にはOpenFOAMが無いため、`checkMesh` 相当のチェックは自前実装
> （`cutmesh/quality.py`、OpenFOAMと同じ式）で行っています。手元で必ず `checkMesh` を
> 実行して確認してください。

## 使い方

```bash
cd examples/stirred_tank
python3 make_geometry.py              # サンプルSTL（攪拌槽＋4枚パドル翼＋シャフト）
python3 -m cutmesh meshDict.json      # → case/constant/polyMesh, case/constant/MRFProperties
# 断面図も欲しい場合
python3 run.py                        # → slice_z.png, slice_y.png
```

`python3 -m cutmesh <meshDict.json> [-case DIR] [-noCheck]`（プロジェクト直下から実行するか
`PYTHONPATH` に `cutmesh` の親ディレクトリを通してください）

## meshDict.json

| キー | 意味 | 既定値 |
|---|---|---|
| `domain` | 流体領域のSTL（閉曲面、複数solid可。solid名がパッチ名） | 必須 |
| `baseCellSize` | レベル0セルの辺長 [m] | 必須 |
| `surfaceRefinement` | solid名→細分化レベル。`default` で既定値 | `{}` |
| `featureLevel` | 特徴線近傍の細分化レベル（`null`=表面最大レベル） | `null` |
| `featureAngle` | 特徴線抽出の角度 [deg] | 30 |
| `bufferLayers` | レベル遷移ごとのバッファセル数 | 2 |
| `snapTolerance` | 節点スナップ距離（セル辺長比）。スライバー防止 | 0.3 |
| `mergeRatio` | 元hex体積比でこれ未満のカットセルを隣接セルに統合 | 0.25 |
| `mergeMaxNonOrtho` | 統合後の非直交角の上限 [deg]。満たす相手が無ければ統合しない | 65 |
| `refinementRegions` | 体積細分化領域 `[{"cylinder": {...}, "level": n}]` | `[]` |
| `featureCaps` | 凸の鋭角部でカット面に特徴点を挿入（`"domain"`/`false`） | `"domain"` |
| `patchTypes` | パッチ名→タイプ（`wall`,`patch`,…） | 全て `wall` |
| `rotatingZones` | 回転領域のリスト（下記） | `[]` |
| `case` | 出力先ケースディレクトリ | `.` |

回転領域:

```json
{ "name": "rotor",
  "cylinder": { "point1": [0,0,0.06], "point2": [0,0,0.14], "radius": 0.07 },
  "surfaceLevel": 2, "volumeLevel": 2, "omega": 31.416, "nonRotatingPatches": [] }
```

STLで与える場合は `"cylinder"` の代わりに `"stl": "rotor.stl", "origin": [...], "axis": [...]`
（`examples/stirred_tank/meshDict_stlRotor.json`）。複数ゾーン可（重ならないこと）。

## アルゴリズム

1. **八分木**: 根グリッド → 表面・特徴線・円筒面・円筒内部の距離基準で細分化 →
   26近傍2:1バランス。ぶら下がり節点は多角形面の頂点として取り込み、面単位で完全適合。
2. **レベルセット**: STLは光線パリティで内外、近傍のみ厳密最近点距離。円筒は解析式。
3. **節点スナップ**: 特徴点 → 特徴線 → 面 の順に、`snapTolerance×h` 以内の節点を面上へ移動。
   面上の節点は φ=0 として扱うためスライバーセルが出ない。ぶら下がり節点は粗い側の面を
   平面に保つため辺/面に沿ってのみ移動し、届かない場合は φ=0 とみなす（形状誤差 ≤ 0.3h）。
4. **カットセル**（`cutcell.py`）: 多面体を頂点のφで切断。面の分割は**その面の頂点値だけ**
   で決め、交点は辺キーで共有するので隣接セル間で必ず一致（適合性が構成的に保証）。
   カット面＝負側面集合の開いた境界を閉じるループ。
   - 1回目: STLで切断し内側のみ残す → カット面が壁パッチになる
   - 2回目以降: 回転ゾーン面で切断し両側を残す → カット面が**界面の内部面**になる
   - 交点は面へ投影。STLと円筒の交線上の点（シャフト貫通部など）は交互射影で交線へ。
5. **小セル・非凸セル統合**: 体積比 `mergeRatio` 未満のカット断片と、自分の重心が見えない面を持つ
   （非凸の）セルを、同一ゾーン内の隣接セルへ統合。候補ごとに統合後の形状を評価し、全面の
   ピラミッド体積が正で、隣接セルとの非直交角が最も小さい相手を選ぶ（`mergeMaxNonOrtho` 超は不採用）。
6. **出力**: 上三角順の内部面、パッチごとの境界面、cellZone/faceZone、未使用点の除去。

## 検証結果（examples/stirred_tank、2コアVM、Pure Python）

| 設定 | セル数 | 時間 | 非直交 max / avg | >70° | skew>4 | 体積誤差 |
|---|---|---|---|---|---|---|
| `meshDict.json`（界面L2・内部L2） | 102,210 | 36 s | 61.7 / 4.2 | 0 | 1 | −0.04% |
| `meshDict_coarseRotor.json`（界面L1・内部L0） | 62,500 | 29 s | 56.5 / 5.5 | 0 | 1 | −0.04% |

- セル閉合性 1e-16、セル内の開いた辺 0、負体積 0、面ピラミッド負 0、面順序OK
- 境界面中心とSTLの最大距離 0.71 mm（最小セル1.56 mmの0.46倍。凹の鋭角部の面取り分）
- 円筒界面は粗いセルでも厳密に円筒面上にあるので、**界面のために細分化する必要がない**
  → snappyHexMeshで界面faceZoneを綺麗に取るための細分化に比べ、セル数を抑えられます

## 既知の制約

- STLは水密（閉じた・向きの揃った）である必要あり
- セルより薄い形状（薄翼・バッフル板）は、その厚さ程度まで細分化が必要
- 円筒の縁（リム）や凹の鋭角部はセル寸法程度で面取りされる（界面としては実用上問題小）
- skew>4 の境界面が少数残る（checkMeshでは "highly skew faces" として1項目失敗扱いになる可能性あり）
- レベル遷移が壁にかかる所では、ぶら下がり節点を壁上へ動かせない場合 φ=0 とみなすため、
  壁位置に最大0.3×最小セル程度の凹凸が出る。
  その壁を一様なレベルで覆う（隣り合うパッチの `surfaceRefinement` を同じレベルにする）と解消
- 統合相手が見つからない小セルはそのまま残る（最小体積が小さくなる）
- Pure Pythonなので遅い。速度比較はC++版で行うべき（下記）

## ロードマップ

1. **境界層（後付け）**: 境界面はSTLのsolid単位のパッチになっているので、
   - 即試せる案: 本メッシュに対し snappyHexMesh を `castellatedMesh false; snap false; addLayers true;`
     で実行し、レイヤー追加フェーズだけを使う（要検証）
   - 自前実装案: 壁パッチ節点の法線を求め、内側へ押し込み（ラプラシアン平滑化付き）、
     元の壁面との間にプリズム層を挿入する独立モジュール（polyMeshを入力にするので本体と分離可能）
2. **品質改善**: レベル遷移とカットが重なるセルの追加細分化、薄い（扁平）セルの統合基準
3. **C++化**: 八分木・距離計算（BVH）・カットはセル単位で独立なのでOpenMPで並列化しやすい。
   Pythonで時間の大半を占めるのはレベルセット（距離計算）と面生成
4. **AMI（スライディングメッシュ）への移行**: `<name>_interface` faceZone を `createBaffles` で
   切り離し、cyclicAMIにする手順はsnappyHexMeshのチュートリアルと同じ
