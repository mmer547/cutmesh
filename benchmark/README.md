# benchmark — cutmesh の対応力テスト

種類の違う STL をまとめて cutmesh に通し、どの形状で失敗・警告・品質低下が起きるかを一覧にします。
変更の前後で結果を比べて、悪くなったケースだけを表示することもできます（回帰チェック）。

```bash
python benchmark/make_synthetic.py                              # 人工形状の STL を生成（ダウンロード不要）
python benchmark/run_bench.py benchmark/suites/synthetic.json -j 4
python benchmark/run_bench.py benchmark/suites/showcase.json -j 4  # 配管・ダクト・外部流れ（実行例。どれも PASS）
python benchmark/run_bench.py benchmark/suites/local.json       # 手元の実形状（無いファイルはスキップ）
python benchmark/run_bench.py benchmark/suites/large.json -j 1  # 約 100 万セル・回転ゾーン（速さの確認。1 ケースずつ）
python benchmark/run_bench.py benchmark/suites/scale.json -j 1 --only tilted_1m   # 高速化の確認（100 万 / 800 万セル）
python benchmark/mesh_hash.py CASE_A CASE_B                     # 2 つのメッシュがバイト単位で同じか
python benchmark/same_mesh.py BASE_TREE HEAD_TREE               # 2 つのチェックアウトのコードで同じメッシュになるか
python benchmark/stl_check.py some.stl                          # 入力 STL の診断だけ
```

結果は `benchmark/results/<suite>-<日時>/` に出ます（`summary.md`, `results.csv`, `results.json`,
ケースごとの `cases/<name>/` に `log.txt`、`overview.png`、`cut.png`、polyMesh）。
PASS したケースの polyMesh は削除します（`--keep` で残す）。画像は残ります。

## ファイル

| ファイル | 内容 |
|---|---|
| `run_bench.py` | スイートを実行して判定・集計する。1 ケース 1 プロセス（落ちても固まっても他のケースは続く）、タイムアウト付き |
| `render_mesh.py` | メッシュの画像: `overview.png`（壁面を 2 方向から、パッチごとに色分け）と `cut.png`（x 方向の中央より奥のセルだけを残した 3D 図 + x・y・z の中央断面）。checkMesh の基準を満たさないセル（体積 ≤ 0、非直交性 > 70°、歪度 > 4）は赤。`run_bench.py` が毎回呼ぶ（`--no-images` で省略）。単独でも `python benchmark/render_mesh.py <case>` で使える |
| `mesh_hash.py` | 書き出した polyMesh の各ファイルの SHA-256。2 つのケースが同じメッシュかを比べる（`--save` / `--check` で保存した値とも比べられる）。`run_bench.py` が毎回記録する。値はマシンごとに違う（CPU・NumPy・OS で丸めが変わる）ので、同じマシンで作ったものどうしを比べる |
| `check_openfoam.py` | 攪拌槽の例と `showcase` のメッシュを作り、OpenFOAM の `checkMesh` をかける（既定は Docker の `opencfd/openfoam-default:2512`、`--native` で手元の OpenFOAM）。`Mesh OK.` にならないケースが `checkmesh_known.json` にないと終了コード 1。CI（`.github/workflows/checkmesh.yml`）が毎回実行する |
| `same_mesh.py` | 2 つのチェックアウト（例: main とプルリクエスト）のコードで、攪拌槽の例と `showcase` を同じマシンで作って比べる。CI（`.github/workflows/same-mesh.yml`）がプルリクエストごとに実行する。メッシュを意図して変えるプルリクエストはラベル `mesh-change` を付けると、違いを報告したうえで通る |
| `stl_check.py` | 入力 STL の診断: 開いた辺、非多様体の辺、向きの不一致、潰れた三角形、重複した三角形、部品の数、オイラー標数、符号付き体積 |
| `make_synthetic.py` | 人工形状 25 個と実用的な形状 8 個（`showcase`）を `data/synthetic/` に生成 |
| `fetch_thingi10k.py` | Thingi10K から段階ごとに抜き出して `data/thingi10k/` と `suites/thingi10k.json` を作る（`pip install thingi10k` が必要。初回はデータセット全体をダウンロードする） |
| `suites/*.json` | スイート定義（STL、期待する結果、meshDict の上書き） |
| `baselines/*.json` | 比較の基準にする過去の結果（`--save-baseline` で更新） |

`data/` と `results/` は git の管理外です（第三者のデータにはそれぞれのライセンスがあるため）。

## 判定

| 状態 | 意味 |
|---|---|
| PASS | 正しいメッシュで、警告も品質フラグもなく、体積が STL の ±1% 以内（`--vol-tol`） |
| WARN | メッシュは正しいが、ログに警告がある（リーク、分かれた領域など）か、品質フラグ（非直交性 > 70°、歪度 > 4）があるか、体積が合わない |
| INVALID | メッシュは書き出されたが壊れている: 体積が 0 以下のセル、checkMesh の face pyramid が負、閉じていないセル、STL から離れた境界 |
| ERROR / CRASH / TIMEOUT | 例外 / プロセスの異常終了 / 時間切れ |

ケースごとの `expect`: `pass`（PASS のみ可）、`valid`（PASS か WARN）、`reject`（壊れた入力。ERROR か
WARN なら可、黙って PASS したり INVALID になったりしたら不可）。合わないものは verdict が `UNEXPECTED` になります。
face-triangle tets（checkMesh より厳しい tet 分解の判定）は記録だけして、`--strict` のときだけ WARN に数えます。

## よく使うオプション

```bash
run_bench.py SUITE --only "thin_*" --only cube          # ケース名で絞る（fnmatch）
run_bench.py SUITE --cells-across 48                    # 全ケースの解像度を上げる
run_bench.py SUITE --set addLayers='{"nLayers": 3}'     # 全ケースの meshDict を上書き（境界層つきで回す）
run_bench.py SUITE --baseline benchmark/baselines/synthetic.json   # 基準との差分（悪化があれば終了コード 1）
run_bench.py SUITE --save-baseline                      # 基準を更新
run_bench.py SUITE --baseline OLD/results.json --same-mesh   # メッシュが変わったら悪化として扱う（高速化の確認）
run_bench.py SUITE --baseline OLD/results.json --gate        # 品質ゲート（メッシュが変わってよい変更の確認）
```

### 基準との比較（`--baseline`）

`WORSE` で始まる行が悪化で、1 つでもあれば終了コード 1 になります。それ以外の行は参考情報です。

| 行 | 条件 |
|---|---|
| `WORSE ... PASS -> WARN` など | 判定が悪くなった |
| `WORSE ... faces skewness > 4 100 -> 586` | 歪度 > 4・非直交 > 70° の面、体積が 0 以下のセル、bad pyramid の数が、10% を超え、かつ 5 以上増えた（それ未満の増減は `more` / `fewer`） |
| `cells` | セル数が変わった |
| `minVol` | 最小セル体積が 1/10 未満になった |
| `slower` / `faster` | 時間が、30 秒を超えるケースでは 5%、2 秒を超えるケースでは 30% を超えて変わった |
| `slower <case> / <step>` | 1 秒以上かかる段階が 20% を超えて遅くなった（段階名の括弧内の件数は除いて突き合わせる） |
| `same` / `mesh` | メッシュのハッシュが同じ / 変わった（`--same-mesh` のときは変わったら `WORSE`）。基準と CPU・OS・NumPy の版が違うと比べない（`note` の行が出る） |

時間は他のケースと並列に走らせると 10〜30% ぶれるので、時間を比べるときは `-j 1` にします。
`summary.md` の末尾に、段階ごとの時間の表が出ます。

### 品質ゲート（`--gate`）

メッシュが変わってよい変更（ロードマップの v0.7.0 以降）を入れるかどうかの判定です。`--baseline` と一緒に使い、
**全ケースが次をすべて満たす**ときだけ合格です。満たさないケースは `WORSE <case>: gate: ...`、満たすケースは
`gate <case>: ok` と出ます。基準は変更前のコードを同じマシンで回した `results.json` にします（メッシュの細部は
マシンごとに違うため）。

| 項目 | 条件 |
|---|---|
| 非直交性 | 70° を超える面が 0（基準ですでにあるケースは、増えないこと） |
| 歪度 | 4 を超える面が増えない |
| 体積 | 0 以下のセルが 0 |
| セル数 | 基準の ±3% 以内 |
| 境界層 | 全厚の層が付いた壁の点の割合（ログの `wall points with full thickness`）が 1 ポイント以上下がらない |
| 特徴線 | STL の特徴線のずれが悪くならない: 最小セルの 0.2 倍より離れた点の割合が +0.5 ポイント以内、95 パーセンタイルが +0.02 hmin 以内 |

特徴線のずれは `tools/edge_check.py` と同じ測り方です（特徴線の上に最小セルの 1/4 ごとに点を置き、境界面の辺までの
距離を最小セルで割る。`mesh_metrics.py`）。結果の `featureEdges`・`layers` に記録され、`summary.md` の表にも出ます
（測るのは 1 ケース 1 秒未満。`--no-metrics` で省けます）。基準と同じく判定状態が悪くなったケース（`PASS -> WARN` など）も
`WORSE` になります。両方で拒否される入力（`dirty_open_box_strict` など）は判定状態の比較に任せます。

最小セルは、円柱ベースで小さくなったとき（ログの `smallest cell of the cylindrical base`）はその値を使います。
境界層や特徴線がないケースでは、`layers`・`featureEdges` が `null` と記録され、その項目は判定しません。
基準かこの実行のどちらかで測れていない項目（測定の失敗 `metricsError`、`--no-metrics`、この記録より前の `results.json`）は、
合格にせず `WORSE ... not measured` になります。その場合は基準を測り直してください。
基準で判定したケースがこの実行で `SKIP` になった（STL がない環境など）ときも、`WORSE ... skipped in this run` になります
（両方で `SKIP` のケースは判定しません）。

`baseCellSize` を指定しないケースは「外接箱の最長辺 / `cellsAcross`」（既定 24）で決めます。
Thingi10K などは単位や大きさがばらばらなので、この方法で揃えます。

## 人工形状スイート（`suites/synthetic.json`）

一つの形状で一つの壊れ方を狙っています。

| 種類 | ケース |
|---|---|
| きれいな基準 | `sphere`, `cube`, `cube_rotated`（格子に対して傾いた鋭いエッジ）, `cube_aligned`（面が八分木の格子面にぴったり乗る）, `cylinder`, `duct_patches`（ASCII・複数 solid） |
| 鋭い特徴 | `l_bracket`（凹エッジ）, `stepped_shaft`, `star_prism`（36° の凸の角）, `cone_tip`（尖った頂点）, `wedge_10deg` |
| トポロジー | `torus`（穴 1 つ）, `shell_nested`（入れ子の球の間）, `two_spheres`（離れた 2 つの流体） |
| 薄い形状 | `thin_channels`（幅 0.02 の流路）, `thin_walls`（厚さ 0.004 の固体の壁 < 最小セル）, `tapered_slot`（先端の幅が 0 になる溝）, `solid_spike` |
| 三角形の質 | `sliver_cylinder`（縦横比 約300 の細長い三角形） |
| 汚れた入力 | `dirty_flipped`, `dirty_mixed_orient`, `dirty_duplicates`, `dirty_edge_touch`, `dirty_open_box`, `dirty_overlap` |

## 実用的な形状のスイート（`suites/showcase.json`）

実際の流れの問題に近い形状で、入口・出口を別の solid（パッチ）にしてあります。壊れ方を探す人工形状スイートとは逆に、
**きれいにメッシュができる例**を集めたもので、実行例（`docs/examples.html#showcase`、画像は `tools/make_docs_images.py --only showcase`）
にも使っています。8 ケースとも PASS（非直交性 > 70°・歪度 > 4 の面が 0、体積の誤差 0.3% 以内）で、`expect` は `pass` です。

| ケース | 内容 | セル数 | 最大非直交 [°] | 最大歪度 |
|---|---|---|---|---|
| `pipe_elbow` | 円管の 90° エルボ（曲率半径 1.5 D） | 27,869 | 53.3 | 2.24 |
| `helical_coil` | 2.5 巻きのらせん管（コイル式熱交換器） | 98,359 | 62.1 | 2.90 |
| `venturi` | ベンチュリ管（のど 0.4 D、7° のディフューザー） | 24,484 | 57.9 | 2.50 |
| `orifice_pipe` | 鋭いエッジのオリフィス板（開口径 0.5 D） | 19,376 | 57.9 | 1.41 |
| `backward_step` | バックステップ流路（拡大比 2、側面は `symmetry`） | 19,440 | 25.2 | 0.33 |
| `t_junction` | 正方形ダクトの T 字合流（入口 2 つ） | 15,736 | 25.2 | 0.33 |
| `pin_fins` | ダクトを横切る 13 本のピンの千鳥配列（ピンだけレベル 3） | 63,287 | 39.2 | 1.59 |
| `body_in_tunnel` | 風洞の中の鈍頭物体（外部流れ。物体だけレベル 4） | 92,311 | 58.1 | 1.61 |

`helical_coil` は `baseCellSize` 0.025・0.0275 だと歪度 > 4 の面が 1 つ出る（0.02・0.0225・0.03 では出ない）ので、
0.0225 にしています。基準は `baselines/showcase.json`。

## 大きなケースのスイート（`suites/large.json`）

速さ優先の方針（[PLAN.md](PLAN.md)）で、変更の良し悪しを実際の使い方に近い大きさで確かめるためのスイートです。
`examples/stirred_tank` の攪拌槽（4 枚パドル翼＋シャフト）を使い、データのダウンロードは要りません。

| ケース | 内容 |
|---|---|
| `tank_rotor_1m` | 円筒の回転ゾーン、約 100 万セル |
| `tank_stl_rotor` | STL で与えた回転ゾーン |
| `tank_layers` | 全壁に境界層 3 層 |
| `tank_cylindrical` | 円筒ベースメッシュ |
| `tank_tilted` | 攪拌槽全体を傾け、格子に沿うエッジをなくした形（`python make_geometry.py --tilt`）。実際の形状と同じく凸エッジの厳密カットが働くので、その品質と時間の悪化を見張る |
| `tank_stator_1mm` | ローター・ステーター: 翼の外側 1 mm に固定リング、回転ゾーンの界面は隙間の中央（実際に計算したい問題に近い形） |
| `tank_stator_gap` | 同じ形で、`autoGapRefinement` の代わりに `gapRefinement`（流体が 2 mm より薄い所の壁をレベル 5 に）。隙間の判定の時間を見張る |
| `tank_stator_0p5mm` | 同、隙間 0.5 mm（約 200 万セル。15 GB のメモリでは足りず、今は ERROR になる。メモリ削減の目標） |

ステーターのケースには `probes` を付けています。翼の先端の正面で隙間を横切る線分を内側半分と外側半分に分け、
内側半分にセルがあって回転ゾーンに入っていること、外側半分にセルがあってゾーンに入っていないことを確かめます
（隙間がつぶれても体積の差はわずかで、リーク検査にも出ないため）。足りなければ WARN になります。

## 高速化の確認用スイート（`suites/scale.json`）

1000 万セルを 1 分で作る目標に向けて、高速化の変更を確かめるスイートです。
傾けた攪拌槽（`tank_tilted` と同じ形状と設定）の解像度だけを上げています。凸エッジの厳密カットと円筒の回転ゾーンが
働くので、公開データの中では実際の対象に一番近い形です。

| ケース | baseCellSize | セル数 | 時間（書き出しと品質チェックまで） | メモリ |
|---|---|---|---|---|
| `tilted_1m` | 0.0052 | 983,169 | 54.7 s | 6.8 GB |
| `tilted_8m` | 0.0024 | 8,256,407 | 339.7 s | 46.3 GB |

（2026-11-21、16 コア Ryzen AI MAX+ 395、Windows、Rust 拡張あり。基準は `baselines/scale.json`）

高速化の PR では、変更前と変更後を同じマシンで回し、`--baseline 変更前/results.json --same-mesh` でメッシュが
バイト単位で同じことを確かめます。`tilted_8m` は約 46 GB のメモリを使うので、最終確認のときだけ回します。

## 現在の結果（2026-11-13、入力 STL の自動修正を入れた後）

synthetic: **PASS 18 / WARN 6 / INVALID 1 / ERROR 1**（ERROR は `dirty_open_box_strict` で期待どおり）、
期待と違うもの 2（`cube_rotated` の歪度、`dirty_open_box`）。local: cyclone・stirred_tank は PASS、bunny は WARN
（歪度 > 4 が 3 面、開いた辺 3 本）。基準は `baselines/*.json`、修正の前の基準は `results/archive/` にある。

cutmesh が入力を自動で直したケースは、notes 欄に `repaired: ...` と出る（ログの `REPAIR:` の行）。
残っている課題と進め方は [PLAN.md](PLAN.md)、これまでの修正とその前後の結果は PLAN.md の「修正履歴」にある。

## 実形状データの入手先

| データ | 用途 | メモ |
|---|---|---|
| [Thingi10K](https://pypi.org/project/thingi10k/) | 量のテスト。メタデータ（閉じているか、自己交差、部品の数）で段階分けできる | `fetch_thingi10k.py`。モデルごとにライセンスが違う。PyPI の 1.2.0 は取り下げ済み |
| Stanford 3D Scanning Repository | 細かいディテール・規模（Dragon, Armadillo, Lucy） | 元のスキャンは穴あり。閉じた版を使うか穴埋めが必要 |
| AhmedML / WindsorML / DrivAerML | CFD 形状 | 外部流れなので「箱 − 車体」の領域を作る必要あり。CC-BY-SA（DrivAerML は公式ページで確認） |
| AAA-100（Zenodo） | 内部流れ・枝分かれした細い管 | 全モデルのオイラー標数が 2（閉曲面） |
