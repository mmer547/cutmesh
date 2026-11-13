# benchmark — cutmesh の対応力テスト

種類の違う STL をまとめて cutmesh に通し、どの形状で失敗・警告・品質低下が起きるかを一覧にします。
変更の前後で結果を比べて、悪くなったケースだけを表示することもできます（回帰チェック）。

```bash
python benchmark/make_synthetic.py                              # 人工形状の STL を生成（ダウンロード不要）
python benchmark/run_bench.py benchmark/suites/synthetic.json -j 4
python benchmark/run_bench.py benchmark/suites/local.json       # 手元の実形状（無いファイルはスキップ）
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
| `stl_check.py` | 入力 STL の診断: 開いた辺、非多様体の辺、向きの不一致、潰れた三角形、重複した三角形、部品の数、オイラー標数、符号付き体積 |
| `make_synthetic.py` | 人工形状 25 個を `data/synthetic/` に生成 |
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

特徴線（鋭角エッジ）の取れ具合も記録します（`edge > 0.2 hmin`）。STL の特徴線に沿って最小セルの 1/4 間隔で点を取り、
最も近いメッシュの境界辺までの距離が最小セルの 0.2 倍を超える点の割合です（`tools/edge_check.py` と同じ測り方）。
判定には使わず、基準との比較で 2 ポイント以上変わったときに `edges` の行で知らせます。

## よく使うオプション

```bash
run_bench.py SUITE --only "thin_*" --only cube          # ケース名で絞る（fnmatch）
run_bench.py SUITE --cells-across 48                    # 全ケースの解像度を上げる
run_bench.py SUITE --set addLayers='{"nLayers": 3}'     # 全ケースの meshDict を上書き（境界層つきで回す）
run_bench.py SUITE --baseline benchmark/baselines/synthetic.json   # 基準との差分（悪化があれば終了コード 1）
run_bench.py SUITE --save-baseline                      # 基準を更新
```

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
