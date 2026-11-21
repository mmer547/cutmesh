# AGENTS.md: このリポジトリで作業するエージェント向けの決まりごと

cutmesh は、閉じた STL の内側に OpenFOAM の polyMesh（多面体セル、回転ゾーン、境界層）を作るメッシャーです。
Python（`cutmesh/`）が本体で、重い処理は Rust 拡張（`rust/`、pyo3 + rayon、`cutmesh._core`）が受け持ちます。
複数のエージェント（Claude、Codex）が並行して作業します。人への説明・PR の本文は日本語で書きます。

## 環境・ビルド・テスト

```bash
pip install -e .                       # numpy, scipy
python tools/build_rust.py             # cargo build --release + cutmesh/_core.pyd（.so）へコピー
python -m unittest discover -s tests  # 全テスト（Rust と Python の一致テストを含む）
python -m cutmesh meshDict.json        # 1 ケースを実行
CUTMESH_NO_RUST=1 python -m cutmesh meshDict.json   # Python だけで実行（比較用）
```

- `_core.pyd` は実行中のプロセスがロックするので、メッシュ生成を走らせている間はコピーできません。
- 既存ファイルの改行コード（CRLF / LF）は変えません。改行だけの差分を出さないよう `git diff --stat` で確かめます。

## いまの最優先: 速さ

目標は **1000 万セルを 1 分**（16 コア、96 GB）。方針は「速さ優先、形状への忠実さは二の次」です
（[benchmark/PLAN.md](benchmark/PLAN.md)）。ただし壁越しのリークや体積 0 以下のセルなど、計算が破綻する不具合は別扱いで直します。

### 高速化の変更で守ること

1. **出力メッシュを変えない。** 高速化の PR は、変更前とバイト単位で同じ polyMesh を出すことを確かめてから出します。
   変える場合は、PR の題と本文でそう明記し、セル数と品質（歪度 > 4、非直交 > 70° の面の数）の前後を載せます。
2. **確かめ方**（同じマシンで、変更前と変更後を回す）:
   ```bash
   # 変更前（main をビルドした状態）と変更後で、同じケースを回す
   python benchmark/run_bench.py benchmark/suites/large.json --only tank_tilted -j 1 --no-images --keep --out benchmark/results/before
   python benchmark/run_bench.py benchmark/suites/large.json --only tank_tilted -j 1 --no-images --keep --out benchmark/results/after \
       --baseline benchmark/results/before/results.json
   # polyMesh の全ファイルを比べる（何も表示されず終了コード 0 なら同じメッシュ）
   git diff --no-index --stat benchmark/results/before/cases/tank_tilted/case/constant/polyMesh \
       benchmark/results/after/cases/tank_tilted/case/constant/polyMesh
   ```
   `tank_tilted`（約 32 万セル）は凸エッジの厳密カットと回転ゾーンが働く、公開データで実際の対象に一番近い形です。
   計算結果は CPU・OS・NumPy の版で丸めが変わりうるので、変更前と変更後は同じマシンで回します。
   100 万・800 万セルに上げた確認用スイートと、メッシュのハッシュ比較（`--same-mesh`）は別の PR で足します。
3. **Python の処理を代替経路として残す。** Rust に移した処理は、`_core` が無いときの Python 版を残し、
   両者の結果が一致するテストを `tests/test_fast_paths.py` などに足します。
4. **浮動小数点の計算順を合わせる。** メッシュを変えないために、Rust 側は NumPy と同じ順で足します。
   - `np.einsum("ij,ij->i")`（3 要素）は `(a0*b0 + a2*b2) + a1*b1`。
   - `np.linalg.norm(axis=1)` は `sqrt((x0² + x1²) + x2²)`。
   - `(n,1,3) @ (n,3,1)` の行列積は FMA の連鎖。
   - 1 次元の `sum` は pairwise。axis=0 の和は先頭から順に足す。
   - Python の `set` を回す順で結果が決まる所は、`merge.rs` の `PySet` が CPython 3.11 の順を再現しています。
   - これらは `tests/test_fast_paths.py` の `NumpyOrder` で固定しています。並列化で足す順が変わる処理は、
     メッシュが変わることを前提に別の PR にします。
5. **時間は単独で測る**（`-j 1`、ほかの重い処理を同時に走らせない）。並列に走らせると 10〜30% ぶれます。
6. `log.txt` の段階ごとの時間（`summary.md` の末尾の表）を、PR に前後で載せます。

## 並行作業の決まり

- 1 つの作業 = 1 つのブランチ（または worktree）= 1 つの PR。main に直接コミットしません。
- 同じファイルを 2 つのエージェントが同時に書き換えないよう、下の分担表で担当を決めます。担当外のファイルに
  手を入れる必要が出たら、変更を最小にして PR の本文に書きます（`rust/src/lib.rs` の登録行など）。
- PR はもう一方のエージェントがレビューします。push・PR 作成・マージは人の了承を得てから。

### 分担（2026-11-21〜）

| 段階 | 内容 | 担当 | 主なファイル |
|---|---|---|---|
| A | カット片を CSR 配列のまま、カット → 統合 → 組み立て → パッチまで Rust で通す | Claude | `cutmesh/mesher.py`, `rust/src/dcut.rs`, `rust/src/merge.rs`, `rust/src/cut.rs` |
| B1 | 品質チェック（`quality.py`）を Rust + rayon へ | Codex | `cutmesh/quality.py`, `rust/src/quality.rs`（新規） |
| B2 | geometry 段階（gap のサンプルの重複、TriMesh 検索のまとめ問い合わせ） | Codex | `cutmesh/geometry.py`, `rust/src/lib.rs` の TriMesh |
| C | octree と snapping の並列化 | A・B の後に決める | `cutmesh/octree.py`, `rust/src/octree.rs` |

## してはいけないこと

- リポジトリの管理外にあるデータ（`.git/info/exclude` で除外されたフォルダ、ユーザーの非公開ケース）を
  読んだり、コミット・PR・Issue・外部サービスに出したりしない。公開のベンチマーク（`benchmark/`、`examples/`）を使う。
- 速さのためにテストを消したり、しきい値を緩めたりしない。
