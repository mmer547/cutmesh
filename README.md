# cutmesh — OpenFOAM 用の八分木カットセル・メッシャー

**日本語** | [English](README.en.md)

水密な STL の**内部**に八分木ベースの多面体メッシュを生成し、OpenFOAM の `constant/polyMesh` を
直接書き出すメッシャーです。八分木のセルを STL で直接切断（カットセル）して
壁面を作ります。

使い方のマニュアルと実行例のメッシュ画像は、ドキュメントのページ（`docs/`、GitHub Pages:
https://mmer547.github.io/cutmesh/ ）にあります。

## 特徴

- **STL 内部のメッシュ**: solid 名ごとに壁パッチを作り、パッチ・特徴線・体積領域ごとに細分化レベルを指定できる
- **鋭角エッジの保持**: 凸・凹の特徴線に沿ってカット面を折り曲げ、節点をエッジ上へ寄せる
- **薄い形状への対処**: セルより薄い隙間・壁を検出して細分化し、流体の閉塞や壁の消失（リーク）を防ぐ。
  隙間の幅に合わせた自動細分化（`autoGapRefinement`）もある
- **境界層**: 壁パッチにプリズム層を追加（cutmesh 以外で作った polyMesh にも使える）
- **回転ゾーン（任意）**: 円筒（寸法または STL で指定）を**メッシュ生成の一部として**切断し、
  円筒面そのものを内部面にする。`cellZone`・向き付きの界面 `faceZone`・`MRFProperties` の雛形を出力する。
  スライディングメッシュ（AMI）用に、界面を 2 つのパッチに分けて `dynamicMeshDict` を書くこともできる
- **円筒ベースメッシュ（任意）**: 回転機械向けに、八分木の根を軸まわりの同心ブロックにできる
- **依存が少ない**: Python 3.9+, numpy, scipy（断面図に matplotlib）。メッシュ生成に OpenFOAM は不要。
  Rust 拡張でメッシュ生成のほぼ全段階を並列化できる（任意、攪拌槽の例で約 100 倍速い）

> メッシュ品質は `checkMesh` と同じ式の自前チェック（`cutmesh/quality.py`）で確認しています。OpenFOAM 本体
> （v2512）の `checkMesh` は公開の例にだけ CI でかけているので、自分の形状では使用前に手元で `checkMesh` を実行してください。

## インストール

**wheel（おすすめ）**: [GitHub の Releases](https://github.com/mmer547/cutmesh/releases) から、自分の OS 用の
wheel（`cutmesh-<版>-cp39-abi3-<OS>.whl`。Linux・Windows・macOS、Python 3.9 以上で共通）をダウンロードしてインストールします。
Rust 拡張が入っているので、ビルドは要りません。PyPI には公開していません。

```bash
pip install cutmesh-1.0.2-cp39-abi3-win_amd64.whl     # ダウンロードした wheel のファイル名
cutmesh --version
```

**ソースから**: Rust ツールチェーン（cargo）があれば、`pip install .`（`maturin` でビルドされる）で Rust 拡張ごと入ります。

```bash
pip install -e .                      # 開発用（Rust 拡張もビルドされる）
python tools/build_rust.py            # インストールせずに、リポジトリ直下から使う場合の Rust 拡張のビルド（[詳細ガイド](docs/guide.md#高速化rust-拡張任意)の「高速化」）
```

インストールしない場合はリポジトリ直下から `python -m cutmesh` で実行するか、`PYTHONPATH` にリポジトリのディレクトリを通してください。
Rust 拡張がなくても Python/NumPy の処理で動きます（大幅に遅い）。

## 使い方

```bash
cutmesh meshDict.json [-case DIR] [-noCheck]    # python -m cutmesh でも同じ
```

最小限の `meshDict.json`（STL 内部だけのメッシュ）:

```json
{
  "domain": "model.stl",
  "baseCellSize": 0.01,
  "surfaceRefinement": { "default": 2 },
  "patchTypes": { "inlet": "patch", "outlet": "patch" },
  "case": "case"
}
```

`case/constant/polyMesh` が書き出され、続けて品質チェックの結果がログに出ます（`-noCheck` で省略）。

### 例題（攪拌槽）

`examples/stirred_tank` は、槽＋4枚パドル翼＋シャフトの形状に回転ゾーンを置いた例です。

```bash
cd examples/stirred_tank
python make_geometry.py               # サンプル STL を生成
cutmesh meshDict.json                 # → case/constant/polyMesh, case/constant/MRFProperties
python run.py                         # 断面図も欲しい場合 → slice_z.png, slice_y.png
```

`rotatingZones` を消せば、同じ形状で回転ゾーンのないメッシュになります。ほかの例（スライディングメッシュ、円筒ベースメッシュ、境界層、ローター・ステーター）の一覧は[詳細ガイド](docs/guide.md#例題のファイル)にあります。

## ドキュメント

| 内容 | 場所 |
|---|---|
| マニュアル（インストール、使い方、meshDict の全キー、回転ゾーン、スライディングメッシュ、円筒ベースメッシュ、境界層、コマンドとオプション、既知の制約） | [ドキュメントのサイト](https://mmer547.github.io/cutmesh/manual.html) |
| meshDict のキーの一覧（検索・入力の補助つき） | [meshDict リファレンス](https://mmer547.github.io/cutmesh/meshdict.html)、仕様の JSON スキーマ [`docs/meshdict.schema.json`](docs/meshdict.schema.json) |
| 実行例とメッシュの画像 | [実行例](https://mmer547.github.io/cutmesh/examples.html) |
| 詳しい説明（例題のファイル、meshDict の表、各機能の詳細、Rust 拡張と書き出し形式、アルゴリズム、検証結果、診断ツール、入力の検査と STL の修正、既知の制約） | [詳細ガイド](docs/guide.md) |
| 版ごとの変更 | [CHANGELOG](CHANGELOG.md)、[Releases](https://github.com/mmer547/cutmesh/releases) |
| meshDict の互換性（1.x のあいだキーの削除・改名・意味の変更をしない） | [マニュアルの「meshDict の互換性」](https://mmer547.github.io/cutmesh/manual.html#compat) |
| ベンチマークと品質ゲート | [`benchmark/README.md`](benchmark/README.md) |
| 今後の計画 | [issue #68](https://github.com/mmer547/cutmesh/issues/68) |

## ライセンス

MIT ライセンスです（[LICENSE](LICENSE)）。

## 商標

cutmesh は OpenFOAM の一部ではなく、OpenFOAM 用のメッシュ（`constant/polyMesh`）を書き出す独立したツールです。

This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the
OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks.
