# cutmesh — Octree cut-cell mesher for OpenFOAM

[日本語](README.md) | **English**

cutmesh generates octree-based polyhedral meshes **inside** watertight STL geometry and writes OpenFOAM's
`constant/polyMesh` directly. It cuts octree cells against the STL to create wall faces.

The documentation site includes a manual and mesh images from the examples. [English documentation](https://mmer547.github.io/cutmesh/en/)

## Features

- **Meshing inside an STL.** Create a wall patch per solid name and choose refinement levels by patch, feature edge, and volume region.
- **Sharp edge preservation.** Bend cut faces along convex and concave feature lines and snap vertices onto the edges.
- **Thin geometry.** Detect and refine gaps and walls thinner than a cell to prevent blocked flow paths and leaks through missing walls.
  `autoGapRefinement` can choose refinement based on the width of each gap.
- **Boundary layers.** Add prismatic layers to wall patches, including on polyMesh files created by other tools.
- **Optional rotating zones.** Cut a cylinder defined by dimensions or STL **during meshing** so the cylinder surface becomes
  internal mesh faces. Export a `cellZone`, an oriented interface `faceZone`, and an `MRFProperties` template.
  For a sliding mesh (AMI), cutmesh can split the interface into two patches and write `dynamicMeshDict`.
- **Optional cylindrical base mesh.** Use concentric blocks around the rotation axis as the roots of the octree for rotating machines.
- **Few dependencies.** Requires Python 3.9+, NumPy, and SciPy, plus Matplotlib for cross-section plots.
  Mesh generation works without OpenFOAM. The optional Rust extension parallelizes nearly every meshing stage
  and makes the stirred tank example about 100× faster.

> The built-in mesh quality checks (`cutmesh/quality.py`) use the same formulas as `checkMesh`. CI runs OpenFOAM's own
> `checkMesh` (v2512) on the public examples only, so run `checkMesh` locally on meshes of your own geometry.

## Installation

**Wheel (recommended).** Download the wheel for your OS from the
[GitHub releases](https://github.com/mmer547/cutmesh/releases) (`cutmesh-<version>-cp39-abi3-<platform>.whl`, for Linux,
Windows, or macOS and any Python 3.9 or later) and install it. The Rust extension is built in. cutmesh is not on PyPI.

```bash
pip install cutmesh-1.0.2-cp39-abi3-win_amd64.whl     # the file name of the wheel you downloaded
cutmesh --version
```

**From source.** With a Rust toolchain (cargo), `pip install .` builds the Rust extension with `maturin`.

```bash
pip install -e .                      # Development install (builds the Rust extension too)
python tools/build_rust.py            # Rust build for running from the source tree without installing (see [Rust acceleration](docs/en/guide.md#rust-acceleration-optional) in the detailed guide)
```

To run from the source tree, use `python -m cutmesh` from the repository root or add it to `PYTHONPATH`. Without the Rust
extension, cutmesh falls back to slower Python/NumPy code.

## Usage

```bash
cutmesh meshDict.json [-case DIR] [-noCheck]    # python -m cutmesh also works
```

A minimal `meshDict.json` for meshing the inside of an STL follows.

```json
{
  "domain": "model.stl",
  "baseCellSize": 0.01,
  "surfaceRefinement": { "default": 2 },
  "patchTypes": { "inlet": "patch", "outlet": "patch" },
  "case": "case"
}
```

cutmesh writes `case/constant/polyMesh` and then logs the quality checks. Use `-noCheck` to skip them.

### Example (stirred tank)

`examples/stirred_tank` defines a rotating zone around a four-paddle impeller and shaft in a tank.

```bash
cd examples/stirred_tank
python make_geometry.py               # Generate the sample STL
cutmesh meshDict.json                 # → case/constant/polyMesh, case/constant/MRFProperties
python run.py                         # Also plot cross-sections → slice_z.png, slice_y.png
```

Remove `rotatingZones` to mesh the same geometry without a rotating zone. The other examples (sliding mesh, cylindrical base mesh, boundary layers, rotor–stator) are listed in the [detailed guide](docs/en/guide.md#example-files).

## Documentation

| Topic | Where |
|---|---|
| Manual: installation, usage, every meshDict key, rotating zones, sliding meshes, cylindrical base meshes, boundary layers, commands and options, known limitations | [Documentation site](https://mmer547.github.io/cutmesh/en/manual.html) |
| meshDict keys, searchable, with an input helper | [meshDict reference](https://mmer547.github.io/cutmesh/en/meshdict.html); the JSON schema [`docs/en/meshdict.schema.json`](docs/en/meshdict.schema.json) is the specification |
| Examples and mesh images | [Examples](https://mmer547.github.io/cutmesh/en/examples.html) |
| Details: example files, the meshDict table, each feature in depth, Rust acceleration and export format, the algorithm, validation results, diagnostic tools, input checks and STL repair, known limitations | [Detailed guide](docs/en/guide.md) |
| Changes by version | [CHANGELOG](CHANGELOG.en.md), [Releases](https://github.com/mmer547/cutmesh/releases) |
| meshDict compatibility: no keys removed, renamed, or given a new meaning within 1.x | [Manual: meshDict compatibility](https://mmer547.github.io/cutmesh/en/manual.html#compat) |
| Benchmarks and the quality gate | [`benchmark/README.md`](benchmark/README.md) |
| Plans | [Issue #68](https://github.com/mmer547/cutmesh/issues/68) |

## License

cutmesh is released under the MIT License (see [LICENSE](LICENSE)).

## Trademarks

cutmesh is an independent tool that writes meshes (`constant/polyMesh`) for OpenFOAM.

This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the
OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks.
