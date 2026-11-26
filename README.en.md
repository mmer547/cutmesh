# cutmesh — Octree cut-cell mesher for OpenFOAM

[日本語](README.md) | **English**

cutmesh generates octree-based polyhedral meshes **inside** watertight STL geometry and writes OpenFOAM's
`constant/polyMesh` directly. It cuts octree cells against the STL to create wall faces. The mesher is a prototype under development.

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

```bash
pip install -e .                      # Offline: use --no-build-isolation (requires wheel)
python tools/build_rust.py            # Optional: build the Rust extension (see Rust acceleration below)
```

To run from the source tree, use the repository root as the working directory or add it to `PYTHONPATH`.

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

| File | Description |
|---|---|
| `meshDict.json` | Cartesian base grid with a rotating zone defined by dimensions |
| `meshDict_coarseRotor.json` | Coarser mesh inside the rotating zone |
| `meshDict_stlRotor.json` | Rotating zone defined by an STL |
| `meshDict_ami.json` | Sliding mesh (AMI) with a pair of `cyclicAMI` interface patches and a `dynamicMeshDict` |
| `meshDict_cylindrical.json` | Cylindrical base mesh |
| `meshDict_layers.json` | 3 boundary layers on all walls |
| `meshDict_stator_1mm.json` | Rotor–stator with a stationary ring 1 mm outside the blades. `python make_geometry.py --stator 1` generates `tank_stator_1mm.stl`. The interface lies halfway across the gap. About 600,000 cells and 10 seconds. |
| `meshDict_stator.json` | Rotor–stator with a 0.5 mm gap (`--stator 0.5`). The smallest cells (0.195 mm) line the entire interface, producing about 2 million cells (about 30 seconds and 3.5 GB of memory). |

Remove `rotatingZones` to mesh the same geometry without a rotating zone.

## meshDict.json

| Key | Description | Default |
|---|---|---|
| `domain` | Watertight STL of the fluid domain. Multiple solids are supported and their names become patch names. | Required |
| `baseCellSize` | Level 0 cell edge length [m] | Required |
| `surfaceRefinement` | Solid name → refinement level. Set `default` for other solids. | `{}` |
| `featureLevel` | Refinement level near feature edges (`null` uses the maximum surface level) | `null` |
| `gapRefinement` | With `{"thickness": t, "level": L}`, refine walls to `L` where fluid is thinner than `t`, as at sharp slot tips or in narrow channels. This helps retain fluid thinner than a cell. A first look samples every wall every `2t` (counting gaps narrower than `3t`); only the triangles it finds and their neighbours are then tested at the spacing of level `L`. | None |
| `autoGapRefinement` | With `{"cellsAcross": n, "maxLevel": L}` (or `true` for defaults), measure the gap width w along the wall normal to the opposing wall. Refine the wall to fit `n` cells in the gap (`baseCellSize / 2^level ≤ w / n`, up to level `L`). This helps preserve grooves, slits, and narrow gaps without a prescribed thickness. Faces whose surface level already fits `n` cells are skipped. The log counts locations that still fall short at the maximum level. Each wall is first sampled at half its cell size, and only triangles with narrow gaps are resampled at the finest spacing. | Off (`n` = 3, `L` = maximum surface level + 2) |
| `featureAngle` | Angle threshold for detecting feature edges [deg] | 30 |
| `bufferLayers` | Number of buffer cells per level transition | 2 |
| `snapTolerance` | Vertex snapping distance, as a fraction of the cell edge length. Helps prevent sliver cells. | 0.3 |
| `featureSnapTolerance` | Vertex snapping distance to sharp STL edges and cylinder rims, as a fraction of the cell edge length | 0.5 |
| `featureSnapMax` | Experimental. Snap to feature edges within this distance and undo snaps that invalidate octree cells. Some cut cells can invert, so the default is the same as `featureSnapTolerance`. | 0.5 |
| `concaveSnapMax` | At sharp concave edges (solid corners protruding into the fluid), snap solid-side vertices onto the edge within this fraction of the cell edge length. Undo snaps that invalidate adjacent cells or collapse face corners. At or below `featureSnapMax`, this is disabled. | 0.75 |
| `mergeRatio` | Merge cut cells below this fraction of the original hexahedron volume into a neighbor. | 0.25 |
| `mergeMaxNonOrtho` | Maximum post-merge non-orthogonality angle [deg]. Skip the merge if no neighbor meets this limit. | 65 |
| `mergeWallFaces` | Join nearly coplanar, edge-sharing wall-face fragments of a cell into one face. See step 6 of [Algorithm](#algorithm). | true |
| `hangingSnap` | Treatment of hanging vertices that cannot reach the surface along an edge or face. `"free"` projects them directly to the surface. `"constrained"` leaves them in place and treats them as φ=0. | `"free"` |
| `fillThinGaps` | Number of extra splits, each one level beyond the maximum, for cells that contain fluid but have no corners in fluid. This captures thin gaps between grid planes. Set to 0 to disable. Cartesian base only. | 2 |
| `fillThinWalls` | Number of extra splits, each one level beyond the maximum, when both sides of a thin wall cross the cell but no corner is in the solid. This guards against a missing wall joining the fluid on both sides. Set to 0 to disable. Cartesian base only. | 3 |
| `thinCutCells` | Number of splits for wall cut cells with fluid fraction below `thinCutFraction` and an adjacent finer cell. Moves the level transition away from the wall to avoid high non-orthogonality when cell centers lie close to the wall. Set to 0 to disable. Cartesian base only. | 2 |
| `thinCutFraction` | Fluid volume fraction threshold for `thinCutCells`, estimated by interpolating signed distances at octree cell corners | 0.35 |
| `checkLeaks` | Check for fluid cells that connect through a solid wall and log warnings with locations. | true |
| `strictInput` | Stop with an error if the STL has open or non-manifold edges. See [Input STL repair and checks](#input-stl-repair-and-checks). | false |
| `volumeCheckTolerance` | Warn if the final mesh volume differs from the signed STL volume by more than this fraction. | 0.05 |
| `minRegionVolume` | Remove disconnected fluid regions below this fraction of the total volume, such as fragments at groove tips thinner than a cell. Larger regions produce a warning. | 1e-3 |
| `refinementRegions` | Volume refinement regions `[{"cylinder": {...}, "level": n}]` | `[]` |
| `featureCaps` | Align cut faces with sharp convex feature edges. Use `"domain"`, `"zone"` for rotating zones only, `"both"`, or `false`. | `"domain"` |
| `exactFeatureCut` | Cut cells near sharp convex edges directly with STL planes to keep off-grid edges straight. Works with `featureCaps` set to `"domain"` or `"both"`. See step 4 of [Algorithm](#algorithm). | true |
| `patchTypes` | Patch name → type (`wall`, `patch`, etc.) | All `wall` |
| `rotatingZones` | List of [rotating zones](#rotating-zones-rotatingzones-optional). Omit for a mesh without zones. | `[]` |
| `writeMRFProperties` | Write a `constant/MRFProperties` template when rotating zones are present. | true |
| `writeFormat` | [polyMesh export format](#export-format-writeformat), `"binary"` or `"ascii"` | `"binary"` |
| `openfoam` | [Sliding-mesh dialect](#sliding-meshes-ami-cutmeshamipy) for zones with `"ami": true`. Use `"com"` for openfoam.com / ESI, or `"org10"`, `"org11"`, or `"org12"` for openfoam.org. | `"com"` |
| `writeDynamicMeshDict` | Write `constant/dynamicMeshDict` with the rotation settings when a zone has `"ami": true`. | true |
| `amiExactRims` | Make a sliding-mesh interface follow the top and bottom cylinder rims exactly by splitting cells outside the rim in two. | true |
| `case` | Output case directory | `.` |
| `padding` | Number of base cells added around the STL bounding box | 1 |
| `addLayers` | [Boundary-layer settings](#boundary-layers-cutmeshlayerspy). Omit for a mesh without layers. | None |
| `baseMesh` | [Base mesh settings](#cylindrical-base-mesh-cutmeshcylbasepy). Omit for a Cartesian grid. | Cartesian grid |

## Rotating zones (`rotatingZones`, optional)

Define rotating regions for MRF or sliding meshes (AMI) with cylinders. The octree is cut against each cylinder during meshing
so that the cylinder surface becomes internal mesh faces.

```json
{ "name": "rotor",
  "cylinder": { "point1": [0,0,0.06], "point2": [0,0,0.14], "radius": 0.07 },
  "surfaceLevel": 2, "volumeLevel": 2, "omega": 31.416, "nonRotatingPatches": [] }
```

- Outputs include `cellZones` for rotating regions, oriented `faceZones` for the cylinder interfaces (`<name>_interface`),
  and a `constant/MRFProperties` template. Set `"writeMRFProperties": false` to omit the template.
- The interface follows the cylinder even in coarse cells, so **it needs no extra refinement**. This can save cells compared
  with refining around a faceZone to capture the interface in snappyHexMesh.

To use a closed STL, replace `"cylinder"` with `"stl": "rotor.stl", "origin": [...], "axis": [...]`.
See `examples/stirred_tank/meshDict_stlRotor.json`. Multiple zones are supported if they do not overlap.

`"wallProximity": { "distance": 0.001, "level": 4 }` optionally refines only the part of the zone surface within `distance`
of the STL, up to `level`. Placing an interface within one cell of a wall, as with an MRF cylinder close to a rotor tip,
requires cells smaller than the clearance. Increasing `surfaceLevel` refines the entire zone surface and adds many cells.
`wallProximity` keeps refinement local. Add `"bufferLayers": 1` to narrow this refinement transition and further reduce cell count.

`"ami": true` optionally splits the interface into two [sliding-mesh patches](#sliding-meshes-ami-cutmeshamipy).

`"conforming": false` optionally selects cells for the cellZone by whether their centers lie inside the cylinder.
This is analogous to MRF zone selection in snappyHexMesh. MRF only needs the cellZone, so this option works when the
interface cannot conform to the cylinder in a narrow gap. AMI requires a conforming zone. This non-conforming interface
does not constrain boundary-layer movement because its faceZone is rebuilt from the cellZone after layer insertion.

## Sliding meshes (AMI, `cutmesh/ami.py`)

Set `"ami": true` on a rotating zone to split the internal faces of `<name>_interface` into two patches, one owned by
rotating cells and one by stationary cells. cutmesh duplicates the interface points on the rotating side, as in
OpenFOAM's `createBaffles` followed by `mergeOrSplitBaffles -split` or `splitBaffles`. It adds any boundary layers before
splitting and writes `constant/dynamicMeshDict` for rigid-body rotation of the cellZone. AMI zones are omitted from
`MRFProperties`. Select the dialect for your OpenFOAM distribution with `openfoam`.

| `openfoam` | Patches (rotating / stationary) | dynamicMeshDict | OpenFOAM step |
|---|---|---|---|
| `"com"` (default) | `<name>_AMI1` / `<name>_AMI2` of type `cyclicAMI` (`neighbourPatch`, `transform noOrdering`) | `dynamicMotionSolverFvMesh` + `solidBody` | Use directly |
| `"org10"`, `"org11"`, `"org12"` | `<name>_nonCouple1` / `<name>_nonCouple2` of type `patch` | `mover { type motionSolver; ... }` | Run `createNonConformalCouples -overwrite <name>_nonCouple1 <name>_nonCouple2` (also shown in the log). |

- The output follows `interFoam/RAS/mixerVesselAMI` for openfoam.com and `mixerVessel2D` for openfoam.org 10–12.
  `examples/stirred_tank/meshDict_ami.json` (about 100,000 cells with 90° of laminar rotation) completed in openfoam.com
  v2606 `pimpleFoam` and OpenFOAM-13 `foamRun` using the `"org12"` dialect. In OpenFOAM-12, both this mesh and a
  rimless cylindrical zone halted with a floating-point exception in `cyclicFvPatch::makeWeights` when rotation began.
  The same mesh runs in OpenFOAM-13, pointing to an OpenFOAM-12 issue.
- Cut cells wrapping around the top or bottom rim of a cylinder can have bent interface faces. OpenFOAM uses face centers,
  effectively chamfering the rim and causing the two sides to diverge as the zone rotates. The cyclicAMI weights
  dropped to 0.18 in this case. `amiExactRims` (default true) inserts rim points on crossing edges and splits outer cells
  in two along one plane of the bent face so the interface follows the rim. Weights were 1.0000 before rotation and
  0.78–1.01 during rotation. cutmesh skips and logs splits that would produce poor-quality cells.
- The interface must be a surface of revolution around the rotation axis. Cylinders satisfy this. For an STL-defined zone,
  cutmesh rotates interface faces by 7° and checks whether their distance from the zone surface changes. It warns if
  the STL is not a surface of revolution or if `origin` or `axis` is misaligned.
- AMI requires conforming zones. The openfoam.org development dialect (`mover { type solidBody; ... }`) is unsupported.
- For multiple zones, cutmesh uses `multiSolidBodyMotionSolver` for openfoam.com and `motionSolverList` for openfoam.org.

An existing mesh with faceZone `<name>_interface` and cellZone `<name>` can be converted. The command reads rotation settings from
`constant/MRFProperties` and removes the converted zone entries so the rotating mesh has no overlapping MRF definition.
Meshing with `"ami": true` also removes entries from prior runs.

```bash
python -m cutmesh.ami CASE [-zone NAME ...] [-openfoam com|org10|org11|org12] [-out CASE2] [-noDynamicMeshDict] [-ascii | -binary]
```

## Cylindrical base mesh (`cutmesh/cylbase.py`)

For rotating machinery, use concentric blocks around the rotation axis as the octree roots. cutmesh lays out the central
square, ring radii, and division counts automatically. A minimal configuration follows.

```json
"baseMesh": { "type": "cylindrical", "origin": [0, 0, 0], "axis": [0, 0, 1] }
```

| Key | Description | Default |
|---|---|---|
| `origin`, `axis` | Rotation axis (origin and direction) | `[0,0,0]`, `[0,0,1]` |
| `zRange` | Axial range `[z0, z1]` measured from `origin` | STL extent ± `baseCellSize` |
| `nz` / `dz` | Number or width of axial divisions (shared by all blocks) | `dz = baseCellSize` |
| `rOuter` | Outer radius of the outermost block in the automatic layout | STL radius + `2/3 × cellSize` |
| `cellSize` | Target radial and circumferential cell width for the automatic layout | Smaller of `baseCellSize` and `dz` |
| `core`, `blocks` | Supply both keys for manual layout. Omit both for automatic layout. | Automatic layout |
| `maxSizeRatio` | Maximum cell size ratio across block boundaries. The larger cells are refined when the ratio is exceeded. | 2.5 |

**Automatic layout** (omit `core` and `blocks`)

- A small 2×2 central square has a half-width of 0.5–0.95 times `cellSize`. An O-grid ring with 8 circumferential
  divisions connects it to a circle of twice that radius. Beyond this, concentric rings grow by about 1.5625× in radius until `rOuter`.
- Each ring's circumferential division count `nTheta` is a multiple of 4 chosen to make the width near mid-ring close
  to `cellSize`. Radial width is at most `1.15 × cellSize`. Every edge is at most 1.3 times `baseCellSize`, avoiding extra refinement levels.
- Only the 4 central cells and those in the first ring lack circumferential alignment. Where a solid shaft or hub lies on the axis,
  most of these cells fall inside it.
- cutmesh finds coaxial cylinders in the domain and zone STLs by searching for vertices at a common radius (shafts, hubs, tank walls,
  and blade-tip circles, for example). It also checks coaxial rotating or refinement cylinders. Radial vertices are positioned so each
  cylinder surface falls 1/3 of the way through a cell. When refinement halves the cell, the surface remains at 1/3 or 2/3 of the
  new cell, avoiding thin cut cells. The central square is sized to keep a similar separation between the inner cylinder (shaft)
  and the midpoint of its edge.
- The log shows the layout and detected radii. Reuse the `"core": ..., "blocks": [...]` entries as a starting point for manual
  layout. Blocks with nonuniform spacing have `nr + 1` radial vertices in `radii`.

**Manual layout**

```json
"core": { "halfWidth": 0.0375, "n": 6 },
"blocks": [
  { "rOuter": 0.06, "nr": 2 },
  { "rOuter": 0.10, "nr": 3, "nTheta": 38 },
  { "rOuter": 0.155, "nr": 4 }
]
```

- `core` is the central square block, with half-width `halfWidth` and `n` divisions on each side. `blocks` lists the concentric blocks
  outward, each with outer radius `rOuter`, radial divisions `nr`, and circumferential divisions `nTheta`.
- The first block is an O-grid connecting the square to a circle, with `nTheta` fixed at `4 × core.n`.
- Later blocks can use any `nTheta`, even if the adjacent counts are not integer multiples. If omitted, the value makes the
  circumferential and radial widths approximately equal at the mean radius.
- The optional `radii` lists `nr + 1` radial vertices from the inside out. The default is uniform spacing.
- Set the outermost `rOuter` greater than the STL radius. A smaller value triggers a warning and clips the STL.

**Both layouts**

- At square or cylindrical block boundaries, cutmesh overlays the faces from both sides to form shared faces and inserts
  the edge vertices into neighboring faces. The resulting polyhedral mesh conforms for arbitrary division ratios without AMI.
- Surface, feature, zone, and region refinement work as usual. Because base cells vary in size, level L represents a target
  size of `baseCellSize / 2^L`. cutmesh splits until the longest edge is at most 1.3 times that target. Refinement halves all
  3 dimensions, so base cells narrower than the axial width `dz` can become too fine in cross-section. For manual layouts,
  choose divisions that give near-cubic cells, as the automatic layout does.
- The automatic layout in `examples/stirred_tank/meshDict_cylindrical.json` has 137,000 cells, max non-orthogonality 54°,
  max skewness 2.8, and 0 negative face pyramids.

## Boundary layers (`cutmesh/layers.py`)

Add prismatic layers to wall patches. This runs as a separate post-processing step that takes a polyMesh as input.

```bash
# Add "addLayers": {...} to meshDict.json to add layers automatically after meshing
python -m cutmesh meshDict.json
# Add 3 layers on all walls → case_layers (about 160,000 cells, 98% of wall vertices reach target thickness)
cutmesh examples/stirred_tank/meshDict_layers.json
# Add layers to an existing case, including meshes from other tools (ASCII, or binary with faceCompactList faces)
python -m cutmesh.layers CASE [layerDict.json] [-out CASE2] [-noCheck] [-ascii | -binary]
```

| Key | Description | Default |
|---|---|---|
| `patches` | List of patches to receive layers | All patches of type `wall` |
| `nLayers` | Number of layers | 3 |
| `expansionRatio` | Ratio of successive layer thicknesses away from the wall | 1.2 |
| `finalLayerThickness` | Thickness of the innermost layer, as a fraction of the local cell size if `relativeSizes` is enabled | 0.3 |
| `relativeSizes` | Specify thickness relative to the local cell size (the largest dimension of the wall cell). | true |
| `minThickness` | Omit layers below this total thickness. | 0.05 |
| `minNormalCos` | Omit layers where the cosine between vertex and face normals is below this threshold, at sharp convex edges. | 0.25 |
| `maxNonOrtho` | Treat cells above this non-orthogonality angle [deg] as invalid. Existing internal faces may retain their original angle plus 2°. | 65 |
| `maxSkewness` | Treat cells adjacent to faces above this skewness limit (using the checkMesh definition) as invalid. Existing faces may retain their original skewness plus 0.2. `null` disables the check. | 4 |
| `wallNonOrthMargin` | Allow the original wall face, now an internal face, up to its boundary-face angle plus this margin [deg]. Coverage improves, but faces above 70° can appear. | None |
| `maxThicknessSlope` | Maximum change in total thickness per unit length along the wall, so layer ends taper gradually | 0.5 |
| `cornerStretch` | Move corner vertices by T/cos to maintain thickness T at corners. | true |
| `maxGapFraction` | Limit total thickness to this fraction of the distance to the opposing wall to prevent layers on the two sides from colliding. Set to 0 to disable. | 0.3 |
| `mergeWallPoints` | Before adding layers, merge wall vertices closer than this value times the local wall edge length. Set to 0 to disable. | 0.1 |
| `concaveSlide` | At concave corners, slide vertices that would intersect the opposite wall's layers along their own wall. A vertex at distance s0 from the other wall moves to c + k·s0, where c is the other wall's layer thickness and k is this setting. `null` disables it. | 0.5 |
| `nRings` / `nSmooth` | Number of edges from the wall over which interior vertices are displaced and number of Laplacian smoothing iterations | 6 / 60 |
| `nIter` / `nCollapseIter` | Iterations that halve thickness at invalid cells and subsequent iterations that suppress layers | 8 / 60 |

Use `"addLayers": {}` to add layers with defaults. When a gap is about as wide as a wall cell, the default `finalLayerThickness`
of 0.3 (total thickness about 0.76 times the wall cell size) often results in thinner layers. A value around 0.15
(about 0.38 times the cell size in total) lets many more vertices reach the target thickness.

0. Cuts can leave nearly coincident vertices along concave edges, such as an octree vertex snapped to the edge and a nearby cut
   point. They may be separated by a few thousandths to a few hundredths of the cell size. The direction of these near-zero
   edges is poorly defined, and the corresponding layer side faces or narrow wall faces can flip when a layer is inserted.
   This stops layers along the edge. cutmesh merges vertices closer than `mergeWallPoints` times the local wall edge length
   into the vertex belonging to more wall faces (the corner vertex). It removes faces left with fewer than 3 vertices
   and excludes vertices on other patches or faceZones from merging. If a merge would make a face touch itself, leave a cell
   with fewer than 4 faces, or invalidate a cell (non-positive volume, less than half its original volume, or a non-positive
   face pyramid), cutmesh keeps those vertices and retests the remaining merges. Thin cells made up entirely of short wall
   edges can collapse during this process.
1. Move wall vertices inward by the total layer thickness T along the area-weighted vertex normal. At edges and corners,
   use the position at distance T from every incident plane, the corner of the offset surfaces. For two planes the offset
   is T/cos(θ/2) along the bisector. An area-weighted normal at a corner would bias the motion toward the larger face.
   At a concave corner, moving one wall vertex along its normal can carry it past the corner vertex and invert the prism
   between them. Vertices closer to the other wall plane than c + k·s0 slide away along their own wall, where c is the other
   wall's layer thickness, s0 is the original distance, and k is `concaveSlide`. With k < 1, vertices within c/(1−k) of the
   corner crowd into a fan while keeping their order. Laplacian smoothing weighted by inverse edge length propagates the
   displacement to interior vertices within `nRings` edges. It propagates only along edges, so it stays on the same side of
   a thin wall and thin cut cells near that wall move almost rigidly.
2. Constrain vertices on unlayered patches (such as inlets or outlets) or faceZones (rotating-zone interfaces) to move along
   those surfaces. Project the displacement of shared wall vertices into the constraint surface. The interface stays
   cylindrical and layer side faces on the interface join the faceZone.
3. Between the original wall and its displaced position, insert `nLayers` prisms for each wall face. The displaced original
   mesh face becomes the internal face between the prisms and the original cell.
4. Check volume, face pyramids, and non-orthogonality. Where invalid cells appear, halve the thickness at the corresponding
   wall vertices. If cells are still invalid, set the thickness to 0, collapsing all layer points there to one vertex and
   ending the layers with wedge-shaped cells, as in snappyHexMesh. Use `maxThicknessSlope` to keep thickness changes gradual.

The log reports the wall vertices at the target thickness, with thinner layers, and with no layers.

Coverage is limited mainly by the cut-cell centroid's offset from the wall-face normal, measured by the angle between the
normal and the vector from cell centroid to face center. The face is originally a boundary face and exempt from the
checkMesh non-orthogonality check. Once layers make it internal, the angle becomes approximately its non-orthogonality.
The layer stops where it exceeds `maxNonOrtho`.

## Rust acceleration (optional)

The Rust extension in `rust/` covers nearly every meshing stage. It uses a BVH for geometry queries (nearest points,
inside/outside tests, signed distances) and Rayon for parallelism. It also handles octree refinement and balancing,
vertex snapping, STL and rotating-zone cuts, merging, assembly, leak and quality checks, and boundary-layer smoothing.
With the Rust toolchain (cargo) installed, the build installs `cutmesh/_core.pyd` (`.so` on Linux and macOS).
The Python/NumPy fallback produces the same results.

```bash
python tools/build_rust.py            # Run cargo build --release and copy into the package
CUTMESH_NO_RUST=1 python -m cutmesh meshDict.json   # Force the Python path for comparison
python -m unittest discover -s tests  # Check that Rust and Python agree
```

| Example | Python | With Rust extension |
|---|---|---|
| examples/stirred_tank (`meshDict.json`, about 100,000 cells) | 142 s | 1.0 s |
| Stirred tank with layers (`meshDict_layers.json`) | 148 s | 2.6 s |
| Stirred tank with a 1 mm gap (`meshDict_stator_1mm.json`, about 600,000 cells) | 2080 s (about 35 min) | 7.3 s |

Measured with v0.6.0 (2026-11-26) on a 16-core machine, including quality checks. A real-world geometry of about 8.5 million
cells takes about 70 seconds and about 13 GB of memory. In v0.7.0 the public benchmark `tilted_1m` (about 1 million cells)
went from 28.3 to about 20 seconds on a 4-core machine. Its mesh changes slightly and every benchmark case passes the
quality gate (see `benchmark/README.md`).

### Export format (`writeFormat`)

cutmesh writes polyMesh in binary by default, using the OpenFOAM `writeFormat binary` format. The large `points`, `faces`,
`owner`, and `neighbour` files are binary. The smaller `boundary`, `cellZones`, and `faceZones` files are ASCII. Each file's
header declares its format, so these formats can be mixed. Set `"writeFormat": "ascii"` to inspect the files with a text
editor or pass them to an ASCII-only tool. OpenFOAM's `foamFormatConvert` can also convert them. The layout follows OpenFOAM's
`UListIO`, `CompactIOList`, and `ISstream` sources. CI checks on every change that OpenFOAM v2512's `checkMesh` reads the
stirred-tank examples and the showcase (`benchmark/check_openfoam.py`). If another version cannot read them, use
`"writeFormat": "ascii"`.

- The header specifies `format binary;` and `arch "LSB;label=32;scalar=64";` (32-bit labels and double precision).
  Binary `faces` uses OpenFOAM's `faceCompactList` format, with arrays of offsets and vertex indices. Meshes whose
  labels exceed 32 bits use `label=64` and require a 64-bit-label OpenFOAM build.
- Binary coordinates are written without rounding. ASCII uses 12 significant digits.
- `python -m cutmesh.layers` writes the input format by default. Override it with `-ascii` or `-binary`.
  `cutmesh.diagnose` and `render_mesh.py` can read both formats.

| About 690,000 cells (examples/stirred_tank, `baseCellSize` 0.006) | ASCII | Binary |
|---|---|---|
| Write | 4.3–4.5 s | 0.14–0.19 s |
| File size | 122 MB | 80 MB |
| Read (`cutmesh.readfoam`) | 3.1 s | 0.18 s |

This saves about 6% of total meshing time (about 70 s for the same case). OpenFOAM generally reads binary meshes faster too,
including in `checkMesh`, solver startup, and `decomposePar` for parallel runs.

## Algorithm

1. **Octree.** Refine the root grid based on distance to surfaces, feature edges, cylinder surfaces, and cylinder interiors.
   Balance 2:1 over the 26 neighbors. Incorporate hanging vertices directly into polygonal faces so the mesh conforms face to face.
2. **Level set.** Use ray parity for STL inside/outside tests and exact nearest-point distances near the surface. Use
   analytic formulas for cylinders.
3. **Vertex snapping.** Snap to feature points and edges within `featureSnapTolerance × h`, then surfaces within
   `snapTolerance × h`. Treat surface vertices as φ=0 to avoid sliver cells. First move hanging vertices along their edge
   or face to keep the coarse face planar. If this cannot reach the surface because the edge or face is nearly parallel,
   `hangingSnap: "free"` projects the vertex directly to the surface, which can bend the coarse face slightly. Undo
   projections that invalidate octree cells. `"constrained"` leaves such vertices fixed and treats them as φ=0, leaving
   depressions up to 0.3h in the wall.
4. **Cut cells (`cutcell.py`).** Cut each polyhedron using φ at its vertices. Determine each face split **only from the
   values at the vertices of that face** and share intersections by edge key. Neighboring cells therefore agree on shared
   faces. A cut face closes a loop along the open boundary of the negative-side faces.
   - First cut against the STL and retain the interior, turning cut faces into wall patches.
   - For each rotating zone, cut against its cylinder and keep both sides, turning cut faces into **internal interface faces**.
   - Project intersections onto the surfaces. Points where the STL meets a cylinder, such as at a shaft crossing, are
     projected alternately onto both surfaces to reach their intersection.
   - **Preserving sharp edges (`featureCaps`).** Where a feature edge crosses a cell face, insert the crossing into that
     face's cut line (chord). Cache it by face key so neighboring cells agree. Bend the cut face inside the cell along the
     feature between the entry and exit points. Follow curves by recursive midpoint projection and split at corners
     using spokes. If bending makes a cell nonconvex, recut it and its neighbors sharing that face using the standard
     chamfered cut. This bending captures convex edges. Concave edges use `concaveSnapMax` and the wall-contact handling
     below. The outside of a cylinder rim is concave and stays chamfered.
   - **Exact cuts for convex edges (`exactFeatureCut`, `cutmesh/featclip.py`).** The bending method derives the cut face from
     vertex signs. If vertices snap to both walls next to an edge, the fluid between the chord and the edge can be lost
     cell by cell, producing a sawtooth edge off the grid (up to 0.8 times the smallest cell size on cube_rotated).
     For locally convex cells near convex edges, clip faces directly with the local STL triangle planes (Sutherland–Hodgman).
     This gives exact intersections of the feature with the face and of each edge leaving the fluid, irrespective of
     vertex values. If the two cutting methods disagree on a shared face, plane-clip the neighbor if it is convex.
     If it uses special wall-contact handling or face splitting, let it adopt the plane-clipped face. Where the
     neighborhood is nonconvex, accepting the face can create zero- or negative-volume fragments, so use the standard
     chamfered cut there.
   - **Wall contact near concave edges (STL only).** Near re-entrant edges, such as a hole rim or an inside step corner,
     consecutive vertices can all lie on the wall (φ=0). Treating them all as fluid can create grid-dependent
     chamfers of 0–h. (a) For a run of φ=0 between N and P, test interior points on its edges for solid or fluid and
     break the cut line at vertices where it enters the solid. The face-local test keeps neighbors in agreement. Exclude
     edges with hanging vertices that did not reach the wall. (b) If every vertex of a face lies on the wall, remove
     that face from the cell on the solid side so it becomes a wall face of the fluid cell. (c) For faces with both
     fluid vertices and a run of 3 or more wall vertices, such as a face extending onto the plane outside a hole, split
     the fluid and contact portions along a chord joining the ends of the run. Remove the contact part from the solid-side
     cell so it becomes a wall face of the fluid-side cell. Even an uncut cell then has a split face. Skip the split when
     the contact portion is narrower than 0.1h. All three operations use existing vertices.
5. **Merging small and nonconvex cells.** Merge fragments below `mergeRatio` and cells with a face that cannot see the
   cell centroid into an adjacent cell in the same zone. Evaluate the merged geometry for each candidate. Require positive
   pyramids at every face and choose the candidate with the smallest non-orthogonality against neighboring cells, subject
   to `mergeMaxNonOrtho`. If a small wall-side fragment cannot merge and has negative face-triangulation tetrahedra,
   such as a lens between a curved wall and a face on it, remove it and turn the exposed face into the neighboring cell's
   wall. Remove fragments with negative face pyramids in all cases. If only the tetrahedra are negative, remove the
   fragment only when the exposed face lies within 0.4h of the STL.
6. **Output.** Write internal faces in upper-triangular order, boundary faces grouped by patch, and cellZone and faceZone
   data. Remove unused points. Join edge-sharing wall fragments of a single cell, such as the wall faces of merged small
   cells. On curved walls (such as a cone), merge neighboring fragments if their normals differ by at most 5° and
   each is within 2.5° of the merged normal. This merges two faces with a bend up to 5° while keeping continuously
   curving strips separate. Small fragments otherwise have face centers far from the cell-center normal, so inserting
   layers can make their non-orthogonality exceed `maxNonOrtho`. Layer termination can then spread along the wall in strips.

## Validation results (examples/stirred_tank, 2026-11-19, 4 cores, with Rust)

| Configuration | Cells | Time | Non-orthogonality max / avg | >70° | skew>4 | Volume error |
|---|---|---|---|---|---|---|
| `meshDict.json` (interface L2, interior L2) | 102,433 | 13 s | 61.7 / 4.3 | 0 | 0 | −0.03% |
| `meshDict_coarseRotor.json` (interface L1, interior L0) | 62,760 | 11 s | 59.8 / 5.7 | 0 | 0 | −0.03% |

- Cell closure 1e-16, 0 open cell edges, 0 negative cell volumes, 0 negative face pyramids, and correct face ordering.
- Max distance from a boundary face center to the STL is 0.36 mm (0.23 times the smallest cell size of 1.56 mm).
- Times include quality checks. For the other cases and the rotor–stator images and cell counts, see [Examples](https://mmer547.github.io/cutmesh/en/examples.html).

## Diagnostic tools

`python tools/edge_check.py <meshDict.json> [-case DIR]` samples STL feature edges and cylinder zone rims and reports
the distance to the nearest mesh edge on a boundary or interface faceZone in units of the smallest cell size.
A captured edge has distance 0. A chamfer can be several tenths of a cell wide.

`python -m cutmesh.diagnose meshDict.json [CASE]` checks an existing mesh for fluid connections across solid walls,
using the same test as `checkLeaks`.

`python benchmark/render_mesh.py CASE [-o OUT]` renders the 3D wall view (`overview.png`), the cutaway and 3 sections
(`cut.png`), and marks poor-quality cells in red. `python benchmark/stl_check.py model.stl` diagnoses the input STL.
Use `benchmark/run_bench.py` to compare a batch of STLs. See `benchmark/README.md` for benchmark details and
[Commands and options](https://mmer547.github.io/cutmesh/en/manual.html#tools) for the full command reference.

## Configuration and input checks

cutmesh validates `meshDict.json` before meshing (`cutmesh/config.py`).

- **Errors stop meshing.** One message collects missing required keys (`domain`, `baseCellSize`), missing files (`domain`
  or a zone's `stl`), invalid value types or ranges (negative levels, `mergeRatio` ≥ 1, or `hangingSnap` other than
  `"free"` or `"constrained"`), duplicate rotating-zone names, zones with both or neither of `cylinder` and `stl`,
  overlapping zones (touching surfaces are allowed), and invalid JSON.
- **Warnings (`WARNING: meshDict:`).** Unknown keys, including nested keys in `addLayers`, rotating zones, and `baseMesh`,
  are reported with suggestions so typos are visible. Names in `surfaceRefinement`, `patchTypes`, `nonRotatingPatches`,
  and `addLayers.patches` are also checked against the STL solids.
- Keys starting with `_` or `//` are treated as comments and skipped.

For an empty or truncated STL, malformed vertex lines, missing triangles, or NaN coordinates, the error identifies the
file, the problem, and the line number for ASCII files. `cutmesh` prints `ERROR: ...` without a traceback and returns
exit code 1. Unknown options and invalid command-line arguments return 2.

## Input STL repair and checks

**Whenever cutmesh automatically repairs an input, it logs a `REPAIR:` line with the repair type and count.**
`cutmesh.diagnose` and `tools/edge_check.py` also print this line when reading and repairing STLs.

`geometry.repair_triangles` applies these repairs on input.

| Repair | Reason |
|---|---|
| Remove zero-area triangles. | Their normals are undefined, which disrupts feature-edge and intersection calculations. |
| Keep one copy of duplicates (same 3 vertices and orientation). | Ray parity counts duplicate intersections twice, reversing inside/outside along the ray. |
| Remove both triangles of an overlapping pair with opposite orientations (a zero-thickness membrane). | As above. An interior membrane contributes nothing to inside/outside classification. |

```
REPAIR: domain STL: removed 20 zero-area triangles, 50 duplicated triangles (one copy kept) (1350 -> 1280 triangles)
```

Open edges, non-manifold edges (shared by 3 or more triangles), and inconsistent orientations produce `WARNING:`
messages with example locations and require repair in the source STL. With `"strictInput": true`, open or non-manifold
edges stop meshing. A valid STL logs `domain STL check: closed, consistently oriented, N triangles`.

After meshing, cutmesh compares the mesh volume to the STL's signed volume (`volume check: mesh ..., STL ... (+x%)`).
It warns when the difference exceeds `volumeCheckTolerance` (default 5%). This can indicate open or self-intersecting
STLs, ray-parity misclassification, or loss of fluid features thinner than a cell. The check is skipped if triangle
orientation is inconsistent.

## Known limitations

- STLs must be watertight and consistently oriented. Zero-area and duplicate triangles are removed on input and logged as
  `REPAIR:`. Open edges and self-intersections require source fixes. Open edges generate warnings and self-intersections
  often show up in the volume check.
- Geometry thinner than a cell, such as thin blades or baffles, needs refinement to around its thickness.
- To preserve sub-cell gaps, `fillThinGaps` refines gaps between grid planes. If a grid plane crosses a gap and snapping
  a vertex to the wall would leave the cell with no fluid corners, the vertex is moved to the gap center. Vertices
  too close to a wall to reach the center are snapped as usual. `fillThinGaps` is available for Cartesian grids.
- To preserve sub-cell walls, `fillThinWalls` refines cells with no corner in the solid. If snapping solid-side
  vertices to both faces would leave a cell with no solid corners, move the vertex to the middle of the wall thickness.
  Cells with solid corners are split too where a thin wall meets another wall (a plate under a ceiling) and separates two
  of their fluid corners. Remaining connections through the wall generate `checkLeaks` warnings. Run
  `python -m cutmesh.diagnose meshDict.json [CASE]` for an existing mesh. A line next to the free end of a wall (a sharp
  edge where the solid juts into the fluid) within the cell's reach does not count: around that end the fluid meets within
  one cell anyway. Re-entrant corners where walls meet can remain chamfered.
- Cylinder rims and concave edges can be chamfered at cell scale, which usually has little effect on interfaces.
  `exactFeatureCut` captures convex edges. Where a convex edge meets a concave edge or a curved surface near a nonconvex
  cell, the standard cut can leave a chamfer. At sharp concave edges, solid-side vertices are snapped onto the edge
  within `concaveSnapMax × h`. Where no solid-side vertex is nearby, such as where an edge meets a level transition,
  slide wall vertices along the wall to the edge and remove the solid wedge enclosed by the edge and wall vertices.
  When a corner meets a coarse face at a hanging vertex, bend the cut line at that vertex and split the cell in two
  along the edge. Chamfering remains where no vertex reaches the edge, such as when the corner is near the center
  of a cell and the nearby wall vertices are hanging vertices or their parents.
- A few boundary faces can have skewness > 4, which can fail the checkMesh “highly skew faces” check.
- Hanging vertices at wall level transitions are projected to avoid wall irregularities. A projection that invalidates
  the octree cell is undone, leaving a depression up to about 0.3 times the smallest cell size at that location.
- Small cells without a valid merge neighbor remain, reducing the minimum cell volume.
- The Python/NumPy fallback is substantially slower than the Rust extension (about 100× in the stirred tank example).
- Memory use is about 1.5–2 GB per million cells. An interface in a narrow gap can sharply increase cell count and memory.
  `meshDict_stator.json` has about 2 million cells and uses about 3.5 GB.

## Roadmap

1. **Boundary layers.** Improve the initial implementation's coverage, particularly thickness allocation in narrow gaps
   and at sharp edges. Smooth normals and report layer quality metrics such as aspect ratio.
2. **Mesh quality.** Add refinement where cuts meet level transitions and improve merge criteria for thin, flat cells.
3. **Performance.** v0.5.0 moved nearly every meshing stage to Rust. v0.7.0 removed the refinement's sample points, the
   fine sampling of every wall for gap tests and the full repeats of the snapping. Next are reducing memory use and checks
   on large real cases. The target is 10 million cells in 1 minute.
4. **Sliding meshes.** Interfaces can already be split into AMI or non-conformal coupling patches. Next are more checks
   in OpenFOAM and `dynamicMeshDict` support for the openfoam.org development version.

## License

cutmesh is released under the MIT License (see [LICENSE](LICENSE)).

## Trademarks

cutmesh is an independent tool that writes meshes (`constant/polyMesh`) for OpenFOAM.

This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the
OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks.
