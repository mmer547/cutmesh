# Changelog

Changes in each version of cutmesh, the same as the News on the documentation site. See "meshDict compatibility" in the manual for versioning and meshDict compatibility.

## v1.0.1 (2026-11-28)

**Bug fixes.** Fixes for the bugs found in a review of v1.0.0 ([#68](https://github.com/mmer547/cutmesh/issues/68)).
A case meshed with a sliding mesh and then remeshed with MRF zones kept the old `constant/dynamicMeshDict`, so the zone turned with the mesh and in a rotating frame. cutmesh now removes the `dynamicMeshDict` and `MRFProperties` it wrote when the mesh has no zones of that kind (a file you wrote is kept, with a warning).
With `relativeSizes: false` in `addLayers` (absolute thicknesses), the default `minThickness` of 0.05 was read as 5 cm and no layers were added. Left out, it is now 0.05 x the local cell size. **Meshes change in this case only.**
Most values of `addLayers` and `baseMesh` were not checked, and bad values ended in a traceback, an endless loop or running out of memory. They are now checked against the schema (the specification) and stop with an error naming the key. Values the schema rejects (such as numbers written as strings) are errors now, also where they happened to work. `baseMesh.maxSizeRatio` must be greater than 1 (up to 1 never finished).
Also fixed: a traceback when `case` is a file, `featureCaps: 1` read as true, and integer values written as `3.0` rejected. The meshes of the public examples are unchanged.

## v1.0.0 (2026-11-27)

**First stable version.** The meshDict specification is frozen: within 1.x, keys are not removed, renamed, or given a new meaning ([meshDict compatibility](https://mmer547.github.io/cutmesh/en/manual.html#compat)). A test holds the 1.0 keys with their types, allowed values, and required keys (`tests/data/meshdict_1.0.json`) and stops changes that would break them.
All the v1.0 conditions are met: CI passes, no known leaks, the public examples pass OpenFOAM's `checkMesh`, wheels are distributed, and the meshDict specification is frozen. This CHANGELOG was added. Meshes are unchanged.

## v0.9.0 (2026-11-27)

**Packaging.** Wheels with the Rust extension built in (Linux, Windows, and macOS, for any Python 3.9 or later) are attached to the [GitHub releases](https://github.com/mmer547/cutmesh/releases). Download one and `pip install` it to get the fast version without building anything. cutmesh is not published on PyPI.
The [meshDict compatibility policy](https://mmer547.github.io/cutmesh/en/manual.html#compat) from 1.0 is set, and retired or renamed keys are read as their replacement with a warning. Meshes are unchanged.

## v0.8.0 (2026-11-27)

**Reliability.** CI now runs OpenFOAM's `checkMesh` (v2512) on all 15 stirred-tank and showcase examples, and every one reports "Mesh OK". This also confirms that OpenFOAM reads cutmesh's binary format.
Thin slivers with wall faces skewed over 4, including cells merged from small pieces, are removed where the wall does not dent (the 1 mm gap example went from 8 such faces to 0).
Leaks where a wall thinner than the smallest cell meets another wall, such as a plate under a ceiling, are fixed (the synthetic `thin_walls` case went from 518 to 0).
Pull requests that change meshes on purpose can pass the mesh-comparison check with the `mesh-change` label.

## v0.7.1 (2026-11-26)

**Housekeeping.** The log and the benchmark results now break the long stages (octree, snapping, STL and rotating-zone cuts, assembly and others) down into their parts.
The version is kept in one place and shown by `cutmesh --version` and on the first line of the log. Meshes are unchanged.
Speed work ends with v0.7.0; next come reliability and packaging toward v1.0.0 (see the [roadmap](https://github.com/mmer547/cutmesh/issues/59)).

## v0.7.0 (2026-11-26)

**Performance improvements, step 3.** Speed-ups that may change the mesh slightly, each checked by a quality gate. Octree refinement tests the exact distance to the surfaces, feature edges and cylinders instead of sample points. The `gapRefinement` test looks at every wall coarsely first and samples finely only where it finds a thin gap. A snap that breaks a cell is redone only around that cell. The octree tests, cell assembly and leak check also do less work.
On the public tilted stirred-tank benchmark (about 1 million cells, 4 cores), runtime fell from 28.3 to about 20 seconds. The cylindrical base mesh example went from 74.6 to 24.4 seconds, and geometry preparation for a gap example with `gapRefinement` from 50.6 to 8.8 seconds (memory 6.2 to 1.4 GB).
Every benchmark case meets the gate for the changes that alter the mesh: no faces over 70 degrees non-orthogonality, no more faces with skewness over 4, no non-positive volumes, cell count within 3%, and no worse boundary layers or feature edges (`run_bench.py --gate`).

## v0.6.0 (2026-11-26)

**Performance improvements, step 2.** The remaining serial steps now run in parallel or in the Rust extension, with the same meshes as before. These include snapping, merging of small cut cells, the octree 2:1 balance and cell assembly, rotating-zone cuts, face pairing, and cell geometry.
On a real-world geometry with about 8.5 million cells, runtime fell from about 100 seconds to about 70 seconds, and peak memory stays at about 13 GB. The meshes of this geometry, the public benchmarks, and the examples were verified byte for byte.
The [example](https://mmer547.github.io/cutmesh/en/examples.html) timings were updated. The stirred tank went from 1.3 to 1.0 seconds and the 1 mm gap example from 9.2 to 7.3 seconds.

## v0.5.1 (2026-11-23)

**Lower memory usage.** Each stage now releases data it has finished using, including octree faces, intermediate cut results, search structures, and caches.
Peak memory dropped from 30 GB to about 13 GB on a real-world geometry with about 8.5 million cells, and from 7.8 GB to 3.4 GB in the 0.5 mm gap example (about 2 million cells). Usage is about 1.5–2 GB per million cells.
The mesh and runtime are unchanged.

## v0.5.0 (2026-11-22)

**Performance improvements, step 1.** The octree, snapping, STL and rotating-zone cuts, cell merging, assembly, and quality checks moved to the Rust extension and now run in parallel.
On a real-world geometry with about 8.5 million cells, runtime fell from 1209 seconds to about 100 seconds (about 12× faster) and memory use fell from 65 GB to about 30 GB. At each step, the mesh was verified byte for byte on this geometry and the public benchmarks.
The [example](https://mmer547.github.io/cutmesh/en/examples.html) timings were updated. The stirred tank went from 14.1 to 1.3 seconds and the 1 mm gap example from 131 to 9.2 seconds. The test machine also changed from 4 cores to 16.
This release also added automatic block layout for cylindrical base meshes and the [meshDict builder](https://mmer547.github.io/cutmesh/en/meshdict.html).

## v0.4.0 (2026-11-21)

Added exact cuts for sharp convex edges that do not align with the grid, sliding-mesh (AMI) interfaces for rotating zones, showcase cases for pipes, ducts, and bluff bodies,
and STL and meshDict input checks. The documentation site went live.

## v0.3.0 (2026-11-13)

Added splitting of thin cut cells at refinement transitions, improved boundary-layer coverage on curved walls and concave corners, and sped up octree and boundary-layer processing.
Also added automatic removal of duplicate and zero-area STL triangles and a benchmark suite.

## v0.2.0 (2026-10-25)

Added rotating-zone interfaces in narrow gaps, non-conforming rotating zones, and cylindrical base meshes. This release also added preservation of sub-cell gaps and walls,
automatic gap refinement, reduced chamfering of concave edges, and a boundary-layer thickness limit in narrow gaps.

## v0.1.0 (2026-10-21)

First release with octree cut-cell meshing inside STLs, OpenFOAM polyMesh export, rotating-zone cuts,
prismatic boundary layers, and a Rust extension for faster geometry calculations.
