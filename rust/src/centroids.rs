//! Radius queries on the centroids of the longest-edge subdivision. Nearby
//! centroids share a BVH primitive so the index stays small on large flat walls.

use std::ops::Range;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{box_d2, dist2, sampling, Bvh, V3};

fn subdivide(
    mut tri: [V3; 3],
    limit: f64,
    centres: &mut Vec<V3>,
    radius2: &mut f64,
) -> PyResult<()> {
    if let Some(k) = sampling::split_edge(tri, limit) {
        tri.rotate_left(k);
        let [a, b, c] = tri;
        let mid = std::array::from_fn(|i| 0.5 * (a[i] + b[i]));
        if mid == a || mid == b || mid.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "cannot bisect edge at this spacing and coordinate scale",
            ));
        }
        subdivide([a, mid, c], limit, centres, radius2)?;
        subdivide([mid, b, c], limit, centres, radius2)?;
    } else {
        let centre = sampling::centroid(tri);
        centres.push(centre);
        for v in tri {
            *radius2 = radius2.max(dist2(v, centre));
        }
    }
    Ok(())
}

#[pyclass(frozen)]
pub(super) struct CentroidSearch {
    centres: Vec<V3>,
    blocks: Vec<Range<usize>>,
    bvh: Bvh,
    #[pyo3(get)]
    rmax: f64,
}

impl CentroidSearch {
    pub(super) fn build(tris: &[[V3; 3]], spacing: f64) -> PyResult<Self> {
        let pieces = tris
            .par_iter()
            .map(|&tri| {
                let mut points = Vec::new();
                let mut radius2 = 0.0;
                subdivide(tri, spacing * spacing, &mut points, &mut radius2)?;
                Ok((points, radius2))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut centres = Vec::with_capacity(pieces.iter().map(|(p, _)| p.len()).sum());
        let mut blocks = Vec::new();
        let (mut lo, mut hi) = (Vec::new(), Vec::new());
        let mut rmax2 = 0.0_f64;
        for (points, radius2) in pieces {
            rmax2 = rmax2.max(radius2);
            // Depth-first order groups siblings. Keep blocks within their input
            // triangle even if consecutive triangles are on opposite walls.
            for block in points.chunks(16) {
                let (mut lower, mut upper) = (block[0], block[0]);
                for p in &block[1..] {
                    for k in 0..3 {
                        lower[k] = lower[k].min(p[k]);
                        upper[k] = upper[k].max(p[k]);
                    }
                }
                lo.push(lower);
                hi.push(upper);
                blocks.push(centres.len()..centres.len() + block.len());
                centres.extend_from_slice(block);
            }
        }
        Ok(Self {
            centres,
            blocks,
            bvh: Bvh::build(&lo, &hi),
            rmax: rmax2.sqrt(),
        })
    }

    fn any_below(&self, node: u32, p: V3, radius: f64, radius2: f64) -> bool {
        let n = &self.bvh.nodes[node as usize];
        if n.count > 0 {
            return self.bvh.prims[n.start as usize..(n.start + n.count) as usize]
                .iter()
                .any(|&b| {
                    self.centres[self.blocks[b as usize].clone()]
                        .iter()
                        .any(|&c| {
                            let d2 = dist2(c, p);
                            // Compare the rounded distance, as in cKDTree.query(...) < radius.
                            d2 <= radius2 && d2.sqrt() < radius
                        })
                });
        }
        let left = (n.start, box_d2(&self.bvh.nodes[n.start as usize], p));
        let right = (n.right, box_d2(&self.bvh.nodes[n.right as usize], p));
        let (near, far) = if left.1 <= right.1 {
            (left, right)
        } else {
            (right, left)
        };
        (near.1 <= radius2 && self.any_below(near.0, p, radius, radius2))
            || (far.1 <= radius2 && self.any_below(far.0, p, radius, radius2))
    }
}

#[pymethods]
impl CentroidSearch {
    fn below<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        thresholds: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        if points.shape()[1] != 3
            || !points.is_c_contiguous()
            || points.shape()[0] != thresholds.len()
        {
            return Err(PyValueError::new_err(
                "expected contiguous (n, 3) points and n thresholds",
            ));
        }
        let pts = points.as_slice()?.as_chunks::<3>().0;
        let thresholds = thresholds.as_array(); // also accepts a broadcast scalar
        let result: Vec<bool> = py.detach(|| {
            pts.par_iter()
                .enumerate()
                .map(|(i, &p)| {
                    let r = thresholds[i];
                    let r2 = r * r;
                    r > 0.0
                        && !self.bvh.nodes.is_empty()
                        && box_d2(&self.bvh.nodes[0], p) <= r2
                        && self.any_below(0, p, r, r2)
                })
                .collect()
        });
        Ok(result.into_pyarray(py))
    }
}
