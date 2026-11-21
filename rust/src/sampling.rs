//! Longest-edge bisection in the order used by geometry._sample_triangles.
//! Keep vertex IDs throughout: adjacent pieces within one input triangle share
//! their midpoint, while coincident points from different triangles keep their IDs.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{V3, dist2};

#[derive(Clone, Copy)]
struct Piece {
    verts: [usize; 3],
    parent: i64,
}

pub(super) fn samples(tris: &[[V3; 3]], selected: &[i64], spacing: f64) -> (Vec<V3>, Vec<i64>) {
    let mut points = Vec::with_capacity(3 * selected.len());
    let mut ids = Vec::with_capacity(3 * selected.len());
    let mut pieces = Vec::with_capacity(selected.len());
    for &parent in selected {
        let start = points.len();
        points.extend(tris[parent as usize]);
        ids.extend([parent; 3]);
        pieces.push(Piece {
            verts: [start, start + 1, start + 2],
            parent,
        });
    }
    let mut centroids = Vec::new();
    let mut centroid_ids = Vec::new();
    let mut unpaired = HashMap::new();
    let limit = spacing * spacing;
    while !pieces.is_empty() {
        let mut split = Vec::new();
        let mut edges = Vec::new();
        for piece in pieces.drain(..) {
            let [a, b, c] = piece.verts.map(|v| points[v]);
            let lengths = [dist2(a, b), dist2(b, c), dist2(c, a)];
            // argmax picks the first longest edge. A NaN makes NumPy's size test false.
            let mut k = usize::from(lengths[1] > lengths[0]);
            if lengths[2] > lengths[k] {
                k = 2;
            }
            if lengths[k] > limit && !lengths.iter().any(|d| d.is_nan()) {
                let mut rotated = piece;
                rotated.verts.rotate_left(k);
                let [u, v, _] = rotated.verts;
                edges.push((u.min(v), u.max(v), split.len()));
                split.push(rotated);
            } else {
                centroids.push(std::array::from_fn::<_, 3, _>(|i| {
                    ((a[i] + b[i]) + c[i]) / 3.0
                }));
                centroid_ids.push(piece.parent);
            }
        }
        // NumPy numbers new midpoints in sorted edge order at each depth.
        edges.par_sort_unstable();
        let mut mids = vec![0; split.len()];
        for group in edges.chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)) {
            let (u, v, first) = group[0];
            let id = unpaired.remove(&(u, v)).unwrap_or_else(|| {
                let id = points.len();
                points.push(std::array::from_fn(|i| 0.5 * (points[u][i] + points[v][i])));
                ids.push(split[first].parent);
                // An edge has at most two incident pieces in this triangle's subdivision.
                if group.len() == 1 {
                    unpaired.insert((u, v), id);
                }
                id
            });
            for &(_, _, j) in group {
                mids[j] = id;
            }
        }
        // All first children precede all second children in the next breadth-first step.
        pieces.reserve(2 * split.len());
        for (p, &m) in split.iter().zip(&mids) {
            pieces.push(Piece {
                verts: [p.verts[0], m, p.verts[2]],
                parent: p.parent,
            });
        }
        for (p, m) in split.into_iter().zip(mids) {
            pieces.push(Piece {
                verts: [m, p.verts[1], p.verts[2]],
                parent: p.parent,
            });
        }
    }
    points.append(&mut centroids);
    ids.append(&mut centroid_ids);
    (points, ids)
}
