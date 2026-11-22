//! Longest-edge bisection in the order used by geometry._sample_triangles.
//! Keep vertex IDs throughout: adjacent pieces within one input triangle share
//! their midpoint, while coincident points from different triangles keep their IDs.

use rayon::prelude::*;

use crate::cut::FxMap as HashMap;

use crate::{V3, dist2};

#[derive(Clone, Copy)]
struct Piece {
    verts: [usize; 3],
    parent: i64,
}

/// NumPy argmax takes the first longest edge. A NaN makes the size test false.
pub(super) fn split_edge([a, b, c]: [V3; 3], limit: f64) -> Option<usize> {
    let lengths = [dist2(a, b), dist2(b, c), dist2(c, a)];
    let mut k = usize::from(lengths[1] > lengths[0]);
    if lengths[2] > lengths[k] {
        k = 2;
    }
    (lengths[k] > limit && !lengths.iter().any(|d| d.is_nan())).then_some(k)
}

pub(super) fn centroid([a, b, c]: [V3; 3]) -> V3 {
    std::array::from_fn(|i| ((a[i] + b[i]) + c[i]) / 3.0)
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
    let mut unpaired: HashMap<(usize, usize), usize> = HashMap::default();
    let limit = spacing * spacing;
    while !pieces.is_empty() {
        // each piece: split along its longest edge (rotated to start there) or done (its
        // centroid), in parallel; then taken in order
        let outcome: Vec<Result<Piece, V3>> = pieces
            .par_iter()
            .map(|piece| {
                let tri = piece.verts.map(|v| points[v]);
                match split_edge(tri, limit) {
                    Some(k) => {
                        let mut rotated = *piece;
                        rotated.verts.rotate_left(k);
                        Ok(rotated)
                    }
                    None => Err(centroid(tri)),
                }
            })
            .collect();
        let mut split = Vec::with_capacity(outcome.len());
        for (piece, o) in pieces.drain(..).zip(outcome) {
            match o {
                Ok(rotated) => split.push(rotated),
                Err(c) => {
                    centroids.push(c);
                    centroid_ids.push(piece.parent);
                }
            }
        }
        let mut edges: Vec<(usize, usize, usize)> = split
            .par_iter()
            .enumerate()
            .map(|(j, p)| {
                let [u, v, _] = p.verts;
                (u.min(v), u.max(v), j)
            })
            .collect();
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
