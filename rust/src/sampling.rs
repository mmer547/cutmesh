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
        // the edges by key: an edge met before (unpaired) keeps its midpoint, the others
        // get new ones in key order (looked up and made in parallel, the table updated
        // after)
        let starts: Vec<usize> = (0..edges.len())
            .into_par_iter()
            .filter(|&k| k == 0 || (edges[k].0, edges[k].1) != (edges[k - 1].0, edges[k - 1].1))
            .collect();
        let ng = starts.len();
        let end = |g: usize| if g + 1 < ng { starts[g + 1] } else { edges.len() };
        let found: Vec<Option<usize>> = starts
            .par_iter()
            .map(|&k| unpaired.get(&(edges[k].0, edges[k].1)).copied())
            .collect();
        let base = points.len();
        let mut gid = vec![0usize; ng];
        let mut newg: Vec<usize> = Vec::new();
        for g in 0..ng {
            gid[g] = match found[g] {
                Some(id) => id,
                None => {
                    newg.push(g);
                    base + newg.len() - 1
                }
            };
        }
        let newp: Vec<V3> = newg
            .par_iter()
            .map(|&g| {
                let (u, v, _) = edges[starts[g]];
                std::array::from_fn(|i| 0.5 * (points[u][i] + points[v][i]))
            })
            .collect();
        points.extend(newp);
        ids.extend(newg.iter().map(|&g| split[edges[starts[g]].2].parent));
        let mut mids = vec![0; split.len()];
        for g in 0..ng {
            let key = (edges[starts[g]].0, edges[starts[g]].1);
            if found[g].is_some() {
                unpaired.remove(&key);
            } else if end(g) - starts[g] == 1 {
                // an edge has at most two incident pieces in this triangle's subdivision
                unpaired.insert(key, gid[g]);
            }
            for &(_, _, j) in &edges[starts[g]..end(g)] {
                mids[j] = gid[g];
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

/// samples() of each triangle on its own, in parallel, joined in triangle order: the
/// same points with the same triangle ids (a triangle's subdivision does not depend on
/// the others), in another order (by triangle, then breadth first)
pub(super) fn samples_by_triangle(tris: &[[V3; 3]], selected: &[i64], spacing: f64) -> (Vec<V3>, Vec<i64>) {
    let per: Vec<(Vec<V3>, Vec<i64>)> = selected.par_iter().map(|&t| samples(tris, &[t], spacing)).collect();
    let mut off = Vec::with_capacity(per.len() + 1);
    off.push(0usize);
    for (p, _) in &per {
        off.push(off.last().unwrap() + p.len());
    }
    let tot = *off.last().unwrap();
    let mut pts = vec![[0.0f64; 3]; tot];
    let mut ids = vec![0i64; tot];
    let mut ps: Vec<&mut [V3]> = Vec::with_capacity(per.len());
    let mut is: Vec<&mut [i64]> = Vec::with_capacity(per.len());
    let (mut rp, mut ri) = (&mut pts[..], &mut ids[..]);
    for (p, _) in &per {
        let (a, b) = std::mem::take(&mut rp).split_at_mut(p.len());
        ps.push(a);
        rp = b;
        let (a, b) = std::mem::take(&mut ri).split_at_mut(p.len());
        is.push(a);
        ri = b;
    }
    per.into_par_iter().zip(ps).zip(is).for_each(|(((p, i), dp), di)| {
        dp.copy_from_slice(&p);
        di.copy_from_slice(&i);
    });
    (pts, ids)
}
