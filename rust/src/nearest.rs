//! Nearest kept point of a few query points among many (mesher._nearest_in), without
//! building a KD-tree over all of them: a uniform grid searched ring by ring.
//!
//! The distance is scipy cKDTree's for p = 2, sqrt(((0 + dx^2) + dy^2) + dz^2).  Where
//! the nearest point is the only one at that distance the answer is the KD-tree's;
//! where several are equally near the caller asks the KD-tree (which picks one of them
//! by its own order).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::V3;

#[inline]
fn sqdist(a: V3, b: V3) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = a[k] - b[k];
        s += d * d;
    }
    s
}

/// nearest_kept(points[n, 3], keep[n], q[m, 3]) -> (distance[m], index into points[m],
/// unique[m]): per query the nearest point with keep set, and whether no other kept
/// point is at the same distance.  Requires some kept point.
#[pyfunction]
pub fn nearest_kept<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    keep: PyReadonlyArray1<'py, bool>,
    q: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<bool>>)> {
    let p = crate::points(&points)?;
    let q = crate::points(&q)?;
    let keep = keep.as_slice()?;
    if keep.len() != p.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("one keep flag per point"));
    }
    let idx: Vec<usize> = (0..p.len()).into_par_iter().filter(|&i| keep[i]).collect();
    if idx.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no kept point"));
    }
    if idx.iter().any(|&i| p[i].iter().any(|x| !x.is_finite())) || q.iter().any(|x| x.iter().any(|v| !v.is_finite())) {
        return Err(pyo3::exceptions::PyValueError::new_err("points must be finite"));
    }
    let res: Vec<(f64, i64, bool)> = py.detach(|| {
        // grid over the kept points, about two per cell
        let mut lo = p[idx[0]];
        let mut hi = lo;
        for &i in &idx {
            for k in 0..3 {
                lo[k] = lo[k].min(p[i][k]);
                hi[k] = hi[k].max(p[i][k]);
            }
        }
        let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let emax = ext[0].max(ext[1]).max(ext[2]);
        let nk = idx.len() as f64;
        let mut s = (ext.iter().map(|&e| e.max(emax * 1e-3)).product::<f64>() * 2.0 / nk).cbrt();
        if !(s > 0.0) || !s.is_finite() {
            s = if emax > 0.0 { emax } else { 1.0 };
        }
        let mut g = [0i64; 3];
        loop {
            for k in 0..3 {
                g[k] = ((ext[k] / s).floor() as i64 + 1).max(1);
            }
            if (g[0] as f64) * (g[1] as f64) * (g[2] as f64) <= 4.0 * nk + 8.0 {
                break;
            }
            s *= 1.5;
        }
        let cell_of = |x: V3| -> [i64; 3] {
            let mut c = [0i64; 3];
            for k in 0..3 {
                c[k] = ((x[k] - lo[k]) / s).floor() as i64;
            }
            c
        };
        let lin = |c: [i64; 3]| ((c[0] * g[1] + c[1]) * g[2] + c[2]) as usize;
        let ncell = (g[0] * g[1] * g[2]) as usize;
        let mut cp: Vec<(usize, usize)> = idx
            .par_iter()
            .map(|&i| {
                let mut c = cell_of(p[i]);
                for k in 0..3 {
                    c[k] = c[k].clamp(0, g[k] - 1);
                }
                (lin(c), i)
            })
            .collect();
        cp.par_sort_unstable();
        let mut start = vec![0usize; ncell + 1];
        for &(c, _) in &cp {
            start[c + 1] += 1;
        }
        for c in 0..ncell {
            start[c + 1] += start[c];
        }
        let members: Vec<usize> = cp.into_iter().map(|e| e.1).collect();
        q.par_iter()
            .map(|&x| {
                let qc = cell_of(x);
                // rings beyond this one hold no cell of the grid
                let rmax = (0..3).map(|k| (qc[k]).max(g[k] - 1 - qc[k]).max(0)).max().unwrap();
                let (mut b1, mut j1, mut b2) = (f64::INFINITY, usize::MAX, f64::INFINITY);
                let mut r = 0i64;
                loop {
                    let rng = |k: usize| ((qc[k] - r).max(0), (qc[k] + r).min(g[k] - 1));
                    let (r0, r1, r2) = (rng(0), rng(1), rng(2));
                    for a in r0.0..=r0.1 {
                        for b in r1.0..=r1.1 {
                            for c in r2.0..=r2.1 {
                                let cheb = (a - qc[0]).abs().max((b - qc[1]).abs()).max((c - qc[2]).abs());
                                if cheb != r {
                                    continue; // inner rings are done
                                }
                                let ci = lin([a, b, c]);
                                for &i in &members[start[ci]..start[ci + 1]] {
                                    let d = sqdist(x, p[i]);
                                    if d < b1 || (d == b1 && i < j1) {
                                        if d < b1 {
                                            b2 = b1;
                                        } else {
                                            b2 = d;
                                        }
                                        b1 = d;
                                        j1 = i;
                                    } else if d < b2 {
                                        b2 = d;
                                    }
                                }
                            }
                        }
                    }
                    // a point in a ring after r is at least r * s away (with a margin
                    // for the rounding of the computed distances)
                    let lb = (r as f64 * s) * (1.0 - 1e-9);
                    if (lb > 0.0 && lb * lb > b2) || r >= rmax {
                        break;
                    }
                    r += 1;
                }
                (b1.sqrt(), j1 as i64, b2 > b1)
            })
            .collect()
    });
    let d: Vec<f64> = res.iter().map(|r| r.0).collect();
    let j: Vec<i64> = res.iter().map(|r| r.1).collect();
    let u: Vec<bool> = res.iter().map(|r| r.2).collect();
    Ok((d.into_pyarray(py), j.into_pyarray(py), u.into_pyarray(py)))
}
