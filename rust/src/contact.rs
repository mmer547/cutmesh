//! Candidates of mesher.contact_splits: faces partly in the fluid and partly lying on
//! the wall (a contiguous run of >= 3 vertices with phi == 0, the others < 0).  The
//! faces are visited in the same order as the Python loop (first cell first, a face
//! once), and the means and widths use NumPy's arithmetic (see wallmerge.rs).

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cut::FxSet;
use crate::wallmerge::{cross, mean, norm1, rows_sum, sub};
use crate::V3;

/// 2 x area / longest edge of the polygon q (contact_splits.width)
fn width(p: &[V3], q: &[i64]) -> f64 {
    let pts: Vec<V3> = q.iter().map(|&v| p[v as usize]).collect();
    let m = pts.len();
    let c = mean(&pts);
    let x: Vec<V3> = pts.iter().map(|&a| sub(a, c)).collect();
    let t: Vec<V3> = (0..m).map(|k| cross(x[k], x[(k + 1) % m])).collect();
    let a = norm1(rows_sum(&t)) * 0.5;
    let mut emax = f64::NEG_INFINITY;
    for k in 0..m {
        let e = sub(pts[(k + 1) % m], pts[k]);
        let l = ((e[0] * e[0] + e[1] * e[1]) + e[2] * e[2]).sqrt();
        if l.is_nan() || emax.is_nan() {
            emax = f64::NAN;
        } else if l > emax {
            emax = l;
        }
    }
    // Python's max(x, 1e-300) returns x unless 1e-300 > x (so a NaN stays)
    let den = if emax.is_nan() || emax >= 1e-300 { emax } else { 1e-300 };
    2.0 * a / den
}

/// (face ids, faces, fluid parts, contact parts, h, fluid-part means, contact-part
/// means, fluid widths, contact widths) for contact_splits
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn contact_candidates<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    phi: PyReadonlyArray1<'py, f64>,
    cells: Vec<usize>,
    cptr: PyReadonlyArray1<'py, i64>,
    fptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
    f_key: PyReadonlyArray1<'py, i64>,
    csize: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    Vec<i64>,
    Vec<Vec<i64>>,
    Vec<Vec<i64>>,
    Vec<Vec<i64>>,
    Vec<f64>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Vec<f64>,
    Vec<f64>,
)> {
    let p = crate::points(&points)?;
    let phi = phi.as_slice()?;
    let cptr = cptr.as_slice()?;
    let fptr = fptr.as_slice()?;
    let fverts = fverts.as_slice()?;
    let f_key = f_key.as_slice()?;
    let csize = csize.as_slice()?;
    let mut seen: FxSet<i64> = FxSet::default();
    let (mut fids, mut polys, mut rest, mut run, mut hs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &ci in &cells {
        for f in cptr[ci] as usize..cptr[ci + 1] as usize {
            let fid = f_key[f];
            if seen.contains(&fid) {
                continue;
            }
            let poly = &fverts[fptr[f] as usize..fptr[f + 1] as usize];
            let m = poly.len();
            let ph: Vec<f64> = poly.iter().map(|&v| phi[v as usize]).collect();
            let nz = ph.iter().filter(|&&x| x == 0.0).count();
            if ph.iter().any(|&x| x > 0.0) || !ph.iter().any(|&x| x < 0.0) || nz < 3 {
                continue;
            }
            let z: Vec<bool> = ph.iter().map(|&x| x == 0.0).collect();
            let starts: Vec<usize> = (0..m).filter(|&i| z[i] && !z[(i + m - 1) % m]).collect();
            if starts.len() != 1 {
                continue;
            }
            let i0 = starts[0];
            let mut k = 0;
            while z[(i0 + k) % m] {
                k += 1;
            }
            seen.insert(fid);
            fids.push(fid);
            polys.push(poly.to_vec());
            run.push((0..k).map(|t| poly[(i0 + t) % m]).collect::<Vec<i64>>());
            rest.push((0..m - k + 2).map(|t| poly[(i0 + k - 1 + t) % m]).collect::<Vec<i64>>());
            hs.push(csize[ci]);
        }
    }
    let (ca, cb, wa, wb): (Vec<V3>, Vec<V3>, Vec<f64>, Vec<f64>) = py.detach(|| {
        let r: Vec<(V3, V3, f64, f64)> = rest
            .par_iter()
            .zip(run.par_iter())
            .map(|(a, b)| {
                let qa: Vec<V3> = a.iter().map(|&v| p[v as usize]).collect();
                let qb: Vec<V3> = b.iter().map(|&v| p[v as usize]).collect();
                (mean(&qa), mean(&qb), width(&p, a), width(&p, b))
            })
            .collect();
        let mut out = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (x, y, u, w) in r {
            out.0.push(x);
            out.1.push(y);
            out.2.push(u);
            out.3.push(w);
        }
        out
    });
    Ok((fids, polys, rest, run, hs, crate::vec3_array(py, ca)?, crate::vec3_array(py, cb)?, wa, wb))
}
