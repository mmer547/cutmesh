//! Candidates of mesher.contact_splits: faces partly in the fluid and partly lying on
//! the wall (a contiguous run of >= 3 vertices with phi == 0, the others < 0).  The
//! faces are visited in the same order as the Python loop (first cell first, a face
//! once), and the means and widths use NumPy's arithmetic (see wallmerge.rs).

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cut::FxSet;
use crate::wallmerge::{cross, dot1, mean, norm1, rows_sum, sub};
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

/// The sampling loop of mesher.surface_contact over the cells (same order and
/// arithmetic): faces lying on the surface (all phi == 0) with their centre, the cell
/// size and the sample points just inside the cell; and the face edges between two
/// surface vertices of faces that are cut, with a sample point pushed into the face.
/// face_ok: per face occurrence, whether it can matter.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn surface_contact_samples<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    phi: PyReadonlyArray1<'py, f64>,
    cells: Vec<usize>,
    cptr: PyReadonlyArray1<'py, i64>,
    fptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
    f_key: PyReadonlyArray1<'py, i64>,
    csize: PyReadonlyArray1<'py, f64>,
    face_ok: PyReadonlyArray1<'py, bool>,
    push: f64,
) -> PyResult<(
    Vec<i64>,
    Vec<i64>,
    Bound<'py, PyArray2<f64>>,
    Vec<f64>,
    Bound<'py, PyArray2<f64>>,
    Vec<i64>,
    Vec<(i64, i64, i64)>,
    Bound<'py, PyArray2<f64>>,
    Vec<f64>,
    Vec<(i64, i64)>,
)> {
    let p = crate::points(&points)?;
    let phi = phi.as_slice()?;
    let cptr = cptr.as_slice()?;
    let fptr = fptr.as_slice()?;
    let fverts = fverts.as_slice()?;
    let f_key = f_key.as_slice()?;
    let csize = csize.as_slice()?;
    let fok = face_ok.as_slice()?;
    let (mut dcell, mut dfid, mut dctr, mut dtol, mut dsamp, mut dgrp) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut ekeys, mut esamp, mut etol, mut eends) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut seen: FxSet<(i64, i64, i64)> = FxSet::default();
    for &ci in &cells {
        let (fa, fb) = (cptr[ci] as usize, cptr[ci + 1] as usize);
        let h = csize[ci];
        let mut cc: Option<V3> = None;
        for f in fa..fb {
            if !fok[f] {
                continue;
            }
            let fid = f_key[f];
            let poly = &fverts[fptr[f] as usize..fptr[f + 1] as usize];
            let m = poly.len();
            let z: Vec<bool> = poly.iter().map(|&v| phi[v as usize] == 0.0).collect();
            if z.iter().all(|&x| x) {
                let q: Vec<V3> = poly.iter().map(|&v| p[v as usize]).collect();
                let cf = mean(&q);
                let x: Vec<V3> = q.iter().map(|&a| sub(a, cf)).collect();
                let t: Vec<V3> = (0..m).map(|k| cross(x[k], x[(k + 1) % m])).collect();
                let mut nrm = rows_sum(&t);
                let nn = norm1(nrm);
                if nn == 0.0 {
                    continue;
                }
                nrm = [nrm[0] / nn, nrm[1] / nn, nrm[2] / nn];
                let c = *cc.get_or_insert_with(|| {
                    let all: Vec<V3> = (fptr[fa] as usize..fptr[fb] as usize).map(|j| p[fverts[j] as usize]).collect();
                    mean(&all)
                });
                if dot1(sub(c, cf), nrm) < 0.0 {
                    nrm = [-nrm[0], -nrm[1], -nrm[2]];
                }
                let g = dcell.len() as i64;
                dcell.push(ci as i64);
                dfid.push(fid);
                dctr.push(cf);
                dtol.push(h);
                let s = 0.1 * h;
                let off = [s * nrm[0], s * nrm[1], s * nrm[2]];
                dsamp.push([cf[0] + off[0], cf[1] + off[1], cf[2] + off[2]]);
                dgrp.push(g);
                for a in &q {
                    let y = [0.5 * (a[0] + cf[0]), 0.5 * (a[1] + cf[1]), 0.5 * (a[2] + cf[2])];
                    dsamp.push([y[0] + off[0], y[1] + off[1], y[2] + off[2]]);
                    dgrp.push(g);
                }
                continue;
            }
            let nz = z.iter().filter(|&&x| x).count();
            let neg = poly.iter().any(|&v| phi[v as usize] < 0.0);
            let pos = poly.iter().any(|&v| phi[v as usize] > 0.0);
            if nz < 2 || !neg || !pos {
                continue;
            }
            let mut cf: Option<V3> = None;
            for k in 0..m {
                let (a, b) = (poly[k], poly[(k + 1) % m]);
                if !(z[k] && z[(k + 1) % m]) {
                    continue;
                }
                let key = if a < b { (fid, a, b) } else { (fid, b, a) };
                if !seen.insert(key) {
                    continue;
                }
                let c = *cf.get_or_insert_with(|| {
                    let q: Vec<V3> = poly.iter().map(|&v| p[v as usize]).collect();
                    mean(&q)
                });
                let (pa, pb) = (p[a as usize], p[b as usize]);
                let mid = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])];
                ekeys.push(key);
                esamp.push([
                    mid[0] + push * (c[0] - mid[0]),
                    mid[1] + push * (c[1] - mid[1]),
                    mid[2] + push * (c[2] - mid[2]),
                ]);
                etol.push(h);
                eends.push((a, b));
            }
        }
    }
    Ok((
        dcell,
        dfid,
        crate::vec3_array(py, dctr)?,
        dtol,
        crate::vec3_array(py, dsamp)?,
        dgrp,
        ekeys,
        crate::vec3_array(py, esamp)?,
        etol,
        eends,
    ))
}
