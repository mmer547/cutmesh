//! Per-face reductions over the octree faces' vertex lists, in one parallel pass
//! instead of several NumPy reduceat passes over every face vertex.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// (vertices with phi < 0, > 0, == 0) of every face (fptr, fverts)
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn face_sign_counts<'py>(
    py: Python<'py>,
    phi: PyReadonlyArray1<'py, f64>,
    fptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
    let (phi, fp, fv) = (phi.as_slice()?, fptr.as_slice()?, fverts.as_slice()?);
    if fp.is_empty() || *fp.last().unwrap() as usize > fv.len() || fv.par_iter().any(|&v| v < 0 || v as usize >= phi.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent face arrays"));
    }
    let r: Vec<(i32, i32, i32)> = py.detach(|| {
        (0..fp.len() - 1)
            .into_par_iter()
            .map(|f| {
                let (mut n, mut p, mut z) = (0, 0, 0);
                for &v in &fv[fp[f] as usize..fp[f + 1] as usize] {
                    let x = phi[v as usize];
                    if x < 0.0 {
                        n += 1;
                    } else if x > 0.0 {
                        p += 1;
                    } else if x == 0.0 {
                        z += 1;
                    }
                }
                (n, p, z)
            })
            .collect()
    });
    let (mut a, mut b, mut c) = (Vec::with_capacity(r.len()), Vec::with_capacity(r.len()), Vec::with_capacity(r.len()));
    for (n, p, z) in r {
        a.push(n);
        b.push(p);
        c.push(z);
    }
    Ok((a.into_pyarray(py), b.into_pyarray(py), c.into_pyarray(py)))
}

/// np.minimum.at(out, fverts, np.repeat(val, face sizes)) on out = full(n, inf)
#[pyfunction]
pub fn vertex_min<'py>(
    py: Python<'py>,
    n: usize,
    fptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
    val: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (fp, fv, val) = (fptr.as_slice()?, fverts.as_slice()?, val.as_slice()?);
    if fp.len() != val.len() + 1 || *fp.last().unwrap_or(&0) as usize > fv.len() || fv.par_iter().any(|&v| v < 0 || v as usize >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent face arrays"));
    }
    let mut out = vec![f64::INFINITY; n];
    for f in 0..val.len() {
        let x = val[f];
        for &v in &fv[fp[f] as usize..fp[f + 1] as usize] {
            let o = &mut out[v as usize];
            // np.minimum: a NaN wins
            if x < *o || x.is_nan() {
                *o = x;
            }
        }
    }
    Ok(out.into_pyarray(py))
}

/// per leaf: some vertex occurrence of its faces in the mask
/// (np.logical_or.reduceat(mask[fverts], vptr[:-1]) with vptr[i]..vptr[i+1] the leaf's
/// occurrences, all non-empty)
#[pyfunction]
pub fn leaf_any<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray1<'py, bool>,
    vptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (m, vp, fv) = (mask.as_slice()?, vptr.as_slice()?, fverts.as_slice()?);
    let bad = || pyo3::exceptions::PyValueError::new_err("inconsistent leaf arrays");
    if vp.is_empty() || vp[0] < 0 || *vp.last().unwrap() as usize > fv.len() || vp.par_windows(2).any(|w| w[1] <= w[0]) {
        return Err(bad());
    }
    // the vertex ids are checked as they are read (one pass over fverts): a leaf with
    // an id out of range is None
    let r: Vec<Option<bool>> = py.detach(|| {
        (0..vp.len() - 1)
            .into_par_iter()
            .map(|c| {
                let occ = &fv[vp[c] as usize..vp[c + 1] as usize];
                if occ.iter().any(|&v| v < 0 || v as usize >= m.len()) {
                    return None;
                }
                Some(occ.iter().any(|&v| m[v as usize]))
            })
            .collect()
    });
    let r: Vec<bool> = r.into_iter().collect::<Option<_>>().ok_or_else(bad)?;
    Ok(r.into_pyarray(py))
}

/// the vertices (sorted, once each) of the leaves in leaf_mask that are in vmask (all
/// when vmask is None): unique(fverts[occ][vmask[fverts[occ]]]) with occ the leaves'
/// vertex occurrences (vptr[i]..vptr[i+1])
#[pyfunction]
#[pyo3(signature = (leaf_mask, vptr, fverts, vmask=None))]
pub fn leaf_vertices<'py>(
    py: Python<'py>,
    leaf_mask: PyReadonlyArray1<'py, bool>,
    vptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
    vmask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (lm, vp, fv) = (leaf_mask.as_slice()?, vptr.as_slice()?, fverts.as_slice()?);
    let vm = match &vmask {
        Some(m) => Some(m.as_slice()?),
        None => None,
    };
    if vp.len() != lm.len() + 1 || *vp.last().unwrap_or(&0) as usize > fv.len()
        || vp.par_windows(2).any(|w| w[1] < w[0])
        || vm.is_some_and(|m| fv.par_iter().any(|&v| v < 0 || v as usize >= m.len()))
    {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent leaf arrays"));
    }
    let mut out: Vec<i64> = py.detach(|| {
        let mut v: Vec<i64> = (0..lm.len())
            .into_par_iter()
            .filter(|&c| lm[c])
            .flat_map_iter(|c| {
                fv[vp[c] as usize..vp[c + 1] as usize].iter().copied().filter(move |&x| vm.is_none_or(|m| m[x as usize]))
            })
            .collect();
        v.par_sort_unstable();
        v
    });
    out.dedup();
    Ok(out.into_pyarray(py))
}

/// per leaf: the smallest value at the vertex occurrences of its faces
/// (np.minimum.reduceat over the faces, then over the leaves; vptr[i]..vptr[i+1] the
/// leaf's occurrences, all non-empty).  NaN wins, as in np.minimum.
#[pyfunction]
pub fn leaf_min<'py>(
    py: Python<'py>,
    vals: PyReadonlyArray1<'py, f64>,
    vptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (x, vp, fv) = (vals.as_slice()?, vptr.as_slice()?, fverts.as_slice()?);
    if vp.is_empty() || *vp.last().unwrap() as usize > fv.len() || vp.par_windows(2).any(|w| w[1] <= w[0])
        || fv.par_iter().any(|&v| v < 0 || v as usize >= x.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent leaf arrays"));
    }
    let r: Vec<f64> = py.detach(|| {
        (0..vp.len() - 1)
            .into_par_iter()
            .map(|c| {
                let mut m = f64::INFINITY;
                for &v in &fv[vp[c] as usize..vp[c + 1] as usize] {
                    let y = x[v as usize];
                    if y.is_nan() || m.is_nan() {
                        m = f64::NAN;
                    } else if y < m {
                        m = y;
                    }
                }
                m
            })
            .collect()
    });
    Ok(r.into_pyarray(py))
}

/// per leaf of `leaves`: the mean of the points at the vertex occurrences of its faces,
/// added in order from 0 and divided by their number (np.bincount per coordinate, then
/// divided by the counts)
#[pyfunction]
pub fn leaf_mean<'py>(
    py: Python<'py>,
    points: numpy::PyReadonlyArray2<'py, f64>,
    leaves: PyReadonlyArray1<'py, i64>,
    vptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let p = crate::points(&points)?;
    let (lv, vp, fv) = (leaves.as_slice()?, vptr.as_slice()?, fverts.as_slice()?);
    if lv.iter().any(|&c| c < 0 || c as usize + 1 >= vp.len()) || *vp.last().unwrap_or(&0) as usize > fv.len()
        || fv.par_iter().any(|&v| v < 0 || v as usize >= p.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent leaf arrays"));
    }
    let r: Vec<crate::V3> = py.detach(|| {
        lv.par_iter()
            .map(|&c| {
                let occ = &fv[vp[c as usize] as usize..vp[c as usize + 1] as usize];
                let mut s = [0.0f64; 3];
                for &v in occ {
                    for k in 0..3 {
                        s[k] += p[v as usize][k];
                    }
                }
                let n = occ.len() as f64;
                [s[0] / n, s[1] / n, s[2] / n]
            })
            .collect()
    });
    crate::vec3_array(py, r)
}

/// piece_geometry(points, pfptr, pvptr, pverts, h per piece) -> (volume, bad) per piece:
/// the piece geometry of mesher.generate's merging stage (face centres and area vectors
/// as quality.face_geometry, sums per piece in face order as np.bincount, einsum rows
/// as (p0 + p2) + p1, norms as sqrt((x0^2 + x1^2) + x2^2)); bad: some face does not
/// clearly see the centre
#[pyfunction]
pub fn piece_geometry<'py>(
    py: Python<'py>,
    points: numpy::PyReadonlyArray2<'py, f64>,
    pfptr: PyReadonlyArray1<'py, i64>,
    pvptr: PyReadonlyArray1<'py, i64>,
    pverts: PyReadonlyArray1<'py, i64>,
    hp: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<bool>>)> {
    use crate::octree::einsum3;
    let p = crate::points(&points)?;
    let (pf, vp, pv, hp) = (pfptr.as_slice()?, pvptr.as_slice()?, pverts.as_slice()?, hp.as_slice()?);
    let n = pf.len().saturating_sub(1);
    if hp.len() != n || *pf.last().unwrap_or(&0) as usize + 1 > vp.len() || *vp.last().unwrap_or(&0) as usize > pv.len()
        || pv.par_iter().any(|&v| v < 0 || v as usize >= p.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent piece arrays"));
    }
    let r: Vec<(f64, bool)> = py.detach(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let faces = pf[i] as usize..pf[i + 1] as usize;
                let cs: Vec<(crate::V3, crate::V3)> = faces
                    .clone()
                    .map(|f| {
                        let idx: Vec<usize> = pv[vp[f] as usize..vp[f + 1] as usize].iter().map(|&v| v as usize).collect();
                        crate::face_cs(&p, &idx)
                    })
                    .collect();
                let cnt = cs.len().max(1) as f64;
                let mut est = [0.0f64; 3];
                for (c, _) in &cs {
                    for k in 0..3 {
                        est[k] += c[k];
                    }
                }
                for e in est.iter_mut() {
                    *e /= cnt;
                }
                let mut vol = 0.0f64;
                let mut w = [0.0f64; 3];
                for (c, s) in &cs {
                    let pvf = einsum3(*s, crate::sub(*c, est));
                    vol += pvf;
                    for k in 0..3 {
                        w[k] += pvf * (0.75 * c[k] + 0.25 * est[k]);
                    }
                }
                let vol = vol / 3.0;
                let den = if vol.abs() > 0.0 { 3.0 * vol } else { 1.0 };
                let cc = [w[0] / den, w[1] / den, w[2] / den];
                let bad = cs.iter().any(|(c, s)| {
                    let nrm = ((s[0] * s[0] + s[1] * s[1]) + s[2] * s[2]).sqrt();
                    einsum3(*s, crate::sub(*c, cc)) < 0.05 * nrm * hp[i]
                });
                (vol, bad)
            })
            .collect()
    });
    let (vol, bad): (Vec<f64>, Vec<bool>) = r.into_iter().unzip();
    Ok((vol.into_pyarray(py), bad.into_pyarray(py)))
}
