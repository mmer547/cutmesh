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
    if fp.is_empty() || *fp.last().unwrap() as usize > fv.len() || fv.iter().any(|&v| v < 0 || v as usize >= phi.len()) {
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
    if fp.len() != val.len() + 1 || *fp.last().unwrap_or(&0) as usize > fv.len() || fv.iter().any(|&v| v < 0 || v as usize >= n) {
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
    if vp.is_empty() || *vp.last().unwrap() as usize > fv.len() || vp.windows(2).any(|w| w[1] <= w[0])
        || fv.iter().any(|&v| v < 0 || v as usize >= m.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent leaf arrays"));
    }
    let r: Vec<bool> = py.detach(|| {
        (0..vp.len() - 1)
            .into_par_iter()
            .map(|c| fv[vp[c] as usize..vp[c + 1] as usize].iter().any(|&v| m[v as usize]))
            .collect()
    });
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
        || vp.windows(2).any(|w| w[1] < w[0])
        || vm.is_some_and(|m| fv.iter().any(|&v| v < 0 || v as usize >= m.len()))
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
