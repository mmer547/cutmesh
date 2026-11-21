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
