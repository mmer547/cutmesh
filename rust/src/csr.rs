//! Cells (lists of faces) as flat arrays, in and out of the kernels: cell i has the
//! faces fptr[i]..fptr[i+1], face f the id key[f] and the vertices
//! verts[vptr[f]..vptr[f+1]].  Saves building a Python list per face.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cut::Face;

pub(crate) fn read_cells(
    fptr: &PyReadonlyArray1<i64>,
    key: &PyReadonlyArray1<i64>,
    vptr: &PyReadonlyArray1<i64>,
    verts: &PyReadonlyArray1<i64>,
) -> PyResult<Vec<Vec<Face>>> {
    let (fp, k, vp, v) = (fptr.as_slice()?, key.as_slice()?, vptr.as_slice()?, verts.as_slice()?);
    let bad = fp.is_empty()
        || fp[0] != 0
        || fp.windows(2).any(|w| w[1] < w[0])
        || *fp.last().unwrap() as usize > k.len()
        || vp.len() != k.len() + 1
        || vp.windows(2).any(|w| w[1] < w[0])
        || *vp.last().unwrap_or(&0) as usize > v.len();
    if bad {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent cell arrays"));
    }
    Ok((0..fp.len() - 1)
        .into_par_iter()
        .map(|i| {
            (fp[i] as usize..fp[i + 1] as usize)
                .map(|f| (k[f], v[vp[f] as usize..vp[f + 1] as usize].to_vec()))
                .collect()
        })
        .collect())
}

pub(crate) type CsrOut<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// per cell (negative components, positive components) -> (item of each component,
/// side 0 negative / 1 positive, face pointer, face ids, vertex pointer, vertices); the
/// components of a cell are listed negative first, then positive, cells in order
pub(crate) fn write_results<'py>(
    py: Python<'py>,
    res: &[(Vec<Vec<Face>>, Vec<Vec<Face>>)],
    neg_only: bool,
) -> CsrOut<'py> {
    let (mut item, mut side, mut fptr, mut key, mut vptr, mut verts) =
        (Vec::new(), Vec::new(), vec![0i64], Vec::new(), vec![0i64], Vec::new());
    for (k, (negc, posc)) in res.iter().enumerate() {
        let sides: [(&Vec<Vec<Face>>, i64); 2] = [(negc, 0), (posc, 1)];
        for (comps, s) in sides {
            if s == 1 && neg_only {
                continue;
            }
            for comp in comps {
                item.push(k as i64);
                side.push(s);
                for (fid, p) in comp {
                    key.push(*fid);
                    verts.extend_from_slice(p);
                    vptr.push(verts.len() as i64);
                }
                fptr.push(key.len() as i64);
            }
        }
    }
    (
        item.into_pyarray(py),
        side.into_pyarray(py),
        fptr.into_pyarray(py),
        key.into_pyarray(py),
        vptr.into_pyarray(py),
        verts.into_pyarray(py),
    )
}
