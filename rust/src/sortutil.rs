//! Parallel stable argsort of int64 keys: the permutation np.argsort(kind="stable")
//! gives (a stable sort has only one answer), on all cores.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn argsort_stable<'py>(py: Python<'py>, keys: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a = keys.as_slice()?;
    let idx: Vec<i64> = py.detach(|| {
        let mut idx: Vec<i64> = (0..a.len() as i64).collect();
        idx.par_sort_by_key(|&i| a[i as usize]);
        idx
    });
    Ok(idx.into_pyarray(py))
}

/// pair_faces(o_key, o_cell) -> (i_occ, i_own, i_nei, b_occ, b_own, n_multi): the face
/// assembly of mesher.generate.  The occurrences with a cell (o_cell >= 0) in a stable
/// order of their keys; a key held twice by two different cells is an internal face
/// (the occurrence of the lower cell, owner < neighbour), held once a boundary face;
/// the internal faces then stably sorted by (owner, neighbour) as np.lexsort does.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn pair_faces<'py>(
    py: Python<'py>,
    o_key: PyReadonlyArray1<'py, i64>,
    o_cell: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    usize,
)> {
    let (k, c) = (o_key.as_slice()?, o_cell.as_slice()?);
    if k.len() != c.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("one cell per face occurrence"));
    }
    let (io, iown, inei, bo, bown, nm) = py.detach(|| {
        let mut srt: Vec<i64> = (0..k.len() as i64).filter(|&i| c[i as usize] >= 0).collect();
        srt.par_sort_by_key(|&i| k[i as usize]);
        let mut pairs: Vec<(i64, i64, i64)> = Vec::new(); // (occ, own, nei)
        let (mut bo, mut bown) = (Vec::new(), Vec::new());
        let mut nm = 0usize;
        let mut s = 0;
        while s < srt.len() {
            let key = k[srt[s] as usize];
            let mut e = s + 1;
            while e < srt.len() && k[srt[e] as usize] == key {
                e += 1;
            }
            match e - s {
                1 => {
                    bo.push(srt[s]);
                    bown.push(c[srt[s] as usize]);
                }
                2 => {
                    let (a1, a2) = (srt[s], srt[s + 1]);
                    let (c1, c2) = (c[a1 as usize], c[a2 as usize]);
                    if c1 != c2 {
                        pairs.push((if c1 < c2 { a1 } else { a2 }, c1.min(c2), c1.max(c2)));
                    }
                }
                _ => nm += 1,
            }
            s = e;
        }
        pairs.par_sort_by_key(|p| (p.1, p.2));
        let io: Vec<i64> = pairs.iter().map(|p| p.0).collect();
        let iown: Vec<i64> = pairs.iter().map(|p| p.1).collect();
        let inei: Vec<i64> = pairs.iter().map(|p| p.2).collect();
        (io, iown, inei, bo, bown, nm)
    });
    Ok((io.into_pyarray(py), iown.into_pyarray(py), inei.into_pyarray(py), bo.into_pyarray(py), bown.into_pyarray(py), nm))
}

/// first_two(keys, vals, n) -> (first, second): per key in 0..n, the value of its first
/// occurrence and of its last later one (-1 if none), as the stable sort by key with
/// first[k[first_mask]] = v, second[k[~first_mask]] = v of mesher.generate does
#[pyfunction]
pub fn first_two<'py>(
    py: Python<'py>,
    keys: PyReadonlyArray1<'py, i64>,
    vals: PyReadonlyArray1<'py, i64>,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let (k, v) = (keys.as_slice()?, vals.as_slice()?);
    if k.len() != v.len() || k.iter().any(|&x| x < 0 || x as usize >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err("keys out of range"));
    }
    let mut a = vec![-1i64; n];
    let mut b = vec![-1i64; n];
    let mut seen = vec![false; n];
    for (&x, &y) in k.iter().zip(v) {
        let x = x as usize;
        if seen[x] {
            b[x] = y;
        } else {
            seen[x] = true;
            a[x] = y;
        }
    }
    Ok((a.into_pyarray(py), b.into_pyarray(py)))
}
