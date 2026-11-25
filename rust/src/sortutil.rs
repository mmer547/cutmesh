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
    if k.len() >= u32::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err("too many face occurrences"));
    }
    let (io, iown, inei, bo, bown, nm) = py.detach(|| {
        // (key, occurrence) sorted: the order of the stable sort by key.  Packed into one
        // u64 where the keys fit in 32 bits (the memory of the index array it replaces).
        let small = k.par_iter().all(|&x| (0..u32::MAX as i64).contains(&x));
        let (packed, wide): (Vec<u64>, Vec<(i64, i64)>) = if small {
            let mut v: Vec<u64> = (0..k.len()).filter(|&i| c[i] >= 0).map(|i| (k[i] as u64) << 32 | i as u64).collect();
            v.par_sort_unstable();
            (v, Vec::new())
        } else {
            let mut v: Vec<(i64, i64)> = (0..k.len()).filter(|&i| c[i] >= 0).map(|i| (k[i], i as i64)).collect();
            v.par_sort_unstable();
            (Vec::new(), v)
        };
        let len = if small { packed.len() } else { wide.len() };
        let at = |j: usize| -> (i64, i64) {
            if small {
                let x = packed[j];
                ((x >> 32) as i64, (x & 0xffff_ffff) as i64)
            } else {
                wide[j]
            }
        };
        // (own, nei, position in key order, occurrence) sorted: the stable sort by (own, nei)
        let mut pairs: Vec<(i64, i64, u32, u32)> = Vec::new();
        let (mut bo, mut bown) = (Vec::new(), Vec::new());
        let mut nm = 0usize;
        let mut s = 0;
        while s < len {
            let (key, a1) = at(s);
            let mut e = s + 1;
            while e < len && at(e).0 == key {
                e += 1;
            }
            match e - s {
                1 => {
                    bo.push(a1);
                    bown.push(c[a1 as usize]);
                }
                2 => {
                    let a2 = at(s + 1).1;
                    let (c1, c2) = (c[a1 as usize], c[a2 as usize]);
                    if c1 != c2 {
                        pairs.push((c1.min(c2), c1.max(c2), pairs.len() as u32, (if c1 < c2 { a1 } else { a2 }) as u32));
                    }
                }
                _ => nm += 1,
            }
            s = e;
        }
        drop((packed, wide));
        pairs.par_sort_unstable();
        let io: Vec<i64> = pairs.par_iter().map(|p| p.3 as i64).collect();
        let iown: Vec<i64> = pairs.par_iter().map(|p| p.0).collect();
        let inei: Vec<i64> = pairs.par_iter().map(|p| p.1).collect();
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
