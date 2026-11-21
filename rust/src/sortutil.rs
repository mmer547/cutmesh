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
