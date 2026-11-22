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
    res: Vec<(Vec<Vec<Face>>, Vec<Vec<Face>>)>,
    neg_only: bool,
) -> CsrOut<'py> {
    // per cell: its components flattened (in parallel), then joined with offsets
    struct Block {
        side: Vec<i64>,
        fcnt: Vec<i64>,
        key: Vec<i64>,
        vcnt: Vec<i64>,
        verts: Vec<i64>,
    }
    // consumed in parallel: the many small vectors are freed by the workers too
    let blocks: Vec<Block> = py.detach(|| {
        res.into_par_iter()
            .map(|(negc, posc)| {
                let mut b = Block { side: Vec::new(), fcnt: Vec::new(), key: Vec::new(), vcnt: Vec::new(), verts: Vec::new() };
                let sides: [(&Vec<Vec<Face>>, i64); 2] = [(&negc, 0), (&posc, 1)];
                for (comps, s) in sides {
                    if s == 1 && neg_only {
                        continue;
                    }
                    for comp in comps {
                        b.side.push(s);
                        b.fcnt.push(comp.len() as i64);
                        for (fid, p) in comp {
                            b.key.push(*fid);
                            b.vcnt.push(p.len() as i64);
                            b.verts.extend_from_slice(p);
                        }
                    }
                }
                b
            })
            .collect()
    });
    // joined in parallel (ordered collects), the offsets as running sums
    let ptr = |cnt: Vec<i64>| -> Vec<i64> {
        let mut p = Vec::with_capacity(cnt.len() + 1);
        p.push(0i64);
        let mut t = 0i64;
        for c in cnt {
            t += c;
            p.push(t);
        }
        p
    };
    let (item, side, fptr, key, vptr, verts) = py.detach(|| {
        let item: Vec<i64> = blocks
            .par_iter()
            .enumerate()
            .flat_map_iter(|(k, b)| std::iter::repeat_n(k as i64, b.side.len()))
            .collect();
        let side: Vec<i64> = blocks.par_iter().flat_map_iter(|b| b.side.iter().copied()).collect();
        let fptr = ptr(blocks.par_iter().flat_map_iter(|b| b.fcnt.iter().copied()).collect());
        let key: Vec<i64> = blocks.par_iter().flat_map_iter(|b| b.key.iter().copied()).collect();
        let vptr = ptr(blocks.par_iter().flat_map_iter(|b| b.vcnt.iter().copied()).collect());
        let verts: Vec<i64> = blocks.par_iter().flat_map_iter(|b| b.verts.iter().copied()).collect();
        blocks.into_par_iter().for_each(drop);
        (item, side, fptr, key, vptr, verts)
    });
    (
        item.into_pyarray(py),
        side.into_pyarray(py),
        fptr.into_pyarray(py),
        key.into_pyarray(py),
        vptr.into_pyarray(py),
        verts.into_pyarray(py),
    )
}
