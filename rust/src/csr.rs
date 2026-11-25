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

/// (item, side, face ptr, face ids, vertex ptr, vertices) as vectors
pub(crate) type CsrVecs = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>);

pub(crate) fn csr_out(py: Python<'_>, a: CsrVecs) -> CsrOut<'_> {
    (a.0.into_pyarray(py), a.1.into_pyarray(py), a.2.into_pyarray(py), a.3.into_pyarray(py), a.4.into_pyarray(py), a.5.into_pyarray(py))
}

/// per cell (negative components, positive components) -> (item of each component,
/// side 0 negative / 1 positive, face pointer, face ids, vertex pointer, vertices); the
/// components of a cell are listed negative first, then positive, cells in order
pub(crate) fn write_results<'py>(
    py: Python<'py>,
    res: Vec<(Vec<Vec<Face>>, Vec<Vec<Face>>)>,
    neg_only: bool,
) -> CsrOut<'py> {
    let res = if neg_only { res.into_par_iter().map(|(n, _)| (n, Vec::new())).collect() } else { res };
    let a = py.detach(|| write_results_mapped(res, |_, v| v, |_, f| f));
    csr_out(py, a)
}

/// write_results with the vertex and face ids of cell c mapped by vmap(c, v), fmap(c, f):
/// the sizes counted first, then every group of cells written to its place in parallel
/// (no per-cell vectors), the cells' results freed as they are written
pub(crate) fn write_results_mapped(
    res: Vec<(Vec<Vec<Face>>, Vec<Vec<Face>>)>,
    vmap: impl Fn(usize, i64) -> i64 + Sync,
    fmap: impl Fn(usize, i64) -> i64 + Sync,
) -> CsrVecs {
    const G: usize = 1024;
    let n = res.len();
    // per group of cells: (components, faces, vertices)
    let gcnt: Vec<(usize, usize, usize)> = res
        .par_chunks(G)
        .map(|cs| {
            let mut t = (0, 0, 0);
            for (neg, pos) in cs {
                for comp in neg.iter().chain(pos.iter()) {
                    t.0 += 1;
                    t.1 += comp.len();
                    t.2 += comp.iter().map(|f| f.1.len()).sum::<usize>();
                }
            }
            t
        })
        .collect();
    let (mut oc, mut of, mut ov) = (vec![0usize], vec![0usize], vec![0usize]);
    for &(c, f, v) in &gcnt {
        oc.push(oc.last().unwrap() + c);
        of.push(of.last().unwrap() + f);
        ov.push(ov.last().unwrap() + v);
    }
    let (nc, nf, nv) = (*oc.last().unwrap(), *of.last().unwrap(), *ov.last().unwrap());
    let mut item = vec![0i64; nc];
    let mut side = vec![0i64; nc];
    let mut fptr = vec![0i64; nc + 1];
    let mut key = vec![0i64; nf];
    let mut vptr = vec![0i64; nf + 1];
    let mut verts = vec![0i64; nv];
    {
        let sizes = |o: &Vec<usize>| -> Vec<usize> { o.windows(2).map(|w| w[1] - w[0]).collect() };
        let (sc, sf, sv) = (sizes(&oc), sizes(&of), sizes(&ov));
        let it = split(&mut item, &sc);
        let sd = split(&mut side, &sc);
        let fp = split(&mut fptr[1..], &sc);
        let ky = split(&mut key, &sf);
        let vp = split(&mut vptr[1..], &sf);
        let vs = split(&mut verts, &sv);
        // the groups' cells, moved out (each group frees its cells once written)
        let mut groups: Vec<Vec<(Vec<Vec<Face>>, Vec<Vec<Face>>)>> = Vec::with_capacity(n.div_ceil(G));
        let mut cells = res.into_iter();
        loop {
            let g: Vec<_> = cells.by_ref().take(G).collect();
            if g.is_empty() {
                break;
            }
            groups.push(g);
        }
        groups
            .into_par_iter()
            .zip(it)
            .zip(sd)
            .zip(fp)
            .zip(ky)
            .zip(vp)
            .zip(vs)
            .enumerate()
            .for_each(|(g, ((((((cs, it), sd), fp), ky), vp), vs))| {
                let (mut ic, mut jf, mut kv) = (0usize, 0usize, 0usize);
                for (j, (neg, pos)) in cs.into_iter().enumerate() {
                    let c = g * G + j;
                    for (comps, s) in [(neg, 0i64), (pos, 1i64)] {
                        for comp in comps {
                            it[ic] = c as i64;
                            sd[ic] = s;
                            for (fid, p) in comp {
                                ky[jf] = fmap(c, fid);
                                for v in p {
                                    vs[kv] = vmap(c, v);
                                    kv += 1;
                                }
                                vp[jf] = (ov[g] + kv) as i64;
                                jf += 1;
                            }
                            fp[ic] = (of[g] + jf) as i64;
                            ic += 1;
                        }
                    }
                }
            });
    }
    (item, side, fptr, key, vptr, verts)
}

fn split<'a>(mut s: &'a mut [i64], sizes: &[usize]) -> Vec<&'a mut [i64]> {
    let mut out = Vec::with_capacity(sizes.len());
    for &k in sizes {
        let (a, b) = std::mem::take(&mut s).split_at_mut(k);
        out.push(a);
        s = b;
    }
    out
}
