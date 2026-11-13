//! Kernels of the boundary layer iterations (layers.py): the Jacobi smoothing of the
//! displacement in the band of interior points (interior()) and the validity test of
//! the cells (_cells_bad()), in parallel.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{cell_cv, dot, face_cs, points, sub, vec3_array, V3};

fn err(msg: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.to_string())
}

fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// cells_bad(points[n, 3], sizes[F], verts[sum sizes], owner[F], neighbour[nI], nC,
///           max_no[nI] or None, max_sk[F] or None, fixed[nC] or None, fixed_c[nC, 3] or None)
///     -> bad[nC]
///
/// neighbour may also be per face (length F, -1 for boundary faces, in any order); max_no
/// is then per face as well.
///
/// layers._cells_bad() over all faces given: cells with a non-positive volume or face
/// pyramid (checkMesh), a face more non-orthogonal than max_no [deg] or more skewed than
/// max_sk (quality.face_skewness).  Cells with fixed[c] take the centre fixed_c[c]
/// (only some of their faces are in the list) and are not tested for their volume.
#[pyfunction]
#[pyo3(signature = (points, sizes, verts, owner, neighbour, n_cells, max_no=None, max_sk=None, fixed=None, fixed_c=None))]
#[allow(clippy::too_many_arguments)]
pub fn cells_bad<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    sizes: PyReadonlyArray1<'py, i64>,
    verts: PyReadonlyArray1<'py, i64>,
    owner: PyReadonlyArray1<'py, i64>,
    neighbour: PyReadonlyArray1<'py, i64>,
    n_cells: usize,
    max_no: Option<PyReadonlyArray1<'py, f64>>,
    max_sk: Option<PyReadonlyArray1<'py, f64>>,
    fixed: Option<PyReadonlyArray1<'py, bool>>,
    fixed_c: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let pts = crate::points(&points)?;
    let n = sizes.as_array();
    let nf = n.len();
    let mut ptr = Vec::with_capacity(nf + 1);
    ptr.push(0usize);
    for &k in n.iter() {
        ptr.push(ptr.last().unwrap() + k as usize);
    }
    let vv: Vec<usize> = verts.as_array().iter().map(|&x| x as usize).collect();
    if *ptr.last().unwrap() != vv.len() || vv.iter().any(|&x| x >= pts.len()) {
        return Err(err("inconsistent face list"));
    }
    let own: Vec<usize> = owner.as_array().iter().map(|&v| v as usize).collect();
    // boundary faces: usize::MAX
    let nei: Vec<usize> = neighbour.as_array().iter().map(|&v| if v < 0 { usize::MAX } else { v as usize }).collect();
    let ni = nei.len();
    if own.len() != nf || ni > nf || own.iter().any(|&c| c >= n_cells)
        || nei.iter().any(|&c| c >= n_cells && c != usize::MAX)
    {
        return Err(err("inconsistent owner / neighbour"));
    }
    let max_no: Option<Vec<f64>> = max_no.map(|a| a.as_array().to_vec());
    let max_sk: Option<Vec<f64>> = max_sk.map(|a| a.as_array().to_vec());
    if max_no.as_ref().is_some_and(|a| a.len() != ni) || max_sk.as_ref().is_some_and(|a| a.len() != nf) {
        return Err(err("limits must be per internal face (max_no) / per face (max_sk)"));
    }
    let fixed: Option<(Vec<bool>, Vec<V3>)> = match (fixed, fixed_c) {
        (Some(m), Some(c)) => {
            let m = m.as_array().to_vec();
            let c = crate::points(&c)?;
            if m.len() != n_cells || c.len() != n_cells {
                return Err(err("fixed cells must be per cell"));
            }
            Some((m, c))
        }
        (None, None) => None,
        _ => return Err(err("fixed and fixed_c go together")),
    };
    let bad = py.detach(|| {
        let fg: Vec<(V3, V3)> = (0..nf).into_par_iter().map(|i| face_cs(&pts, &vv[ptr[i]..ptr[i + 1]])).collect();
        let (cf, sf): (Vec<V3>, Vec<V3>) = fg.into_iter().unzip();
        let (mut cc, vol) = cell_cv(n_cells, &own, &nei, &cf, &sf);
        let mut bad: Vec<bool> = vol.iter().map(|&v| v <= 0.0).collect();
        if let Some((m, c)) = &fixed {
            for i in 0..n_cells {
                if m[i] {
                    cc[i] = c[i];
                    bad[i] = false;
                }
            }
        }
        // per face: (owner bad, neighbour bad)
        let flags: Vec<(bool, bool)> = (0..nf)
            .into_par_iter()
            .map(|f| {
                let o = own[f];
                let s = sf[f];
                let mut bo = dot(s, sub(cf[f], cc[o])) <= 0.0;
                let mut bn = false;
                let inner = f < ni && nei[f] != usize::MAX;
                if inner {
                    bn = dot(s, sub(cc[nei[f]], cf[f])) <= 0.0;
                    if let Some(lim) = &max_no {
                        let d = sub(cc[nei[f]], cc[o]);
                        let c = dot(d, s) / (norm(d) * norm(s)).max(1e-300);
                        if c.clamp(-1.0, 1.0).acos().to_degrees() > lim[f] {
                            bo = true;
                            bn = true;
                        }
                    }
                }
                if let Some(lim) = &max_sk {
                    // quality.face_skewness with the limit: the upper bound first
                    let cpf = sub(cf[f], cc[o]);
                    let d = if inner {
                        sub(cc[nei[f]], cc[o])
                    } else {
                        let ms = norm(s).max(1e-300);
                        let nb = [s[0] / ms, s[1] / ms, s[2] / ms];
                        let t = dot(nb, cpf);
                        [nb[0] * t, nb[1] * t, nb[2] * t]
                    };
                    let a = dot(s, cpf) / (dot(s, d) + 1e-300);
                    let sv = [cpf[0] - a * d[0], cpf[1] - a * d[1], cpf[2] - a * d[2]];
                    let msv = norm(sv);
                    let fd = (if inner { 0.2 } else { 0.4 }) * norm(d);
                    let mut sk = msv / (fd + 1e-300);
                    if sk > lim[f] {
                        let q = msv + 1e-300;
                        let svh = [sv[0] / q, sv[1] / q, sv[2] / q];
                        let mut ext = f64::NEG_INFINITY;
                        for &v in &vv[ptr[f]..ptr[f + 1]] {
                            ext = ext.max(dot(svh, sub(pts[v], cf[f])).abs());
                        }
                        sk = msv / (fd.max(ext) + 1e-300);
                        if sk > lim[f] {
                            bo = true;
                            bn |= inner;
                        }
                    }
                }
                (bo, bn)
            })
            .collect();
        for (f, &(bo, bn)) in flags.iter().enumerate() {
            if bo {
                bad[own[f]] = true;
            }
            if bn {
                bad[nei[f]] = true;
            }
        }
        bad
    });
    Ok(bad.into_pyarray(py))
}

/// jacobi_band(indptr[n+1], indices[nnz], weights[nnz], c[n, 3], deg[n], proj_rows[k],
///             proj[k, 3, 3], n_iter) -> x[n, 3]
///
/// x starts at 0; each sweep x <- (A x + c) / deg (A: CSR matrix of the band, rows summed
/// in stored order as scipy does), then x[r] <- M x[r] at the rows with a projector.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn jacobi_band<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    c: PyReadonlyArray2<'py, f64>,
    deg: PyReadonlyArray1<'py, f64>,
    proj_rows: PyReadonlyArray1<'py, i64>,
    proj: PyReadonlyArray3<'py, f64>,
    n_iter: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ip: Vec<usize> = indptr.as_array().iter().map(|&v| v as usize).collect();
    let ix: Vec<usize> = indices.as_array().iter().map(|&v| v as usize).collect();
    let w: Vec<f64> = weights.as_array().to_vec();
    let c = points(&c)?;
    let deg: Vec<f64> = deg.as_array().to_vec();
    let n = c.len();
    if ip.len() != n + 1 || deg.len() != n || *ip.last().unwrap() != ix.len() || ix.len() != w.len()
        || ix.iter().any(|&j| j >= n)
    {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent band matrix"));
    }
    let pa = proj.as_array();
    let pr = proj_rows.as_array();
    if pa.shape() != [pr.len(), 3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err("projectors must be [k, 3, 3]"));
    }
    // projector per row (None: free)
    let mut pid = vec![usize::MAX; n];
    for (k, &r) in pr.iter().enumerate() {
        let r = r as usize;
        if r >= n {
            return Err(pyo3::exceptions::PyValueError::new_err("projector row out of range"));
        }
        pid[r] = k;
    }
    let mats: Vec<[[f64; 3]; 3]> = (0..pr.len())
        .map(|k| {
            let mut m = [[0.0; 3]; 3];
            for (i, row) in m.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = pa[[k, i, j]];
                }
            }
            m
        })
        .collect();
    let x = py.detach(|| {
        let mut x = vec![[0.0f64; 3]; n];
        let mut y = vec![[0.0f64; 3]; n];
        for _ in 0..n_iter {
            y.par_iter_mut().enumerate().for_each(|(i, yi)| {
                let mut s: V3 = [0.0; 3];
                for jj in ip[i]..ip[i + 1] {
                    let a = w[jj];
                    let xj = &x[ix[jj]];
                    s[0] += a * xj[0];
                    s[1] += a * xj[1];
                    s[2] += a * xj[2];
                }
                let d = deg[i];
                let v = [(s[0] + c[i][0]) / d, (s[1] + c[i][1]) / d, (s[2] + c[i][2]) / d];
                *yi = if pid[i] == usize::MAX {
                    v
                } else {
                    let m = &mats[pid[i]];
                    [
                        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
                    ]
                };
            });
            std::mem::swap(&mut x, &mut y);
        }
        x
    });
    vec3_array(py, x)
}
