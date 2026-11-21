//! Joining coplanar wall faces of one cell: port of mesher.merge_coplanar_wall_faces.
//!
//! The arithmetic follows NumPy's as the Python code runs it, so the same faces join:
//! np.cross (three products and differences, no FMA), axis-0 sums and means of small
//! arrays row by row, 1-D np.dot and np.linalg.norm as an FMA chain, a matrix-vector
//! product row as fma(a2,b2, fma(a0,b0, a1*b1)), and 1-D sums pairwise.  Dicts are
//! replaced by insertion-ordered maps.  The runs of faces of one cell are independent
//! and are joined in parallel; the output keeps the face order.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cut::FxMap;
use crate::dcut::np_sum;
use crate::V3;

#[inline]
pub(crate) fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// np.dot of two 1-D 3-vectors
#[inline]
pub(crate) fn dot1(a: V3, b: V3) -> f64 {
    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
}

/// one row of an (m,3) @ (3,) product, m >= 2
#[inline]
pub(crate) fn gemv_row(a: V3, b: V3) -> f64 {
    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
}

/// np.linalg.norm of a 1-D 3-vector
#[inline]
pub(crate) fn norm1(a: V3) -> f64 {
    dot1(a, a).sqrt()
}

/// axis-0 sum of rows (first row, then the others added in order)
pub(crate) fn rows_sum(r: &[V3]) -> V3 {
    let mut s = r[0];
    for x in &r[1..] {
        for k in 0..3 {
            s[k] += x[k];
        }
    }
    s
}

pub(crate) fn mean(q: &[V3]) -> V3 {
    let s = rows_sum(q);
    let n = q.len() as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

/// insertion-ordered map (a Python dict): keys in first-insertion order, values updated
struct OMap<K: std::hash::Hash + Eq + Copy, V> {
    idx: FxMap<K, usize>,
    keys: Vec<K>,
    vals: Vec<V>,
}

impl<K: std::hash::Hash + Eq + Copy, V> OMap<K, V> {
    fn new() -> Self {
        OMap { idx: FxMap::default(), keys: Vec::new(), vals: Vec::new() }
    }
    fn get(&self, k: &K) -> Option<&V> {
        self.idx.get(k).map(|&i| &self.vals[i])
    }
    fn get_mut_or(&mut self, k: K, v: V) -> &mut V {
        let i = match self.idx.get(&k) {
            Some(&i) => i,
            None => {
                self.idx.insert(k, self.keys.len());
                self.keys.push(k);
                self.vals.push(v);
                self.keys.len() - 1
            }
        };
        &mut self.vals[i]
    }
}

fn merged_face_ok(p: &[V3], lp: &[i64], parts: &[&[i64]], nrms: &[V3], cmin: f64, flat_tol: f64) -> bool {
    let q: Vec<V3> = lp.iter().map(|&v| p[v as usize]).collect();
    let m = q.len();
    let c0 = mean(&q);
    let tri: Vec<V3> = (0..m).map(|k| cross(sub(q[k], c0), sub(q[(k + 1) % m], c0))).collect();
    let ts = rows_sum(&tri);
    let s = [0.5 * ts[0], 0.5 * ts[1], 0.5 * ts[2]];
    let a = norm1(s);
    if a <= 0.0 {
        return false;
    }
    let n = [s[0] / a, s[1] / a, s[2] / a];
    if nrms.iter().any(|&mm| dot1(mm, n) <= cmin) {
        return false;
    }
    let mut vs: Vec<i64> = parts.iter().flat_map(|x| x.iter().copied()).collect();
    vs.sort_unstable();
    vs.dedup();
    // np.max propagates a NaN, and NaN > x is False
    let mut dmax = f64::NEG_INFINITY;
    for &v in &vs {
        let d = gemv_row(sub(p[v as usize], c0), n).abs();
        if d.is_nan() || dmax.is_nan() {
            dmax = f64::NAN;
        } else if d > dmax {
            dmax = d;
        }
    }
    if dmax > flat_tol * a.sqrt() {
        return false;
    }
    let at: Vec<f64> = tri.iter().map(|&t| gemv_row(t, n)).collect();
    if at.iter().any(|&x| x <= 1e-9 * a) {
        return false;
    }
    let w: Vec<V3> = (0..m)
        .map(|k| {
            let b = q[(k + 1) % m];
            let s_ = [(q[k][0] + b[0]) + c0[0], (q[k][1] + b[1]) + c0[1], (q[k][2] + b[2]) + c0[2]];
            [at[k] * s_[0], at[k] * s_[1], at[k] * s_[2]]
        })
        .collect();
    let ws = rows_sum(&w);
    let den = 3.0 * np_sum(&at);
    let cf = [ws[0] / den, ws[1] / den, ws[2] / den];
    (0..m).all(|k| gemv_row(cross(sub(q[k], cf), sub(q[(k + 1) % m], cf)), n) > 1e-9 * a)
}

/// _join_cell_faces: the faces of one cell in one patch -> (faces, number joined away)
fn join_cell(p: &[V3], polys: &[&[i64]], cmin: f64, cpart: f64, flat_tol: f64) -> (Vec<Vec<i64>>, usize) {
    let nf = polys.len();
    let nrm: Vec<V3> = polys
        .iter()
        .map(|poly| {
            let q: Vec<V3> = poly.iter().map(|&v| p[v as usize]).collect();
            let m = q.len();
            let c = mean(&q);
            let t: Vec<V3> = (0..m).map(|k| cross(sub(q[k], c), sub(q[(k + 1) % m], c))).collect();
            let s = rows_sum(&t);
            let d = norm1(s).max(1e-300);
            [s[0] / d, s[1] / d, s[2] / d]
        })
        .collect();
    let mut par: Vec<usize> = (0..nf).collect();
    fn fnd(par: &mut [usize], mut i: usize) -> usize {
        while par[i] != i {
            par[i] = par[par[i]];
            i = par[i];
        }
        i
    }
    let mut emap: OMap<(i64, i64), usize> = OMap::new();
    for (i, q) in polys.iter().enumerate() {
        let m = q.len();
        for k in 0..m {
            *emap.get_mut_or((q[k], q[(k + 1) % m]), i) = i;
        }
    }
    for t in 0..emap.keys.len() {
        let (a, b) = emap.keys[t];
        let i = emap.vals[t];
        if let Some(&j) = emap.get(&(b, a)) {
            if j != i && dot1(nrm[i], nrm[j]) > cmin {
                let rj = fnd(&mut par, j);
                let ri = fnd(&mut par, i);
                par[ri] = rj;
            }
        }
    }
    let mut groups: OMap<usize, Vec<usize>> = OMap::new();
    for i in 0..nf {
        let r = fnd(&mut par, i);
        groups.get_mut_or(r, Vec::new()).push(i);
    }
    let mut res: Vec<Vec<i64>> = Vec::with_capacity(nf);
    let mut joined = 0usize;
    for ids in &groups.vals {
        if ids.len() == 1 {
            res.push(polys[ids[0]].to_vec());
            continue;
        }
        let mut de: OMap<(i64, i64), usize> = OMap::new();
        for &i in ids {
            let q = polys[i];
            let m = q.len();
            for k in 0..m {
                *de.get_mut_or((q[k], q[(k + 1) % m]), 0) += 1;
            }
        }
        let segs: Vec<(i64, i64)> = de.keys.iter().copied().filter(|&(a, b)| de.get(&(b, a)).is_none()).collect();
        let mut nxt: FxMap<i64, i64> = FxMap::default();
        let mut ok = true;
        for &(a, b) in &segs {
            if nxt.contains_key(&a) {
                ok = false;
                break;
            }
            nxt.insert(a, b);
        }
        let mut lp: Vec<i64> = Vec::new();
        let mut a: Option<i64> = None;
        let mut a0: Option<i64> = None;
        if ok && !segs.is_empty() {
            a0 = Some(segs[0].0);
            a = a0;
            for _ in 0..segs.len() + 1 {
                lp.push(a.unwrap());
                a = nxt.get(&a.unwrap()).copied();
                if a.is_none() || a == a0 {
                    break;
                }
            }
        }
        let parts: Vec<&[i64]> = ids.iter().map(|&i| polys[i]).collect();
        let nrms: Vec<V3> = ids.iter().map(|&i| nrm[i]).collect();
        if ok
            && a0.is_some()
            && a == a0
            && lp.len() == segs.len()
            && lp.len() >= 3
            && merged_face_ok(p, &lp, &parts, &nrms, cpart, flat_tol)
        {
            res.push(lp);
            joined += ids.len() - 1;
        } else {
            for &i in ids {
                res.push(polys[i].to_vec());
            }
        }
    }
    (res, joined)
}

/// merge_coplanar_wall_faces(P, nn, vv, owner, nI, patches) with the patches given as
/// (startFace, nFaces) lists: (nn, vv, owner, nFaces per patch, number joined)
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn merge_wall_faces<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    nn: PyReadonlyArray1<'py, i64>,
    vv: PyReadonlyArray1<'py, i64>,
    owner: PyReadonlyArray1<'py, i64>,
    n_internal: usize,
    starts: Vec<usize>,
    counts: Vec<usize>,
    cmin: f64,
    cpart: f64,
    flat_tol: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Vec<usize>, usize)> {
    let p = crate::points(&points)?;
    let nn = nn.as_slice()?;
    let vv = vv.as_slice()?;
    let owner = owner.as_slice()?;
    let mut ptr = vec![0usize; nn.len() + 1];
    for (i, &n) in nn.iter().enumerate() {
        ptr[i + 1] = ptr[i] + n as usize;
    }
    // runs of several faces of one cell, per patch
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (&f0, &nf) in starts.iter().zip(&counts) {
        let mut s = f0;
        for f in f0 + 1..=f0 + nf {
            if f == f0 + nf || owner[f] != owner[f - 1] {
                if f - s > 1 {
                    runs.push((s, f));
                }
                s = f;
            }
        }
    }
    let joined_runs: Vec<(Vec<Vec<i64>>, usize)> = py.detach(|| {
        runs.par_iter()
            .map(|&(f, g)| {
                let polys: Vec<&[i64]> = (f..g).map(|k| &vv[ptr[k]..ptr[k + 1]]).collect();
                join_cell(&p, &polys, cmin, cpart, flat_tol)
            })
            .collect()
    });
    let mut on: Vec<i64> = Vec::with_capacity(nn.len());
    let mut ov: Vec<i64> = Vec::with_capacity(vv.len());
    let mut oo: Vec<i64> = Vec::with_capacity(owner.len());
    on.extend_from_slice(&nn[..n_internal]);
    ov.extend_from_slice(&vv[..ptr[n_internal]]);
    oo.extend_from_slice(&owner[..n_internal]);
    let mut new_counts = Vec::with_capacity(counts.len());
    let mut joined = 0usize;
    let mut r = 0usize;
    for (&f0, &nf) in starts.iter().zip(&counts) {
        let mut f = f0;
        let mut cnt = 0usize;
        while f < f0 + nf {
            if r < runs.len() && runs[r].0 == f {
                let (polys, nj) = &joined_runs[r];
                joined += nj;
                for q in polys {
                    on.push(q.len() as i64);
                    ov.extend_from_slice(q);
                    oo.push(owner[f]);
                }
                cnt += polys.len();
                f = runs[r].1;
                r += 1;
            } else {
                on.push(nn[f]);
                ov.extend_from_slice(&vv[ptr[f]..ptr[f + 1]]);
                oo.push(owner[f]);
                cnt += 1;
                f += 1;
            }
        }
        new_counts.push(cnt);
    }
    Ok((on.into_pyarray(py), ov.into_pyarray(py), oo.into_pyarray(py), new_counts, joined))
}
