//! Plane clipping of the cells at convex sharp edges: port of featclip.PlaneClip's
//! planes, groups, convex_at / convex_many, clip_face and ls_compatible.
//!
//! The arithmetic is NumPy's as the Python code runs it: means and sums of rows in
//! order, row norms as sqrt((x0^2 + x1^2) + x2^2), 1-D dot products as an FMA chain
//! fma(a2,b2, fma(a1,b1, a0 b0)), and an (m,3) @ (3,k) product per element as that
//! chain when m >= 2 and k >= 2 or m = k = 1, else as fma(a2,b2, fma(a0,b0, a1 b1))
//! (pinned in tests/test_fast_paths.py NumpyOrder).  NumPy's max, minimum and clip
//! pass NaN on, argmin returns the first NaN.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

use crate::cut::FxMap;
use crate::wallmerge::{dot1, gemv_row, mean};
use crate::{TriMesh, V3};

/// one element of an (m,3) @ (3,k) product
#[inline]
fn mm(m: usize, k: usize, a: V3, b: V3) -> f64 {
    if (m >= 2 && k >= 2) || (m == 1 && k == 1) { dot1(a, b) } else { gemv_row(a, b) }
}

#[inline]
fn row_norm(x: V3) -> f64 {
    ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]).sqrt()
}

/// np.max of a 1-D array (NaN wins)
fn np_max(v: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::NEG_INFINITY;
    let mut first = true;
    for x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        if first || x > m {
            m = x;
            first = false;
        }
    }
    m
}

/// np.minimum(a, b) for scalars
#[inline]
fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if a < b { a } else { b }
}

/// np.clip(x, lo, hi) for scalars
#[inline]
fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() { f64::NAN } else if x < lo { lo } else if x > hi { hi } else { x }
}

/// np.argmin (the first NaN if any, else the first smallest)
fn np_argmin(t: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in t.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x < t[best] {
            best = i;
        }
    }
    best
}

#[inline]
fn axpy(a: V3, t: f64, b: V3) -> V3 {
    // a + t * (b - a)
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}

#[derive(Clone)]
enum Rec {
    V(i64),
    E(i64, i64, i64),
    F(i64, Vec<i64>),
}

struct FaceClip {
    recs: Vec<Rec>,
    p: Vec<V3>,
    poly: Vec<i64>,
}

#[pyclass(unsendable)]
pub struct PlaneClipRs {
    mesh: Py<TriMesh>,
    n: Vec<V3>,
    d: Vec<f64>,
    eps: f64,
    hmin: f64,
    cos_feat: f64,
    pts: Vec<V3>,
    phi: Vec<f64>,
    zero_edge: FxMap<(i64, i64, i64), i64>,
    cache: FxMap<i64, FaceClip>,
    face_planes: FxMap<i64, Vec<i64>>,
}

impl PlaneClipRs {
    fn tris_near(&self, c: V3, r: f64) -> Vec<i64> {
        self.mesh.get().near_one(c, r * (1.0 + 1e-9) + self.eps)
    }

    fn planes_of(&self, tris: &[i64]) -> Vec<i64> {
        let mut keep: Vec<i64> = Vec::new();
        for &t in tris {
            let (nt, dt) = (self.n[t as usize], self.d[t as usize]);
            if !keep.iter().any(|&k| (dt - self.d[k as usize]).abs() < self.eps && dot1(nt, self.n[k as usize]) > 1.0 - 1e-12) {
                keep.push(t);
            }
        }
        keep
    }

    fn groups_of(&self, pl: &[i64]) -> Vec<i64> {
        let k = pl.len();
        let mut g: Vec<usize> = (0..k).collect();
        fn find(g: &mut [usize], mut i: usize) -> usize {
            while g[i] != i {
                g[i] = g[g[i]];
                i = g[i];
            }
            i
        }
        for i in 0..k {
            for j in i + 1..k {
                // N @ N.T: (k,3) @ (3,k), k >= 2
                if mm(k, k, self.n[pl[i] as usize], self.n[pl[j] as usize]) > self.cos_feat {
                    let rj = find(&mut g, j);
                    let ri = find(&mut g, i);
                    g[ri] = rj;
                }
            }
        }
        let roots: Vec<usize> = (0..k).map(|i| find(&mut g, i)).collect();
        let mut u: Vec<usize> = roots.clone();
        u.sort_unstable();
        u.dedup();
        roots.iter().map(|r| u.binary_search(r).unwrap() as i64).collect()
    }

    /// the cell points P with their convexity at radius r (convex_at, given the triangles)
    fn convex_with(&self, p: &[V3], tris: &[i64]) -> (bool, Vec<i64>) {
        if tris.is_empty() {
            return (false, Vec::new());
        }
        let pl = self.planes_of(tris);
        let g = self.groups_of(&pl);
        let gmax = *g.iter().max().unwrap();
        if gmax > 0 && g.len() as i64 > gmax + 1 {
            return (false, pl);
        }
        let mut lo = p[0];
        let mut hi = p[0];
        for x in &p[1..] {
            for k in 0..3 {
                // np.min / np.max along the axis (NaN wins)
                if x[k].is_nan() || lo[k].is_nan() {
                    lo[k] = f64::NAN;
                } else if x[k] < lo[k] {
                    lo[k] = x[k];
                }
                if x[k].is_nan() || hi[k].is_nan() {
                    hi[k] = f64::NAN;
                } else if x[k] > hi[k] {
                    hi[k] = x[k];
                }
            }
        }
        let step = 1.0 / 3.0;
        let lin = [0.0, step, 2.0 * step, 1.0];
        let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let mut x: Vec<V3> = p.to_vec();
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let u = [lin[a], lin[b], lin[c]];
                    x.push([lo[0] + ext[0] * u[0], lo[1] + ext[1] * u[1], lo[2] + ext[2] * u[2]]);
                }
            }
        }
        let (m, k) = (x.len(), pl.len());
        let tol = 1e-6 * self.hmin;
        let mesh = self.mesh.get();
        let bad = x.iter().any(|&q| {
            let s = np_max(pl.iter().map(|&j| mm(m, k, q, self.n[j as usize]) - self.d[j as usize]));
            let ph = mesh.sdf_one(q);
            (ph < -tol && s > tol) || (ph > tol && s < -tol)
        });
        (!bad, pl)
    }

    fn common_edge(r1: &Rec, r2: &Rec, edge_of_v: &FxMap<(i64, i64), (i64, i64)>) -> Option<(i64, i64)> {
        // edges(r): {(lo, hi)} for 'e', None for 'v', {} for 'f'
        let edges = |r: &Rec| -> Option<Vec<(i64, i64)>> {
            match r {
                Rec::E(a, b, _) => Some(vec![(*a, *b)]),
                Rec::V(_) => None,
                Rec::F(..) => Some(Vec::new()),
            }
        };
        let id = |r: &Rec| match r {
            Rec::V(v) => *v,
            Rec::E(a, ..) => *a,
            Rec::F(f, _) => *f,
        };
        match (edges(r1), edges(r2)) {
            (None, None) => edge_of_v.get(&(id(r1), id(r2))).copied(),
            (None, Some(e2)) => e2.into_iter().find(|e| e.0 == id(r1) || e.1 == id(r1)),
            (Some(e1), None) => e1.into_iter().find(|e| e.0 == id(r2) || e.1 == id(r2)),
            (Some(e1), Some(e2)) => e1.into_iter().find(|e| e2.contains(e)),
        }
    }

    fn clip_face_rs(&mut self, fid: i64, poly: &[i64]) -> (Vec<Rec>, Vec<V3>) {
        if let Some(h) = self.cache.get(&fid) {
            if h.poly == poly {
                return (h.recs.clone(), h.p.clone());
            }
            // the other cell: the same face reversed
            let mut r = h.recs.clone();
            let mut p = h.p.clone();
            r.reverse();
            p.reverse();
            return (r, p);
        }
        let x: Vec<V3> = poly.iter().map(|&v| self.pts[v as usize]).collect();
        let m = poly.len();
        let c = mean(&x);
        let r = np_max(x.iter().map(|&q| row_norm([q[0] - c[0], q[1] - c[1], q[2] - c[2]])));
        let tris = self.tris_near(c, r);
        let pl = self.planes_of(&tris);
        let store = |me: &mut Self, recs: Vec<Rec>, p: Vec<V3>, pl: Vec<i64>| {
            me.cache.insert(fid, FaceClip { recs: recs.clone(), p: p.clone(), poly: poly.to_vec() });
            me.face_planes.insert(fid, pl);
            (recs, p)
        };
        if pl.is_empty() {
            // no wall within reach: the whole face is on the side of its vertices
            let inside = poly.iter().all(|&v| self.phi[v as usize] <= 0.0);
            let (recs, p) = if inside { (poly.iter().map(|&v| Rec::V(v)).collect(), x.clone()) } else { (Vec::new(), Vec::new()) };
            return store(self, recs, p, Vec::new());
        }
        let grp = self.groups_of(&pl);
        let gmax = *grp.iter().max().unwrap();
        if gmax == 0 {
            // no sharp crease near the face: the level-set rule with exact crossings
            let ph: Vec<f64> = poly.iter().map(|&v| self.phi[v as usize]).collect();
            let (mut recs, mut p) = (Vec::new(), Vec::new());
            if !ph.iter().any(|&v| v > 0.0) {
                recs = poly.iter().map(|&v| Rec::V(v)).collect();
                p = x.clone();
            } else if !ph.iter().any(|&v| v < 0.0) {
            } else {
                let k = pl.len();
                for i in 0..m {
                    let j = (i + 1) % m;
                    if ph[i] <= 0.0 {
                        recs.push(Rec::V(poly[i]));
                        p.push(x[i]);
                    }
                    if ph[i] * ph[j] < 0.0 {
                        let (a, b) = if ph[i] < 0.0 { (i, j) } else { (j, i) };
                        // Np @ X[a] - Dp: (k,3) @ (3,)
                        let mv = |q: V3, t: i64| (if k >= 2 { gemv_row(self.n[t as usize], q) } else { dot1(self.n[t as usize], q) }) - self.d[t as usize];
                        let sa: Vec<f64> = pl.iter().map(|&t| mv(x[a], t)).collect();
                        let sb: Vec<f64> = pl.iter().map(|&t| mv(x[b], t)).collect();
                        let (kk, tt) = if sb.iter().any(|&v| v > 0.0) {
                            let t: Vec<f64> = (0..k)
                                .map(|q| {
                                    if sb[q] > 0.0 {
                                        let ca = np_minimum(sa[q], 0.0);
                                        ca / (ca - sb[q])
                                    } else {
                                        2.0
                                    }
                                })
                                .collect();
                            let kk = np_argmin(&t);
                            (kk, np_clip(t[kk], 0.0, 1.0))
                        } else {
                            (0, 0.5)
                        };
                        let (lo, hi) = (poly[i].min(poly[j]), poly[i].max(poly[j]));
                        recs.push(Rec::E(lo, hi, pl[kk]));
                        p.push(axpy(x[a], tt, x[b]));
                    }
                }
            }
            return store(self, recs, p, pl);
        }
        let mut recs: Vec<Rec> = poly.iter().map(|&v| Rec::V(v)).collect();
        let mut p: Vec<V3> = x.clone();
        let mut edge_of_v: FxMap<(i64, i64), (i64, i64)> = FxMap::default();
        for i in 0..m {
            let (a, b) = (poly[i], poly[(i + 1) % m]);
            let e = (a.min(b), a.max(b));
            edge_of_v.insert((a, b), e);
            edge_of_v.insert((b, a), e);
        }
        let eps = self.eps;
        let tol6 = 1e-6 * self.hmin;
        for q in 0..=gmax {
            if recs.is_empty() {
                break;
            }
            let ks: Vec<i64> = pl.iter().zip(&grp).filter(|p_| *p_.1 == q).map(|p_| *p_.0).collect();
            let (l, kq) = (p.len(), ks.len());
            // S = np.array(P) @ Nq.T - Dq  (vertex, plane)
            let s_mat: Vec<Vec<f64>> =
                p.iter().map(|&y| ks.iter().map(|&t| mm(l, kq, y, self.n[t as usize]) - self.d[t as usize]).collect()).collect();
            let s: Vec<f64> = s_mat
                .iter()
                .map(|row| {
                    let v = np_max(row.iter().copied());
                    if v.abs() < eps { 0.0 } else { v }
                })
                .collect();
            if s.iter().all(|&v| v <= 0.0) {
                continue;
            }
            let (mut nr, mut np_) = (Vec::new(), Vec::new());
            for i in 0..l {
                let j = (i + 1) % l;
                if s[i] <= 0.0 {
                    nr.push(recs[i].clone());
                    np_.push(p[i]);
                }
                if (s[i] < 0.0 && 0.0 < s[j]) || (s[j] < 0.0 && 0.0 < s[i]) {
                    let (a, b) = if s[i] < 0.0 { (i, j) } else { (j, i) };
                    let t: Vec<f64> = (0..kq)
                        .map(|w| {
                            let sa = np_minimum(s_mat[a][w], 0.0);
                            let sb = s_mat[b][w];
                            if sb > 0.0 { -sa / (sb - sa) } else { f64::INFINITY }
                        })
                        .collect();
                    let kk = np_argmin(&t);
                    let y = axpy(p[a], np_clip(t[kk], 0.0, 1.0), p[b]);
                    match Self::common_edge(&recs[i], &recs[j], &edge_of_v) {
                        Some(e) => nr.push(Rec::E(e.0, e.1, ks[kk])),
                        None => {
                            let mut on: Vec<i64> = pl
                                .iter()
                                .copied()
                                .filter(|&k| (dot1(self.n[k as usize], y) - self.d[k as usize]).abs() < tol6)
                                .collect();
                            on.sort_unstable();
                            nr.push(Rec::F(fid, on));
                        }
                    }
                    np_.push(y);
                }
            }
            recs = nr;
            p = np_;
        }
        if recs.len() < 3 {
            recs.clear();
            p.clear();
        } else if !(recs.len() == m && recs.iter().all(|r| matches!(r, Rec::V(_)))) {
            // a part lying in one wall plane has no fluid on either side
            let (l, k) = (p.len(), pl.len());
            let on_all = pl.iter().any(|&t| {
                p.iter().all(|&y| (mm(l, k, y, self.n[t as usize]) - self.d[t as usize]).abs() < tol6)
            });
            if on_all {
                recs.clear();
                p.clear();
            }
        }
        store(self, recs, p, pl)
    }

    fn rec_py<'py>(py: Python<'py>, r: &Rec) -> PyResult<Bound<'py, PyTuple>> {
        match r {
            Rec::V(v) => PyTuple::new(py, [("v").into_pyobject(py)?.into_any(), v.into_pyobject(py)?.into_any()]),
            Rec::E(a, b, k) => PyTuple::new(
                py,
                [
                    ("e").into_pyobject(py)?.into_any(),
                    a.into_pyobject(py)?.into_any(),
                    b.into_pyobject(py)?.into_any(),
                    k.into_pyobject(py)?.into_any(),
                ],
            ),
            Rec::F(f, ks) => {
                let mut items = vec![("f").into_pyobject(py)?.into_any(), f.into_pyobject(py)?.into_any()];
                for k in ks {
                    items.push(k.into_pyobject(py)?.into_any());
                }
                PyTuple::new(py, items)
            }
        }
    }
}

#[pymethods]
impl PlaneClipRs {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mesh: Py<TriMesh>,
        n: PyReadonlyArray2<f64>,
        d: PyReadonlyArray1<f64>,
        eps: f64,
        hmin: f64,
        cos_feat: f64,
        pts: PyReadonlyArray2<f64>,
        phi: PyReadonlyArray1<f64>,
        zero_edge: FxMap<(i64, i64, i64), i64>,
    ) -> PyResult<Self> {
        let n = crate::points(&n)?;
        let d = d.as_slice()?.to_vec();
        let pts = crate::points(&pts)?;
        let phi = phi.as_slice()?.to_vec();
        if n.len() != d.len() || phi.len() != pts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("inconsistent plane clip data"));
        }
        Ok(PlaneClipRs {
            mesh,
            n,
            d,
            eps,
            hmin,
            cos_feat,
            pts,
            phi,
            zero_edge,
            cache: FxMap::default(),
            face_planes: FxMap::default(),
        })
    }

    fn planes(&self, tris: Vec<i64>) -> PyResult<Vec<i64>> {
        if tris.iter().any(|&t| t < 0 || t as usize >= self.n.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("triangle id out of range"));
        }
        Ok(self.planes_of(&tris))
    }

    fn groups(&self, pl: Vec<i64>) -> PyResult<Vec<i64>> {
        if pl.iter().any(|&t| t < 0 || t as usize >= self.n.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("triangle id out of range"));
        }
        Ok(self.groups_of(&pl))
    }

    /// convex_at(P, r) -> (convex, planes)
    fn convex_at(&self, p: PyReadonlyArray2<f64>, r: f64) -> PyResult<(bool, Vec<i64>)> {
        let p = crate::points(&p)?;
        if p.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("no points"));
        }
        let tris = self.tris_near(mean(&p), r);
        Ok(self.convex_with(&p, &tris))
    }

    /// convex_many([P], [r]) -> [convex] (in parallel)
    fn convex_many(&self, py: Python<'_>, ps: Vec<PyReadonlyArray2<f64>>, rs: Vec<f64>) -> PyResult<Vec<bool>> {
        if ps.len() != rs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("one radius per cell"));
        }
        let ps: Vec<Vec<V3>> = ps.iter().map(crate::points).collect::<PyResult<_>>()?;
        if ps.iter().any(|p| p.is_empty()) {
            return Err(pyo3::exceptions::PyValueError::new_err("no points"));
        }
        Ok(py.detach(|| {
            ps.par_iter().zip(rs.par_iter()).map(|(p, &r)| self.convex_with(p, &self.tris_near(mean(p), r)).0).collect()
        }))
    }

    /// clip_face(fid, poly) -> (records, points), cached per face
    fn clip_face<'py>(&mut self, py: Python<'py>, fid: i64, poly: Vec<i64>) -> PyResult<(Vec<Bound<'py, PyTuple>>, Vec<[f64; 3]>)> {
        if poly.iter().any(|&v| v < 0 || v as usize >= self.pts.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
        }
        let (recs, p) = self.clip_face_rs(fid, &poly);
        let r: Vec<Bound<'py, PyTuple>> = recs.iter().map(|r| Self::rec_py(py, r)).collect::<PyResult<_>>()?;
        Ok((r, p))
    }

    fn face_planes(&self, fid: i64) -> Option<Vec<i64>> {
        self.face_planes.get(&fid).cloned()
    }

    /// ls_compatible(fid, poly)
    fn ls_compatible(&mut self, fid: i64, poly: Vec<i64>) -> PyResult<bool> {
        if poly.iter().any(|&v| v < 0 || v as usize >= self.pts.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
        }
        let (recs, _) = self.clip_face_rs(fid, &poly);
        let ph: Vec<f64> = poly.iter().map(|&v| self.phi[v as usize]).collect();
        let m = poly.len();
        if !ph.iter().any(|&v| v > 0.0) {
            return Ok(recs.len() == m && recs.iter().all(|r| matches!(r, Rec::V(_))));
        }
        if !ph.iter().any(|&v| v < 0.0) {
            return Ok(recs.is_empty());
        }
        if recs.iter().any(|r| matches!(r, Rec::F(..))) {
            return Ok(false);
        }
        if !self.zero_edge.is_empty() {
            for i in 0..m {
                let (a, b) = (poly[i], poly[(i + 1) % m]);
                if ph[i] == 0.0 && ph[(i + 1) % m] == 0.0 && self.zero_edge.contains_key(&(fid, a.min(b), a.max(b))) {
                    return Ok(false);
                }
            }
        }
        let mut want_v: Vec<i64> = poly.iter().zip(&ph).filter(|p_| *p_.1 <= 0.0).map(|p_| *p_.0).collect();
        let mut want_e: Vec<(i64, i64)> = (0..m)
            .filter(|&i| ph[i] * ph[(i + 1) % m] < 0.0)
            .map(|i| (poly[i].min(poly[(i + 1) % m]), poly[i].max(poly[(i + 1) % m])))
            .collect();
        let mut got_v: Vec<i64> = recs.iter().filter_map(|r| if let Rec::V(v) = r { Some(*v) } else { None }).collect();
        let got_e: Vec<(i64, i64)> = recs.iter().filter_map(|r| if let Rec::E(a, b, _) = r { Some((*a, *b)) } else { None }).collect();
        for v in [&mut want_v, &mut got_v] {
            v.sort_unstable();
            v.dedup();
        }
        want_e.sort_unstable();
        want_e.dedup();
        let mut ge = got_e.clone();
        ge.sort_unstable();
        ge.dedup();
        Ok(got_v == want_v && got_e.len() == want_e.len() && ge == want_e)
    }
}
