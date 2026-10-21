//! Fast geometry queries for cutmesh: closest point / inside test / signed distance on
//! a triangulated surface, and closest point on a set of segments (feature edges).
//!
//! Each query runs over the points in parallel (rayon) with the GIL released.  The
//! results reproduce the Python implementation in cutmesh/geometry.py:
//! - `inside` uses the same +x ray parity, perturbation and hit test (identical bits),
//! - `closest` is exact (branch and bound on a BVH), the same closest point formula.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

mod cut;

pub(crate) type V3 = [f64; 3];

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn axpy(a: V3, s: f64, b: V3) -> V3 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}
#[inline]
fn dist2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

// ------------------------------------------------------------------------- BVH
#[derive(Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    // leaf: count > 0, prims[start..start+count]; inner: children start, start+1... (left, right)
    start: u32,
    count: u32,
    right: u32,
}

struct Bvh {
    nodes: Vec<Node>,
    prims: Vec<u32>,
}

const LEAF: usize = 4;

impl Bvh {
    fn build(lo: &[V3], hi: &[V3]) -> Bvh {
        let n = lo.len();
        let cen: Vec<V3> = (0..n)
            .map(|i| [0.5 * (lo[i][0] + hi[i][0]), 0.5 * (lo[i][1] + hi[i][1]), 0.5 * (lo[i][2] + hi[i][2])])
            .collect();
        let mut prims: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n / LEAF + 1);
        if n > 0 {
            Self::split(&mut nodes, &mut prims, 0, n, lo, hi, &cen);
        }
        Bvh { nodes, prims }
    }

    fn split(nodes: &mut Vec<Node>, prims: &mut [u32], s: usize, e: usize, lo: &[V3], hi: &[V3], cen: &[V3]) -> usize {
        let mut blo = [f64::INFINITY; 3];
        let mut bhi = [f64::NEG_INFINITY; 3];
        let mut clo = [f64::INFINITY; 3];
        let mut chi = [f64::NEG_INFINITY; 3];
        for &p in &prims[s..e] {
            let p = p as usize;
            for k in 0..3 {
                blo[k] = blo[k].min(lo[p][k]);
                bhi[k] = bhi[k].max(hi[p][k]);
                clo[k] = clo[k].min(cen[p][k]);
                chi[k] = chi[k].max(cen[p][k]);
            }
        }
        let id = nodes.len();
        nodes.push(Node { lo: blo, hi: bhi, start: s as u32, count: (e - s) as u32, right: 0 });
        let ext = [chi[0] - clo[0], chi[1] - clo[1], chi[2] - clo[2]];
        let ax = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 } else if ext[1] >= ext[2] { 1 } else { 2 };
        if e - s <= LEAF || ext[ax] <= 0.0 {
            return id;
        }
        let m = (s + e) / 2;
        prims[s..e].select_nth_unstable_by(m - s, |a, b| {
            cen[*a as usize][ax].partial_cmp(&cen[*b as usize][ax]).unwrap()
        });
        Self::split(nodes, prims, s, m, lo, hi, cen);
        let r = Self::split(nodes, prims, m, e, lo, hi, cen);
        nodes[id].count = 0;
        nodes[id].start = (id + 1) as u32;
        nodes[id].right = r as u32;
        id
    }

    /// closest primitive: f(prim) -> (distance^2, closest point)
    fn closest<F: Fn(usize) -> (f64, V3)>(&self, p: V3, f: F) -> (f64, V3, usize) {
        let mut best = (f64::INFINITY, [0.0; 3], 0usize);
        if self.nodes.is_empty() {
            return best;
        }
        let mut stack: Vec<(u32, f64)> = Vec::with_capacity(64);
        stack.push((0, box_d2(&self.nodes[0], p)));
        while let Some((ni, bd)) = stack.pop() {
            if bd >= best.0 {
                continue;
            }
            let nd = &self.nodes[ni as usize];
            if nd.count > 0 {
                for &q in &self.prims[nd.start as usize..(nd.start + nd.count) as usize] {
                    let (d2, cp) = f(q as usize);
                    if d2 < best.0 {
                        best = (d2, cp, q as usize);
                    }
                }
            } else {
                let (a, b) = (nd.start, nd.right);
                let (da, db) = (box_d2(&self.nodes[a as usize], p), box_d2(&self.nodes[b as usize], p));
                // nearer child popped first
                if da <= db {
                    stack.push((b, db));
                    stack.push((a, da));
                } else {
                    stack.push((a, da));
                    stack.push((b, db));
                }
            }
        }
        best
    }
}

impl Bvh {
    /// primitives whose centre (given by `centre`) lies within r of p, ascending
    fn within<F: Fn(usize) -> V3>(&self, p: V3, r: f64, centre: F, out: &mut Vec<usize>) {
        if self.nodes.is_empty() {
            return;
        }
        let r2 = r * r;
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if box_d2(n, p) > r2 {
                continue;
            }
            if n.count > 0 {
                for &q in &self.prims[n.start as usize..(n.start + n.count) as usize] {
                    if dist2(centre(q as usize), p) <= r2 {
                        out.push(q as usize);
                    }
                }
            } else {
                stack.push(n.start);
                stack.push(n.right);
            }
        }
        out.sort_unstable();
    }
}

#[inline]
fn box_d2(n: &Node, p: V3) -> f64 {
    let mut d = 0.0;
    for k in 0..3 {
        let v = if p[k] < n.lo[k] { n.lo[k] - p[k] } else if p[k] > n.hi[k] { p[k] - n.hi[k] } else { 0.0 };
        d += v * v;
    }
    d
}

// ------------------------------------------------------------------------- primitives
/// closest point on triangle abc (Ericson, Real-Time Collision Detection 5.1.5),
/// same case order as closest_point_triangles() in geometry.py
fn closest_tri(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let den = d1 - d3;
        let t = d1 / if den != 0.0 { den } else { 1.0 };
        return axpy(a, t, ab);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let den = d2 - d6;
        let t = d2 / if den != 0.0 { den } else { 1.0 };
        return axpy(a, t, ac);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let e = (d4 - d3) + (d5 - d6);
        let t = (d4 - d3) / if e != 0.0 { e } else { 1.0 };
        return axpy(b, t, sub(c, b));
    }
    let den = va + vb + vc;
    let den = if den.abs() > 0.0 { den } else { 1.0 };
    let v = vb / den;
    let w = vc / den;
    axpy(axpy(a, v, ab), w, ac)
}

fn closest_seg(p: V3, a: V3, b: V3) -> V3 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = (dot(sub(p, a), ab) / l2.max(1e-300)).clamp(0.0, 1.0);
    axpy(a, t, ab)
}

#[inline]
fn cr(u: V3, v: V3, w: V3) -> f64 {
    // 2D cross in (y, z) of (v - u) x (w - u), as in Surface.inside()
    (v[1] - u[1]) * (w[2] - u[2]) - (v[2] - u[2]) * (w[1] - u[1])
}

// ------------------------------------------------------------------------- helpers
pub(crate) fn points(p: &PyReadonlyArray2<f64>) -> PyResult<Vec<V3>> {
    let a = p.as_array();
    if a.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("points must have shape (n, 3)"));
    }
    Ok(a.outer_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

pub(crate) fn vec3_array<'py>(py: Python<'py>, v: Vec<V3>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = v.len();
    let flat: Vec<f64> = v.into_iter().flatten().collect();
    flat.into_pyarray(py).reshape([n, 3])
}

// ------------------------------------------------------------------------- TriMesh
/// Triangulated closed surface: TriMesh(tris[n, 3, 3])
#[pyclass(frozen)]
struct TriMesh {
    t: Vec<[V3; 3]>,
    bvh: Bvh,
    pert: V3,
}

impl TriMesh {
    fn closest_one(&self, p: V3) -> (f64, V3, usize) {
        self.bvh.closest(p, |i| {
            let [a, b, c] = self.t[i];
            let q = closest_tri(p, a, b, c);
            (dist2(p, q), q)
        })
    }

    fn inside_one(&self, p: V3) -> bool {
        let q = [p[0] + self.pert[0], p[1] + self.pert[1], p[2] + self.pert[2]];
        let mut parity = false;
        if self.bvh.nodes.is_empty() {
            return false;
        }
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let n = &self.bvh.nodes[ni as usize];
            if q[1] < n.lo[1] || q[1] > n.hi[1] || q[2] < n.lo[2] || q[2] > n.hi[2] || n.hi[0] < q[0] {
                continue;
            }
            if n.count > 0 {
                for &i in &self.bvh.prims[n.start as usize..(n.start + n.count) as usize] {
                    let [a, b, c] = self.t[i as usize];
                    let w0 = cr(a, b, q);
                    let w1 = cr(b, c, q);
                    let w2 = cr(c, a, q);
                    let hit = (w0 > 0.0 && w1 > 0.0 && w2 > 0.0) || (w0 < 0.0 && w1 < 0.0 && w2 < 0.0);
                    if hit {
                        let sw = w0 + w1 + w2;
                        let sw = if sw != 0.0 { sw } else { 1.0 };
                        let x = (w1 * a[0] + w2 * b[0] + w0 * c[0]) / sw;
                        if x > q[0] {
                            parity = !parity;
                        }
                    }
                }
            } else {
                stack.push(n.start);
                stack.push(n.right);
            }
        }
        parity
    }
}

#[pymethods]
impl TriMesh {
    #[new]
    fn new(tris: PyReadonlyArray3<f64>) -> PyResult<Self> {
        let a = tris.as_array();
        if a.shape()[1] != 3 || a.shape()[2] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("tris must have shape (n, 3, 3)"));
        }
        let t: Vec<[V3; 3]> = a
            .outer_iter()
            .map(|m| [[m[[0, 0]], m[[0, 1]], m[[0, 2]]], [m[[1, 0]], m[[1, 1]], m[[1, 2]]], [m[[2, 0]], m[[2, 1]], m[[2, 2]]]])
            .collect();
        let mut lo = Vec::with_capacity(t.len());
        let mut hi = Vec::with_capacity(t.len());
        let mut blo = [f64::INFINITY; 3];
        let mut bhi = [f64::NEG_INFINITY; 3];
        for tri in &t {
            let mut l = [f64::INFINITY; 3];
            let mut h = [f64::NEG_INFINITY; 3];
            for v in tri {
                for k in 0..3 {
                    l[k] = l[k].min(v[k]);
                    h[k] = h[k].max(v[k]);
                }
            }
            for k in 0..3 {
                blo[k] = blo[k].min(l[k]);
                bhi[k] = bhi[k].max(h[k]);
            }
            lo.push(l);
            hi.push(h);
        }
        // the same perturbation of the ray origin as Surface.inside()
        let d = sub(bhi, blo);
        let len = dot(d, d).sqrt();
        let pert = [len * 0.0, len * 3.14159265e-7, len * 2.71828183e-7];
        let bvh = Bvh::build(&lo, &hi);
        Ok(TriMesh { t, bvh, pert })
    }

    /// (closest points[n, 3], distances[n], triangle ids[n])
    fn closest<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
        let pts = points(&p)?;
        let res: Vec<(f64, V3, usize)> = py.detach(|| pts.par_iter().map(|&q| self.closest_one(q)).collect());
        let d: Vec<f64> = res.iter().map(|r| r.0.sqrt()).collect();
        let id: Vec<i64> = res.iter().map(|r| r.2 as i64).collect();
        let cp: Vec<V3> = res.into_iter().map(|r| r.1).collect();
        Ok((vec3_array(py, cp)?, d.into_pyarray(py), id.into_pyarray(py)))
    }

    /// parity of +x ray crossings: true inside the closed surface
    fn inside<'py>(&self, py: Python<'py>, p: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let pts = points(&p)?;
        let r: Vec<bool> = py.detach(|| pts.par_iter().map(|&q| self.inside_one(q)).collect());
        Ok(r.into_pyarray(py))
    }

    /// signed distance (negative inside)
    fn sdf<'py>(&self, py: Python<'py>, p: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pts = points(&p)?;
        let r: Vec<f64> = py.detach(|| {
            pts.par_iter()
                .map(|&q| {
                    let d = self.closest_one(q).0.sqrt();
                    if self.inside_one(q) { -d } else { d }
                })
                .collect()
        });
        Ok(r.into_pyarray(py))
    }

    fn __len__(&self) -> usize {
        self.t.len()
    }
}

// ------------------------------------------------------------------------- SegSet
/// Set of segments (feature edges): SegSet(seg[n, 2, 3])
#[pyclass(frozen)]
struct SegSet {
    s: Vec<[V3; 2]>,
    bvh: Bvh,
    mid: Vec<V3>,
    half: f64,
}

#[pymethods]
impl SegSet {
    #[new]
    fn new(seg: PyReadonlyArray3<f64>) -> PyResult<Self> {
        let a = seg.as_array();
        if a.shape()[1] != 2 || a.shape()[2] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("segments must have shape (n, 2, 3)"));
        }
        let s: Vec<[V3; 2]> = a
            .outer_iter()
            .map(|m| [[m[[0, 0]], m[[0, 1]], m[[0, 2]]], [m[[1, 0]], m[[1, 1]], m[[1, 2]]]])
            .collect();
        let lo: Vec<V3> = s.iter().map(|e| [e[0][0].min(e[1][0]), e[0][1].min(e[1][1]), e[0][2].min(e[1][2])]).collect();
        let hi: Vec<V3> = s.iter().map(|e| [e[0][0].max(e[1][0]), e[0][1].max(e[1][1]), e[0][2].max(e[1][2])]).collect();
        let bvh = Bvh::build(&lo, &hi);
        let mid: Vec<V3> = s.iter().map(|e| [0.5 * (e[0][0] + e[1][0]), 0.5 * (e[0][1] + e[1][1]), 0.5 * (e[0][2] + e[1][2])]).collect();
        let half = s.iter().map(|e| 0.5 * dist2(e[0], e[1]).sqrt()).fold(0.0, f64::max);
        Ok(SegSet { s, bvh, mid, half })
    }

    /// (closest points[n, 3], distances[n]) - exact
    fn closest<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        let pts = points(&p)?;
        let query = |q: V3| {
            self.bvh.closest(q, |i| {
                let c = closest_seg(q, self.s[i][0], self.s[i][1]);
                (dist2(q, c), c)
            })
        };
        // small batches (one cell at a time) are cheaper without the thread pool
        let res: Vec<(f64, V3, usize)> = if pts.len() < 256 {
            pts.iter().map(|&q| query(q)).collect()
        } else {
            py.detach(|| pts.par_iter().map(|&q| query(q)).collect())
        };
        let d: Vec<f64> = res.iter().map(|r| r.0.sqrt()).collect();
        let cp: Vec<V3> = res.into_iter().map(|r| r.1).collect();
        Ok((vec3_array(py, cp)?, d.into_pyarray(py)))
    }

    /// Where a feature edge pierces a face that is cut along the chord a-b: the point
    /// to bend the chord through, or None (port of make_chord_hook in mesher.py).
    /// face: polygon vertices; kept: the face vertices on the kept (phi < 0) side.
    fn pierce(&self, face: Vec<V3>, a: V3, b: V3, kept: Vec<V3>) -> Option<V3> {
        let m = face.len();
        if m < 3 || self.s.is_empty() {
            return None;
        }
        let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut c0 = [0.0; 3];
        let mut nrm = [0.0; 3];
        for i in 0..m {
            let p = face[i];
            let c = cross(p, face[(i + 1) % m]);
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
                c0[k] += p[k];
                nrm[k] += c[k];
            }
        }
        for k in 0..3 {
            c0[k] /= m as f64;
        }
        let diam = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
        let mut idx = Vec::new();
        self.bvh.within(mid, diam + self.half, |i| self.mid[i], &mut idx);
        if idx.is_empty() {
            return None;
        }
        let nn = dot(nrm, nrm).sqrt();
        if nn == 0.0 {
            return None;
        }
        let n = [nrm[0] / nn, nrm[1] / nn, nrm[2] / nn];
        let u = sub(b, a);
        let len = dot(u, u).sqrt();
        if len == 0.0 {
            return None;
        }
        // side of the chord the kept vertices are on (mean)
        let sn = kept.iter().map(|&v| dot(cross(u, sub(v, a)), n)).sum::<f64>() / kept.len().max(1) as f64;
        let tol = -1e-9 * diam * diam;
        let mut best: Option<(f64, V3)> = None;
        for &i in &idx {
            let [s0, s1] = self.s[i];
            let sa = dot(sub(s0, c0), n);
            let sb = dot(sub(s1, c0), n);
            if !(sa * sb <= 0.0 && sa != sb) {
                continue;
            }
            let t = sa / (sa - sb);
            let x0 = axpy(s0, t, sub(s1, s0));
            let x = axpy(x0, -dot(sub(x0, c0), n), n);
            // inside the face polygon
            if (0..m).any(|k| dot(cross(sub(face[(k + 1) % m], face[k]), sub(x, face[k])), n) < tol) {
                continue;
            }
            // off the chord, but not farther than its length
            let w = cross(sub(x, a), u);
            let dl = dot(w, w).sqrt() / len;
            if !(dl > 1e-3 * diam && dl < len) {
                continue;
            }
            // a convex corner: on the other side of the chord than the kept vertices
            if !(dot(cross(u, sub(x, a)), n) * sn < 0.0) {
                continue;
            }
            let dm = dist2(x, mid);
            if best.is_none_or(|(d, _)| dm < d) {
                best = Some((dm, x));
            }
        }
        best.map(|(_, x)| x)
    }

    fn __len__(&self) -> usize {
        self.s.len()
    }
}


// ------------------------------------------------------------------------- face / cell geometry
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// centre and area vector of one face (quality.face_geometry: fan around the vertex
/// average, triangles weighted by their area projected on the face normal)
fn face_cs(pts: &[V3], f: &[usize]) -> (V3, V3) {
    let n = f.len() as f64;
    let mut e = [0.0; 3];
    for &v in f {
        for k in 0..3 {
            e[k] += pts[v][k];
        }
    }
    for k in 0..3 {
        e[k] /= n;
    }
    let m = f.len();
    let mut sum_n = [0.0; 3];
    let mut tri: Vec<(V3, V3)> = Vec::with_capacity(m);
    for i in 0..m {
        let a = pts[f[i]];
        let b = pts[f[(i + 1) % m]];
        let nv = cross(sub(b, a), sub(e, a));
        let c = [a[0] + b[0] + e[0], a[1] + b[1] + e[1], a[2] + b[2] + e[2]];
        for k in 0..3 {
            sum_n[k] += nv[k];
        }
        tri.push((nv, c));
    }
    let mag = dot(sum_n, sum_n).sqrt();
    let nh = if mag > 1e-300 { [sum_n[0] / mag, sum_n[1] / mag, sum_n[2] / mag] } else { [0.0; 3] };
    let mut sa = 0.0;
    let mut sac = [0.0; 3];
    for (nv, c) in &tri {
        let w = dot(*nv, nh);
        sa += w;
        for k in 0..3 {
            sac[k] += w * c[k];
        }
    }
    let ctr = if mag > 1e-300 && sa > 1e-300 { [sac[0] / (3.0 * sa), sac[1] / (3.0 * sa), sac[2] / (3.0 * sa)] } else { e };
    (ctr, [0.5 * sum_n[0], 0.5 * sum_n[1], 0.5 * sum_n[2]])
}

fn all_points(p: &PyReadonlyArray2<f64>) -> PyResult<Vec<V3>> {
    points(p)
}

/// face_geometry(points[n, 3], sizes[F], verts[sum sizes]) -> (centres[F, 3], area vectors[F, 3])
#[pyfunction]
fn face_geometry<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    sizes: PyReadonlyArray1<'py, i64>,
    verts: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let pts = all_points(&points)?;
    let n = sizes.as_array();
    let v = verts.as_array();
    let mut ptr = Vec::with_capacity(n.len() + 1);
    ptr.push(0usize);
    for &k in n.iter() {
        ptr.push(ptr.last().unwrap() + k as usize);
    }
    let vv: Vec<usize> = v.iter().map(|&x| x as usize).collect();
    if *ptr.last().unwrap() != vv.len() || vv.iter().any(|&x| x >= pts.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent face list"));
    }
    let res: Vec<(V3, V3)> = py.detach(|| (0..n.len()).into_par_iter().map(|i| face_cs(&pts, &vv[ptr[i]..ptr[i + 1]])).collect());
    let (c, s): (Vec<V3>, Vec<V3>) = res.into_iter().unzip();
    Ok((vec3_array(py, c)?, vec3_array(py, s)?))
}

/// cell_geometry(nC, owner[F], neighbour[nI], Cf[F, 3], Sf[F, 3]) -> (centres[nC, 3], volumes[nC])
#[pyfunction]
fn cell_geometry<'py>(
    py: Python<'py>,
    n_cells: usize,
    owner: PyReadonlyArray1<'py, i64>,
    neighbour: PyReadonlyArray1<'py, i64>,
    cf: PyReadonlyArray2<'py, f64>,
    sf: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let own = owner.as_array();
    let nei = neighbour.as_array();
    let cf = points(&cf)?;
    let sf = points(&sf)?;
    let ni = nei.len();
    let mut est = vec![[0.0; 3]; n_cells];
    let mut cnt = vec![0usize; n_cells];
    for f in 0..cf.len() {
        let o = own[f] as usize;
        for k in 0..3 {
            est[o][k] += cf[f][k];
        }
        cnt[o] += 1;
        if f < ni {
            let q = nei[f] as usize;
            for k in 0..3 {
                est[q][k] += cf[f][k];
            }
            cnt[q] += 1;
        }
    }
    for c in 0..n_cells {
        let m = cnt[c].max(1) as f64;
        for k in 0..3 {
            est[c][k] /= m;
        }
    }
    let mut vol = vec![0.0; n_cells];
    let mut cc = vec![[0.0; 3]; n_cells];
    for f in 0..cf.len() {
        let o = own[f] as usize;
        let po = dot(sf[f], sub(cf[f], est[o]));
        vol[o] += po;
        for k in 0..3 {
            cc[o][k] += po * (0.75 * cf[f][k] + 0.25 * est[o][k]);
        }
        if f < ni {
            let q = nei[f] as usize;
            let pn = -dot(sf[f], sub(cf[f], est[q]));
            vol[q] += pn;
            for k in 0..3 {
                cc[q][k] += pn * (0.75 * cf[f][k] + 0.25 * est[q][k]);
            }
        }
    }
    for c in 0..n_cells {
        let d = if vol[c].abs() > 1e-300 { vol[c] } else { 1.0 };
        for k in 0..3 {
            cc[c][k] /= d;
        }
        vol[c] /= 3.0;
    }
    Ok((vec3_array(py, cc)?, vol.into_pyarray(py)))
}

/// poly_cell(points[n, 3], polys: list of vertex lists) -> (Cf[k, 3], Sf[k, 3], volume, centre[3])
/// geometry of one cell given by its faces (outward), as poly_geom() in mesher.py
#[pyfunction]
fn poly_cell<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    polys: Vec<Vec<usize>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, f64, Bound<'py, PyArray1<f64>>)> {
    let a = points.as_array();
    let np_ = a.nrows();
    let get = |v: usize| [a[[v, 0]], a[[v, 1]], a[[v, 2]]];
    let mut cs = Vec::with_capacity(polys.len());
    for f in &polys {
        if f.iter().any(|&v| v >= np_) {
            return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
        }
        let loc: Vec<V3> = f.iter().map(|&v| get(v)).collect();
        let idx: Vec<usize> = (0..loc.len()).collect();
        cs.push(face_cs(&loc, &idx));
    }
    let k = cs.len().max(1) as f64;
    let mut est = [0.0; 3];
    for (c, _) in &cs {
        for j in 0..3 {
            est[j] += c[j] / k;
        }
    }
    let mut sv = 0.0;
    let mut acc = [0.0; 3];
    for (c, s) in &cs {
        let pv = dot(*s, sub(*c, est));
        sv += pv;
        for j in 0..3 {
            acc[j] += pv * (0.75 * c[j] + 0.25 * est[j]);
        }
    }
    let vol = sv / 3.0;
    let cc = if vol > 0.0 { [acc[0] / sv, acc[1] / sv, acc[2] / sv] } else { est };
    let (c, s): (Vec<V3>, Vec<V3>) = cs.into_iter().unzip();
    Ok((vec3_array(py, c)?, vec3_array(py, s)?, vol, cc.to_vec().into_pyarray(py)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TriMesh>()?;
    m.add_class::<SegSet>()?;
    m.add_function(wrap_pyfunction!(face_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(cell_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(poly_cell, m)?)?;
    m.add_function(wrap_pyfunction!(cut::cut_batch, m)?)?;
    Ok(())
}
