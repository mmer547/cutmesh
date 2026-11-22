//! The cap hook of the feature cut: port of mesher.make_feature_hook's hook and
//! _split_on_features.  A cap loop is split along the feature edges through the cell
//! (entry and exit points on the feature, or spokes to a feature corner inside), or a
//! feature point bulging out of the cap is fanned to.  New points get the ids
//! base, base + 1, ... in the order the Python code adds them.  NumPy's arithmetic:
//! means and sums of rows in order, 1-D norms and dots as an FMA chain, row norms as
//! sqrt((x0^2 + x1^2) + x2^2), a matrix-vector row as fma(a2,b2, fma(a0,b0, a1 b1)).

use numpy::{PyReadonlyArray2};
use pyo3::prelude::*;

use crate::cut::FxMap;
use crate::wallmerge::{cross, dot1, gemv_row, mean, norm1, rows_sum, sub};
use crate::{SegSet, V3};

pub(crate) struct HookOut {
    pub polys: Vec<Vec<i64>>,
    pub newp: Vec<V3>,
    pub newf: Vec<i64>,
    pub did: bool,
    pub fan: bool,
}

#[inline]
fn row_norm(x: V3) -> f64 {
    ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]).sqrt()
}

/// _newell(Q - Q.mean(0))
fn newell_c(q: &[V3]) -> V3 {
    let c = mean(q);
    let x: Vec<V3> = q.iter().map(|&a| sub(a, c)).collect();
    let m = x.len();
    let t: Vec<V3> = (0..m).map(|k| cross(x[k], x[(k + 1) % m])).collect();
    rows_sum(&t)
}

struct Split<'a> {
    segs: &'a SegSet,
    h: f64,
    base: i64,
    newp: Vec<V3>,
    newf: Vec<i64>,
}

impl Split<'_> {
    const TOL: f64 = 0.05;
    const DEPTH: u32 = 4;

    /// points strictly between A and B along the feature, or None
    fn path(&self, a: V3, b: V3, lev: u32) -> Option<Vec<V3>> {
        let m = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
        let (cp, d) = self.segs.closest_pd(m);
        if d < Self::TOL * self.h {
            return Some(Vec::new());
        }
        if lev >= Self::DEPTH || d > 0.5 * norm1(sub(b, a)) {
            return None;
        }
        let mut l = self.path(a, cp, lev + 1)?;
        let r = self.path(cp, b, lev + 1)?;
        l.push(cp);
        l.extend(r);
        Some(l)
    }

    /// non-degenerate pieces that fold convexly (outward normals, kept side inside)
    fn area_ok(&self, polys: &[Vec<V3>]) -> bool {
        let amin = 1e-3 * self.h * self.h;
        let mut nn: Vec<V3> = Vec::new();
        let mut cc: Vec<V3> = Vec::new();
        for q in polys {
            let nq = newell_c(q);
            let l = norm1(nq);
            if q.len() < 3 || l < 2.0 * amin {
                return false;
            }
            nn.push([nq[0] / l, nq[1] / l, nq[2] / l]);
            cc.push(mean(q));
        }
        for i in 0..polys.len() {
            for j in 0..polys.len() {
                if i != j && dot1(sub(cc[j], cc[i]), nn[i]) > 1e-3 * self.h {
                    return false;
                }
            }
        }
        true
    }

    fn add(&mut self, x: V3) -> i64 {
        let vid = self.base + self.newp.len() as i64;
        self.newp.push(x);
        self.newf.push(vid);
        vid
    }
}

/// np.argmin (first NaN, else first smallest)
fn argmin(t: &[f64]) -> usize {
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

fn split_on_features(sp: &mut Split, corners: &[V3], lp: &[i64], p: &[V3], onfeat: &[bool], reach: f64) -> Option<Vec<Vec<i64>>> {
    let m = lp.len();
    let h = sp.h;
    let f: Vec<usize> = (0..m).filter(|&k| onfeat[k] || sp.segs.closest_pd(p[k]).1 < 1e-6 * h).collect();
    if f.len() < 2 {
        return None;
    }
    // feature corner inside the cell: spokes from the corner to the entry points
    if f.len() >= 3 && !corners.is_empty() {
        let c = mean(p);
        let dc: Vec<f64> = corners.iter().map(|&x| row_norm(sub(x, c))).collect();
        let j = argmin(&dc);
        let x = corners[j];
        // np.min of the row norms (NaN wins)
        let mut dmin = f64::INFINITY;
        for &q in p {
            let d = row_norm(sub(q, x));
            if d.is_nan() || dmin.is_nan() {
                dmin = f64::NAN;
            } else if d < dmin {
                dmin = d;
            }
        }
        if dc[j] < reach * h && dmin > 1e-6 * h {
            let mut spk: Vec<usize> = Vec::new();
            let mut spath: FxMap<usize, Vec<V3>> = FxMap::default();
            for &k in &f {
                if let Some(t) = sp.path(p[k], x, 0) {
                    spk.push(k);
                    spath.insert(k, t);
                }
            }
            if spk.len() >= 2 {
                let ns = spk.len();
                let mut sect: Vec<(Vec<usize>, usize, usize)> = Vec::new();
                for i in 0..ns {
                    let (s0, s1) = (spk[i], spk[(i + 1) % ns]);
                    let span = (s1 + m - s0) % m;
                    let span = if span == 0 { m } else { span };
                    let q: Vec<usize> = (0..=span).map(|kk| (s0 + kk) % m).collect();
                    sect.push((q, s1, s0));
                }
                let geo: Vec<Vec<V3>> = sect
                    .iter()
                    .map(|(q, s1, s0)| {
                        let mut g: Vec<V3> = q.iter().map(|&k| p[k]).collect();
                        g.extend(spath[s1].iter().copied());
                        g.push(x);
                        g.extend(spath[s0].iter().rev().copied());
                        g
                    })
                    .collect();
                if sp.area_ok(&geo) {
                    let vx = sp.add(x);
                    let mut ids: FxMap<usize, Vec<i64>> = FxMap::default();
                    for &k in &spk {
                        let pts = spath[&k].clone();
                        let v: Vec<i64> = pts.into_iter().map(|y| sp.add(y)).collect();
                        ids.insert(k, v);
                    }
                    return Some(
                        sect.iter()
                            .map(|(q, s1, s0)| {
                                let mut poly: Vec<i64> = q.iter().map(|&k| lp[k]).collect();
                                poly.extend(ids[s1].iter().copied());
                                poly.push(vx);
                                poly.extend(ids[s0].iter().rev().copied());
                                poly
                            })
                            .collect(),
                    );
                }
            }
        }
    }
    // entry point -> exit point along the feature; accept non-crossing splits
    let mut polys: Vec<Vec<i64>> = vec![lp.to_vec()];
    let mut used = 0;
    let mut lpos: FxMap<i64, V3> = FxMap::default();
    for (k, &v) in lp.iter().enumerate() {
        lpos.insert(v, p[k]);
    }
    for a_ in 0..f.len() {
        for b_ in a_ + 1..f.len() {
            let (vi, vj) = (lp[f[a_]], lp[f[b_]]);
            for pi in 0..polys.len() {
                let q = polys[pi].clone();
                let (Some(ii), Some(jj)) = (q.iter().position(|&v| v == vi), q.iter().position(|&v| v == vj)) else {
                    continue;
                };
                let (i, j) = if ii < jj { (ii, jj) } else { (jj, ii) };
                if j - i == 1 || (i == 0 && j == q.len() - 1) {
                    break;
                }
                let Some(t) = sp.path(lpos[&vi], lpos[&vj], 0) else {
                    break;
                };
                let pos_of = |v: i64, sp: &Split| -> V3 {
                    match lpos.get(&v) {
                        Some(&x) => x,
                        None => sp.newp[(v - sp.base) as usize],
                    }
                };
                let mut g1: Vec<V3> = q[i..=j].iter().map(|&v| pos_of(v, sp)).collect();
                g1.extend(t.iter().rev().copied());
                let q2: Vec<i64> = q[j..].iter().chain(q[..=i].iter()).copied().collect();
                let mut g2: Vec<V3> = q2.iter().map(|&v| pos_of(v, sp)).collect();
                g2.extend(t.iter().copied());
                if sp.area_ok(&[g1, g2]) {
                    let tid: Vec<i64> = t.iter().map(|&y| sp.add(y)).collect();
                    let mut n1: Vec<i64> = q[i..=j].to_vec();
                    n1.extend(tid.iter().rev().copied());
                    let mut n2 = q2;
                    n2.extend(tid.iter().copied());
                    polys.splice(pi..pi + 1, [n1, n2]);
                    used += 1;
                }
                break;
            }
        }
    }
    if used == 0 {
        return None;
    }
    Some(polys)
}

/// make_feature_hook's hook (not forbidden) for the loop lp with the points p and the
/// feature flags onfeat of its vertices, in a cell of size h; new points from base on
#[allow(clippy::too_many_arguments)]
pub(crate) fn feature_hook(
    segs: &SegSet,
    corners: &[V3],
    lp: &[i64],
    p: &[V3],
    onfeat: &[bool],
    base: i64,
    h: f64,
    reach: f64,
    min_bend: f64,
) -> HookOut {
    let mut out = HookOut { polys: vec![lp.to_vec()], newp: Vec::new(), newf: Vec::new(), did: false, fan: false };
    let m = lp.len();
    if m < 3 {
        return out;
    }
    let mut sp = Split { segs, h, base, newp: Vec::new(), newf: Vec::new() };
    if let Some(polys) = split_on_features(&mut sp, corners, lp, p, onfeat, reach) {
        out.polys = polys;
        out.newp = sp.newp;
        out.newf = sp.newf;
        out.did = true;
        return out;
    }
    let c = mean(p);
    let (fp, d) = segs.closest_pd(c);
    if d > reach * h {
        return out;
    }
    // plane of the cap (Newell normal)
    let x: Vec<V3> = p.iter().map(|&a| sub(a, c)).collect();
    let t: Vec<V3> = (0..m).map(|k| cross(x[k], x[(k + 1) % m])).collect();
    let nrm = rows_sum(&t);
    let nn = norm1(nrm);
    if nn == 0.0 {
        return out;
    }
    let u = [nrm[0] / nn, nrm[1] / nn, nrm[2] / nn];
    let bend = dot1(sub(fp, c), u);
    if bend < min_bend * h {
        return out;
    }
    let lim = 0.1 * nn / m as f64;
    if (0..m).any(|k| gemv_row(cross(sub(p[k], fp), sub(p[(k + 1) % m], fp)), u) <= lim) {
        return out;
    }
    let vid = base;
    out.newp = vec![fp];
    out.did = true;
    out.fan = true;
    out.polys = (0..m).map(|i| vec![lp[i], lp[(i + 1) % m], vid]).collect();
    out
}

/// feature_hook(segs, corners, loop, points, onfeat flags, base, h, reach, min_bend)
///   -> (polys, new points, new feature vertex ids, did, fan)
#[pyfunction]
#[pyo3(name = "feature_hook")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn feature_hook_py(
    segs: Py<SegSet>,
    corners: PyReadonlyArray2<f64>,
    lp: Vec<i64>,
    points: PyReadonlyArray2<f64>,
    onfeat: Vec<bool>,
    base: i64,
    h: f64,
    reach: f64,
    min_bend: f64,
) -> PyResult<(Vec<Vec<i64>>, Vec<[f64; 3]>, Vec<i64>, bool, bool)> {
    let corners = crate::points(&corners)?;
    let p = crate::points(&points)?;
    if p.len() != lp.len() || onfeat.len() != lp.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("one point and flag per loop vertex"));
    }
    let o = feature_hook(segs.get(), &corners, &lp, &p, &onfeat, base, h, reach, min_bend);
    Ok((o.polys, o.newp, o.newf, o.did, o.fan))
}
