//! Small-cell merging: port of the greedy merge loop of mesher.generate.
//!
//! Small (and non-convex) cut pieces are merged, in the given order, into the neighbour
//! whose union has the best worst-case non-orthogonality.  The arithmetic is that of
//! the Python loop (poly_cell for the geometry, NumPy's summation orders for the
//! row-wise products), and the candidates are tried in the iteration order of the
//! Python set that holds them (PySet below), so ties go the same way.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::cut::{Face, FxMap, FxSet};
use crate::{face_cs, sub, V3};

/// insertion and iteration order of a CPython (3.11) set of non-negative ints
pub(crate) struct PySet {
    table: Vec<Option<i64>>,
    fill: usize,
}

const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

impl PySet {
    pub(crate) fn new() -> Self {
        PySet { table: vec![None; 8], fill: 0 }
    }

    fn insert_clean(table: &mut [Option<i64>], key: i64) {
        let mask = table.len() - 1;
        let mut perturb = key as u64;
        let mut i = (key as u64 as usize) & mask;
        loop {
            if table[i].is_none() {
                table[i] = Some(key);
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if table[i + j].is_none() {
                        table[i + j] = Some(key);
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    pub(crate) fn add(&mut self, key: i64) {
        let mask = self.table.len() - 1;
        let mut perturb = key as u64;
        let mut i = (key as u64 as usize) & mask;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in 0..=probes {
                match self.table[i + j] {
                    None => {
                        self.table[i + j] = Some(key);
                        self.fill += 1;
                        if self.fill * 5 >= mask * 3 {
                            let used = self.fill;
                            let minused = if used > 50000 { used * 2 } else { used * 4 };
                            let mut newsize = 8;
                            while newsize <= minused {
                                newsize <<= 1;
                            }
                            let mut t = vec![None; newsize];
                            for k in self.table.iter().flatten() {
                                Self::insert_clean(&mut t, *k);
                            }
                            self.table = t;
                        }
                        return;
                    }
                    Some(k) if k == key => return,
                    _ => {}
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.table.iter().flatten().copied()
    }
}

/// list(set(keys)) as CPython orders it (for the tests)
#[pyfunction]
pub fn pyset_order(keys: Vec<i64>) -> PyResult<Vec<i64>> {
    if keys.iter().any(|&k| k < 0) {
        return Err(pyo3::exceptions::PyValueError::new_err("non-negative keys only"));
    }
    let mut s = PySet::new();
    for k in keys {
        s.add(k);
    }
    Ok(s.iter().collect())
}

/// row-wise d @ s of numpy's matmul for (n, 1, 3) @ (n, 3, 1): a fused multiply-add chain
#[inline]
fn mm3(a: V3, b: V3) -> f64 {
    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
}

#[inline]
fn einsum3(a: V3, b: V3) -> f64 {
    crate::octree::einsum3(a, b)
}

#[inline]
fn norm(a: V3) -> f64 {
    crate::dot(a, a).sqrt()
}

/// _core.poly_cell: (face centres, area vectors, volume, centre)
fn poly_cell(pos: &[V3], polys: &[&Vec<i64>]) -> (Vec<V3>, Vec<V3>, f64, V3) {
    let mut cs = Vec::with_capacity(polys.len());
    for f in polys {
        let loc: Vec<V3> = f.iter().map(|&v| pos[v as usize]).collect();
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
        let pv = crate::dot(*s, sub(*c, est));
        sv += pv;
        for j in 0..3 {
            acc[j] += pv * (0.75 * c[j] + 0.25 * est[j]);
        }
    }
    let vol = sv / 3.0;
    let cc = if vol > 0.0 { [acc[0] / sv, acc[1] / sv, acc[2] / sv] } else { est };
    let (c, s): (Vec<V3>, Vec<V3>) = cs.into_iter().unzip();
    (c, s, vol, cc)
}

struct Merger<'a> {
    pos: &'a [V3],
    pieces: &'a [Vec<Face>],
    zone_of: &'a [i64],
    bulk_zone: &'a [i64],
    cptr: &'a [i64],
    fptr: &'a [i64],
    fverts: &'a [i64],
    fkey: &'a [i64],
    kb1: &'a [i64],
    kb2: &'a [i64],
    np_: i64,
    occ: FxMap<i64, Vec<i64>>,
    parent: Vec<i64>,
    members: FxMap<i64, Vec<i64>>,
    bulk_members: FxMap<i64, Vec<i64>>,
    gcache: FxMap<i64, V3>,
}

impl Merger<'_> {
    fn find(&mut self, mut i: i64) -> i64 {
        while i < self.np_ && self.parent[i as usize] != i {
            let p = self.parent[i as usize];
            self.parent[i as usize] = if p < self.np_ { self.parent[p as usize] } else { p };
            i = self.parent[i as usize];
        }
        i
    }

    fn zone_of_node(&self, r: i64) -> i64 {
        if r < self.np_ { self.zone_of[r as usize] } else { self.bulk_zone[(r - self.np_) as usize] }
    }

    fn node_faces(&self, r: i64) -> Vec<Face> {
        let (mut fl, extra) = if r >= self.np_ {
            let leaf = (r - self.np_) as usize;
            let (a, b) = (self.cptr[leaf] as usize, self.cptr[leaf + 1] as usize);
            let fl: Vec<Face> = (a..b)
                .map(|j| (self.fkey[j], self.fverts[self.fptr[j] as usize..self.fptr[j + 1] as usize].to_vec()))
                .collect();
            (fl, self.bulk_members.get(&(r - self.np_)))
        } else {
            (self.pieces[r as usize].clone(), self.members.get(&r))
        };
        if let Some(m) = extra {
            for &q in m {
                fl.extend(self.pieces[q as usize].iter().cloned());
            }
        }
        fl
    }

    fn face_neighbour(&mut self, fid: i64, me: &[i64]) -> i64 {
        if let Some(o) = self.occ.get(&fid).cloned() {
            for q in o {
                let r = self.find(q);
                if !me.contains(&r) {
                    return r;
                }
            }
        }
        if (fid as usize) < self.kb1.len() && fid >= 0 {
            for b in [self.kb1[fid as usize], self.kb2[fid as usize]] {
                if b >= 0 {
                    let r = self.np_ + b;
                    if !me.contains(&r) {
                        return r;
                    }
                }
            }
        }
        -1
    }

    /// poly_geom: the faces of fl that occur once, and their geometry
    fn poly_geom(&self, fl: &[Face]) -> (Vec<i64>, Vec<V3>, Vec<V3>, f64, V3) {
        let mut cnt: FxMap<i64, u32> = FxMap::default();
        for (fid, _) in fl {
            *cnt.entry(*fid).or_insert(0) += 1;
        }
        let keep: Vec<&Face> = fl.iter().filter(|(f, _)| cnt[f] == 1).collect();
        let polys: Vec<&Vec<i64>> = keep.iter().map(|(_, p)| p).collect();
        let (cf, sf, v, cc) = poly_cell(self.pos, &polys);
        (keep.iter().map(|(f, _)| *f).collect(), cf, sf, v, cc)
    }

    fn node_centre(&mut self, r: i64) -> V3 {
        if let Some(c) = self.gcache.get(&r) {
            return *c;
        }
        let fl = self.node_faces(r);
        let c = self.poly_geom(&fl).4;
        self.gcache.insert(r, c);
        c
    }

    fn merge_score(&mut self, fa: &[Face], r_small: i64, r_target: i64, hcell: f64) -> Option<f64> {
        let mut fl = fa.to_vec();
        fl.extend(self.node_faces(r_target));
        let (keep, cf, sf, v, cc) = self.poly_geom(&fl);
        if v <= 0.0 {
            return None;
        }
        for i in 0..keep.len() {
            if einsum3(sf[i], sub(cf[i], cc)) <= 0.02 * norm(sf[i]) * hcell {
                return None;
            }
        }
        let me = [r_small, r_target];
        let mut cmin = f64::INFINITY;
        let mut any = false;
        for (i, &fid) in keep.iter().enumerate() {
            let nb = self.face_neighbour(fid, &me);
            if nb >= 0 {
                let c = self.node_centre(nb);
                let d = sub(c, cc);
                let s = sf[i];
                let x = mm3(d, s) / (mm3(d, d).sqrt() * mm3(s, s).sqrt() + 1e-300);
                if !any || x < cmin {
                    cmin = x;
                }
                any = true;
            }
        }
        if !any {
            return Some(1.0);
        }
        Some(if cmin < 1.0 { cmin } else { 1.0 })
    }
}

type MergeOut = (Vec<i64>, Vec<(i64, Vec<i64>)>, Vec<(i64, Vec<i64>)>, Vec<f64>, usize, usize);

/// merge_cells(points, pieces [faces], piece zone, bulk zone, cptr, fptr, fverts, f_key,
///             kb1, kb2, small, nsmall, gvol, hexvol, beta, piece h, cos_lim)
///   -> (parent, members [(root, pieces)], bulk_members [(leaf, pieces)], gvol, merged, rejected)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn merge_cells(
    points: PyReadonlyArray2<f64>,
    pieces: Vec<Vec<Face>>,
    zone_of: Vec<i64>,
    bulk_zone: PyReadonlyArray1<i64>,
    cptr: PyReadonlyArray1<i64>,
    fptr: PyReadonlyArray1<i64>,
    fverts: PyReadonlyArray1<i64>,
    fkey: PyReadonlyArray1<i64>,
    kb1: PyReadonlyArray1<i64>,
    kb2: PyReadonlyArray1<i64>,
    small: Vec<i64>,
    nsmall: usize,
    gvol: Vec<f64>,
    hexvol: Vec<f64>,
    beta: f64,
    hpiece: Vec<f64>,
    cos_lim: f64,
) -> PyResult<MergeOut> {
    let pos = crate::points(&points)?;
    let np_ = pieces.len();
    let (bz, cp, fp, fv, fk, k1, k2) =
        (bulk_zone.as_slice()?, cptr.as_slice()?, fptr.as_slice()?, fverts.as_slice()?, fkey.as_slice()?, kb1.as_slice()?, kb2.as_slice()?);
    let bad = |c: bool| if c { Err(pyo3::exceptions::PyValueError::new_err("inconsistent merge data")) } else { Ok(()) };
    bad(zone_of.len() != np_ || gvol.len() != np_ || hexvol.len() != np_ || hpiece.len() != np_ || k1.len() != k2.len())?;
    bad(pieces.iter().flatten().any(|(_, p)| p.iter().any(|&v| v < 0 || v as usize >= pos.len())))?;
    bad(fv.iter().any(|&v| v < 0 || v as usize >= pos.len()) || cp.len() != bz.len() + 1)?;
    bad(*cp.last().unwrap_or(&0) as usize >= fp.len() || fk.len() + 1 < fp.len() || *fp.last().unwrap_or(&0) as usize > fv.len())?;
    bad(small.iter().any(|&s| s < 0 || s as usize >= np_))?;
    bad(k1.iter().chain(k2.iter()).any(|&b| b >= bz.len() as i64))?;
    let mut occ: FxMap<i64, Vec<i64>> = FxMap::default();
    for (pi, fl) in pieces.iter().enumerate() {
        for (fid, _) in fl {
            occ.entry(*fid).or_default().push(pi as i64);
        }
    }
    let mut m = Merger {
        pos: &pos,
        pieces: &pieces,
        zone_of: &zone_of,
        bulk_zone: bz,
        cptr: cp,
        fptr: fp,
        fverts: fv,
        fkey: fk,
        kb1: k1,
        kb2: k2,
        np_: np_ as i64,
        occ,
        parent: (0..np_ as i64).collect(),
        members: FxMap::default(),
        bulk_members: FxMap::default(),
        gcache: FxMap::default(),
    };
    let mut gvol = gvol;
    let (mut merged, mut rejected) = (0usize, 0usize);
    for (qi, &pi) in small.iter().enumerate() {
        if m.find(pi) != pi || (qi < nsmall && gvol[pi as usize] >= beta * hexvol[pi as usize]) {
            continue;
        }
        let fa = m.node_faces(pi);
        let me = [pi];
        let mut cands = PySet::new();
        let mut seen: FxSet<i64> = FxSet::default();
        for (fid, _) in &fa {
            let nb = m.face_neighbour(*fid, &me);
            if nb >= 0 && m.zone_of_node(nb) == zone_of[pi as usize] && seen.insert(nb) {
                cands.add(nb);
            }
        }
        let hcell = hpiece[pi as usize];
        let (mut best, mut bscore) = (-1i64, cos_lim);
        let order: Vec<i64> = cands.iter().collect();
        for &ro in &order {
            if let Some(sc) = m.merge_score(&fa, pi, ro, hcell) {
                if sc > bscore {
                    best = ro;
                    bscore = sc;
                }
            }
        }
        if best >= 0 {
            m.parent[pi as usize] = best;
            let mut mem = vec![pi];
            if let Some(x) = m.members.remove(&pi) {
                mem.extend(x);
            }
            if best < np_ as i64 {
                gvol[best as usize] += gvol[pi as usize];
                m.members.entry(best).or_default().extend(mem);
            } else {
                m.bulk_members.entry(best - np_ as i64).or_default().extend(mem);
            }
            m.gcache.remove(&best);
            m.gcache.remove(&pi);
            merged += 1;
        } else if !order.is_empty() {
            rejected += 1;
        }
    }
    let members: Vec<(i64, Vec<i64>)> = m.members.drain().collect();
    let bulk_members: Vec<(i64, Vec<i64>)> = m.bulk_members.drain().collect();
    Ok((m.parent, members, bulk_members, gvol, merged, rejected))
}
