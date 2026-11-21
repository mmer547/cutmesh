//! Small-cell merging: port of the greedy merge loop of mesher.generate.
//!
//! Small (and non-convex) cut pieces are merged, in the given order, into the neighbour
//! whose union has the best worst-case non-orthogonality.  The arithmetic is that of
//! the Python loop (poly_cell for the geometry, NumPy's summation orders for the
//! row-wise products), and the candidates are tried in the iteration order of the
//! Python set that holds them (PySet below), so ties go the same way.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

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

/// a face of a node: (face id, handle).  Handles below n_pf are faces of the pieces
/// (index into the piece face arrays), the others faces of the octree leaves
/// (n_pf + index into the leaf face arrays).
type FRef = (i64, usize);

struct Merger<'a> {
    pos: &'a [V3],
    // the pieces' faces: piece i has pf[i]..pf[i+1], face f the id pkey[f] and the
    // vertices pv[pvp[f]..pvp[f+1]]
    pf: &'a [i64],
    pkey: &'a [i64],
    pvp: &'a [i64],
    pv: &'a [i64],
    n_pf: usize,
    zone_of: &'a [i64],
    bulk_zone: &'a [i64],
    cptr: &'a [i64],
    fptr: &'a [i64],
    fverts: &'a [i64],
    fkey: &'a [i64],
    kb1: &'a [i64],
    kb2: &'a [i64],
    np_: i64,
    // face id -> pieces holding it (in piece order): ids sorted, and their pieces
    occ_k: &'a [i64],
    occ_p: &'a [i64],
    parent: Vec<i64>,
    members: FxMap<i64, Vec<i64>>,
    bulk_members: FxMap<i64, Vec<i64>>,
    gcache: FxMap<i64, V3>,
    // centre and area vector of a face (face_cs depends on its vertices only): of the
    // pieces' faces computed up front, of the leaves' faces when needed
    pcs: &'a [(V3, V3)],
    cs_cache: FxMap<usize, (V3, V3)>,
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

    fn poly(&self, h: usize) -> &[i64] {
        if h < self.n_pf {
            &self.pv[self.pvp[h] as usize..self.pvp[h + 1] as usize]
        } else {
            let j = h - self.n_pf;
            &self.fverts[self.fptr[j] as usize..self.fptr[j + 1] as usize]
        }
    }

    fn push_piece(&self, fl: &mut Vec<FRef>, q: i64) {
        let (a, b) = (self.pf[q as usize] as usize, self.pf[q as usize + 1] as usize);
        fl.extend((a..b).map(|f| (self.pkey[f], f)));
    }

    fn node_faces(&self, r: i64) -> Vec<FRef> {
        let mut fl: Vec<FRef> = Vec::new();
        let extra = if r >= self.np_ {
            let leaf = (r - self.np_) as usize;
            let (a, b) = (self.cptr[leaf] as usize, self.cptr[leaf + 1] as usize);
            fl.extend((a..b).map(|j| (self.fkey[j], self.n_pf + j)));
            self.bulk_members.get(&(r - self.np_))
        } else {
            self.push_piece(&mut fl, r);
            self.members.get(&r)
        };
        if let Some(m) = extra {
            for &q in m {
                self.push_piece(&mut fl, q);
            }
        }
        fl
    }

    fn face_neighbour(&mut self, fid: i64, me: &[i64]) -> i64 {
        let a = self.occ_k.partition_point(|&k| k < fid);
        let mut b = a;
        while b < self.occ_k.len() && self.occ_k[b] == fid {
            b += 1;
        }
        for j in a..b {
            let r = self.find(self.occ_p[j]);
            if !me.contains(&r) {
                return r;
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

    /// the merge partner of piece pi (the loop body of the Python code): (best node or
    /// -1, whether it had candidates)
    fn evaluate(&mut self, pi: i64, hcell: f64, cos_lim: f64) -> (i64, bool) {
        let fa = self.node_faces(pi);
        let me = [pi];
        let mut cands = PySet::new();
        let mut seen: FxSet<i64> = FxSet::default();
        let zpi = self.zone_of[pi as usize];
        for &(fid, _) in &fa {
            let nb = self.face_neighbour(fid, &me);
            if nb >= 0 && self.zone_of_node(nb) == zpi && seen.insert(nb) {
                cands.add(nb);
            }
        }
        let (mut best, mut bscore) = (-1i64, cos_lim);
        let order: Vec<i64> = cands.iter().collect();
        for &ro in &order {
            if let Some(sc) = self.merge_score(&fa, pi, ro, hcell) {
                if sc > bscore {
                    best = ro;
                    bscore = sc;
                }
            }
        }
        (best, !order.is_empty())
    }

    fn cs(&mut self, h: usize) -> (V3, V3) {
        if h < self.n_pf {
            return self.pcs[h];
        }
        if let Some(x) = self.cs_cache.get(&h) {
            return *x;
        }
        let loc: Vec<V3> = self.poly(h).iter().map(|&v| self.pos[v as usize]).collect();
        let idx: Vec<usize> = (0..loc.len()).collect();
        let x = face_cs(&loc, &idx);
        self.cs_cache.insert(h, x);
        x
    }

    /// poly_geom: the faces of fl that occur once, and their geometry (_core.poly_cell)
    fn poly_geom(&mut self, fl: &[FRef]) -> (Vec<i64>, Vec<V3>, Vec<V3>, f64, V3) {
        // the faces whose id occurs once (a node has a few dozen faces at most: compared
        // pairwise, no hash map)
        let keep: Vec<FRef> = if fl.len() <= 64 {
            fl.iter().copied().filter(|&(f, _)| fl.iter().filter(|&&(g, _)| g == f).count() == 1).collect()
        } else {
            let mut cnt: FxMap<i64, u32> = FxMap::default();
            for (fid, _) in fl {
                *cnt.entry(*fid).or_insert(0) += 1;
            }
            fl.iter().copied().filter(|(f, _)| cnt[f] == 1).collect()
        };
        let cs: Vec<(V3, V3)> = keep.iter().map(|&(_, h)| self.cs(h)).collect();
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
        (keep.iter().map(|(f, _)| *f).collect(), c, s, vol, cc)
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

    fn merge_score(&mut self, fa: &[FRef], r_small: i64, r_target: i64, hcell: f64) -> Option<f64> {
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
    let (mut pf, mut pk, mut pvp, mut pv) = (vec![0i64], Vec::new(), vec![0i64], Vec::new());
    for fl in &pieces {
        for (fid, p) in fl {
            pk.push(*fid);
            pv.extend_from_slice(p);
            pvp.push(pv.len() as i64);
        }
        pf.push(pk.len() as i64);
    }
    merge_core(
        &pos, &pf, &pk, &pvp, &pv, &zone_of, bulk_zone.as_slice()?, cptr.as_slice()?, fptr.as_slice()?,
        fverts.as_slice()?, fkey.as_slice()?, kb1.as_slice()?, kb2.as_slice()?, &small, nsmall, gvol, &hexvol, beta,
        &hpiece, cos_lim,
    )
}

/// merge_cells with the pieces as flat arrays: piece i has the faces pfptr[i]..pfptr[i+1],
/// face f has the id pkey[f] and the vertices pverts[pvptr[f]..pvptr[f+1]]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn merge_cells_csr(
    points: PyReadonlyArray2<f64>,
    pfptr: PyReadonlyArray1<i64>,
    pkey: PyReadonlyArray1<i64>,
    pvptr: PyReadonlyArray1<i64>,
    pverts: PyReadonlyArray1<i64>,
    zone_of: PyReadonlyArray1<i64>,
    bulk_zone: PyReadonlyArray1<i64>,
    cptr: PyReadonlyArray1<i64>,
    fptr: PyReadonlyArray1<i64>,
    fverts: PyReadonlyArray1<i64>,
    fkey: PyReadonlyArray1<i64>,
    kb1: PyReadonlyArray1<i64>,
    kb2: PyReadonlyArray1<i64>,
    small: PyReadonlyArray1<i64>,
    nsmall: usize,
    gvol: PyReadonlyArray1<f64>,
    hexvol: PyReadonlyArray1<f64>,
    beta: f64,
    hpiece: PyReadonlyArray1<f64>,
    cos_lim: f64,
) -> PyResult<MergeOut> {
    let pos = crate::points(&points)?;
    merge_core(
        &pos, pfptr.as_slice()?, pkey.as_slice()?, pvptr.as_slice()?, pverts.as_slice()?, zone_of.as_slice()?,
        bulk_zone.as_slice()?, cptr.as_slice()?, fptr.as_slice()?, fverts.as_slice()?, fkey.as_slice()?,
        kb1.as_slice()?, kb2.as_slice()?, small.as_slice()?, nsmall, gvol.as_slice()?.to_vec(), hexvol.as_slice()?,
        beta, hpiece.as_slice()?, cos_lim,
    )
}

#[allow(clippy::too_many_arguments)]
fn merge_core(
    pos: &[V3],
    pf: &[i64],
    pk: &[i64],
    pvp: &[i64],
    pv: &[i64],
    zone_of: &[i64],
    bz: &[i64],
    cp: &[i64],
    fp: &[i64],
    fv: &[i64],
    fk: &[i64],
    k1: &[i64],
    k2: &[i64],
    small: &[i64],
    nsmall: usize,
    gvol: Vec<f64>,
    hexvol: &[f64],
    beta: f64,
    hpiece: &[f64],
    cos_lim: f64,
) -> PyResult<MergeOut> {
    let bad = |c: bool| if c { Err(pyo3::exceptions::PyValueError::new_err("inconsistent merge data")) } else { Ok(()) };
    bad(pf.is_empty() || pf[0] != 0 || pf.windows(2).any(|w| w[1] < w[0]) || *pf.last().unwrap() as usize != pk.len())?;
    bad(pvp.len() != pk.len() + 1 || pvp.windows(2).any(|w| w[1] < w[0]) || *pvp.last().unwrap_or(&0) as usize > pv.len())?;
    let np_ = pf.len() - 1;
    bad(zone_of.len() != np_ || gvol.len() != np_ || hexvol.len() != np_ || hpiece.len() != np_ || k1.len() != k2.len())?;
    bad(pv.par_iter().any(|&v| v < 0 || v as usize >= pos.len()))?;
    bad(fv.par_iter().any(|&v| v < 0 || v as usize >= pos.len()) || cp.len() != bz.len() + 1)?;
    bad(*cp.last().unwrap_or(&0) as usize >= fp.len() || fk.len() + 1 < fp.len() || *fp.last().unwrap_or(&0) as usize > fv.len())?;
    bad(small.iter().any(|&s| s < 0 || s as usize >= np_) || nsmall > small.len())?;
    bad(k1.iter().chain(k2.iter()).any(|&b| b >= bz.len() as i64))?;
    // face id -> pieces, sorted by id (stable: pieces in order)
    let mut fl_p: Vec<i64> = Vec::with_capacity(pk.len());
    for i in 0..np_ {
        for _ in pf[i]..pf[i + 1] {
            fl_p.push(i as i64);
        }
    }
    let mut idx: Vec<usize> = (0..pk.len()).collect();
    idx.par_sort_by_key(|&i| pk[i]);
    let occ_k: Vec<i64> = idx.iter().map(|&i| pk[i]).collect();
    let occ_p: Vec<i64> = idx.iter().map(|&i| fl_p[i]).collect();
    drop((fl_p, idx));
    let pcs: Vec<(V3, V3)> = (0..pk.len())
        .into_par_iter()
        .map(|f| {
            let loc: Vec<V3> = pv[pvp[f] as usize..pvp[f + 1] as usize].iter().map(|&v| pos[v as usize]).collect();
            let idx: Vec<usize> = (0..loc.len()).collect();
            face_cs(&loc, &idx)
        })
        .collect();
    let mut m = Merger {
        pos,
        pcs: &pcs,
        pf,
        pkey: pk,
        pvp,
        pv,
        n_pf: pk.len(),
        zone_of,
        bulk_zone: bz,
        cptr: cp,
        fptr: fp,
        fverts: fv,
        fkey: fk,
        kb1: k1,
        kb2: k2,
        np_: np_ as i64,
        occ_k: &occ_k,
        occ_p: &occ_p,
        parent: (0..np_ as i64).collect(),
        members: FxMap::default(),
        bulk_members: FxMap::default(),
        gcache: FxMap::default(),
        cs_cache: FxMap::default(),
    };
    let mut gvol = gvol;
    let (mut merged, mut rejected) = (0usize, 0usize);
    for (qi, &pi) in small.iter().enumerate() {
        if m.find(pi) != pi || (qi < nsmall && gvol[pi as usize] >= beta * hexvol[pi as usize]) {
            continue;
        }
        let (best, any) = m.evaluate(pi, hpiece[pi as usize], cos_lim);
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
        } else if any {
            rejected += 1;
        }
    }
    let members: Vec<(i64, Vec<i64>)> = m.members.drain().collect();
    let bulk_members: Vec<(i64, Vec<i64>)> = m.bulk_members.drain().collect();
    Ok((m.parent, members, bulk_members, gvol, merged, rejected))
}
