//! Cut kernel: port of cutcell.CutContext.cut (without the feature hooks) and
//! split_components, applied to a batch of cells in order, so that face ids, crossing
//! vertices and cap loops come out exactly as from the Python code.

use std::collections::HashMap;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{vec3_array, V3};

pub type Face = (i64, Vec<i64>);

/// insertion-ordered map (Python dict iteration order)
struct OMap<K: std::hash::Hash + Eq + Clone, V> {
    idx: HashMap<K, usize>,
    keys: Vec<K>,
    vals: Vec<V>,
}

impl<K: std::hash::Hash + Eq + Clone, V> OMap<K, V> {
    fn new() -> Self {
        OMap { idx: HashMap::new(), keys: Vec::new(), vals: Vec::new() }
    }
    fn entry(&mut self, k: K, v: impl FnOnce() -> V) -> &mut V {
        let i = match self.idx.get(&k) {
            Some(&i) => i,
            None => {
                self.idx.insert(k.clone(), self.keys.len());
                self.keys.push(k);
                self.vals.push(v());
                self.keys.len() - 1
            }
        };
        &mut self.vals[i]
    }
    fn get(&self, k: &K) -> Option<&V> {
        self.idx.get(k).map(|&i| &self.vals[i])
    }
    fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        match self.idx.get(k) {
            Some(&i) => Some(&mut self.vals[i]),
            None => None,
        }
    }
}

fn dedupe(p: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(p.len());
    for &v in p {
        if out.last() != Some(&v) {
            out.push(v);
        }
    }
    while out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

fn n_unique(p: &[i64]) -> usize {
    let mut q = p.to_vec();
    q.sort_unstable();
    q.dedup();
    q.len()
}

/// cutcell._simple_loops
fn simple_loops(lp: &[i64]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let mut stack: Vec<i64> = Vec::new();
    let mut pos: HashMap<i64, usize> = HashMap::new();
    for &v in lp {
        match pos.get(&v).copied() {
            None => {
                pos.insert(v, stack.len());
                stack.push(v);
            }
            Some(i) => {
                let sub = stack[i..].to_vec();
                for u in &sub[1..] {
                    pos.remove(u);
                }
                stack.truncate(i + 1);
                out.push(sub);
            }
        }
    }
    out.push(stack);
    out.into_iter().filter(|s| s.len() >= 3).collect()
}

/// cutcell._chain_loops
fn chain_loops(segs: &[(i64, i64)]) -> Vec<Vec<i64>> {
    let mut out: OMap<i64, Vec<i64>> = OMap::new();
    for &(a, b) in segs {
        out.entry(a, Vec::new).push(b);
    }
    let starts = out.keys.clone();
    let mut loops = Vec::new();
    for start in starts {
        while let Some(v0) = out.get_mut(&start).and_then(|l| l.pop()) {
            let mut lp = vec![start];
            let mut v = v0;
            let mut guard = 0;
            while v != start && guard < 100000 {
                lp.push(v);
                match out.get_mut(&v).and_then(|l| l.pop()) {
                    Some(w) => v = w,
                    None => break,
                }
                guard += 1;
            }
            if v == start {
                loops.extend(simple_loops(&lp));
            }
        }
    }
    loops
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// cutcell.split_components
fn split_components(faces: Vec<Face>) -> Vec<Vec<Face>> {
    let n = faces.len();
    if n <= 4 {
        return vec![faces];
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let mut owner: HashMap<(i64, i64), usize> = HashMap::new();
    for (i, (_, p)) in faces.iter().enumerate() {
        let m = p.len();
        for k in 0..m {
            let (a, b) = (p[k], p[(k + 1) % m]);
            let key = if a < b { (a, b) } else { (b, a) };
            match owner.get(&key).copied() {
                None => {
                    owner.insert(key, i);
                }
                Some(j) => {
                    let ri = find(&mut parent, i);
                    let rj = find(&mut parent, j);
                    if ri != rj {
                        parent[ri] = rj;
                    }
                }
            }
        }
    }
    let roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    let mut groups: OMap<usize, Vec<Face>> = OMap::new();
    for (f, r) in faces.into_iter().zip(roots) {
        groups.entry(r, Vec::new).push(f);
    }
    groups.vals
}

struct Cutter<'a> {
    pos: &'a mut Vec<V3>,
    phi: &'a mut Vec<f64>,
    cache: HashMap<(i64, i64), i64>,
    subids: HashMap<(i64, Vec<i64>), i64>,
    next_id: i64,
    new_parents: Vec<(i64, i64)>,
    zero_edge: &'a HashMap<(i64, i64, i64), i64>,
}

impl Cutter<'_> {
    fn new_face_id(&mut self) -> i64 {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    fn sub_id(&mut self, fid: i64, pp: &[i64]) -> i64 {
        let mut k = pp.to_vec();
        k.sort_unstable();
        let key = (fid, k);
        if let Some(&i) = self.subids.get(&key) {
            return i;
        }
        let i = self.new_face_id();
        self.subids.insert(key, i);
        i
    }

    fn cross(&mut self, a: i64, b: i64) -> i64 {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if let Some(&c) = self.cache.get(&(lo, hi)) {
            return c;
        }
        let (pl, ph) = (self.phi[lo as usize], self.phi[hi as usize]);
        let c = if pl == 0.0 {
            lo
        } else if ph == 0.0 {
            hi
        } else {
            let t = pl / (pl - ph);
            let (pa, pb) = (self.pos[lo as usize], self.pos[hi as usize]);
            let c = self.pos.len() as i64;
            self.pos.push([pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2])]);
            self.phi.push(0.0);
            self.new_parents.push((lo, hi));
            c
        };
        self.cache.insert((lo, hi), c);
        c
    }

    fn xing(&mut self, a: i64, b: i64, a_neg: bool) -> i64 {
        if self.phi[a as usize] == 0.0 && self.phi[b as usize] == 0.0 {
            return if a_neg { a } else { b };
        }
        self.cross(a, b)
    }

    /// CutContext._zero_runs
    fn zero_runs(&self, fid: i64, poly: &[i64], cls: &mut [bool]) {
        let n = poly.len();
        let z: Vec<bool> = poly.iter().map(|&v| self.phi[v as usize] == 0.0).collect();
        for i in 0..n {
            let im = (i + n - 1) % n;
            if !z[i] || z[im] {
                continue;
            }
            let mut k = 0;
            while k < n && z[(i + k) % n] {
                k += 1;
            }
            if k < 2 || k == n {
                continue;
            }
            let mut run: Vec<usize> = (0..k).map(|t| (i + t) % n).collect();
            let pn = self.phi[poly[im] as usize] < 0.0;
            if pn == (self.phi[poly[(i + k) % n] as usize] < 0.0) {
                continue;
            }
            if !pn {
                run.reverse();
            }
            for t in 0..k - 1 {
                let (a, b) = (poly[run[t]], poly[run[t + 1]]);
                let key = if a < b { (fid, a, b) } else { (fid, b, a) };
                let e = self.zero_edge.get(&key).copied();
                if e == Some(1) {
                    for &r in &run[t + 1..] {
                        cls[r] = false;
                    }
                }
                if e != Some(-1) {
                    break;
                }
            }
        }
    }

    /// CutContext.cut without hooks -> (neg faces, pos faces)
    fn cut(&mut self, faces: Vec<Face>, shared_cap: bool) -> (Vec<Face>, Vec<Face>) {
        let mut neg: Vec<Face> = Vec::new();
        let mut pos: Vec<Face> = Vec::new();
        for (fid, poly) in faces {
            let mut has_n = false;
            let mut has_p = false;
            for &v in &poly {
                let x = self.phi[v as usize];
                if x < 0.0 {
                    has_n = true;
                } else if x > 0.0 {
                    has_p = true;
                }
            }
            if !has_p {
                neg.push((fid, poly));
                continue;
            }
            if !has_n {
                pos.push((fid, poly));
                continue;
            }
            let n = poly.len();
            let mut cls: Vec<bool> = poly.iter().map(|&v| self.phi[v as usize] <= 0.0).collect();
            if !self.zero_edge.is_empty() {
                self.zero_runs(fid, &poly, &mut cls);
            }
            let s = (0..n).find(|&i| cls[i] && !cls[(i + n - 1) % n]).unwrap_or(0);
            let mut seq_n: Vec<i64> = Vec::new();
            let mut ppieces: Vec<Vec<i64>> = Vec::new();
            let mut cur: Vec<i64> = Vec::new();
            for step in 0..n {
                let i = (s + step) % n;
                let j = (i + 1) % n;
                let a = poly[i];
                if cls[i] {
                    seq_n.push(a);
                } else {
                    cur.push(a);
                }
                if cls[i] && !cls[j] {
                    let c = self.xing(a, poly[j], true);
                    seq_n.push(c);
                    cur = vec![c];
                } else if !cls[i] && cls[j] {
                    let c = self.xing(a, poly[j], false);
                    cur.push(c);
                    ppieces.push(std::mem::take(&mut cur));
                    seq_n.push(c);
                }
            }
            let mut pieces: Vec<(u8, Vec<i64>)> = Vec::new();
            let np = dedupe(&seq_n);
            if n_unique(&np) >= 3 {
                pieces.push((0, np));
            }
            for pp in ppieces {
                let pp = dedupe(&pp);
                if n_unique(&pp) >= 3 {
                    pieces.push((1, pp));
                }
            }
            if pieces.len() == 1 {
                if pieces[0].0 == 0 {
                    neg.push((fid, poly));
                } else {
                    pos.push((fid, poly));
                }
                continue;
            }
            for (k, pp) in pieces {
                let f2 = self.sub_id(fid, &pp);
                if k == 0 {
                    neg.push((f2, pp));
                } else {
                    pos.push((f2, pp));
                }
            }
        }
        // cap = reversed open boundary of the N-side faces
        let mut de: OMap<(i64, i64), i64> = OMap::new();
        for (_, p) in &neg {
            let m = p.len();
            for k in 0..m {
                *de.entry((p[k], p[(k + 1) % m]), || 0) += 1;
            }
        }
        let mut segs = Vec::new();
        for (i, &(a, b)) in de.keys.iter().enumerate() {
            let r = de.get(&(b, a)).copied().unwrap_or(0);
            for _ in 0..(de.vals[i] - r).max(0) {
                segs.push((b, a));
            }
        }
        for lp in chain_loops(&segs) {
            let rev: Vec<i64> = lp.iter().rev().copied().collect();
            if shared_cap {
                let cid = self.new_face_id();
                neg.push((cid, lp));
                pos.push((cid, rev));
            } else {
                let c1 = self.new_face_id();
                neg.push((c1, lp));
                let c2 = self.new_face_id();
                pos.push((c2, rev));
            }
        }
        (neg, pos)
    }
}

type CellResult = (Vec<Vec<Face>>, Vec<Vec<Face>>);

/// cut_batch(points[n, 3], phi[n], next_face_id, cells, shared_cap, zero_edge)
///   -> ([(neg components, pos components)], new points[k, 3], new parents [(lo, hi)], next_face_id)
///
/// cells: list of cells, each a list of (face id, vertex list), cut in this order.
/// New vertices get ids n, n + 1, ... in creation order; pos components are empty
/// unless shared_cap.  zero_edge: CutContext.zero_edge.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn cut_batch<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    phi: PyReadonlyArray1<'py, f64>,
    next_face_id: i64,
    cells: Vec<Vec<Face>>,
    shared_cap: bool,
    zero_edge: HashMap<(i64, i64, i64), i64>,
) -> PyResult<(Vec<CellResult>, Bound<'py, PyArray2<f64>>, Vec<(i64, i64)>, i64)> {
    let mut pos = crate::points(&points)?;
    let mut ph: Vec<f64> = phi.as_array().to_vec();
    if ph.len() != pos.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("phi and points differ in length"));
    }
    let n0 = pos.len();
    if cells.iter().flatten().any(|(_, p)| p.iter().any(|&v| v < 0 || v as usize >= n0)) {
        return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
    }
    let (res, parents, nid) = py.detach(|| {
        let mut ct = Cutter {
            pos: &mut pos,
            phi: &mut ph,
            cache: HashMap::new(),
            subids: HashMap::new(),
            next_id: next_face_id,
            new_parents: Vec::new(),
            zero_edge: &zero_edge,
        };
        let mut res: Vec<CellResult> = Vec::with_capacity(cells.len());
        for c in cells {
            let (neg, pos_) = ct.cut(c, shared_cap);
            let pc = if shared_cap { split_components(pos_) } else { Vec::new() };
            res.push((split_components(neg), pc));
        }
        let parents = std::mem::take(&mut ct.new_parents);
        (res, parents, ct.next_id)
    });
    let newp: Vec<V3> = pos[n0..].to_vec();
    Ok((res, vec3_array(py, newp)?, parents, nid))
}
