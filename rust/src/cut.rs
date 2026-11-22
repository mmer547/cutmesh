//! Cut kernel: port of cutcell.CutContext.cut (without the feature hooks) and
//! split_components, applied to a batch of cells in order, so that face ids, crossing
//! vertices and cap loops come out exactly as from the Python code.

use std::collections::HashMap;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{vec3_array, V3};

pub type Face = (i64, Vec<i64>);

/// FxHash (rustc's hasher): much faster than SipHash for the integer keys used here;
/// nothing depends on the iteration order of these maps
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl std::hash::Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) type FxBuild = std::hash::BuildHasherDefault<Fx>;
pub(crate) type FxMap<K, V> = HashMap<K, V, FxBuild>;
pub(crate) type FxSet<K> = std::collections::HashSet<K, FxBuild>;

/// insertion-ordered map (Python dict iteration order)
pub(crate) struct OMap<K: std::hash::Hash + Eq + Clone, V> {
    idx: FxMap<K, usize>,
    pub(crate) keys: Vec<K>,
    pub(crate) vals: Vec<V>,
}

impl<K: std::hash::Hash + Eq + Clone, V> OMap<K, V> {
    pub(crate) fn new() -> Self {
        OMap { idx: FxMap::default(), keys: Vec::new(), vals: Vec::new() }
    }
    pub(crate) fn entry(&mut self, k: K, v: impl FnOnce() -> V) -> &mut V {
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
    pub(crate) fn get(&self, k: &K) -> Option<&V> {
        self.idx.get(k).map(|&i| &self.vals[i])
    }
    pub(crate) fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        match self.idx.get(k) {
            Some(&i) => Some(&mut self.vals[i]),
            None => None,
        }
    }
}

pub(crate) fn dedupe(p: &[i64]) -> Vec<i64> {
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

pub(crate) fn n_unique(p: &[i64]) -> usize {
    let mut q = p.to_vec();
    q.sort_unstable();
    q.dedup();
    q.len()
}

/// cutcell._simple_loops
fn simple_loops(lp: &[i64]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let mut stack: Vec<i64> = Vec::new();
    let mut pos: FxMap<i64, usize> = FxMap::default();
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
pub(crate) fn chain_loops(segs: &[(i64, i64)]) -> Vec<Vec<i64>> {
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
    split_components_apart(faces, &FxSet::default())
}

/// cutcell.split_components(faces, apart): edges with both ends in apart do not join
pub(crate) fn split_components_apart(faces: Vec<Face>, apart: &FxSet<i64>) -> Vec<Vec<Face>> {
    let n = faces.len();
    if n <= 4 {
        return vec![faces];
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let mut owner: FxMap<(i64, i64), usize> = FxMap::default();
    for (i, (_, p)) in faces.iter().enumerate() {
        let m = p.len();
        for k in 0..m {
            let (a, b) = (p[k], p[(k + 1) % m]);
            if !apart.is_empty() && apart.contains(&a) && apart.contains(&b) {
                continue;
            }
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

/// how a cut gets its new vertex and face ids
trait Alloc {
    fn phi(&self, v: i64) -> f64;
    fn zero_edge(&self) -> &HashMap<(i64, i64, i64), i64>;
    fn cross(&mut self, a: i64, b: i64) -> i64;
    fn sub_id(&mut self, fid: i64, pp: &[i64]) -> i64;
    fn new_face_id(&mut self) -> i64;
}

fn xing<A: Alloc>(al: &mut A, a: i64, b: i64, a_neg: bool) -> i64 {
    if al.phi(a) == 0.0 && al.phi(b) == 0.0 {
        return if a_neg { a } else { b };
    }
    al.cross(a, b)
}

/// CutContext._zero_runs
fn zero_runs<A: Alloc>(al: &A, fid: i64, poly: &[i64], cls: &mut [bool]) {
    let n = poly.len();
    let z: Vec<bool> = poly.iter().map(|&v| al.phi(v) == 0.0).collect();
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
        let pn = al.phi(poly[im]) < 0.0;
        if pn == (al.phi(poly[(i + k) % n]) < 0.0) {
            continue;
        }
        if !pn {
            run.reverse();
        }
        for t in 0..k - 1 {
            let (a, b) = (poly[run[t]], poly[run[t + 1]]);
            let key = if a < b { (fid, a, b) } else { (fid, b, a) };
            let e = al.zero_edge().get(&key).copied();
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

/// the ids of the Python code: crossings and sub-faces numbered when first met
struct Cutter<'a> {
    pos: &'a mut Vec<V3>,
    phi: &'a mut Vec<f64>,
    cache: FxMap<(i64, i64), i64>,
    subids: FxMap<(i64, Vec<i64>), i64>,
    next_id: i64,
    new_parents: Vec<(i64, i64)>,
    zero_edge: &'a HashMap<(i64, i64, i64), i64>,
}

impl Cutter<'_> {
    fn new_vertex(&mut self, lo: i64, hi: i64) -> i64 {
        let (pl, ph) = (self.phi[lo as usize], self.phi[hi as usize]);
        let t = pl / (pl - ph);
        let (pa, pb) = (self.pos[lo as usize], self.pos[hi as usize]);
        let c = self.pos.len() as i64;
        self.pos.push([pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2])]);
        self.phi.push(0.0);
        self.new_parents.push((lo, hi));
        c
    }

    fn sub_key(&mut self, key: (i64, Vec<i64>)) -> i64 {
        if let Some(&i) = self.subids.get(&key) {
            return i;
        }
        let i = self.new_face_id();
        self.subids.insert(key, i);
        i
    }
}

impl Alloc for Cutter<'_> {
    fn phi(&self, v: i64) -> f64 {
        self.phi[v as usize]
    }

    fn zero_edge(&self) -> &HashMap<(i64, i64, i64), i64> {
        self.zero_edge
    }

    fn new_face_id(&mut self) -> i64 {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    fn sub_id(&mut self, fid: i64, pp: &[i64]) -> i64 {
        let mut k = pp.to_vec();
        k.sort_unstable();
        self.sub_key((fid, k))
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
            self.new_vertex(lo, hi)
        };
        self.cache.insert((lo, hi), c);
        c
    }
}

/// an id request of a cut, in the order made (replayed by the Cutter)
enum Ev {
    /// the crossing of the edge (lo, hi), both ends off the surface
    Vertex(i64, i64),
    /// the part pp (symbolic vertices) of face fid
    Sub(i64, Vec<i64>),
    /// a cap face
    New,
}

/// ids of one cell's cut before the global numbering: a new vertex is -(1 + k) for the
/// k-th crossing the cell meets, a new face -(1 + j) for its j-th face request.  The
/// cut only compares ids for equality (the canonical edge and set keys too), so the
/// result maps onto the sequential one id for id.
struct SymAlloc<'a> {
    phi: &'a [f64],
    zero_edge: &'a HashMap<(i64, i64, i64), i64>,
    vcache: FxMap<(i64, i64), i64>,
    nv: i64,
    scache: FxMap<(i64, Vec<i64>), i64>,
    nf: i64,
    events: Vec<Ev>,
}

impl Alloc for SymAlloc<'_> {
    fn phi(&self, v: i64) -> f64 {
        if v < 0 { 0.0 } else { self.phi[v as usize] }
    }

    fn zero_edge(&self) -> &HashMap<(i64, i64, i64), i64> {
        self.zero_edge
    }

    fn new_face_id(&mut self) -> i64 {
        self.nf += 1;
        self.events.push(Ev::New);
        -self.nf
    }

    fn sub_id(&mut self, fid: i64, pp: &[i64]) -> i64 {
        let mut k = pp.to_vec();
        k.sort_unstable();
        let key = (fid, k);
        if let Some(&i) = self.scache.get(&key) {
            return i;
        }
        self.nf += 1;
        self.events.push(Ev::Sub(fid, pp.to_vec()));
        self.scache.insert(key, -self.nf);
        -self.nf
    }

    fn cross(&mut self, a: i64, b: i64) -> i64 {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (pl, ph) = (self.phi(lo), self.phi(hi));
        if pl == 0.0 {
            return lo;
        }
        if ph == 0.0 {
            return hi;
        }
        if let Some(&c) = self.vcache.get(&(lo, hi)) {
            return c;
        }
        self.nv += 1;
        self.events.push(Ev::Vertex(lo, hi));
        self.vcache.insert((lo, hi), -self.nv);
        -self.nv
    }
}

/// CutContext.cut without hooks -> (neg faces, pos faces)
fn cut_with<A: Alloc>(al: &mut A, faces: Vec<Face>, shared_cap: bool) -> (Vec<Face>, Vec<Face>) {
    let mut neg: Vec<Face> = Vec::new();
    let mut pos: Vec<Face> = Vec::new();
    for (fid, poly) in faces {
        let mut has_n = false;
        let mut has_p = false;
        for &v in &poly {
            let x = al.phi(v);
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
        let mut cls: Vec<bool> = poly.iter().map(|&v| al.phi(v) <= 0.0).collect();
        if !al.zero_edge().is_empty() {
            zero_runs(al, fid, &poly, &mut cls);
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
                let c = xing(al, a, poly[j], true);
                seq_n.push(c);
                cur = vec![c];
            } else if !cls[i] && cls[j] {
                let c = xing(al, a, poly[j], false);
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
            let f2 = al.sub_id(fid, &pp);
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
            let cid = al.new_face_id();
            neg.push((cid, lp));
            pos.push((cid, rev));
        } else {
            let c1 = al.new_face_id();
            neg.push((c1, lp));
            let c2 = al.new_face_id();
            pos.push((c2, rev));
        }
    }
    (neg, pos)
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
    let (res, newp, parents, nid) = cut_cells(py, &points, &phi, next_face_id, cells, shared_cap, &zero_edge)?;
    Ok((res, vec3_array(py, newp)?, parents, nid))
}

/// cut_batch with the cells and the results as flat arrays (csr.rs):
///   -> ((item, side, face ptr, face ids, vertex ptr, vertices), new points, new parents,
///       next_face_id)
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn cut_batch_csr<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    phi: PyReadonlyArray1<'py, f64>,
    next_face_id: i64,
    fptr: PyReadonlyArray1<'py, i64>,
    key: PyReadonlyArray1<'py, i64>,
    vptr: PyReadonlyArray1<'py, i64>,
    verts: PyReadonlyArray1<'py, i64>,
    shared_cap: bool,
    zero_edge: HashMap<(i64, i64, i64), i64>,
) -> PyResult<(crate::csr::CsrOut<'py>, Bound<'py, PyArray2<f64>>, Vec<(i64, i64)>, i64)> {
    let cells = crate::csr::read_cells(&fptr, &key, &vptr, &verts)?;
    let (res, newp, parents, nid) = cut_cells_par(py, &points, &phi, next_face_id, cells, shared_cap, &zero_edge)?;
    Ok((crate::csr::write_results(py, &res, false), vec3_array(py, newp)?, parents, nid))
}

#[allow(clippy::type_complexity)]
fn cut_cells(
    py: Python<'_>,
    points: &PyReadonlyArray2<'_, f64>,
    phi: &PyReadonlyArray1<'_, f64>,
    next_face_id: i64,
    cells: Vec<Vec<Face>>,
    shared_cap: bool,
    zero_edge: &HashMap<(i64, i64, i64), i64>,
) -> PyResult<(Vec<CellResult>, Vec<V3>, Vec<(i64, i64)>, i64)> {
    let mut pos = crate::points(points)?;
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
            cache: FxMap::default(),
            subids: FxMap::default(),
            next_id: next_face_id,
            new_parents: Vec::new(),
            zero_edge,
        };
        let mut res: Vec<CellResult> = Vec::with_capacity(cells.len());
        for c in cells {
            let (neg, pos_) = cut_with(&mut ct, c, shared_cap);
            let pc = if shared_cap { split_components(pos_) } else { Vec::new() };
            res.push((split_components(neg), pc));
        }
        let parents = std::mem::take(&mut ct.new_parents);
        (res, parents, ct.next_id)
    });
    let newp: Vec<V3> = pos[n0..].to_vec();
    Ok((res, newp, parents, nid))
}

/// cut_cells with the cells cut in parallel on symbolic ids (SymAlloc), then numbered
/// in cell order by replaying their id requests on a Cutter: the same ids, points and
/// results as the sequential loop
#[allow(clippy::type_complexity)]
fn cut_cells_par(
    py: Python<'_>,
    points: &PyReadonlyArray2<'_, f64>,
    phi: &PyReadonlyArray1<'_, f64>,
    next_face_id: i64,
    cells: Vec<Vec<Face>>,
    shared_cap: bool,
    zero_edge: &HashMap<(i64, i64, i64), i64>,
) -> PyResult<(Vec<CellResult>, Vec<V3>, Vec<(i64, i64)>, i64)> {
    let mut pos = crate::points(points)?;
    let mut ph: Vec<f64> = phi.as_array().to_vec();
    if ph.len() != pos.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("phi and points differ in length"));
    }
    let n0 = pos.len();
    if cells.par_iter().flatten().any(|(_, p)| p.iter().any(|&v| v < 0 || v as usize >= n0)) {
        return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
    }
    let (res, parents, nid) = py.detach(|| {
        let ph0: &[f64] = &ph;
        let sym: Vec<(CellResult, Vec<Ev>)> = cells
            .into_par_iter()
            .map(|c| {
                let mut al = SymAlloc {
                    phi: ph0,
                    zero_edge,
                    vcache: FxMap::default(),
                    nv: 0,
                    scache: FxMap::default(),
                    nf: 0,
                    events: Vec::new(),
                };
                let (neg, pos_) = cut_with(&mut al, c, shared_cap);
                let pc = if shared_cap { split_components(pos_) } else { Vec::new() };
                ((split_components(neg), pc), al.events)
            })
            .collect();
        let mut ct = Cutter {
            pos: &mut pos,
            phi: &mut ph,
            cache: FxMap::default(),
            subids: FxMap::default(),
            next_id: next_face_id,
            new_parents: Vec::new(),
            zero_edge,
        };
        let mut ids: Vec<(CellResult, Vec<i64>, Vec<i64>)> = Vec::with_capacity(sym.len());
        for ((negc, posc), events) in sym {
            let mut vmap: Vec<i64> = Vec::new();
            let mut fmap: Vec<i64> = Vec::new();
            let real_v = |v: i64, vmap: &[i64]| if v < 0 { vmap[(-v - 1) as usize] } else { v };
            for ev in events {
                match ev {
                    Ev::Vertex(lo, hi) => {
                        let c = match ct.cache.get(&(lo, hi)) {
                            Some(&c) => c,
                            None => {
                                let c = ct.new_vertex(lo, hi);
                                ct.cache.insert((lo, hi), c);
                                c
                            }
                        };
                        vmap.push(c);
                    }
                    Ev::Sub(fid, pp) => {
                        let mut k: Vec<i64> = pp.iter().map(|&v| real_v(v, &vmap)).collect();
                        k.sort_unstable();
                        fmap.push(ct.sub_key((fid, k)));
                    }
                    Ev::New => fmap.push(ct.new_face_id()),
                }
            }
            ids.push(((negc, posc), vmap, fmap));
        }
        let res: Vec<CellResult> = ids
            .into_par_iter()
            .map(|((negc, posc), vmap, fmap)| {
                let real_v = |v: i64| if v < 0 { vmap[(-v - 1) as usize] } else { v };
                let map_comps = |comps: Vec<Vec<Face>>| -> Vec<Vec<Face>> {
                    comps
                        .into_iter()
                        .map(|comp| {
                            comp.into_iter()
                                .map(|(f, p)| {
                                    let f = if f < 0 { fmap[(-f - 1) as usize] } else { f };
                                    (f, p.into_iter().map(real_v).collect())
                                })
                                .collect()
                        })
                        .collect()
                };
                (map_comps(negc), map_comps(posc))
            })
            .collect();
        let parents = std::mem::take(&mut ct.new_parents);
        (res, parents, ct.next_id)
    });
    let newp: Vec<V3> = pos[n0..].to_vec();
    Ok((res, newp, parents, nid))
}
