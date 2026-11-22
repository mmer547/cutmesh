//! Stateful cut kernel with the feature hooks: port of cutcell.CutContext.cut as the
//! domain cut (and hooked zone cuts) use it through mesher.cut_with_fallback.
//!
//! The kernel keeps the state of the Python CutContext (points, level set, crossing
//! cache, sub-face ids, chord / pinch caches, feature vertices, next face id) and cuts
//! the items in the order asked, so every id comes out as from the Python code.
//! - chord points: SegSet.pierce (the Rust chord hook of make_chord_hook),
//! - pinch points and cap splits: Python callbacks (make_pinch_hook / make_feature_hook),
//!   the cap callback only where the hook can do more than return the loop (a feature
//!   point within reach or two loop vertices on a feature, decided here exactly as the
//!   hook decides it).
//! The state goes back to the CutContext with export() (featclip continues with it).

use crate::cut::{FxMap as HashMap, FxSet as HashSet};

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cut::{chain_loops, dedupe, n_unique, split_components_apart, Face, OMap};
use crate::{vec3_array, SegSet, V3};

type Key3 = (i64, i64, i64);
type CellRes = (Vec<Vec<Face>>, Vec<Vec<Face>>);

#[pyclass(unsendable)]
pub struct CutKernel {
    pos: Vec<V3>,
    phi: Vec<f64>,
    n0: usize,
    cache: HashMap<(i64, i64), i64>,
    new_cache: Vec<((i64, i64), i64)>,
    subids: HashMap<(i64, Vec<i64>), i64>,
    sub_mark: i64, // sub-face ids from here on are new since the last export
    next_id: i64,
    new_vertices: Vec<i64>,
    new_parents: Vec<(i64, i64)>,
    zero_edge: HashMap<Key3, i64>,
    plain: HashSet<i64>,
    onfeat: HashSet<i64>,
    new_onfeat: Vec<i64>,
    chord_cache: HashMap<Key3, Option<i64>>,
    pinch_cache: HashMap<Key3, Option<i64>>,
    cache_log: Vec<(u8, Key3, Option<i64>)>,
    chord_segs: Option<Py<SegSet>>,
    cap_segs: Option<Py<SegSet>>,
    reach: f64,
    csize: Vec<f64>,
    pinch_cb: Option<Py<PyAny>>,
    cap_cb: Option<Py<PyAny>>,
    pinch_min_area: f64,
    shared_cap: bool,
    items: Vec<(Vec<Face>, i64)>,
    res: Vec<Option<CellRes>>,
    // state of the current cut
    used_chords: Vec<Key3>,
    pinch_now: HashSet<i64>,
    did: bool,
    forbid: bool,
    leaf: i64,
    // the cap hook in Rust (fhook.rs) instead of cap_cb: feature corners, min_bend
    hook_rs: Option<(Vec<V3>, f64)>,
}

fn key3(fid: i64, a: i64, b: i64) -> Key3 {
    if a < b { (fid, a, b) } else { (fid, b, a) }
}

fn index_of(v: &[i64], x: i64) -> usize {
    v.iter().position(|&y| y == x).unwrap()
}

#[pymethods]
impl CutKernel {
    #[new]
    #[pyo3(signature = (points, phi, next_id, shared_cap, zero_edge, plain_faces, onfeat, cache, subids,
                        chord_cache, pinch_cache, chord_segs, cap_segs, reach, csize, pinch_cb, cap_cb,
                        pinch_min_area))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        points: PyReadonlyArray2<f64>,
        phi: PyReadonlyArray1<f64>,
        next_id: i64,
        shared_cap: bool,
        zero_edge: HashMap<Key3, i64>,
        plain_faces: HashSet<i64>,
        onfeat: HashSet<i64>,
        cache: HashMap<(i64, i64), i64>,
        subids: HashMap<(i64, Vec<i64>), i64>,
        chord_cache: HashMap<Key3, Option<i64>>,
        pinch_cache: HashMap<Key3, Option<i64>>,
        chord_segs: Option<Py<SegSet>>,
        cap_segs: Option<Py<SegSet>>,
        reach: f64,
        csize: PyReadonlyArray1<f64>,
        pinch_cb: Option<Py<PyAny>>,
        cap_cb: Option<Py<PyAny>>,
        pinch_min_area: f64,
    ) -> PyResult<Self> {
        let pos = crate::points(&points)?;
        let phi: Vec<f64> = phi.as_array().to_vec();
        if phi.len() != pos.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("phi and points differ in length"));
        }
        let n0 = pos.len();
        Ok(CutKernel {
            pos,
            phi,
            n0,
            cache,
            new_cache: Vec::new(),
            subids,
            sub_mark: next_id,
            next_id,
            new_vertices: Vec::new(),
            new_parents: Vec::new(),
            zero_edge,
            plain: plain_faces,
            onfeat,
            new_onfeat: Vec::new(),
            chord_cache,
            pinch_cache,
            cache_log: Vec::new(),
            chord_segs,
            cap_segs,
            reach,
            csize: csize.as_array().to_vec(),
            pinch_cb,
            cap_cb,
            pinch_min_area,
            shared_cap,
            items: Vec::new(),
            res: Vec::new(),
            used_chords: Vec::new(),
            pinch_now: HashSet::default(),
            did: false,
            forbid: false,
            leaf: 0,
            hook_rs: None,
        })
    }

    /// items: [(faces, parent leaf)]
    fn set_items(&mut self, items: Vec<(Vec<Face>, i64)>) -> PyResult<()> {
        let np_ = self.pos.len();
        for (fl, leaf) in &items {
            if *leaf < 0 || *leaf as usize >= self.csize.len() {
                return Err(pyo3::exceptions::PyValueError::new_err("leaf out of range"));
            }
            if fl.iter().any(|(_, p)| p.iter().any(|&v| v < 0 || v as usize >= np_)) {
                return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
            }
        }
        self.res = vec![None; items.len()];
        self.items = items;
        Ok(())
    }

    /// set_items: item k is leaf leaves[k] with its own faces (leaf CSR: cptr, fptr,
    /// fverts, f_key), unless overrides gives its faces [(k, faces)]
    #[allow(clippy::too_many_arguments)]
    fn set_items_leaves(
        &mut self,
        py: Python<'_>,
        leaves: PyReadonlyArray1<i64>,
        cptr: PyReadonlyArray1<i64>,
        fptr: PyReadonlyArray1<i64>,
        fverts: PyReadonlyArray1<i64>,
        f_key: PyReadonlyArray1<i64>,
        overrides: Vec<(usize, Vec<Face>)>,
    ) -> PyResult<()> {
        use rayon::prelude::*;
        let (lv, cp, fp, fv, fk) = (leaves.as_slice()?, cptr.as_slice()?, fptr.as_slice()?, fverts.as_slice()?, f_key.as_slice()?);
        let nc = cp.len().saturating_sub(1);
        if lv.iter().any(|&c| c < 0 || c as usize >= nc) || *cp.last().unwrap_or(&0) as usize >= fp.len()
            || fk.len() + 1 < fp.len() || overrides.iter().any(|(k, _)| *k >= lv.len())
        {
            return Err(pyo3::exceptions::PyValueError::new_err("inconsistent item arrays"));
        }
        let mut over: HashMap<usize, Vec<Face>> = overrides.into_iter().collect();
        let mut items: Vec<(Vec<Face>, i64)> = py.detach(|| {
            lv.par_iter()
                .map(|&c| {
                    let c = c as usize;
                    let fl: Vec<Face> = (cp[c] as usize..cp[c + 1] as usize)
                        .map(|f| (fk[f], fv[fp[f] as usize..fp[f + 1] as usize].to_vec()))
                        .collect();
                    (fl, c as i64)
                })
                .collect()
        });
        for (k, it) in items.iter_mut().enumerate() {
            if let Some(fl) = over.remove(&k) {
                it.0 = fl;
            }
        }
        self.set_items(items)
    }

    /// the crossing cache, the sub-face ids and the chord / pinch caches, moved out of
    /// the kernel (instead of export()ing them into Python dicts): looked up when Python
    /// asks.  The chord / pinch cache changes are not exported after this.
    fn take_caches(&mut self) -> CutCaches {
        self.cache_log.clear();
        CutCaches {
            cache: std::mem::take(&mut self.cache),
            subids: std::mem::take(&mut self.subids),
            chord: std::mem::take(&mut self.chord_cache),
            pinch: std::mem::take(&mut self.pinch_cache),
        }
    }

    /// export() without the new cache entries and sub-face ids (take_caches() has them)
    #[allow(clippy::type_complexity)]
    fn export_lazy<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<i64>, Bound<'py, PyArray2<i64>>, i64, Vec<(u8, Key3, Option<i64>)>, Vec<i64>)> {
        let newp: Vec<V3> = self.pos[self.n0..].to_vec();
        self.new_cache.clear();
        self.sub_mark = self.next_id;
        let par: Vec<i64> = std::mem::take(&mut self.new_parents).into_iter().flat_map(|(a, b)| [a, b]).collect();
        let n = par.len() / 2;
        let par = {
            use numpy::PyArrayMethods;
            numpy::PyArray1::from_vec(py, par).reshape([n, 2])?
        };
        Ok((
            vec3_array(py, newp)?,
            std::mem::take(&mut self.new_vertices),
            par,
            self.next_id,
            std::mem::take(&mut self.cache_log),
            std::mem::take(&mut self.new_onfeat),
        ))
    }

    /// set_items with the items as flat arrays (csr.rs) and their parent leaves
    fn set_items_csr(
        &mut self,
        leaf: PyReadonlyArray1<i64>,
        fptr: PyReadonlyArray1<i64>,
        key: PyReadonlyArray1<i64>,
        vptr: PyReadonlyArray1<i64>,
        verts: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        let cells = crate::csr::read_cells(&fptr, &key, &vptr, &verts)?;
        let leaf = leaf.as_slice()?;
        if leaf.len() != cells.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("one leaf per item"));
        }
        self.set_items(cells.into_iter().zip(leaf.iter().copied()).collect())
    }

    /// results() as flat arrays (csr.rs), moved out of the kernel; neg_only: only the
    /// negative components
    /// (the items are released: nothing can be cut after this)
    fn results_csr<'py>(&mut self, py: Python<'py>, neg_only: bool) -> PyResult<crate::csr::CsrOut<'py>> {
        use rayon::prelude::*;
        let r = self.results()?;
        let items = std::mem::take(&mut self.items);
        py.detach(|| items.into_par_iter().for_each(drop));
        Ok(crate::csr::write_results(py, r, neg_only))
    }

    /// use the Rust cap hook (make_feature_hook's, with these feature corners and
    /// min_bend) instead of the Python callback
    fn set_rust_hook(&mut self, corners: PyReadonlyArray2<f64>, min_bend: f64) -> PyResult<()> {
        if self.cap_segs.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err("no cap features"));
        }
        self.hook_rs = Some((crate::points(&corners)?, min_bend));
        Ok(())
    }

    /// the items holding face fid (in item order, an item once per face it has)
    fn items_of_face(&self, py: Python<'_>, fid: i64) -> Vec<usize> {
        use rayon::prelude::*;
        // a scan of the items (asked for a few faces only: no map of all of them)
        py.detach(|| {
            self.items
                .par_iter()
                .enumerate()
                .flat_map_iter(|(k, (fl, _))| fl.iter().filter(move |(f, _)| *f == fid).map(move |_| k))
                .collect()
        })
    }

    /// of the items ks (cut), those with a piece some face of which does not clearly see
    /// its centroid (mesher._piece_bad, the same arithmetic as the NumPy code)
    fn bad_items(&self, ks: Vec<usize>) -> PyResult<Vec<usize>> {
        let mut out = Vec::new();
        for k in ks {
            let r = self.res.get(k).and_then(|x| x.as_ref())
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("item not cut"))?;
            let h = self.csize[self.items[k].1 as usize];
            if r.0.iter().chain(r.1.iter()).any(|c| self.piece_bad(c, h)) {
                out.push(k);
            }
        }
        Ok(out)
    }

    /// cut the items ks in this order (forbid: cap hook off for them); returns per item
    /// (chord / pinch keys used, cap hook changed something)
    fn cut(&mut self, py: Python<'_>, ks: Vec<usize>, forbid: Vec<bool>) -> PyResult<Vec<(Vec<Key3>, bool)>> {
        if ks.len() != forbid.len() || ks.iter().any(|&k| k >= self.items.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("bad item list"));
        }
        let mut out = Vec::with_capacity(ks.len());
        for (&k, &fb) in ks.iter().zip(forbid.iter()) {
            let faces = self.items[k].0.clone();
            self.leaf = self.items[k].1;
            self.forbid = fb;
            self.did = false;
            let (neg, pos, apart) = cut_one_env(&mut RealEnv { k: self, py }, faces).map_err(py_err)?;
            let negc = split_components_apart(neg, &apart);
            let posc = if self.shared_cap { split_components_apart(pos, &HashSet::default()) } else { Vec::new() };
            self.res[k] = Some((negc, posc));
            out.push((self.used_chords.clone(), self.did));
        }
        Ok(out)
    }

    /// cut(all items in order, no forbids) with the items cut in parallel on the state
    /// at the start (symbolic ids), then applied in item order; an item that needs a
    /// Python hook, or whose chord points disagree with an earlier item's, is cut there
    /// on the current state.  The same results, ids and state as cut().
    fn cut_all(&mut self, py: Python<'_>) -> PyResult<Vec<(Vec<Key3>, bool)>> {
        use rayon::prelude::*;
        let n = self.items.len();
        let sym: Vec<SymOut> = {
            let me: &CutKernel = self;
            let chord_segs = me.chord_segs.as_ref().map(|s| s.get());
            let cap_segs = me.cap_segs.as_ref().map(|s| s.get());
            py.detach(|| {
                (0..n)
                    .into_par_iter()
                    .map(|k| {
                        let mut e = SymEnv {
                            k: me,
                            chord_segs,
                            cap_segs,
                            leaf: me.items[k].1,
                            newpos: Vec::new(),
                            newfeat: Vec::new(),
                            vcache: HashMap::default(),
                            chord_local: HashMap::default(),
                            pinch_local: HashMap::default(),
                            scache: HashMap::default(),
                            nf: 0,
                            events: Vec::new(),
                            used: Vec::new(),
                            pinch_now: HashSet::default(),
                            did: false,
                        };
                        let (neg, pos, apart) = cut_one_env(&mut e, me.items[k].0.clone()).ok()?;
                        let negc = split_components_apart(neg, &apart);
                        let posc =
                            if me.shared_cap { split_components_apart(pos, &HashSet::default()) } else { Vec::new() };
                        Some(((negc, posc), e.used, e.events, e.newpos, e.did))
                    })
                    .collect()
            })
        };
        // ids in item order (sequential), then the results renumbered in parallel
        let (mut nv, mut np_, mut nch, mut npi, mut nsub) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for (_, _, ev, newpos, _) in sym.iter().flatten() {
            for e in ev {
                match e {
                    Ev::Vertex(..) | Ev::Zero(..) => nv += 1,
                    Ev::Chord(..) => nch += 1,
                    Ev::Pinch(..) => npi += 1,
                    Ev::Sub(..) => nsub += 1,
                    _ => {}
                }
            }
            np_ += newpos.len();
        }
        self.cache.reserve(nv);
        self.new_cache.reserve(nv);
        self.chord_cache.reserve(nch);
        self.pinch_cache.reserve(npi);
        self.cache_log.reserve(nch + npi);
        self.subids.reserve(nsub);
        self.pos.reserve(np_);
        self.phi.reserve(np_);
        self.new_vertices.reserve(np_);
        self.new_parents.reserve(np_);
        let mut out: Vec<Option<(Vec<Key3>, bool)>> = vec![None; n];
        let mut pending: Vec<(usize, CellRes, Vec<Key3>, Vec<i64>, Vec<i64>, bool)> = Vec::new();
        for (k, s) in sym.into_iter().enumerate() {
            let ok = match &s {
                Some((_, _, ev, np_, _)) => self.check_sym(ev, np_),
                None => false,
            };
            if ok {
                let (res, used, events, newpos, did) = s.unwrap();
                let (vmap, fmap) = self.replay_ids(events, &newpos);
                pending.push((k, res, used, vmap, fmap, did));
            } else {
                out[k] = self.cut(py, vec![k], vec![false])?.pop();
            }
        }
        let mapped: Vec<(usize, CellRes, Vec<Key3>, bool)> = py.detach(|| {
            pending
                .into_par_iter()
                .map(|(k, res, used, vmap, fmap, did)| {
                    let (r, u) = Self::map_result(res, used, &vmap, &fmap);
                    (k, r, u, did)
                })
                .collect()
        });
        for (k, r, u, did) in mapped {
            self.res[k] = Some(r);
            out[k] = Some((u, did));
        }
        Ok(out.into_iter().map(|o| o.unwrap()).collect())
    }

    fn result(&self, k: usize) -> PyResult<CellRes> {
        self.res.get(k).cloned().flatten().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("item not cut"))
    }

    /// all results in item order (moved out of the kernel)
    fn results(&mut self) -> PyResult<Vec<CellRes>> {
        let r = std::mem::take(&mut self.res);
        r.into_iter()
            .map(|x| x.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("item not cut")))
            .collect()
    }

    fn points<'py>(&self, py: Python<'py>, ids: Vec<i64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if ids.iter().any(|&v| v < 0 || v as usize >= self.pos.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
        }
        vec3_array(py, ids.iter().map(|&v| self.pos[v as usize]).collect())
    }

    /// CutContext.forbid_chord
    fn forbid_chord(&mut self, key: Key3) {
        self.chord_cache.insert(key, None);
        self.pinch_cache.insert(key, None);
        self.cache_log.push((0, key, None));
        self.cache_log.push((1, key, None));
    }

    /// (new points[k, 3], new vertex ids, new parents, next face id, new cache entries,
    ///  new sub-face ids, chord / pinch cache changes [(0 chord | 1 pinch, key, value)],
    ///  new feature vertices)
    #[allow(clippy::type_complexity)]
    fn export<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Vec<i64>,
        Vec<(i64, i64)>,
        i64,
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
        Vec<(u8, Key3, Option<i64>)>,
        Vec<i64>,
    )> {
        let newp: Vec<V3> = self.pos[self.n0..].to_vec();
        let cd = PyDict::new(py);
        for (k, v) in self.new_cache.drain(..) {
            cd.set_item(k, v)?;
        }
        let sd = PyDict::new(py);
        let mut new_sub: Vec<(&(i64, Vec<i64>), i64)> =
            self.subids.iter().filter(|e| *e.1 >= self.sub_mark).map(|(k, &v)| (k, v)).collect();
        new_sub.sort_unstable_by_key(|e| e.1); // in the order they were given out
        for ((fid, key), v) in new_sub {
            sd.set_item((*fid, pyo3::types::PyTuple::new(py, key)?), v)?;
        }
        self.sub_mark = self.next_id;
        Ok((
            vec3_array(py, newp)?,
            std::mem::take(&mut self.new_vertices),
            std::mem::take(&mut self.new_parents),
            self.next_id,
            cd,
            sd,
            std::mem::take(&mut self.cache_log),
            std::mem::take(&mut self.new_onfeat),
        ))
    }
}

/// numpy's pairwise sum of a 1-D float64 array (np.sum / ndarray.sum)
pub(crate) fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// a face's centre (np.mean of its points) and area vector halved (0.5 * the sum of
/// np.cross(Q - c, np.roll(Q, -1, 0) - c)), the points by pos
fn face_centre_area2(pos: &dyn Fn(i64) -> V3, p: &[i64]) -> (V3, V3) {
    let m = p.len();
    let mut c = [0.0f64; 3];
    for &v in p {
        let q = pos(v);
        for k in 0..3 {
            c[k] += q[k];
        }
    }
    for k in 0..3 {
        c[k] /= m as f64;
    }
    let mut s = [0.0f64; 3];
    for i in 0..m {
        let a = crate::sub(pos(p[i]), c);
        let b = crate::sub(pos(p[(i + 1) % m]), c);
        let x = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        for k in 0..3 {
            s[k] += x[k];
        }
    }
    (c, s)
}

/// mesher._piece_volume: sum over the faces of np.dot(c, sum of the crosses) / 6
pub(crate) fn piece_volume_at(pos: &dyn Fn(i64) -> V3, comp: &[Face]) -> f64 {
    let mut v = 0.0;
    for (_, p) in comp {
        let (c, s) = face_centre_area2(pos, p);
        v += crate::wallmerge::dot1(c, s) / 6.0;
    }
    v
}

/// mesher._piece_bad, the points by pos
pub(crate) fn piece_bad_at(pos: &dyn Fn(i64) -> V3, comp: &[Face], h: f64) -> bool {
    let nf = comp.len();
    let mut cf: Vec<V3> = Vec::with_capacity(nf);
    let mut sf: Vec<V3> = Vec::with_capacity(nf);
    for (_, p) in comp {
        let (c, s) = face_centre_area2(pos, p);
        cf.push(c);
        sf.push([0.5 * s[0], 0.5 * s[1], 0.5 * s[2]]);
    }
    let mut est = [0.0f64; 3];
    for c in &cf {
        for k in 0..3 {
            est[k] += c[k];
        }
    }
    for k in 0..3 {
        est[k] /= nf as f64;
    }
    let pv: Vec<f64> = (0..nf).map(|i| crate::octree::einsum3(sf[i], crate::sub(cf[i], est))).collect();
    let spv = np_sum(&pv);
    if spv <= 0.0 {
        return true;
    }
    let mut cc = [0.0f64; 3];
    for i in 0..nf {
        for k in 0..3 {
            cc[k] += pv[i] * (0.75 * cf[i][k] + 0.25 * est[k]);
        }
    }
    for k in 0..3 {
        cc[k] /= spv;
    }
    (0..nf).any(|i| {
        let n = crate::dot(sf[i], sf[i]).sqrt();
        crate::octree::einsum3(sf[i], crate::sub(cf[i], cc)) < 0.05 * n * h
    })
}

/// [(mesher._piece_volume, mesher._piece_bad)] of the components of a cut, the points
/// as a CutContext's _PosList holds them: rows of the array a, then of the array b,
/// then the list added (only the rows the components use are read)
#[pyfunction]
pub fn pieces_volume_bad(
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    added: &Bound<'_, pyo3::types::PyList>,
    comps: Vec<Vec<Face>>,
    h: f64,
) -> PyResult<Vec<(f64, bool)>> {
    let (a, b) = (a.as_array(), b.as_array());
    if (a.nrows() > 0 && a.ncols() != 3) || (b.nrows() > 0 && b.ncols() != 3) {
        return Err(pyo3::exceptions::PyValueError::new_err("points must be (n, 3)"));
    }
    let (na, nb) = (a.nrows(), b.nrows());
    let ntot = na + nb + added.len();
    let mut at: HashMap<i64, V3> = HashMap::default();
    for comp in &comps {
        for (_, p) in comp {
            for &v in p {
                if at.contains_key(&v) {
                    continue;
                }
                if v < 0 || v as usize >= ntot {
                    return Err(pyo3::exceptions::PyValueError::new_err("vertex id out of range"));
                }
                let u = v as usize;
                let x: V3 = if u < na {
                    [a[[u, 0]], a[[u, 1]], a[[u, 2]]]
                } else if u < na + nb {
                    [b[[u - na, 0]], b[[u - na, 1]], b[[u - na, 2]]]
                } else {
                    let r: Vec<f64> = added.get_item(u - na - nb)?.extract()?;
                    if r.len() != 3 {
                        return Err(pyo3::exceptions::PyValueError::new_err("points must have 3 coordinates"));
                    }
                    [r[0], r[1], r[2]]
                };
                at.insert(v, x);
            }
        }
    }
    let pos = |v: i64| at[&v];
    Ok(comps.iter().map(|c| (piece_volume_at(&pos, c), piece_bad_at(&pos, c, h))).collect())
}

impl CutKernel {
    /// mesher._piece_bad
    fn piece_bad(&self, comp: &[Face], h: f64) -> bool {
        piece_bad_at(&|v| self.pos[v as usize], comp, h)
    }

    fn new_face_id(&mut self) -> i64 {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    fn sub_id(&mut self, fid: i64, pp: &[i64]) -> i64 {
        let mut k = pp.to_vec();
        k.sort_unstable();
        self.sub_id_sorted(fid, k)
    }

    /// sub_id with the vertex ids sorted already (one hash lookup)
    fn sub_id_sorted(&mut self, fid: i64, k: Vec<i64>) -> i64 {
        use std::collections::hash_map::Entry;
        match self.subids.entry((fid, k)) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let i = self.next_id;
                self.next_id += 1;
                e.insert(i);
                i
            }
        }
    }

    fn add_point(&mut self, x: V3, parents: (i64, i64)) -> i64 {
        let c = self.pos.len() as i64;
        self.pos.push(x);
        self.phi.push(0.0);
        self.new_vertices.push(c);
        self.new_parents.push(parents);
        c
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
            self.add_point(
                [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2])],
                (lo, hi),
            )
        };
        self.cache.insert((lo, hi), c);
        self.new_cache.push(((lo, hi), c));
        c
    }

    /// CutContext.chord_point with the SegSet pierce hook
    fn chord_point(&mut self, py: Python<'_>, fid: i64, poly: &[i64], a: i64, b: i64) -> Option<i64> {
        if a == b || self.plain.contains(&fid) {
            return None;
        }
        let key = key3(fid, a, b);
        if let Some(&c) = self.chord_cache.get(&key) {
            if c.is_some() {
                self.used_chords.push(key);
            }
            return c;
        }
        let segs = self.chord_segs.as_ref().unwrap().bind(py).get();
        let face: Vec<V3> = poly.iter().map(|&v| self.pos[v as usize]).collect();
        let kept: Vec<V3> = poly.iter().filter(|&&v| self.phi[v as usize] < 0.0).map(|&v| self.pos[v as usize]).collect();
        let x = segs.pierce(face, self.pos[a as usize], self.pos[b as usize], kept);
        let c = x.map(|x| {
            let c = self.add_point(x, (-1, -1));
            self.onfeat.insert(c);
            self.new_onfeat.push(c);
            self.used_chords.push(key);
            c
        });
        self.chord_cache.insert(key, c);
        self.cache_log.push((0, key, c));
        c
    }


    /// CutContext.pinch_point through the Python pinch hook
    fn pinch_point(&mut self, py: Python<'_>, fid: i64, poly: &[i64], npoly: &[i64], a: i64, b: i64) -> PyResult<Option<i64>> {
        if self.plain.contains(&fid) {
            return Ok(None);
        }
        let key = key3(fid, a, b);
        let c = if let Some(&c) = self.pinch_cache.get(&key) {
            c
        } else {
            let n = npoly.len();
            let ia = index_of(npoly, a);
            let ib = index_of(npoly, b);
            let near = [a, b, npoly[(ia + n - 1) % n], npoly[(ib + 1) % n]];
            let cand: Vec<i64> =
                npoly.iter().copied().filter(|&v| self.phi[v as usize] == 0.0 && !near.contains(&v)).collect();
            let c = if cand.is_empty() {
                None
            } else {
                let cb = self.pinch_cb.as_ref().unwrap();
                let mut ids: Vec<i64> = poly.to_vec();
                ids.extend_from_slice(npoly);
                let pos = PyDict::new(py);
                for &v in &ids {
                    pos.set_item(v, self.pos[v as usize].to_vec())?;
                }
                let phi = PyDict::new(py);
                for &v in poly {
                    phi.set_item(v, self.phi[v as usize])?;
                }
                cb.call1(py, (poly.to_vec(), a, b, cand, pos, phi))?.extract::<Option<i64>>(py)?
            };
            self.pinch_cache.insert(key, c);
            self.cache_log.push((1, key, c));
            c
        };
        if c.is_some() {
            self.used_chords.push(key);
        }
        Ok(c)
    }



    /// make_feature_hook's hook: the loop itself unless a feature can split or bend it
    /// (then the Python hook decides); returns the cap polygons
    fn cap(&mut self, py: Python<'_>, lp: Vec<i64>) -> PyResult<Vec<Vec<i64>>> {
        if self.cap_cb.is_none() || lp.len() < 3 || self.forbid {
            return Ok(vec![lp]);
        }
        let h = self.csize[self.leaf as usize];
        let segs = self.cap_segs.as_ref().unwrap().bind(py).get();
        let pts: Vec<V3> = lp.iter().map(|&v| self.pos[v as usize]).collect();
        // centre of the loop as numpy's P.mean(0), and its distance to the features
        let m = pts.len() as f64;
        let mut c = [0.0f64; 3];
        for p in &pts {
            for k in 0..3 {
                c[k] += p[k];
            }
        }
        for k in 0..3 {
            c[k] /= m;
        }
        let dc = segs.closest_d(c);
        let nfeat = lp.iter().filter(|v| self.onfeat.contains(v)).count();
        // far from the features: no loop vertex within 1e-6 h of one either (the
        // distances differ by at most the loop radius), only the feature vertices count
        let rad = pts.iter().map(|p| crate::dot(crate::sub(*p, c), crate::sub(*p, c)).sqrt()).fold(0.0, f64::max);
        let far = dc > self.reach * h && dc > rad * (1.0 + 1e-9) + 2e-6 * h;
        let nf = if far {
            nfeat
        } else {
            lp.iter().enumerate().filter(|&(k, v)| self.onfeat.contains(v) || segs.closest_d(pts[k]) < 1e-6 * h).count()
        };
        if nf < 2 && dc > self.reach * h {
            return Ok(vec![lp]);
        }
        let flags: Vec<bool> = lp.iter().map(|v| self.onfeat.contains(v)).collect();
        if let Some((corners, min_bend)) = &self.hook_rs {
            let base = self.pos.len() as i64;
            let o = crate::fhook::feature_hook(segs, corners, &lp, &pts, &flags, base, h, self.reach, *min_bend);
            for x in o.newp {
                self.add_point(x, (-1, -1));
            }
            for v in o.newf {
                self.onfeat.insert(v);
                self.new_onfeat.push(v);
            }
            if o.did {
                self.did = true;
            }
            return Ok(o.polys);
        }
        let plist: Vec<Vec<f64>> = pts.iter().map(|p| p.to_vec()).collect();
        let cb = self.cap_cb.as_ref().unwrap();
        let (polys, newp, newf, did): (Vec<Vec<i64>>, Vec<[f64; 3]>, Vec<i64>, bool) = cb
            .call1(py, (lp.clone(), plist, flags, self.pos.len(), self.leaf, false))?
            .extract(py)?;
        let base = self.pos.len() as i64;
        for x in newp {
            self.add_point(x, (-1, -1));
        }
        let top = self.pos.len() as i64;
        for v in newf {
            if v < base || v >= top {
                return Err(pyo3::exceptions::PyValueError::new_err("cap hook: bad feature vertex"));
            }
            self.onfeat.insert(v);
            self.new_onfeat.push(v);
        }
        if polys.iter().flatten().any(|&v| v < 0 || v >= top) {
            return Err(pyo3::exceptions::PyValueError::new_err("cap hook: vertex id out of range"));
        }
        if did {
            self.did = true;
        }
        Ok(polys)
    }

}

/// what a cut may fail with: a Python error, or (symbolic cut) a step that needs the
/// sequential state (a Python hook)
enum CutErr {
    Py(PyErr),
    Bail,
}

impl From<PyErr> for CutErr {
    fn from(e: PyErr) -> Self {
        CutErr::Py(e)
    }
}

fn py_err(e: CutErr) -> PyErr {
    match e {
        CutErr::Py(p) => p,
        CutErr::Bail => pyo3::exceptions::PyRuntimeError::new_err("internal: sequential cut bailed"),
    }
}

/// the state a cut reads and the ids it asks for
trait Env {
    fn phi(&self, v: i64) -> f64;
    fn pos(&self, v: i64) -> V3;
    fn zero_edge(&self) -> &HashMap<Key3, i64>;
    fn chord_on(&self) -> bool;
    fn pinch_on(&self) -> bool;
    fn shared_cap(&self) -> bool;
    fn pinch_min_area(&self) -> f64;
    fn begin(&mut self);
    fn pinch_now(&self) -> &HashSet<i64>;
    fn pinch_now_insert(&mut self, v: i64);
    fn cross(&mut self, a: i64, b: i64) -> i64;
    fn sub_id(&mut self, fid: i64, pp: &[i64]) -> i64;
    fn new_face_id(&mut self) -> i64;
    fn chord_point(&mut self, fid: i64, poly: &[i64], a: i64, b: i64) -> Option<i64>;
    fn pinch_point(&mut self, fid: i64, poly: &[i64], npoly: &[i64], a: i64, b: i64) -> Result<Option<i64>, CutErr>;
    fn cap(&mut self, lp: Vec<i64>) -> Result<Vec<Vec<i64>>, CutErr>;
}

fn xing<E: Env>(e: &mut E, a: i64, b: i64, a_neg: bool) -> i64 {
    if e.phi(a) == 0.0 && e.phi(b) == 0.0 {
        return if a_neg { a } else { b };
    }
    e.cross(a, b)
}

/// the kernel itself (ids as the Python code gives them)
struct RealEnv<'a, 'py> {
    k: &'a mut CutKernel,
    py: Python<'py>,
}

impl Env for RealEnv<'_, '_> {
    fn phi(&self, v: i64) -> f64 {
        self.k.phi[v as usize]
    }
    fn pos(&self, v: i64) -> V3 {
        self.k.pos[v as usize]
    }
    fn zero_edge(&self) -> &HashMap<Key3, i64> {
        &self.k.zero_edge
    }
    fn chord_on(&self) -> bool {
        self.k.chord_segs.is_some()
    }
    fn pinch_on(&self) -> bool {
        self.k.pinch_cb.is_some()
    }
    fn shared_cap(&self) -> bool {
        self.k.shared_cap
    }
    fn pinch_min_area(&self) -> f64 {
        self.k.pinch_min_area
    }
    fn begin(&mut self) {
        self.k.used_chords.clear();
        self.k.pinch_now.clear();
    }
    fn pinch_now(&self) -> &HashSet<i64> {
        &self.k.pinch_now
    }
    fn pinch_now_insert(&mut self, v: i64) {
        self.k.pinch_now.insert(v);
    }
    fn cross(&mut self, a: i64, b: i64) -> i64 {
        self.k.cross(a, b)
    }
    fn sub_id(&mut self, fid: i64, pp: &[i64]) -> i64 {
        self.k.sub_id(fid, pp)
    }
    fn new_face_id(&mut self) -> i64 {
        self.k.new_face_id()
    }
    fn chord_point(&mut self, fid: i64, poly: &[i64], a: i64, b: i64) -> Option<i64> {
        self.k.chord_point(self.py, fid, poly, a, b)
    }
    fn pinch_point(&mut self, fid: i64, poly: &[i64], npoly: &[i64], a: i64, b: i64) -> Result<Option<i64>, CutErr> {
        Ok(self.k.pinch_point(self.py, fid, poly, npoly, a, b)?)
    }
    fn cap(&mut self, lp: Vec<i64>) -> Result<Vec<Vec<i64>>, CutErr> {
        Ok(self.k.cap(self.py, lp)?)
    }
}

/// an id request or cache change of a symbolic cut, in the order made
enum Ev {
    /// new crossing of the edge (lo, hi) (symbolic vertex: the next new point)
    Vertex(i64, i64),
    /// the edge (lo, hi) with an end on the surface: the crossing is that end
    Zero(i64, i64, i64),
    /// chord point of the key (symbolic ids): None, or the next new point
    Chord(Key3, bool),
    /// pinch point of the key found to be None (no candidate)
    Pinch(Key3),
    /// the part pp (symbolic vertices) of face fid
    Sub(i64, Vec<i64>),
    /// a cap face
    New,
    /// a point the cap hook adds (the next new point); on a feature or not
    HookPoint(bool),
}

/// one item cut on the kernel's state at the start of the pass, with symbolic ids for
/// what it creates: new points -(1 + k) (k-th new point of the item), new faces
/// -(1 + j).  The cut compares ids only for equality, so a replay that maps them onto
/// the real ids (CutKernel::replay) gives the sequential result, unless a cache filled
/// by an earlier item disagrees (checked there; the item is then cut again).
struct SymEnv<'a> {
    k: &'a CutKernel,
    chord_segs: Option<&'a SegSet>,
    cap_segs: Option<&'a SegSet>,
    leaf: i64,
    newpos: Vec<V3>,
    newfeat: Vec<bool>,
    vcache: HashMap<(i64, i64), i64>,
    chord_local: HashMap<Key3, Option<i64>>,
    pinch_local: HashMap<Key3, Option<i64>>,
    scache: HashMap<(i64, Vec<i64>), i64>,
    nf: i64,
    events: Vec<Ev>,
    used: Vec<Key3>,
    pinch_now: HashSet<i64>,
    did: bool,
}

impl SymEnv<'_> {
    fn add_point(&mut self, x: V3, feat: bool) -> i64 {
        self.newpos.push(x);
        self.newfeat.push(feat);
        -(self.newpos.len() as i64)
    }
    fn onfeat(&self, v: i64) -> bool {
        if v < 0 { self.newfeat[(-v - 1) as usize] } else { self.k.onfeat.contains(&v) }
    }
}

impl Env for SymEnv<'_> {
    fn phi(&self, v: i64) -> f64 {
        if v < 0 { 0.0 } else { self.k.phi[v as usize] }
    }
    fn pos(&self, v: i64) -> V3 {
        if v < 0 { self.newpos[(-v - 1) as usize] } else { self.k.pos[v as usize] }
    }
    fn zero_edge(&self) -> &HashMap<Key3, i64> {
        &self.k.zero_edge
    }
    fn chord_on(&self) -> bool {
        self.k.chord_segs.is_some()
    }
    fn pinch_on(&self) -> bool {
        self.k.pinch_cb.is_some()
    }
    fn shared_cap(&self) -> bool {
        self.k.shared_cap
    }
    fn pinch_min_area(&self) -> f64 {
        self.k.pinch_min_area
    }
    fn begin(&mut self) {
        self.used.clear();
        self.pinch_now.clear();
    }
    fn pinch_now(&self) -> &HashSet<i64> {
        &self.pinch_now
    }
    fn pinch_now_insert(&mut self, v: i64) {
        self.pinch_now.insert(v);
    }
    fn cross(&mut self, a: i64, b: i64) -> i64 {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if let Some(&c) = self.k.cache.get(&(lo, hi)) {
            return c;
        }
        if let Some(&c) = self.vcache.get(&(lo, hi)) {
            return c;
        }
        let (pl, ph) = (self.phi(lo), self.phi(hi));
        let c = if pl == 0.0 || ph == 0.0 {
            let c = if pl == 0.0 { lo } else { hi };
            self.events.push(Ev::Zero(lo, hi, c));
            c
        } else {
            let t = pl / (pl - ph);
            let (pa, pb) = (self.pos(lo), self.pos(hi));
            self.events.push(Ev::Vertex(lo, hi));
            self.add_point([pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2])], false)
        };
        self.vcache.insert((lo, hi), c);
        c
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
    fn new_face_id(&mut self) -> i64 {
        self.nf += 1;
        self.events.push(Ev::New);
        -self.nf
    }
    fn chord_point(&mut self, fid: i64, poly: &[i64], a: i64, b: i64) -> Option<i64> {
        if a == b || self.k.plain.contains(&fid) {
            return None;
        }
        let key = key3(fid, a, b);
        let hit = if a >= 0 && b >= 0 { self.k.chord_cache.get(&key).copied() } else { None };
        let hit = hit.or_else(|| self.chord_local.get(&key).copied());
        if let Some(c) = hit {
            if c.is_some() {
                self.used.push(key);
            }
            return c;
        }
        let segs = self.chord_segs.unwrap();
        let face: Vec<V3> = poly.iter().map(|&v| self.pos(v)).collect();
        let kept: Vec<V3> = poly.iter().filter(|&&v| self.phi(v) < 0.0).map(|&v| self.pos(v)).collect();
        let x = segs.pierce(face, self.pos(a), self.pos(b), kept);
        self.events.push(Ev::Chord(key, x.is_some()));
        let c = x.map(|x| {
            let c = self.add_point(x, true);
            self.used.push(key);
            c
        });
        self.chord_local.insert(key, c);
        c
    }
    fn pinch_point(&mut self, fid: i64, poly: &[i64], npoly: &[i64], a: i64, b: i64) -> Result<Option<i64>, CutErr> {
        let _ = poly;
        if self.k.plain.contains(&fid) {
            return Ok(None);
        }
        let key = key3(fid, a, b);
        let hit = if a >= 0 && b >= 0 { self.k.pinch_cache.get(&key).copied() } else { None };
        let c = match hit.or_else(|| self.pinch_local.get(&key).copied()) {
            Some(c) => c,
            None => {
                let n = npoly.len();
                let ia = index_of(npoly, a);
                let ib = index_of(npoly, b);
                let near = [a, b, npoly[(ia + n - 1) % n], npoly[(ib + 1) % n]];
                if npoly.iter().any(|&v| self.phi(v) == 0.0 && !near.contains(&v)) {
                    return Err(CutErr::Bail); // the Python pinch hook decides
                }
                self.events.push(Ev::Pinch(key));
                self.pinch_local.insert(key, None);
                None
            }
        };
        if c.is_some() {
            self.used.push(key);
        }
        Ok(c)
    }
    fn cap(&mut self, lp: Vec<i64>) -> Result<Vec<Vec<i64>>, CutErr> {
        if self.k.cap_cb.is_none() || lp.len() < 3 {
            return Ok(vec![lp]);
        }
        let h = self.k.csize[self.leaf as usize];
        let segs = self.cap_segs.unwrap();
        let pts: Vec<V3> = lp.iter().map(|&v| self.pos(v)).collect();
        let m = pts.len() as f64;
        let mut c = [0.0f64; 3];
        for p in &pts {
            for k in 0..3 {
                c[k] += p[k];
            }
        }
        for k in 0..3 {
            c[k] /= m;
        }
        let dc = segs.closest_d(c);
        let nfeat = lp.iter().filter(|&&v| self.onfeat(v)).count();
        let rad = pts.iter().map(|p| crate::dot(crate::sub(*p, c), crate::sub(*p, c)).sqrt()).fold(0.0, f64::max);
        let far = dc > self.k.reach * h && dc > rad * (1.0 + 1e-9) + 2e-6 * h;
        let nf = if far {
            nfeat
        } else {
            lp.iter().enumerate().filter(|&(k, &v)| self.onfeat(v) || segs.closest_d(pts[k]) < 1e-6 * h).count()
        };
        if nf < 2 && dc > self.k.reach * h {
            return Ok(vec![lp]);
        }
        let Some((corners, min_bend)) = &self.k.hook_rs else {
            return Err(CutErr::Bail); // the Python cap hook decides
        };
        // the Rust hook with symbolic ids for the points it adds
        const BASE: i64 = i64::MAX / 4;
        let flags: Vec<bool> = lp.iter().map(|&v| self.onfeat(v)).collect();
        let o = crate::fhook::feature_hook(segs, corners, &lp, &pts, &flags, BASE, h, self.k.reach, *min_bend);
        let ids: Vec<i64> = o
            .newp
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let feat = o.newf.contains(&(BASE + i as i64));
                self.events.push(Ev::HookPoint(feat));
                self.add_point(x, feat)
            })
            .collect();
        if o.did {
            self.did = true;
        }
        Ok(o.polys.into_iter().map(|q| q.into_iter().map(|v| if v >= BASE { ids[(v - BASE) as usize] } else { v }).collect()).collect())
    }
}

/// a symbolic cut's outcome: results, used keys, events, new points, cap hook did
type SymOut = Option<(CellRes, Vec<Key3>, Vec<Ev>, Vec<V3>, bool)>;

/// CutContext._zero_runs
fn zero_runs<E: Env>(e: &E, fid: i64, poly: &[i64], cls: &mut [bool]) {
    let n = poly.len();
    let z: Vec<bool> = poly.iter().map(|&v| e.phi(v) == 0.0).collect();
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
        let pn = e.phi(poly[im]) < 0.0;
        if pn == (e.phi(poly[(i + k) % n]) < 0.0) {
            continue;
        }
        if !pn {
            run.reverse();
        }
        for t in 0..k - 1 {
            let e = e.zero_edge().get(&key3(fid, poly[run[t]], poly[run[t + 1]])).copied();
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

fn area<E: Env>(e: &E, poly: &[i64]) -> f64 {
    let x0 = e.pos(poly[0]);
    let mut s = [0.0f64; 3];
    for k in 1..poly.len().saturating_sub(1) {
        let p = e.pos(poly[k]);
        let q = e.pos(poly[k + 1]);
        let u = [p[0] - x0[0], p[1] - x0[1], p[2] - x0[2]];
        let w = [q[0] - x0[0], q[1] - x0[1], q[2] - x0[2]];
        s[0] += u[1] * w[2] - u[2] * w[1];
        s[1] += u[2] * w[0] - u[0] * w[2];
        s[2] += u[0] * w[1] - u[1] * w[0];
    }
    0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).powf(0.5)
}

/// CutContext._pinch_line
fn pinch_line<E: Env>(e: &E, neg: &[Face]) -> HashSet<i64> {
    let mut pv: Vec<i64> = e.pinch_now().iter().copied().collect();
    pv.sort_unstable();
    let mut line: HashSet<i64> = pv.iter().copied().collect();
    let mut zs: Vec<i64> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::default();
    for (_, p) in neg {
        for &v in p {
            if e.phi(v) == 0.0 && !line.contains(&v) && seen.insert(v) {
                zs.push(v);
            }
        }
    }
    for i in 0..pv.len() {
        let a = e.pos(pv[i]);
        for j in i + 1..pv.len() {
            let b = e.pos(pv[j]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let l2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            if l2 == 0.0 {
                continue;
            }
            for &v in &zs {
                let x = e.pos(v);
                let w = [x[0] - a[0], x[1] - a[1], x[2] - a[2]];
                let t = (w[0] * u[0] + w[1] * u[1] + w[2] * u[2]) / l2;
                if !(0.0 < t && t < 1.0) {
                    continue;
                }
                let d = [w[0] - t * u[0], w[1] - t * u[1], w[2] - t * u[2]];
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 0.01 * l2 {
                    line.insert(v);
                }
            }
        }
    }
    line
}

/// CutContext.cut -> (neg faces, pos faces, last_pinch)
fn cut_one_env<E: Env>(e: &mut E, faces: Vec<Face>) -> Result<(Vec<Face>, Vec<Face>, HashSet<i64>), CutErr> {
    let mut neg: Vec<Face> = Vec::new();
    let mut pos: Vec<Face> = Vec::new();
    e.begin();
    let chord_on = e.chord_on();
    let pinch_on = e.pinch_on();
    for (fid, poly) in faces {
        let mut has_n = false;
        let mut has_p = false;
        for &v in &poly {
            let x = e.phi(v);
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
        let mut cls: Vec<bool> = poly.iter().map(|&v| e.phi(v) <= 0.0).collect();
        if !e.zero_edge().is_empty() {
            zero_runs(e, fid, &poly, &mut cls);
        }
        let s = (0..n).find(|&i| cls[i] && !cls[(i + n - 1) % n]).unwrap_or(0);
        let mut seq_n: Vec<i64> = Vec::new();
        let mut ppieces: Vec<Vec<i64>> = Vec::new();
        let mut cur: Vec<i64> = Vec::new();
        let mut chords: Vec<(i64, i64, Option<i64>)> = Vec::new();
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
                let c = xing(e, a, poly[j], true);
                seq_n.push(c);
                cur = vec![c];
            } else if !cls[i] && cls[j] {
                let c = xing(e, a, poly[j], false);
                let f = if chord_on { e.chord_point(fid, &poly, cur[0], c) } else { None };
                chords.push((cur[0], c, f));
                cur.push(c);
                if let Some(f) = f {
                    cur.push(f);
                    seq_n.push(f);
                }
                ppieces.push(std::mem::take(&mut cur));
                seq_n.push(c);
            }
        }
        let mut pieces: Vec<(u8, Vec<i64>)> = Vec::new();
        let npoly = dedupe(&seq_n);
        let mut tp: Option<i64> = None;
        let mut kk: (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
        if pinch_on && chords.len() == 1 && chords[0].2.is_none() && npoly.len() >= 4 && n_unique(&npoly) == npoly.len()
        {
            let (a, b, _) = chords[0];
            if a != b && npoly.contains(&a) && npoly.contains(&b) {
                let m = npoly.len();
                let ib = index_of(&npoly, b);
                if npoly[(ib + m - 1) % m] == a {
                    tp = e.pinch_point(fid, &poly, &npoly, a, b)?;
                    if let Some(t) = tp {
                        let seq: Vec<i64> = npoly[ib..].iter().chain(npoly[..ib].iter()).copied().collect();
                        let it = index_of(&seq, t);
                        let mut k1 = vec![t];
                        k1.extend_from_slice(&seq[..it]);
                        let k2 = seq[it..].to_vec();
                        let (a1, a2, a0) = (area(e, &k1), area(e, &k2), area(e, &npoly));
                        if a1.min(a2) < e.pinch_min_area() * a0 {
                            tp = None;
                        } else {
                            kk = (k1, k2);
                        }
                    }
                }
            }
        }
        if let Some(t) = tp {
            pieces.push((0, std::mem::take(&mut kk.0)));
            pieces.push((0, std::mem::take(&mut kk.1)));
            ppieces[0].push(t);
            e.pinch_now_insert(t);
        } else if n_unique(&npoly) >= 3 {
            pieces.push((0, npoly));
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
            let f2 = e.sub_id(fid, &pp);
            if k == 0 {
                neg.push((f2, pp));
            } else {
                pos.push((f2, pp));
            }
        }
    }
    // cap = reversed open boundary of the N-side faces (of each part, where the kept
    // side is pinched along a line of surface vertices)
    let apart = if e.pinch_now().len() >= 2 { pinch_line(e, &neg) } else { HashSet::default() };
    let groups: Vec<Vec<Face>> =
        if apart.is_empty() { vec![neg.clone()] } else { split_components_apart(neg.clone(), &apart) };
    let mut caps: Vec<Vec<i64>> = Vec::new();
    for grp in &groups {
        let mut de: OMap<(i64, i64), i64> = OMap::new();
        for (_, p) in grp {
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
        caps.extend(chain_loops(&segs));
    }
    for lp in caps {
        let polys = e.cap(lp)?;
        for poly in polys {
            let rev: Vec<i64> = poly.iter().rev().copied().collect();
            if e.shared_cap() {
                let cid = e.new_face_id();
                neg.push((cid, poly));
                pos.push((cid, rev));
            } else {
                let c1 = e.new_face_id();
                neg.push((c1, poly));
                let c2 = e.new_face_id();
                pos.push((c2, rev));
            }
        }
    }
    Ok((neg, pos, apart))
}

impl CutKernel {
    /// map a symbolic id (new point -(1 + k)) through vmap
    fn real_v(v: i64, vmap: &[i64]) -> i64 {
        if v < 0 { vmap[(-v - 1) as usize] } else { v }
    }

    /// the vertex each symbolic new point stands for as far as the current state knows:
    /// Some(id) if it exists already, None if it will be new (no cache can hold a key
    /// with it then); and whether the item's chord / pinch results agree with the caches
    fn check_sym(&self, events: &[Ev], newpos: &[V3]) -> bool {
        if !events.iter().any(|e| matches!(e, Ev::Chord(..) | Ev::Pinch(..))) {
            return true;
        }
        let mut tv: Vec<Option<i64>> = Vec::with_capacity(newpos.len());
        let res = |v: i64, tv: &[Option<i64>]| if v < 0 { tv[(-v - 1) as usize] } else { Some(v) };
        for ev in events {
            match ev {
                Ev::Vertex(lo, hi) => tv.push(self.cache.get(&(*lo, *hi)).copied()),
                Ev::HookPoint(_) => tv.push(None),
                Ev::Chord((f, a, b), some) => {
                    let key = match (res(*a, &tv), res(*b, &tv)) {
                        (Some(a), Some(b)) => Some(key3(*f, a, b)),
                        _ => None,
                    };
                    let cached = key.and_then(|k| self.chord_cache.get(&k).copied());
                    match cached {
                        None => {
                            if *some {
                                tv.push(None);
                            }
                        }
                        Some(None) => {
                            if *some {
                                return false;
                            }
                        }
                        Some(Some(c)) => {
                            if !*some {
                                return false;
                            }
                            let x = newpos[tv.len()];
                            let y = self.pos[c as usize];
                            if x.iter().zip(y.iter()).any(|(p, q)| p.to_bits() != q.to_bits()) {
                                return false;
                            }
                            tv.push(Some(c));
                        }
                    }
                }
                Ev::Pinch((f, a, b)) => {
                    if let (Some(a), Some(b)) = (res(*a, &tv), res(*b, &tv)) {
                        if let Some(Some(_)) = self.pinch_cache.get(&key3(*f, a, b)) {
                            return false;
                        }
                    }
                }
                _ => {}
            }
        }
        true
    }

    /// apply a checked symbolic cut's id requests to the state: the real ids of its new
    /// points (vmap) and new faces (fmap), in the order of the sequential cut
    fn replay_ids(&mut self, events: Vec<Ev>, newpos: &[V3]) -> (Vec<i64>, Vec<i64>) {
        let mut vmap: Vec<i64> = Vec::with_capacity(newpos.len());
        let mut fmap: Vec<i64> = Vec::new();
        let rk = |(f, a, b): Key3, vmap: &[i64]| key3(f, Self::real_v(a, vmap), Self::real_v(b, vmap));
        for ev in events {
            match ev {
                Ev::Vertex(lo, hi) => {
                    let fresh_id = self.pos.len() as i64;
                    let mut fresh = false;
                    let c = *self.cache.entry((lo, hi)).or_insert_with(|| {
                        fresh = true;
                        fresh_id
                    });
                    if fresh {
                        self.add_point(newpos[vmap.len()], (lo, hi));
                        self.new_cache.push(((lo, hi), c));
                    }
                    vmap.push(c);
                }
                Ev::Zero(lo, hi, c) => {
                    if !self.cache.contains_key(&(lo, hi)) {
                        self.cache.insert((lo, hi), c);
                        self.new_cache.push(((lo, hi), c));
                    }
                }
                Ev::Chord(key, some) => {
                    let key = rk(key, &vmap);
                    match self.chord_cache.get(&key).copied() {
                        Some(c) => {
                            if let Some(c) = c {
                                vmap.push(c);
                            }
                        }
                        None => {
                            let c = if some {
                                let c = self.add_point(newpos[vmap.len()], (-1, -1));
                                self.onfeat.insert(c);
                                self.new_onfeat.push(c);
                                vmap.push(c);
                                Some(c)
                            } else {
                                None
                            };
                            self.chord_cache.insert(key, c);
                            self.cache_log.push((0, key, c));
                        }
                    }
                }
                Ev::Pinch(key) => {
                    let key = rk(key, &vmap);
                    if !self.pinch_cache.contains_key(&key) {
                        self.pinch_cache.insert(key, None);
                        self.cache_log.push((1, key, None));
                    }
                }
                Ev::Sub(fid, mut pp) => {
                    for v in pp.iter_mut() {
                        *v = Self::real_v(*v, &vmap);
                    }
                    pp.sort_unstable();
                    fmap.push(self.sub_id_sorted(fid, pp));
                }
                Ev::New => fmap.push(self.new_face_id()),
                Ev::HookPoint(feat) => {
                    let c = self.add_point(newpos[vmap.len()], (-1, -1));
                    if feat {
                        self.onfeat.insert(c);
                        self.new_onfeat.push(c);
                    }
                    vmap.push(c);
                }
            }
        }
        (vmap, fmap)
    }

    /// a symbolic cut's results and used keys with the real ids (no state involved)
    fn map_result(res: CellRes, used: Vec<Key3>, vmap: &[i64], fmap: &[i64]) -> (CellRes, Vec<Key3>) {
        let rk = |(f, a, b): Key3| key3(f, Self::real_v(a, vmap), Self::real_v(b, vmap));
        let map_comps = |comps: Vec<Vec<Face>>| -> Vec<Vec<Face>> {
            comps
                .into_iter()
                .map(|comp| {
                    comp.into_iter()
                        .map(|(f, p)| {
                            let f = if f < 0 { fmap[(-f - 1) as usize] } else { f };
                            (f, p.into_iter().map(|v| Self::real_v(v, vmap)).collect())
                        })
                        .collect()
                })
                .collect()
        };
        let used = used.into_iter().map(rk).collect();
        ((map_comps(res.0), map_comps(res.1)), used)
    }
}

/// the crossing cache and sub-face ids of a CutKernel (take_caches)
#[pyclass]
pub struct CutCaches {
    cache: HashMap<(i64, i64), i64>,
    subids: HashMap<(i64, Vec<i64>), i64>,
    chord: HashMap<Key3, Option<i64>>,
    pinch: HashMap<Key3, Option<i64>>,
}

#[pymethods]
impl CutCaches {
    fn cache_get(&self, key: (i64, i64)) -> Option<i64> {
        self.cache.get(&key).copied()
    }

    fn subid_get(&self, key: (i64, Vec<i64>)) -> Option<i64> {
        self.subids.get(&key).copied()
    }

    /// (key cached, its point or None)
    fn chord_get(&self, key: Key3) -> (bool, Option<i64>) {
        self.chord.get(&key).map_or((false, None), |&v| (true, v))
    }

    fn pinch_get(&self, key: Key3) -> (bool, Option<i64>) {
        self.pinch.get(&key).map_or((false, None), |&v| (true, v))
    }
}
