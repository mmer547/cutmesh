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
    new_subids: Vec<((i64, Vec<i64>), i64)>,
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
    face_items: HashMap<i64, Vec<usize>>,
    res: Vec<Option<CellRes>>,
    // state of the current cut
    used_chords: Vec<Key3>,
    pinch_now: HashSet<i64>,
    did: bool,
    forbid: bool,
    leaf: i64,
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
            new_subids: Vec::new(),
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
            face_items: HashMap::default(),
            res: Vec::new(),
            used_chords: Vec::new(),
            pinch_now: HashSet::default(),
            did: false,
            forbid: false,
            leaf: 0,
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
        self.face_items.clear();
        for (k, (fl, _)) in items.iter().enumerate() {
            for (fid, _) in fl {
                self.face_items.entry(*fid).or_default().push(k);
            }
        }
        self.items = items;
        Ok(())
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
    fn results_csr<'py>(&mut self, py: Python<'py>, neg_only: bool) -> PyResult<crate::csr::CsrOut<'py>> {
        let r = self.results()?;
        Ok(crate::csr::write_results(py, &r, neg_only))
    }

    /// the items holding face fid (in item order, an item once per face it has)
    fn items_of_face(&self, fid: i64) -> Vec<usize> {
        self.face_items.get(&fid).cloned().unwrap_or_default()
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
            let (neg, pos, apart) = self.cut_one(py, faces)?;
            let negc = split_components_apart(neg, &apart);
            let posc = if self.shared_cap { split_components_apart(pos, &HashSet::default()) } else { Vec::new() };
            self.res[k] = Some((negc, posc));
            out.push((self.used_chords.clone(), self.did));
        }
        Ok(out)
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
        for ((fid, key), v) in self.new_subids.drain(..) {
            sd.set_item((fid, pyo3::types::PyTuple::new(py, key)?), v)?;
        }
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

impl CutKernel {
    /// mesher._piece_bad
    fn piece_bad(&self, comp: &[Face], h: f64) -> bool {
        let nf = comp.len();
        let mut cf: Vec<V3> = Vec::with_capacity(nf);
        let mut sf: Vec<V3> = Vec::with_capacity(nf);
        for (_, p) in comp {
            let m = p.len();
            let mut c = [0.0f64; 3];
            for &v in p {
                let q = self.pos[v as usize];
                for k in 0..3 {
                    c[k] += q[k];
                }
            }
            for k in 0..3 {
                c[k] /= m as f64;
            }
            let mut s = [0.0f64; 3];
            for i in 0..m {
                let a = crate::sub(self.pos[p[i] as usize], c);
                let b = crate::sub(self.pos[p[(i + 1) % m] as usize], c);
                let x = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
                for k in 0..3 {
                    s[k] += x[k];
                }
            }
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
        self.subids.insert(key.clone(), i);
        self.new_subids.push((key, i));
        i
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

    fn xing(&mut self, a: i64, b: i64, a_neg: bool) -> i64 {
        if self.phi[a as usize] == 0.0 && self.phi[b as usize] == 0.0 {
            return if a_neg { a } else { b };
        }
        self.cross(a, b)
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
                let e = self.zero_edge.get(&key3(fid, poly[run[t]], poly[run[t + 1]])).copied();
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

    fn area(&self, poly: &[i64]) -> f64 {
        let x0 = self.pos[poly[0] as usize];
        let mut s = [0.0f64; 3];
        for k in 1..poly.len().saturating_sub(1) {
            let p = self.pos[poly[k] as usize];
            let q = self.pos[poly[k + 1] as usize];
            let u = [p[0] - x0[0], p[1] - x0[1], p[2] - x0[2]];
            let w = [q[0] - x0[0], q[1] - x0[1], q[2] - x0[2]];
            s[0] += u[1] * w[2] - u[2] * w[1];
            s[1] += u[2] * w[0] - u[0] * w[2];
            s[2] += u[0] * w[1] - u[1] * w[0];
        }
        0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).powf(0.5)
    }

    /// CutContext._pinch_line
    fn pinch_line(&self, neg: &[Face]) -> HashSet<i64> {
        let mut pv: Vec<i64> = self.pinch_now.iter().copied().collect();
        pv.sort_unstable();
        let mut line: HashSet<i64> = pv.iter().copied().collect();
        let mut zs: Vec<i64> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::default();
        for (_, p) in neg {
            for &v in p {
                if self.phi[v as usize] == 0.0 && !line.contains(&v) && seen.insert(v) {
                    zs.push(v);
                }
            }
        }
        for i in 0..pv.len() {
            let a = self.pos[pv[i] as usize];
            for j in i + 1..pv.len() {
                let b = self.pos[pv[j] as usize];
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let l2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                if l2 == 0.0 {
                    continue;
                }
                for &v in &zs {
                    let x = self.pos[v as usize];
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

    /// CutContext.cut -> (neg faces, pos faces, last_pinch)
    fn cut_one(&mut self, py: Python<'_>, faces: Vec<Face>) -> PyResult<(Vec<Face>, Vec<Face>, HashSet<i64>)> {
        let mut neg: Vec<Face> = Vec::new();
        let mut pos: Vec<Face> = Vec::new();
        self.used_chords.clear();
        self.pinch_now.clear();
        let chord_on = self.chord_segs.is_some();
        let pinch_on = self.pinch_cb.is_some();
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
                    let c = self.xing(a, poly[j], true);
                    seq_n.push(c);
                    cur = vec![c];
                } else if !cls[i] && cls[j] {
                    let c = self.xing(a, poly[j], false);
                    let f = if chord_on { self.chord_point(py, fid, &poly, cur[0], c) } else { None };
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
                        tp = self.pinch_point(py, fid, &poly, &npoly, a, b)?;
                        if let Some(t) = tp {
                            let seq: Vec<i64> = npoly[ib..].iter().chain(npoly[..ib].iter()).copied().collect();
                            let it = index_of(&seq, t);
                            let mut k1 = vec![t];
                            k1.extend_from_slice(&seq[..it]);
                            let k2 = seq[it..].to_vec();
                            let (a1, a2, a0) = (self.area(&k1), self.area(&k2), self.area(&npoly));
                            if a1.min(a2) < self.pinch_min_area * a0 {
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
                self.pinch_now.insert(t);
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
                let f2 = self.sub_id(fid, &pp);
                if k == 0 {
                    neg.push((f2, pp));
                } else {
                    pos.push((f2, pp));
                }
            }
        }
        // cap = reversed open boundary of the N-side faces (of each part, where the kept
        // side is pinched along a line of surface vertices)
        let apart = if self.pinch_now.len() >= 2 { self.pinch_line(&neg) } else { HashSet::default() };
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
            let polys = self.cap(py, lp)?;
            for poly in polys {
                let rev: Vec<i64> = poly.iter().rev().copied().collect();
                if self.shared_cap {
                    let cid = self.new_face_id();
                    neg.push((cid, poly));
                    pos.push((cid, rev));
                } else {
                    let c1 = self.new_face_id();
                    neg.push((c1, poly));
                    let c2 = self.new_face_id();
                    pos.push((c2, rev));
                }
            }
        }
        Ok((neg, pos, apart))
    }
}
