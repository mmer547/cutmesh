//! Octree to polyhedral cells: port of octree.Octree.build_cells (integer lattice part).
//! The leaves are given by their level and integer coordinates; the result has exactly
//! the arrays (and the orders) of the NumPy code, built in parallel.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{dot, sub, V3};

/// local corner offsets in the order of itertools.product((0, 1), repeat=3)
const CHILD: [[i64; 3]; 8] = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
];

/// outward hex faces: (axis, side, 4 local corners), as HEX_FACES in octree.py
const HEX_FACES: [(usize, bool, [[i64; 3]; 4]); 6] = [
    (0, false, [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
    (0, true, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
    (1, false, [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
    (1, true, [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
    (2, false, [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
    (2, true, [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
];

type I3 = [i64; 3];

struct Tree {
    n: I3,
    lmax: u32,
    periodic: [bool; 3],
    /// keys sorted by (level, key), the leaf of each, and where each level starts
    keys: Vec<i64>,
    idx: Vec<i64>,
    start: Vec<usize>,
}

impl Tree {
    fn ncell(&self, l: u32) -> I3 {
        [self.n[0] << l, self.n[1] << l, self.n[2] << l]
    }
    fn wrap(&self, l: u32, mut q: I3) -> I3 {
        let nn = self.ncell(l);
        for a in 0..3 {
            if self.periodic[a] {
                q[a] = q[a].rem_euclid(nn[a]);
            }
        }
        q
    }
    fn key(&self, l: u32, q: I3) -> i64 {
        let nn = self.ncell(l);
        (q[0].wrapping_mul(nn[1]).wrapping_add(q[1])).wrapping_mul(nn[2]).wrapping_add(q[2])
    }
    fn find_exact(&self, l: u32, q: I3) -> i64 {
        let (a, b) = (self.start[l as usize], self.start[l as usize + 1]);
        let k = self.key(l, q);
        match self.keys[a..b].binary_search(&k) {
            Ok(p) => self.idx[a + p],
            Err(_) => -1,
        }
    }
    /// (inside the box, leaf of level <= l containing the level-l cell q or -1)
    fn covering(&self, l: u32, q: I3) -> (bool, i64) {
        let q = self.wrap(l, q);
        let nn = self.ncell(l);
        if (0..3).any(|a| q[a] < 0 || q[a] >= nn[a]) {
            return (false, -1);
        }
        for c in (0..=l).rev() {
            let s = l - c;
            let r = self.find_exact(c, [q[0] >> s, q[1] >> s, q[2] >> s]);
            if r >= 0 {
                return (true, r);
            }
        }
        (true, -1)
    }
}

struct Verts {
    nv: I3,
    nfine: I3,
    periodic: [bool; 3],
    ukeys: Vec<i64>,
}

impl Verts {
    fn vkey(&self, mut x: I3) -> i64 {
        for a in 0..3 {
            if self.periodic[a] {
                x[a] = x[a].rem_euclid(self.nfine[a]);
            }
        }
        (x[0].wrapping_mul(self.nv[1]).wrapping_add(x[1])).wrapping_mul(self.nv[2]).wrapping_add(x[2])
    }
    fn vid_key(&self, k: i64) -> i64 {
        match self.ukeys.binary_search(&k) {
            Ok(p) => p as i64,
            Err(_) => -1,
        }
    }
    fn vid(&self, x: I3) -> i64 {
        self.vid_key(self.vkey(x))
    }
}

fn half(a: I3, b: I3) -> I3 {
    // (a + b) // 2 with Python's floor division
    [(a[0] + b[0]).div_euclid(2), (a[1] + b[1]).div_euclid(2), (a[2] + b[2]).div_euclid(2)]
}

fn par_unique(mut v: Vec<i64>) -> Vec<i64> {
    v.par_sort_unstable();
    v.dedup();
    v
}

type Out<'py> = (
    Bound<'py, PyArray2<i64>>, // points_lattice
    Bound<'py, PyArray1<bool>>, // hanging
    Bound<'py, PyArray2<i64>>, // hang_par
    Bound<'py, PyArray1<i64>>, // hang_size
    Bound<'py, PyArray1<i64>>, // face_cell
    Bound<'py, PyArray1<i64>>, // face_key
    Bound<'py, PyArray1<i64>>, // fptr
    Bound<'py, PyArray1<i64>>, // fverts
    Bound<'py, PyArray1<i64>>, // cell_fptr
    Bound<'py, PyArray2<i64>>, // corner_vid
);

/// build_cells(ijk[n, 3], lev[n], n[3], lmax, periodic[3]) -> (points_lattice, hanging,
/// hang_par, hang_size, face_cell, face_key, fptr, fverts, cell_fptr, corner_vid)
#[pyfunction]
pub fn build_cells<'py>(
    py: Python<'py>,
    ijk: PyReadonlyArray2<'py, i64>,
    lev: PyReadonlyArray1<'py, i64>,
    n: [i64; 3],
    lmax: u32,
    periodic: [bool; 3],
) -> PyResult<Out<'py>> {
    let (ijk, lev) = leaves_in(&ijk, &lev, lmax)?;
    let nl = lev.len();
    let r = py.detach(|| build(&ijk, &lev, n, lmax, periodic));
    let nv = r.ukeys.len();
    let hp = numpy::ndarray::Array2::from_shape_vec((nv, 4), r.hang_par.into_iter().flatten().collect()).unwrap();
    let lat = numpy::ndarray::Array2::from_shape_vec((nv, 3), r.lat.into_iter().flatten().collect()).unwrap();
    let cvid = numpy::ndarray::Array2::from_shape_vec((nl, 8), r.corner_vid.into_iter().flatten().collect()).unwrap();
    Ok((
        lat.into_pyarray(py),
        r.hanging.into_pyarray(py),
        hp.into_pyarray(py),
        r.hang_size.into_pyarray(py),
        r.face_cell.into_pyarray(py),
        r.face_key.into_pyarray(py),
        r.fptr.into_pyarray(py),
        r.fverts.into_pyarray(py),
        r.cell_fptr.into_pyarray(py),
        cvid.into_pyarray(py),
    ))
}

struct Res {
    ukeys: Vec<i64>,
    lat: Vec<I3>,
    hanging: Vec<bool>,
    hang_par: Vec<[i64; 4]>,
    hang_size: Vec<i64>,
    face_cell: Vec<i64>,
    face_key: Vec<i64>,
    fptr: Vec<i64>,
    fverts: Vec<i64>,
    cell_fptr: Vec<i64>,
    corner_vid: Vec<[i64; 8]>,
}

fn build(ijk: &[I3], lev: &[u32], n: I3, lmax: u32, periodic: [bool; 3]) -> Res {
    let nl = lev.len();
    // ---------------- leaf lookup per level
    let tree = tree_of(ijk, lev, n, lmax, periodic);
    let _ = tree.lmax;

    // ---------------- vertices: sorted distinct corner keys
    let size = |i: usize| 1i64 << (lmax - lev[i]);
    let base = |i: usize| {
        let s = size(i);
        [ijk[i][0] * s, ijk[i][1] * s, ijk[i][2] * s]
    };
    let nfine = tree.ncell(lmax);
    let nvx = [nfine[0] + 1, nfine[1] + 1, nfine[2] + 1];
    let mut vt = Verts { nv: nvx, nfine, periodic, ukeys: Vec::new() };
    let ck: Vec<i64> = (0..nl)
        .into_par_iter()
        .flat_map_iter(|i| {
            let b = base(i);
            let s = size(i);
            let vt = &vt;
            CHILD.iter().map(move |c| vt.vkey([b[0] + c[0] * s, b[1] + c[1] * s, b[2] + c[2] * s]))
        })
        .collect();
    vt.ukeys = par_unique(ck);
    let vt = vt;
    let nv = vt.ukeys.len();
    let (n1, n2) = (nvx[1], nvx[2]);
    let lat: Vec<I3> = vt.ukeys.par_iter().map(|&k| [k / (n1 * n2), (k / n2) % n1, k % n2]).collect();
    let corner_vid: Vec<[i64; 8]> = (0..nl)
        .into_par_iter()
        .map(|i| {
            let b = base(i);
            let s = size(i);
            let mut o = [0i64; 8];
            for (j, c) in CHILD.iter().enumerate() {
                o[j] = vt.vid([b[0] + c[0] * s, b[1] + c[1] * s, b[2] + c[2] * s]);
            }
            o
        })
        .collect();

    // ---------------- hanging vertices (edge midpoints / face centres of a coarser leaf)
    // Candidates of one coarse size agree on the parents; the largest size wins.
    let mut dirs: Vec<(I3, Vec<usize>)> = Vec::new();
    for d0 in 0..3 {
        for d1 in 0..3 {
            for d2 in 0..3 {
                let d = [d0, d1, d2];
                let free: Vec<usize> = (0..3).filter(|&a| d[a] == 1).collect();
                if free.len() == 1 || free.len() == 2 {
                    dirs.push((d, free));
                }
            }
        }
    }
    let cands: Vec<(i64, i64, [i64; 4])> = (0..nl)
        .into_par_iter()
        .filter(|&i| size(i) >= 2)
        .flat_map_iter(|i| {
            let hb = base(i);
            let hs = size(i);
            let h2 = hs / 2;
            let vt = &vt;
            dirs.iter().filter_map(move |(d, free)| {
                let hv = vt.vid([hb[0] + d[0] * h2, hb[1] + d[1] * h2, hb[2] + d[2] * h2]);
                if hv < 0 {
                    return None;
                }
                let corner = |ch: &[i64]| {
                    let mut dd = *d;
                    for (k, &a) in free.iter().enumerate() {
                        dd[a] = ch[k];
                    }
                    vt.vid([hb[0] + dd[0] * h2, hb[1] + dd[1] * h2, hb[2] + dd[2] * h2])
                };
                let p = if free.len() == 1 {
                    let p0 = corner(&[0]);
                    [p0, corner(&[2]), p0, p0]
                } else {
                    [corner(&[0, 0]), corner(&[0, 2]), corner(&[2, 2]), corner(&[2, 0])]
                };
                Some((hv, hs, p))
            })
        })
        .collect();
    let mut hanging = vec![false; nv];
    let mut hang_par = vec![[-1i64; 4]; nv];
    let mut hang_size = vec![0i64; nv];
    for (hv, hs, p) in cands {
        let v = hv as usize;
        hanging[v] = true;
        if hs > hang_size[v] {
            hang_size[v] = hs;
            hang_par[v] = p;
        }
    }

    // ---------------- faces: per leaf and hex face, the own square or 4 sub-squares
    let finer: Vec<u8> = (0..nl)
        .into_par_iter()
        .map(|i| {
            let mut m = 0u8;
            for (f, (axis, side, _)) in HEX_FACES.iter().enumerate() {
                let mut q = ijk[i];
                q[*axis] += if *side { 1 } else { -1 };
                let (inb, cov) = tree.covering(lev[i], q);
                if inb && cov < 0 {
                    m |= 1 << f;
                }
            }
            m
        })
        .collect();
    let mut cell_fptr = Vec::with_capacity(nl + 1);
    cell_fptr.push(0i64);
    for i in 0..nl {
        let k = 6 + 3 * finer[i].count_ones() as i64;
        cell_fptr.push(cell_fptr[i] + k);
    }
    let nf = cell_fptr[nl] as usize;
    // squares: (lattice corners, axis)
    let squares = |i: usize, out: &mut Vec<([I3; 4], usize)>| {
        let b = base(i);
        let s = size(i);
        for (f, (axis, _, loc)) in HEX_FACES.iter().enumerate() {
            let mut p = [[0i64; 3]; 4];
            for k in 0..4 {
                p[k] = [b[0] + loc[k][0] * s, b[1] + loc[k][1] * s, b[2] + loc[k][2] * s];
            }
            if finer[i] & (1 << f) == 0 {
                out.push((p, *axis));
            } else {
                let m01 = half(p[0], p[1]);
                let m12 = half(p[1], p[2]);
                let m23 = half(p[2], p[3]);
                let m30 = half(p[3], p[0]);
                let c = half(p[0], p[2]);
                out.push(([p[0], m01, c, m30], *axis));
                out.push(([m01, p[1], m12, c], *axis));
                out.push(([c, m12, p[2], m23], *axis));
                out.push(([m30, c, m23, p[3]], *axis));
            }
        }
    };
    let span = (1i64 << lmax) + 1;
    // identity key and polygon (corners interleaved with existing mid-edge vertices) of
    // each square, for chunks of cells in parallel
    let chunks: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = (0..nl)
        .collect::<Vec<usize>>()
        .par_chunks(16384)
        .map(|cells| {
            let mut idk = Vec::new();
            let mut fsz = Vec::new();
            let mut fv = Vec::new();
            let mut sq = Vec::with_capacity(18);
            for &i in cells {
                sq.clear();
                squares(i, &mut sq);
                for (q, axis) in &sq {
                    let mut mn = q[0];
                    let mut mx = q[0];
                    for p in &q[1..] {
                        for a in 0..3 {
                            mn[a] = mn[a].min(p[a]);
                            mx[a] = mx[a].max(p[a]);
                        }
                    }
                    let sz = (0..3).map(|a| mx[a] - mn[a]).max().unwrap();
                    idk.push(
                        (vt.vkey(mn).wrapping_mul(3).wrapping_add(*axis as i64))
                            .wrapping_mul(span)
                            .wrapping_add(sz),
                    );
                    let n0 = fv.len();
                    for k in 0..4 {
                        fv.push(vt.vid(q[k]));
                        let nx = q[(k + 1) % 4];
                        let el: i64 = (0..3).map(|a| (nx[a] - q[k][a]).abs()).sum();
                        if el >= 2 {
                            let m = vt.vid(half(q[k], nx));
                            if m >= 0 {
                                fv.push(m);
                            }
                        }
                    }
                    fsz.push((fv.len() - n0) as i64);
                }
            }
            (idk, fsz, fv)
        })
        .collect();
    let mut idk: Vec<i64> = Vec::with_capacity(nf);
    let mut fptr = Vec::with_capacity(nf + 1);
    fptr.push(0i64);
    let nfv: usize = chunks.iter().map(|c| c.2.len()).sum();
    let mut fverts = Vec::with_capacity(nfv);
    for (k, s, v) in chunks {
        idk.extend_from_slice(&k);
        for x in s {
            fptr.push(fptr.last().unwrap() + x);
        }
        fverts.extend_from_slice(&v);
    }
    let ukey = par_unique(idk.clone());
    let face_key: Vec<i64> = idk.par_iter().map(|k| ukey.binary_search(k).unwrap() as i64).collect();
    drop(idk);
    drop(ukey);
    let face_cell: Vec<i64> = (0..nl)
        .into_par_iter()
        .flat_map_iter(|i| std::iter::repeat(i as i64).take((cell_fptr[i + 1] - cell_fptr[i]) as usize))
        .collect();
    Res {
        ukeys: vt.ukeys,
        lat,
        hanging,
        hang_par,
        hang_size,
        face_cell,
        face_key,
        fptr,
        fverts,
        cell_fptr,
        corner_vid,
    }
}

// ------------------------------------------------------------------ 2:1 balance
fn off26() -> Vec<I3> {
    let mut v = Vec::with_capacity(26);
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                if (a, b, c) != (0, 0, 0) {
                    v.push([a, b, c]);
                }
            }
        }
    }
    v
}

/// the cells of the tree that are split (internal nodes), by level: sorted keys.  A
/// cell inside the box is covered by a leaf of its level or coarser, or it is one of these.
fn inner_cells(tree: &Tree) -> Vec<Vec<i64>> {
    let lmax = tree.lmax as usize;
    let mut inner: Vec<Vec<i64>> = vec![Vec::new(); lmax + 1];
    for l in (1..=lmax).rev() {
        let nn = tree.ncell(l as u32);
        let np = tree.ncell(l as u32 - 1);
        let parent = |k: i64| -> i64 {
            let z = k.rem_euclid(nn[2]);
            let y = (k / nn[2]).rem_euclid(nn[1]);
            let x = k / (nn[2] * nn[1]);
            ((x >> 1) * np[1] + (y >> 1)) * np[2] + (z >> 1)
        };
        let leaves = &tree.keys[tree.start[l]..tree.start[l + 1]];
        let mut pk: Vec<i64> = leaves.par_iter().map(|&k| parent(k)).chain(inner[l].par_iter().map(|&k| parent(k))).collect();
        pk.par_sort_unstable();
        pk.dedup();
        inner[l - 1] = pk;
    }
    inner
}

fn tree_of(ijk: &[I3], lev: &[u32], n: I3, lmax: u32, periodic: [bool; 3]) -> Tree {
    let t0 = Tree { n, lmax, periodic, keys: Vec::new(), idx: Vec::new(), start: Vec::new() };
    // one parallel sort by (level, key) (keys are unique within a level)
    let mut kv: Vec<(u32, i64, i64)> =
        (0..lev.len()).into_par_iter().map(|i| (lev[i], t0.key(lev[i], ijk[i]), i as i64)).collect();
    kv.par_sort_unstable();
    let start: Vec<usize> = (0..=lmax + 1).map(|l| kv.partition_point(|e| e.0 < l)).collect();
    let keys: Vec<i64> = kv.par_iter().map(|e| e.1).collect();
    let idx: Vec<i64> = kv.par_iter().map(|e| e.2).collect();
    Tree { keys, idx, start, ..t0 }
}

/// the rows of an (n, 3) array: borrowed when C-contiguous, else copied row by row
fn rows_i3<'a>(a: &'a PyReadonlyArray2<i64>) -> PyResult<std::borrow::Cow<'a, [I3]>> {
    let v = a.as_array();
    if v.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("ijk must be (n, 3)"));
    }
    if numpy::PyUntypedArrayMethods::is_c_contiguous(&**a) {
        return Ok(std::borrow::Cow::Borrowed(a.as_slice()?.as_chunks::<3>().0));
    }
    Ok(std::borrow::Cow::Owned(v.outer_iter().map(|r| [r[0], r[1], r[2]]).collect()))
}

/// the leaves' (ijk, level) of the (n, 3) and (n,) arrays (validated)
fn leaves_in(ijk: &PyReadonlyArray2<i64>, lev: &PyReadonlyArray1<i64>, lmax: u32) -> PyResult<(Vec<I3>, Vec<u32>)> {
    let a = ijk.as_array();
    let lv = lev.as_array();
    if a.nrows() != lv.len() || a.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("ijk must be (n, 3) like lev"));
    }
    let q: Vec<I3> = rows_i3(ijk)?.into_owned();
    let l: Vec<u32> = match lev.as_slice() {
        Ok(v) => v.par_iter().map(|&x| x as u32).collect(),
        Err(_) => lv.iter().map(|&x| x as u32).collect(),
    };
    if l.par_iter().any(|&x| x > lmax) {
        // (a negative level wraps to a large one)
        return Err(pyo3::exceptions::PyValueError::new_err("leaf level above max level"));
    }
    Ok((q, l))
}

const CHUNK: usize = 1 << 15;

/// the slice cut into consecutive pieces of the given lengths
fn split_by<T>(mut s: &mut [T], sizes: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    for k in sizes {
        let (a, b) = std::mem::take(&mut s).split_at_mut(k);
        out.push(a);
        s = b;
    }
    out
}

/// running sums [0, c0, c0 + c1, ...]
fn offsets(c: &[usize]) -> Vec<usize> {
    let mut o = Vec::with_capacity(c.len() + 1);
    o.push(0);
    for &x in c {
        o.push(o.last().unwrap() + x);
    }
    o
}

/// the leaves i with keep(i), in order, as (flat ijk, level) arrays: written in
/// parallel at offsets counted per chunk (no unordered collect to join)
fn compact(n: usize, keep: &(dyn Fn(usize) -> bool + Sync), q: &(dyn Fn(usize) -> I3 + Sync),
           l: &(dyn Fn(usize) -> i64 + Sync)) -> (Vec<i64>, Vec<i64>) {
    let nch = n.div_ceil(CHUNK);
    let cnt: Vec<usize> = (0..nch).into_par_iter().map(|c| (c * CHUNK..((c + 1) * CHUNK).min(n)).filter(|&i| keep(i)).count()).collect();
    let off = offsets(&cnt);
    let tot = off[nch];
    let mut flat = vec![0i64; 3 * tot];
    let mut lv = vec![0i64; tot];
    let fs = split_by(&mut flat, cnt.iter().map(|&k| 3 * k));
    let ls = split_by(&mut lv, cnt.iter().copied());
    fs.into_par_iter().zip(ls).enumerate().for_each(|(c, (f, lo))| {
        let mut k = 0;
        for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
            if keep(i) {
                f[3 * k..3 * k + 3].copy_from_slice(&q(i));
                lo[k] = l(i);
                k += 1;
            }
        }
    });
    (flat, lv)
}

/// octree_centers(ijk[n, 3], lev[n], idx or None, x0[3], h0) -> centres[k, 3] of the
/// leaves idx (all if None): x0 + (ijk + 0.5) * (h0 / 2^lev), as Octree.centers
#[pyfunction]
#[pyo3(signature = (ijk, lev, idx, x0, h0))]
pub fn octree_centers<'py>(
    py: Python<'py>,
    ijk: PyReadonlyArray2<'py, i64>,
    lev: PyReadonlyArray1<'py, i64>,
    idx: Option<PyReadonlyArray1<'py, i64>>,
    x0: [f64; 3],
    h0: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (q, lv) = (rows_i3(&ijk)?, lev.as_slice()?);
    if q.len() != lv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("ijk must be (n, 3) like lev"));
    }
    let q: &[I3] = &q;
    let ids: Option<&[i64]> = match &idx {
        Some(a) => Some(a.as_slice()?),
        None => None,
    };
    if let Some(ids) = ids {
        if ids.par_iter().any(|&i| i < 0 || i as usize >= lv.len()) {
            return Err(pyo3::exceptions::PyValueError::new_err("leaf index out of range"));
        }
    }
    let k = ids.map_or(lv.len(), |v| v.len());
    let mut out = vec![0.0f64; 3 * k];
    py.detach(|| {
        out.par_chunks_mut(3).enumerate().for_each(|(r, o)| {
            let i = ids.map_or(r, |v| v[r] as usize);
            // h0 / 2.0 ** lev (a power of two: exact), then x0 + (ijk + 0.5) * hh
            let hh = h0 / 2f64.powi(lv[i] as i32);
            for a in 0..3 {
                o[a] = x0[a] + (q[i][a] as f64 + 0.5) * hh;
            }
        })
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((k, 3), out).unwrap().into_pyarray(py))
}

/// octree_split(ijk[n, 3], lev[n], mask[n]) -> (ijk, lev): Octree.split's arrays, the
/// unsplit leaves in order, then the eight children of each split leaf (CHILD order)
#[pyfunction]
pub fn octree_split<'py>(
    py: Python<'py>,
    ijk: PyReadonlyArray2<'py, i64>,
    lev: PyReadonlyArray1<'py, i64>,
    mask: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>)> {
    let (q, lv, m) = (rows_i3(&ijk)?, lev.as_slice()?, mask.as_slice()?);
    if q.len() != lv.len() || m.len() != lv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("ijk must be (n, 3) like lev and mask"));
    }
    let q: &[I3] = &q;
    let (flat, l) = py.detach(|| {
        let n = lv.len();
        let (mut flat, mut l) = compact(n, &|i| !m[i], &|i| q[i], &|i| lv[i]);
        // the children, eight per split leaf in leaf order (few: sequential)
        for i in (0..n).filter(|&i| m[i]) {
            let p = q[i];
            for c in CHILD.iter() {
                flat.extend_from_slice(&[2 * p[0] + c[0], 2 * p[1] + c[1], 2 * p[2] + c[2]]);
                l.push(lv[i] + 1);
            }
        }
        (flat, l)
    });
    let nl = l.len();
    let arr = numpy::ndarray::Array2::from_shape_vec((nl, 3), flat).unwrap();
    Ok((arr.into_pyarray(py), l.into_pyarray(py)))
}

/// balance(ijk[n, 3], lev[n], n[3], lmax, periodic[3], new_from, prefix)
///   -> (ijk, lev, splits per pass, survivors)
///
/// Passes of Octree.balance until no leaf needs splitting.  The leaf lookup is built
/// once: a split leaf is marked dead and its children are added to a table, and the
/// leaves come out as the passes of split() leave them (the unsplit leaves in order,
/// then each pass's children), which is all leaves ever made, in the order made, that
/// are still leaves.
///
/// new_from: the tree was balanced before its leaves from new_from on were added by a
/// split.  A leaf that was there before only gets finer neighbours, so only the new
/// leaves (and then the children of each pass) can need a coarse neighbour split: the
/// same splits as checking every leaf, without the scan of the whole tree.
/// prefix: the first `prefix` leaves are followed; survivors: those left unsplit (they
/// come first, in order).
#[pyfunction]
#[pyo3(signature = (ijk, lev, n, lmax, periodic, new_from=None, prefix=0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn balance<'py>(
    py: Python<'py>,
    ijk: PyReadonlyArray2<'py, i64>,
    lev: PyReadonlyArray1<'py, i64>,
    n: [i64; 3],
    lmax: u32,
    periodic: [bool; 3],
    new_from: Option<usize>,
    prefix: usize,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>, Vec<usize>, Bound<'py, PyArray1<i64>>)> {
    let (mut ijk, mut lev) = leaves_in(&ijk, &lev, lmax)?;
    let n0 = lev.len();
    let (passes, alive) = py.detach(|| {
        let tree = tree_of(&ijk, &lev, n, lmax, periodic);
        let inner = inner_cells(&tree);
        let mut added: Vec<crate::cut::FxMap<i64, i64>> = vec![crate::cut::FxMap::default(); lmax as usize + 1];
        let mut alive = vec![true; n0];
        let off = off26();
        let mut passes = Vec::new();
        let mut from = new_from.unwrap_or(0);
        // every leaf checked (new_from None): the (level, parent) of the alive leaves of
        // level >= 2, with the number of alive leaves of each, kept up to date across the
        // passes instead of collected from all the leaves again
        let mut ent: Vec<(u32, I3)> = Vec::new();
        let mut cnt: Vec<u32> = Vec::new();
        let mut extra: crate::cut::FxMap<(u32, I3), u32> = crate::cut::FxMap::default();
        if new_from.is_none() {
            let mut all: Vec<(u32, I3)> = (0..lev.len())
                .into_par_iter()
                .filter(|&i| lev[i] >= 2)
                .map(|i| (lev[i], [ijk[i][0] >> 1, ijk[i][1] >> 1, ijk[i][2] >> 1]))
                .collect();
            all.par_sort_unstable();
            for e in all {
                if ent.last() == Some(&e) {
                    *cnt.last_mut().unwrap() += 1;
                } else {
                    ent.push(e);
                    cnt.push(1);
                }
            }
        }
        loop {
            let bad: Vec<usize> = {
                let (ijk, lev, alive, added, tree, inner) = (&ijk, &lev, &alive, &added, &tree, &inner);
                // the leaf of level <= l covering the level-l cell q (Tree::covering), -1
                // where the cell is split (finer leaves).  The search stops at the first
                // leaf found, alive or split in a pass (then there is no coverer: the
                // levels above were looked at first), and at a cell split in the tree.
                let leaf_at = |c: u32, q: I3| -> Option<i64> {
                    let r = tree.find_exact(c, q);
                    if r >= 0 {
                        return Some(if alive[r as usize] { r } else { -1 });
                    }
                    added[c as usize].get(&tree.key(c, q)).map(|&r| if alive[r as usize] { r } else { -1 })
                };
                let covering = |l: u32, q: I3| -> i64 {
                    let q = tree.wrap(l, q);
                    let nn = tree.ncell(l);
                    if (0..3).any(|k| q[k] < 0 || q[k] >= nn[k]) {
                        return -1;
                    }
                    if let Some(r) = leaf_at(l, q) {
                        return r;
                    }
                    if inner[l as usize].binary_search(&tree.key(l, q)).is_ok() {
                        return -1;
                    }
                    for c in (0..l).rev() {
                        let s = l - c;
                        if let Some(r) = leaf_at(c, [q[0] >> s, q[1] >> s, q[2] >> s]) {
                            return r;
                        }
                    }
                    -1
                };
                // the test depends on the level and the parent only: once per parent
                // (siblings ask the same)
                let par: Vec<(u32, I3)> = if new_from.is_none() {
                    let mut v: Vec<(u32, I3)> = ent.iter().zip(&cnt).filter(|e| *e.1 > 0).map(|e| *e.0).collect();
                    v.extend(extra.iter().filter(|e| *e.1 > 0).map(|e| *e.0));
                    v
                } else {
                    let mut v: Vec<(u32, I3)> = (from.min(lev.len())..lev.len())
                        .into_par_iter()
                        .filter(|&i| alive[i] && lev[i] >= 2)
                        .map(|i| (lev[i], [ijk[i][0] >> 1, ijk[i][1] >> 1, ijk[i][2] >> 1]))
                        .collect();
                    v.par_sort_unstable();
                    v.dedup();
                    v
                };
                par.par_iter()
                    .flat_map_iter(|&(m, p)| {
                        let covering = &covering;
                        off.iter().filter_map(move |d| {
                            let r = covering(m - 1, [p[0] + d[0], p[1] + d[1], p[2] + d[2]]);
                            if r >= 0 && lev[r as usize] < m - 1 { Some(r as usize) } else { None }
                        })
                    })
                    .collect()
            };
            let mut marked = bad;
            marked.sort_unstable();
            marked.dedup();
            passes.push(marked.len());
            if marked.is_empty() {
                break;
            }
            if new_from.is_some() {
                from = lev.len(); // the children of this pass come next
            }
            for &r in &marked {
                alive[r] = false;
                if new_from.is_none() {
                    // r leaves its parent's count; its children make one of their own
                    let l = lev[r];
                    if l >= 2 {
                        let e = (l, [ijk[r][0] >> 1, ijk[r][1] >> 1, ijk[r][2] >> 1]);
                        match extra.get_mut(&e) {
                            Some(c) => *c -= 1,
                            None => {
                                let k = ent.binary_search(&e).expect("parent of an alive leaf");
                                cnt[k] -= 1;
                            }
                        }
                    }
                    if l >= 1 {
                        *extra.entry((l + 1, ijk[r])).or_insert(0) += 8;
                    }
                }
                for c in CHILD.iter() {
                    let q = [2 * ijk[r][0] + c[0], 2 * ijk[r][1] + c[1], 2 * ijk[r][2] + c[2]];
                    let l = lev[r] + 1;
                    added[l as usize].insert(tree.key(l, q), ijk.len() as i64);
                    ijk.push(q);
                    lev.push(l);
                    alive.push(true);
                }
            }
        }
        (passes, alive)
    });
    // the leaves still alive, in order (in parallel)
    let (survivors, flat, lv) = py.detach(|| {
        let (sv, _) = compact(prefix.min(n0), &|i| alive[i], &|i| [i as i64, 0, 0], &|_| 0);
        let survivors: Vec<i64> = sv.par_chunks(3).map(|c| c[0]).collect();
        let (flat, lv) = compact(ijk.len(), &|i| alive[i], &|i| ijk[i], &|i| lev[i] as i64);
        (survivors, flat, lv)
    });
    let nl = lv.len();
    let arr = numpy::ndarray::Array2::from_shape_vec((nl, 3), flat).unwrap();
    Ok((arr.into_pyarray(py), lv.into_pyarray(py), passes, survivors.into_pyarray(py)))
}

// ------------------------------------------------------------------ leaf validity
/// leaf_bad(points, leaves, fptr, fverts, cptr, csize, flat_at or None, sin(flat angle))
/// -> bool per leaf, as mesher._leaf_bad: some face does not clearly see the cell centre
/// (non-convex or inverted), or a flat_at vertex lost its face corner
#[pyfunction]
#[pyo3(signature = (points, leaves, fptr, fverts, cptr, csize, flat_at, flat_sin))]
pub fn leaf_bad<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    leaves: PyReadonlyArray1<'py, i64>,
    fptr: PyReadonlyArray1<'py, i64>,
    fverts: PyReadonlyArray1<'py, i64>,
    cptr: PyReadonlyArray1<'py, i64>,
    csize: PyReadonlyArray1<'py, f64>,
    flat_at: Option<PyReadonlyArray1<'py, bool>>,
    flat_sin: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let pts = crate::points_ref(&points)?;
    let lv = leaves.as_slice()?;
    let fp = fptr.as_slice()?;
    let fv = fverts.as_slice()?;
    let cp = cptr.as_slice()?;
    let cs = csize.as_slice()?;
    let fa = match &flat_at {
        Some(a) => Some(a.as_slice()?),
        None => None,
    };
    let nc = cp.len().saturating_sub(1);
    if lv.iter().any(|&c| c < 0 || c as usize >= nc)
        || cs.len() < nc
        || *cp.last().unwrap_or(&0) as usize >= fp.len()
        || *fp.last().unwrap_or(&0) as usize > fv.len()
        || fv.par_iter().any(|&v| v < 0 || v as usize >= pts.len())
        || fa.map_or(false, |a| a.len() < pts.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent leaf data"));
    }
    let res: Vec<bool> = py.detach(|| {
        lv.par_iter()
            .map(|&c| {
                let c = c as usize;
                let (f0, f1) = (cp[c] as usize, cp[c + 1] as usize);
                let mut idx: Vec<usize> = Vec::new();
                let mut cfs: Vec<V3> = Vec::with_capacity(f1 - f0);
                let mut sfs: Vec<V3> = Vec::with_capacity(f1 - f0);
                let mut est = [0.0f64; 3];
                for f in f0..f1 {
                    idx.clear();
                    idx.extend(fv[fp[f] as usize..fp[f + 1] as usize].iter().map(|&v| v as usize));
                    let (cf, sf) = crate::face_cs(&pts, &idx);
                    for k in 0..3 {
                        est[k] += cf[k];
                    }
                    cfs.push(cf);
                    sfs.push(sf);
                }
                let nf = (f1 - f0) as f64;
                for k in 0..3 {
                    est[k] /= nf;
                }
                let mut vv = 0.0f64;
                let mut cc = [0.0f64; 3];
                for j in 0..cfs.len() {
                    let pv = einsum3(sfs[j], sub(cfs[j], est));
                    vv += pv;
                    for k in 0..3 {
                        cc[k] += pv * (0.75 * cfs[j][k] + 0.25 * est[k]);
                    }
                }
                if vv <= 0.0 {
                    return true;
                }
                for k in 0..3 {
                    cc[k] /= vv;
                }
                let hl = cs[c];
                for (j, f) in (f0..f1).enumerate() {
                    let sp = einsum3(sfs[j], sub(cfs[j], cc));
                    let af = dot(sfs[j], sfs[j]).sqrt();
                    if sp < 0.05 * af * hl || af < 0.05 * hl * hl {
                        return true;
                    }
                    if let Some(fa) = fa {
                        let vs = &fv[fp[f] as usize..fp[f + 1] as usize];
                        let m = vs.len();
                        let am = af.max(1e-300);
                        let nh = [sfs[j][0] / am, sfs[j][1] / am, sfs[j][2] / am];
                        for i in 0..m {
                            let v = vs[i] as usize;
                            if !fa[v] {
                                continue;
                            }
                            let q = pts[v];
                            let e_in = sub(q, pts[vs[(i + m - 1) % m] as usize]);
                            let e_out = sub(pts[vs[(i + 1) % m] as usize], q);
                            let s = einsum3(cross3(e_in, e_out), nh);
                            let lin = dot(e_in, e_in).sqrt() * dot(e_out, e_out).sqrt();
                            if s < flat_sin * lin {
                                return true;
                            }
                        }
                    }
                }
                false
            })
            .collect()
    });
    Ok(res.into_pyarray(py))
}

/// np.einsum("ij,ij->i", a, b) for one row: NumPy adds the products as (0 + 2) + 1
pub(crate) fn einsum3(a: V3, b: V3) -> f64 {
    (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]
}

fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// finer_face_neighbour(ijk[n, 3], lev[n], n[3], lmax, cand) -> per candidate leaf:
/// one of its six face neighbours at its own level is split into finer leaves.  In a
/// balanced tree that is "a finer leaf centre within 0.95 cell sizes" (the face-touching
/// children are 0.83 h away; any other finer leaf is at least 1.06 h away).  No
/// periodic wrap, as the distance test has none.
#[pyfunction]
pub fn finer_face_neighbour<'py>(
    py: Python<'py>,
    ijk: PyReadonlyArray2<'py, i64>,
    lev: PyReadonlyArray1<'py, i64>,
    n: [i64; 3],
    lmax: u32,
    cand: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let cand = cand.as_slice()?;
    let (ijk, lev) = leaves_in(&ijk, &lev, lmax)?;
    if cand.iter().any(|&c| c < 0 || c as usize >= lev.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("inconsistent leaf arrays"));
    }
    let r: Vec<bool> = py.detach(|| {
        let tree = tree_of(&ijk, &lev, n, lmax, [false; 3]);
        cand.par_iter()
            .map(|&c| {
                let (q, l) = (ijk[c as usize], lev[c as usize]);
                (0..3).any(|ax| {
                    [-1i64, 1].iter().any(|&s| {
                        let mut p = q;
                        p[ax] += s;
                        let (inside, r) = tree.covering(l, p);
                        inside && r < 0
                    })
                })
            })
            .collect()
    });
    Ok(r.into_pyarray(py))
}
