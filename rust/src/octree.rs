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
    /// per level: (sorted keys, leaf index)
    lk: Vec<(Vec<i64>, Vec<i64>)>,
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
        let (keys, idx) = &self.lk[l as usize];
        let k = self.key(l, q);
        match keys.binary_search(&k) {
            Ok(p) => idx[p],
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
    let a = ijk.as_array();
    let lv = lev.as_array();
    let nl = lv.len();
    if a.nrows() != nl || a.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("ijk must be (n, 3) like lev"));
    }
    let ijk: Vec<I3> = (0..nl).map(|i| [a[[i, 0]], a[[i, 1]], a[[i, 2]]]).collect();
    let lev: Vec<u32> = lv.iter().map(|&l| l as u32).collect();
    if lev.iter().any(|&l| l > lmax) {
        return Err(pyo3::exceptions::PyValueError::new_err("leaf level above max level"));
    }
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
    let mut lk = Vec::with_capacity(lmax as usize + 1);
    let tree0 = Tree { n, lmax, periodic, lk: Vec::new() };
    for l in 0..=lmax {
        let mut kv: Vec<(i64, i64)> = (0..nl)
            .into_par_iter()
            .filter(|&i| lev[i] == l)
            .map(|i| (tree0.key(l, ijk[i]), i as i64))
            .collect();
        kv.par_sort_unstable();
        let (k, i): (Vec<i64>, Vec<i64>) = kv.into_iter().unzip();
        lk.push((k, i));
    }
    let tree = Tree { lk, ..tree0 };
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

fn tree_of(ijk: &[I3], lev: &[u32], n: I3, lmax: u32, periodic: [bool; 3]) -> Tree {
    let t0 = Tree { n, lmax, periodic, lk: Vec::new() };
    let mut lk = Vec::with_capacity(lmax as usize + 1);
    for l in 0..=lmax {
        let mut kv: Vec<(i64, i64)> = (0..lev.len())
            .into_par_iter()
            .filter(|&i| lev[i] == l)
            .map(|i| (t0.key(l, ijk[i]), i as i64))
            .collect();
        kv.par_sort_unstable();
        lk.push(kv.into_iter().unzip());
    }
    Tree { lk, ..t0 }
}

/// leaves to split so that no leaf of level m has a neighbour (face, edge or vertex)
/// coarser than m - 1, as one pass of Octree.balance
/// (only the leaves from `from` on are checked: the others are known to be fine)
fn balance_marks(ijk: &[I3], lev: &[u32], n: I3, lmax: u32, periodic: [bool; 3], from: usize) -> Vec<bool> {
    let tree = tree_of(ijk, lev, n, lmax, periodic);
    let off = off26();
    let bad: Vec<usize> = (from.min(lev.len())..lev.len())
        .into_par_iter()
        .filter(|&i| lev[i] >= 2)
        .flat_map_iter(|i| {
            let m = lev[i];
            let p = [ijk[i][0] >> 1, ijk[i][1] >> 1, ijk[i][2] >> 1];
            let tree = &tree;
            off.iter().filter_map(move |d| {
                let (_, r) = tree.covering(m - 1, [p[0] + d[0], p[1] + d[1], p[2] + d[2]]);
                if r >= 0 && lev[r as usize] < m - 1 { Some(r as usize) } else { None }
            })
        })
        .collect();
    let mut mark = vec![false; lev.len()];
    for r in bad {
        mark[r] = true;
    }
    mark
}

/// Octree.split: the kept leaves in order, then the 8 children of each split leaf
fn split(ijk: &[I3], lev: &[u32], mark: &[bool]) -> (Vec<I3>, Vec<u32>) {
    let ns = mark.iter().filter(|&&m| m).count();
    let mut a = Vec::with_capacity(ijk.len() + 7 * ns);
    let mut l = Vec::with_capacity(ijk.len() + 7 * ns);
    for i in 0..ijk.len() {
        if !mark[i] {
            a.push(ijk[i]);
            l.push(lev[i]);
        }
    }
    for i in 0..ijk.len() {
        if mark[i] {
            for c in CHILD.iter() {
                a.push([2 * ijk[i][0] + c[0], 2 * ijk[i][1] + c[1], 2 * ijk[i][2] + c[2]]);
                l.push(lev[i] + 1);
            }
        }
    }
    (a, l)
}

/// balance(ijk[n, 3], lev[n], n[3], lmax, periodic[3], new_from) -> (ijk, lev, splits per pass)
///
/// new_from: the tree was balanced before its leaves from new_from on were added by a
/// split.  A leaf that was there before only gets finer neighbours, so only the new
/// leaves (and then the children of each pass) can need a coarse neighbour split: the
/// same splits as checking every leaf, without the scan of the whole tree.
#[pyfunction]
#[pyo3(signature = (ijk, lev, n, lmax, periodic, new_from=None, prefix=0))]
pub fn balance<'py>(
    py: Python<'py>,
    ijk: PyReadonlyArray2<'py, i64>,
    lev: PyReadonlyArray1<'py, i64>,
    n: [i64; 3],
    lmax: u32,
    periodic: [bool; 3],
    new_from: Option<usize>,
    prefix: usize,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>, Vec<usize>, Vec<i64>)> {
    let a = ijk.as_array();
    let lv = lev.as_array();
    if a.nrows() != lv.len() || a.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("ijk must be (n, 3) like lev"));
    }
    let mut ijk: Vec<I3> = (0..lv.len()).map(|i| [a[[i, 0]], a[[i, 1]], a[[i, 2]]]).collect();
    let mut lev: Vec<u32> = lv.iter().map(|&l| l as u32).collect();
    if lev.iter().any(|&l| l > lmax) {
        return Err(pyo3::exceptions::PyValueError::new_err("leaf level above max level"));
    }
    // the first `prefix` leaves are followed: which of them are left unsplit (split()
    // keeps the unsplit leaves first, in order)
    let mut origin: Vec<i64> = (0..lev.len() as i64).map(|i| if (i as usize) < prefix { i } else { -1 }).collect();
    let passes = py.detach(|| {
        let mut passes = Vec::new();
        let mut from = new_from.unwrap_or(0);
        loop {
            let mark = balance_marks(&ijk, &lev, n, lmax, periodic, from);
            let ns = mark.iter().filter(|&&m| m).count();
            passes.push(ns);
            if ns == 0 {
                break;
            }
            if new_from.is_some() {
                from = lev.len() - ns; // split() puts the children after the kept leaves
            }
            if prefix > 0 {
                let mut o2: Vec<i64> = origin.iter().zip(mark.iter()).filter(|p| !*p.1).map(|p| *p.0).collect();
                o2.resize(o2.len() + 8 * ns, -1);
                origin = o2;
            }
            let (a2, l2) = split(&ijk, &lev, &mark);
            ijk = a2;
            lev = l2;
        }
        passes
    });
    let nl = lev.len();
    let arr = numpy::ndarray::Array2::from_shape_vec((nl, 3), ijk.into_iter().flatten().collect()).unwrap();
    let lv: Vec<i64> = lev.into_iter().map(|l| l as i64).collect();
    let survivors: Vec<i64> = origin.into_iter().take_while(|&o| o >= 0).collect();
    Ok((arr.into_pyarray(py), lv.into_pyarray(py), passes, if prefix > 0 { survivors } else { Vec::new() }))
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
    let pts = crate::points(&points)?;
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
        || fv.iter().any(|&v| v < 0 || v as usize >= pts.len())
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
