//! Geometry and directed-edge checks on the final mesh. Faces are independent;
//! for cell checks, visit each cell's owners first, then neighbours, in face order.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use crate::{V3, cross, dot, octree::einsum3, sub};

fn rows<'a>(a: &'a PyReadonlyArray2<'_, f64>) -> PyResult<&'a [V3]> {
    if a.shape()[1] != 3 || !a.is_c_contiguous() {
        return Err(PyValueError::new_err("expected a contiguous (n, 3) array"));
    }
    Ok(a.as_slice()?.as_chunks::<3>().0)
}

fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

// np.maximum propagates NaN, including when a degenerate face yields 0/0.
fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

struct Faces<'a> {
    ptr: &'a [i64],
    verts: &'a [i64],
}

impl Faces<'_> {
    fn edges(&self, f: usize) -> impl Iterator<Item = (i64, i64)> + '_ {
        let v = &self.verts[self.ptr[f] as usize..self.ptr[f + 1] as usize];
        v.iter().copied().zip(v[1..].iter().chain(&v[..1]).copied())
    }
}

// One pass over the face vertices gives both the skewness extent and the worst
// triangle pyramid on each side. Divide by three before testing its sign as in NumPy.
fn face_checks(
    points: &[V3],
    faces: &Faces,
    f: usize,
    cf: V3,
    sf: V3,
    area: f64,
    owner: V3,
    neighbour: Option<V3>,
) -> (f64, usize, usize) {
    let cpf = sub(cf, owner);
    let d = match neighbour {
        Some(c) => sub(c, owner),
        None => {
            let normal = sf.map(|x| x / maximum(area, 1e-300));
            let projection = einsum3(normal, cpf);
            normal.map(|x| x * projection)
        }
    };
    let ratio = einsum3(sf, cpf) / (einsum3(sf, d) + 1e-300);
    let sv = sub(cpf, d.map(|x| ratio * x));
    let msv = norm(sv);
    let direction = sv.map(|x| x / (msv + 1e-300));
    let mut extent = (if neighbour.is_some() { 0.2 } else { 0.4 }) * norm(d);
    let from_neighbour = neighbour.map(|c| sub(cf, c));
    let mut worst_own = f64::NEG_INFINITY;
    let mut worst_nei = f64::NEG_INFINITY;
    for (a, b) in faces.edges(f) {
        let pa = points[a as usize];
        let pb = points[b as usize];
        extent = maximum(extent, einsum3(direction, sub(pa, cf)).abs());
        let tri = cross(sub(pb, pa), sub(cf, pa)).map(|x| x * 0.5);
        worst_own = maximum(worst_own, -einsum3(tri, cpf) / 3.0);
        if let Some(delta) = from_neighbour {
            worst_nei = maximum(worst_nei, einsum3(tri, delta) / 3.0);
        }
    }
    let bad_owner = einsum3(sf, cpf) / 3.0 <= 0.0;
    let bad_nei = neighbour.is_some_and(|c| einsum3(sf, sub(c, cf)) / 3.0 <= 0.0);
    (
        msv / (extent + 1e-300),
        usize::from(bad_owner) + usize::from(bad_nei),
        usize::from(worst_own >= 0.0) + usize::from(worst_nei >= 0.0),
    )
}

// Build cell -> incident faces without a Vec allocation per cell. The low bit
// identifies the neighbour side, even if a malformed face has owner == neighbour.
fn cell_faces(n_cells: usize, owner: &[i64], neighbour: &[i64]) -> (Vec<usize>, Vec<usize>) {
    let mut ptr = vec![0; n_cells + 1];
    for &c in owner.iter().chain(neighbour) {
        ptr[c as usize + 1] += 1;
    }
    for c in 0..n_cells {
        ptr[c + 1] += ptr[c];
    }
    let mut next = ptr[..n_cells].to_vec();
    let mut incident = vec![0; ptr[n_cells]];
    for (side, cells) in [owner, neighbour].iter().enumerate() {
        for (f, &c) in cells.iter().enumerate() {
            let slot = &mut next[c as usize];
            incident[*slot] = 2 * f + side;
            *slot += 1;
        }
    }
    (ptr, incident)
}

fn cell_checks(
    faces: &Faces,
    incident: &[usize],
    sf: &[V3],
    area: &[f64],
    edges: &mut Vec<(i64, i64, bool)>,
) -> (f64, usize, usize) {
    let mut sum = [0.0; 3];
    let mut area_sum = [0.0; 2]; // NumPy adds the two bincounts after summing each side.
    edges.clear();
    for &entry in incident {
        let (f, side) = (entry / 2, entry % 2);
        area_sum[side] += area[f];
        for (s, x) in sum.iter_mut().zip(sf[f]) {
            *s += if side == 0 { x } else { -x };
        }
        for (a, b) in faces.edges(f) {
            let forward = if side == 0 { a < b } else { b < a };
            edges.push((a.min(b), a.max(b), forward));
        }
    }
    // Only edges in the same cell can be partners. Sort a small, reusable
    // buffer instead of numbering and sorting all the mesh edges globally.
    edges.sort_unstable();
    let (mut open, mut duplicates) = (0, 0);
    for group in edges.chunk_by(|a, b| a.0 == b.0 && a.1 == b.1) {
        let paired = group[0].2 != group[group.len() - 1].2;
        if !paired {
            open += group.len();
        }
        duplicates += group.len() - 1 - usize::from(paired);
    }
    (
        norm(sum) / maximum(area_sum[0] + area_sum[1], 1e-300),
        open,
        duplicates,
    )
}

type Metrics<'py> = (
    f64,
    usize,
    usize,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    usize,
);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn quality_metrics<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    ptr: PyReadonlyArray1<'py, i64>,
    verts: PyReadonlyArray1<'py, i64>,
    owner: PyReadonlyArray1<'py, i64>,
    neighbour: PyReadonlyArray1<'py, i64>,
    cf: PyReadonlyArray2<'py, f64>,
    sf: PyReadonlyArray2<'py, f64>,
    centres: PyReadonlyArray2<'py, f64>,
    area: PyReadonlyArray1<'py, f64>,
) -> PyResult<Metrics<'py>> {
    let points = rows(&points)?;
    let faces = Faces {
        ptr: ptr.as_slice()?,
        verts: verts.as_slice()?,
    };
    let (own, nei) = (owner.as_slice()?, neighbour.as_slice()?);
    let (cf, sf, centres, area) = (rows(&cf)?, rows(&sf)?, rows(&centres)?, area.as_slice()?);
    let nf = own.len();
    if faces.ptr.len() != nf + 1
        || faces.ptr[0] != 0
        || faces.ptr[nf] != faces.verts.len() as i64
        || faces.ptr.windows(2).any(|p| p[0] >= p[1])
        || faces.verts.iter().any(|&v| v as usize >= points.len())
    {
        return Err(PyValueError::new_err("inconsistent face vertices"));
    }
    if cf.len() != nf
        || sf.len() != nf
        || area.len() != nf
        || nei.len() > nf
        || own.iter().chain(nei).any(|&c| c as usize >= centres.len())
    {
        return Err(PyValueError::new_err("inconsistent face or cell data"));
    }

    let (ptr, incident) = cell_faces(centres.len(), own, nei);
    let (closed, open, duplicates) = ptr
        .par_windows(2)
        .map_init(Vec::new, |edges, p| {
            cell_checks(&faces, &incident[p[0]..p[1]], sf, area, edges)
        })
        .reduce(
            || (0.0, 0, 0),
            |a, b| (maximum(a.0, b.0), a.1 + b.1, a.2 + b.2),
        );
    let cosang: Vec<_> = nei
        .par_iter()
        .enumerate()
        .map(|(f, &n)| {
            let d = sub(centres[n as usize], centres[own[f] as usize]);
            einsum3(d, sf[f]) / maximum(norm(d) * area[f], 1e-300)
        })
        .collect();
    let mut skew = vec![0.0; nf];
    let (pyramids, tets) = skew
        .par_iter_mut()
        .enumerate()
        .map(|(f, out)| {
            let (sk, p, t) = face_checks(
                points,
                &faces,
                f,
                cf[f],
                sf[f],
                area[f],
                centres[own[f] as usize],
                nei.get(f).map(|&c| centres[c as usize]),
            );
            *out = sk;
            (p, t)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    Ok((
        closed,
        open,
        duplicates,
        cosang.into_pyarray(py),
        skew.into_pyarray(py),
        pyramids,
        tets,
    ))
}
