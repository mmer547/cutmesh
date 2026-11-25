//! geometry.Cylinder's signed distance and closest points, in parallel, with NumPy's
//! arithmetic: t = d @ a as the (m,3) @ (3,) product (gemv order for m >= 2, the 1-D dot
//! for one point), elementwise products and differences rounded one by one, row norms as
//! sqrt((x0^2 + x1^2) + x2^2), np.hypot as the C library's hypot.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::wallmerge::{dot1, gemv_row};
use crate::V3;

/// the cylinder: base point p1, unit axis a, a unit vector e1 normal to it, radius, height,
/// diagonal L (Cylinder.p1, .a, .e1, .R, .H, .L)
#[derive(Clone, Copy)]
struct Cyl {
    p1: V3,
    a: V3,
    e1: V3,
    r: f64,
    h: f64,
    l: f64,
}

impl Cyl {
    /// Cylinder._local: (t, r, dirv)
    #[inline]
    fn local(&self, p: V3, many: bool) -> (f64, f64, V3) {
        let d = [p[0] - self.p1[0], p[1] - self.p1[1], p[2] - self.p1[2]];
        let t = if many { gemv_row(d, self.a) } else { dot1(d, self.a) };
        let rv = [d[0] - t * self.a[0], d[1] - t * self.a[1], d[2] - t * self.a[2]];
        let r = ((rv[0] * rv[0] + rv[1] * rv[1]) + rv[2] * rv[2]).sqrt();
        let dirv = if r > 1e-14 * self.l { [rv[0] / r, rv[1] / r, rv[2] / r] } else { self.e1 };
        (t, r, dirv)
    }

    fn sdf(&self, p: V3, many: bool) -> f64 {
        let (t, r, _) = self.local(p, many);
        let dr = r - self.r;
        let da = np_max(-t, t - self.h);
        np_min(np_max(dr, da), 0.0) + np_max(dr, 0.0).hypot(np_max(da, 0.0))
    }

    fn closest(&self, p: V3, many: bool) -> (V3, f64) {
        let (t, r, dv) = self.local(p, many);
        let dr = r - self.r;
        let da = np_max(-t, t - self.h);
        let outside = dr > 0.0 || da > 0.0;
        let cp = if outside {
            let tc = np_clip(t, 0.0, self.h);
            let rc = np_min(r, self.r);
            std::array::from_fn(|k| (self.p1[k] + tc * self.a[k]) + rc * dv[k])
        } else {
            // the nearest of side / bottom / top (np.argmin: the first smallest, NaN first)
            let c = [self.r - r, t, self.h - t];
            let mut kmin = 0;
            for k in 0..3 {
                if c[k].is_nan() {
                    kmin = k;
                    break;
                }
                if c[k] < c[kmin] {
                    kmin = k;
                }
            }
            match kmin {
                0 => std::array::from_fn(|k| (self.p1[k] + t * self.a[k]) + self.r * dv[k]),
                1 => std::array::from_fn(|k| p[k] - t * self.a[k]),
                _ => std::array::from_fn(|k| p[k] + (self.h - t) * self.a[k]),
            }
        };
        let x = [cp[0] - p[0], cp[1] - p[1], cp[2] - p[2]];
        (cp, ((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]).sqrt())
    }
}

/// np.maximum / np.minimum of two scalars (NaN wins)
#[inline]
fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if a >= b { a } else { b }
}
#[inline]
fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if a <= b { a } else { b }
}
#[inline]
fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    np_min(np_max(x, lo), hi)
}

fn cyl(p1: V3, a: V3, e1: V3, r: f64, h: f64, l: f64) -> Cyl {
    Cyl { p1, a, e1, r, h, l }
}

/// cylinder_sdf(points[n, 3], p1, a, e1, R, H, L) -> Cylinder.sdf
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cylinder_sdf<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    p1: V3,
    a: V3,
    e1: V3,
    radius: f64,
    height: f64,
    diag: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = crate::points_ref(&points)?;
    let c = cyl(p1, a, e1, radius, height, diag);
    let many = p.len() >= 2;
    let r: Vec<f64> = py.detach(|| p.par_iter().map(|&q| c.sdf(q, many)).collect());
    Ok(numpy::IntoPyArray::into_pyarray(r, py))
}

/// cylinder_closest(points[n, 3], p1, a, e1, R, H, L) -> (closest points, distances) of
/// Cylinder.closest
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cylinder_closest<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    p1: V3,
    a: V3,
    e1: V3,
    radius: f64,
    height: f64,
    diag: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let p = crate::points_ref(&points)?;
    let c = cyl(p1, a, e1, radius, height, diag);
    let many = p.len() >= 2;
    let r: Vec<(V3, f64)> = py.detach(|| p.par_iter().map(|&q| c.closest(q, many)).collect());
    let d: Vec<f64> = r.iter().map(|x| x.1).collect();
    let cp: Vec<V3> = r.into_iter().map(|x| x.0).collect();
    Ok((crate::vec3_array(py, cp)?, numpy::IntoPyArray::into_pyarray(d, py)))
}
