//! Results of point queries kept by the exact bits of the point: a point asked again (the
//! same three f64, e.g. a corner shared by neighbouring octree cells and computed the same
//! way from either) gets the value computed before instead of a new query.  The values are
//! the query's own results, so the answers are the same as without the cache.

use pyo3::prelude::*;

use crate::cut::FxMap;
use crate::V3;
use rayon::prelude::*;

#[pyclass]
pub(crate) struct PointCache {
    pub(crate) map: FxMap<[u64; 3], u64>,
}

#[pymethods]
impl PointCache {
    #[new]
    fn new() -> Self {
        PointCache { map: FxMap::default() }
    }

    fn __len__(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map = FxMap::default();
    }
}

#[inline]
fn key(p: V3) -> [u64; 3] {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

/// f(p) for every point (as u64 bits), each distinct point not in the map evaluated once
/// (in parallel) and added to the map
pub(crate) fn cached_eval<F: Fn(V3) -> u64 + Sync>(map: &mut FxMap<[u64; 3], u64>, pts: &[V3], f: F) -> Vec<u64> {
    const NEW: u32 = u32::MAX;
    // the points known already (the map is only read here: in parallel)
    let known: Vec<Option<u64>> = {
        let m = &*map;
        pts.par_iter().map(|&p| m.get(&key(p)).copied()).collect()
    };
    let n = pts.len();
    let mut slot = vec![NEW; n];
    let mut miss: Vec<V3> = Vec::new();
    let mut local: FxMap<[u64; 3], u32> = FxMap::default();
    for i in 0..n {
        if known[i].is_none() {
            let k = key(pts[i]);
            let j = *local.entry(k).or_insert_with(|| {
                miss.push(pts[i]);
                (miss.len() - 1) as u32
            });
            slot[i] = j;
        }
    }
    drop(local);
    let vals: Vec<u64> = miss.par_iter().map(|&q| f(q)).collect();
    map.reserve(miss.len());
    for (q, &v) in miss.iter().zip(vals.iter()) {
        map.insert(key(*q), v);
    }
    known
        .into_par_iter()
        .zip(slot.into_par_iter())
        .map(|(k, s)| k.unwrap_or_else(|| vals[s as usize]))
        .collect()
}
