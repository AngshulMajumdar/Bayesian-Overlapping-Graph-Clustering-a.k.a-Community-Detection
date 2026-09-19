//! Section 13.1: exact-enumeration closure and residual microbenchmark
//! (Tables 4 and 5).

use crate::bethe::{Graph, Mrf};
use crate::model::sigmoid;

pub const N: usize = 10;
pub const J11: f64 = -3.0; // log psi(1,1); other entries are log 1 = 0

pub fn path_edges() -> Vec<(usize, usize)> {
    (0..N - 1).map(|i| (i, i + 1)).collect()
}

pub fn two_tree_edges() -> Vec<(usize, usize)> {
    let mut e = vec![(0, 1), (0, 2), (1, 2)];
    for v in 3..N {
        e.push((v - 1, v));
        e.push((v - 2, v));
    }
    e
}

pub struct Marg {
    pub single: Vec<f64>,
    /// Pair tables per edge, [(0,0),(0,1),(1,0),(1,1)].
    pub pair: Vec<[f64; 4]>,
}

fn log_weight(edges: &[(usize, usize)], x: u32) -> f64 {
    // unary phi = 1/2 for both states: constant, dropped
    let mut s = 0.0;
    for &(i, j) in edges {
        if (x >> i) & 1 == 1 && (x >> j) & 1 == 1 {
            s += J11;
        }
    }
    s
}

pub fn exact(edges: &[(usize, usize)]) -> Marg {
    let mut z = 0.0;
    let mut single = vec![0.0; N];
    let mut pair = vec![[0.0; 4]; edges.len()];
    for x in 0u32..(1 << N) {
        let w = log_weight(edges, x).exp();
        z += w;
        for i in 0..N {
            if (x >> i) & 1 == 1 {
                single[i] += w;
            }
        }
        for (e, &(i, j)) in edges.iter().enumerate() {
            let a = ((x >> i) & 1) as usize;
            let c = ((x >> j) & 1) as usize;
            pair[e][2 * a + c] += w;
        }
    }
    for v in single.iter_mut() {
        *v /= z;
    }
    for p in pair.iter_mut() {
        for v in p.iter_mut() {
            *v /= z;
        }
    }
    Marg { single, pair }
}

pub fn mrf(g: &Graph) -> Mrf<'_> {
    Mrf {
        g,
        h0: vec![0.0; g.n],
        lpsi: vec![[0.0, 0.0, 0.0, J11]; g.edges.len()],
    }
}

/// Product mean field: sequential coordinate updates from q = 1/2 to a
/// fixed point (tolerance 1e-14, cap 1e5 sweeps).
pub fn mean_field(g: &Graph) -> Marg {
    let mut q = vec![0.5; N];
    for _ in 0..100_000 {
        let mut delta: f64 = 0.0;
        for i in 0..N {
            let mut h = 0.0;
            for &(j, _) in &g.adj[i] {
                h += J11 * q[j];
            }
            let v = sigmoid(h);
            delta = delta.max((v - q[i]).abs());
            q[i] = v;
        }
        if delta < 1e-14 {
            break;
        }
    }
    let pair = g
        .edges
        .iter()
        .map(|&(i, j)| {
            [
                (1.0 - q[i]) * (1.0 - q[j]),
                (1.0 - q[i]) * q[j],
                q[i] * (1.0 - q[j]),
                q[i] * q[j],
            ]
        })
        .collect();
    Marg { single: q, pair }
}

/// Region-3 closure on the maximal bags {t, t+1, t+2} of a width-two junction
/// tree (a chain of bags for both the path and the 2-tree), computed by
/// sum–product over the bag chain.
pub fn region3(edges: &[(usize, usize)]) -> Marg {
    let nb = N - 2;
    // assign each edge to the first bag containing both endpoints
    let mut bag_pot = vec![[0.0f64; 8]; nb];
    for &(i, j) in edges {
        let t = (0..nb)
            .find(|&t| (t..t + 3).contains(&i) && (t..t + 3).contains(&j))
            .expect("edge not covered by a bag");
        for s in 0..8usize {
            let bit = |v: usize| (s >> (v - t)) & 1;
            if bit(i) == 1 && bit(j) == 1 {
                bag_pot[t][s] += J11;
            }
        }
    }
    let pot: Vec<[f64; 8]> = bag_pot
        .iter()
        .map(|p| {
            let mut o = [0.0; 8];
            for s in 0..8 {
                o[s] = p[s].exp();
            }
            o
        })
        .collect();
    // Separator between bag t-1 = {t-1,t,t+1} and bag t = {t,t+1,t+2} is
    // {t,t+1}: bits 1,2 of the left bag and bits 0,1 of the right bag.
    let left_sep = |s: usize| (s >> 1) & 3;
    let right_sep = |s: usize| s & 3;
    let mut fwd = vec![[1.0f64; 4]; nb]; // into bag t from the left, by right_sep
    for t in 0..nb - 1 {
        let mut m = [0.0; 4];
        for s in 0..8 {
            m[left_sep(s)] += pot[t][s] * fwd[t][right_sep(s)];
        }
        let z: f64 = m.iter().sum();
        fwd[t + 1] = [m[0] / z, m[1] / z, m[2] / z, m[3] / z];
    }
    let mut bwd = vec![[1.0f64; 4]; nb]; // into bag t from the right, by left_sep
    for t in (1..nb).rev() {
        let mut m = [0.0; 4];
        for s in 0..8 {
            m[right_sep(s)] += pot[t][s] * bwd[t][left_sep(s)];
        }
        let z: f64 = m.iter().sum();
        bwd[t - 1] = [m[0] / z, m[1] / z, m[2] / z, m[3] / z];
    }
    let mut beliefs = vec![[0.0f64; 8]; nb];
    for t in 0..nb {
        let mut z = 0.0;
        for s in 0..8 {
            let v = pot[t][s] * fwd[t][right_sep(s)] * bwd[t][left_sep(s)];
            beliefs[t][s] = v;
            z += v;
        }
        for s in 0..8 {
            beliefs[t][s] /= z;
        }
    }
    let mut single = vec![0.0; N];
    for v in 0..N {
        let t = v.min(nb - 1);
        let off = v - t;
        single[v] = (0..8).filter(|s| (s >> off) & 1 == 1).map(|s| beliefs[t][s]).sum();
    }
    let pair = edges
        .iter()
        .map(|&(i, j)| {
            let t = (0..nb)
                .find(|&t| (t..t + 3).contains(&i) && (t..t + 3).contains(&j))
                .unwrap();
            let mut p = [0.0; 4];
            for s in 0..8 {
                let a = (s >> (i - t)) & 1;
                let c = (s >> (j - t)) & 1;
                p[2 * a + c] += beliefs[t][s];
            }
            p
        })
        .collect();
    Marg { single, pair }
}

pub fn single_mae(a: &Marg, b: &Marg) -> f64 {
    a.single.iter().zip(&b.single).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.single.len() as f64
}

/// Edge-pair MAE: mean absolute error of the edge pair moments q_ij(1,1).
pub fn pair_mae(a: &Marg, b: &Marg) -> f64 {
    a.pair.iter().zip(&b.pair).map(|(p, q)| (p[3] - q[3]).abs()).sum::<f64>() / a.pair.len() as f64
}

pub fn bethe_marg(m: &Mrf, u: &[f64]) -> Marg {
    Marg { single: m.singletons(u), pair: m.pairs(u) }
}
