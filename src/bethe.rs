//! Pairwise binary Bethe closure: normalized sum–product in log-odds form
//! (57)–(61), damping (58), and the fixed-point residual (59).

pub struct Graph {
    pub n: usize,
    pub edges: Vec<(usize, usize)>,
    /// For each node: list of (neighbour, edge index).
    pub adj: Vec<Vec<(usize, usize)>>,
}

impl Graph {
    pub fn new(n: usize, edges: Vec<(usize, usize)>) -> Self {
        let mut adj = vec![Vec::new(); n];
        for (e, &(i, j)) in edges.iter().enumerate() {
            adj[i].push((j, e));
            adj[j].push((i, e));
        }
        Graph { n, edges, adj }
    }
}

/// Directed message index: edge e=(i,j); 2e is i->j, 2e+1 is j->i.
#[inline]
fn dir(e: usize, from_first: bool) -> usize {
    if from_first {
        2 * e
    } else {
        2 * e + 1
    }
}

#[inline]
fn lse2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Binary pairwise model: unary log-odds h0_i = log phi_i(1)/phi_i(0) and
/// log pair potentials lpsi[e] = [ (0,0), (0,1), (1,0), (1,1) ] with the
/// first index referring to edges[e].0.
pub struct Mrf<'a> {
    pub g: &'a Graph,
    pub h0: Vec<f64>,
    pub lpsi: Vec<[f64; 4]>,
}

impl<'a> Mrf<'a> {
    fn node_field(&self, u: &[f64], i: usize) -> f64 {
        let mut h = self.h0[i];
        for &(j, e) in &self.g.adj[i] {
            let (a, _) = self.g.edges[e];
            // message j -> i
            h += u[dir(e, a == j)];
        }
        h
    }

    /// Undamped normalized log-message map T(u).
    pub fn t_map(&self, u: &[f64], out: &mut [f64]) {
        let fields: Vec<f64> = (0..self.g.n).map(|i| self.node_field(u, i)).collect();
        for (e, &(i, j)) in self.g.edges.iter().enumerate() {
            let lp = &self.lpsi[e];
            // i -> j : cavity field of i excluding j->i
            let hi = fields[i] - u[dir(e, false)];
            // m(c) = psi(0,c) + psi(1,c) e^{hi}
            let m0 = lse2(lp[0], lp[2] + hi);
            let m1 = lse2(lp[1], lp[3] + hi);
            out[dir(e, true)] = m1 - m0;
            // j -> i : cavity field of j excluding i->j; psi indexed (a=i, c=j)
            let hj = fields[j] - u[dir(e, true)];
            let m0 = lse2(lp[0], lp[1] + hj);
            let m1 = lse2(lp[2], lp[3] + hj);
            out[dir(e, false)] = m1 - m0;
        }
    }

    pub fn residual(&self, u: &[f64], scratch: &mut [f64]) -> f64 {
        self.t_map(u, scratch);
        u.iter()
            .zip(scratch.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// Damped synchronous iteration. `cb(t, u, r)` is called after each
    /// iteration t = 1, 2, ... with the residual r = ||u - T(u)||_inf.
    /// Returns (iterations, final residual).
    pub fn run<F: FnMut(usize, &[f64], f64)>(
        &self,
        u: &mut Vec<f64>,
        damping: f64,
        tol: f64,
        cap: usize,
        mut cb: F,
    ) -> (usize, f64) {
        let m = u.len();
        let mut raw = vec![0.0; m];
        let mut r = self.residual(u, &mut raw);
        if r <= tol {
            return (0, r);
        }
        for t in 1..=cap {
            self.t_map(u, &mut raw);
            for x in 0..m {
                u[x] = (1.0 - damping) * u[x] + damping * raw[x];
            }
            r = self.residual(u, &mut raw);
            cb(t, u, r);
            if r <= tol {
                return (t, r);
            }
        }
        (cap, r)
    }

    /// Singleton beliefs q_i(1) (60).
    pub fn singletons(&self, u: &[f64]) -> Vec<f64> {
        (0..self.g.n)
            .map(|i| crate::model::sigmoid(self.node_field(u, i)))
            .collect()
    }

    /// Pair beliefs (61), ordered [(0,0),(0,1),(1,0),(1,1)].
    pub fn pairs(&self, u: &[f64]) -> Vec<[f64; 4]> {
        let fields: Vec<f64> = (0..self.g.n).map(|i| self.node_field(u, i)).collect();
        self.g
            .edges
            .iter()
            .enumerate()
            .map(|(e, &(i, j))| {
                let hi = fields[i] - u[dir(e, false)];
                let hj = fields[j] - u[dir(e, true)];
                let lp = &self.lpsi[e];
                let l = [lp[0], lp[1] + hj, lp[2] + hi, lp[3] + hi + hj];
                let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = l.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = ex.iter().sum();
                [ex[0] / z, ex[1] / z, ex[2] / z, ex[3] / z]
            })
            .collect()
    }
}
