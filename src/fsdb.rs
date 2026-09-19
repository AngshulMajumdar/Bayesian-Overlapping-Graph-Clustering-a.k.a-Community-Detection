//! Model-specific FSDB: coordinate pair closure (Section 5.1), Bethe messages
//! (5.3), point-field score closures (5.4), Algorithm 1, product mean field,
//! exact enumeration, and the set-valued Bayes decision (Section 6).

use crate::bethe::{Graph, Mrf};
use crate::model::{sigmoid, Data};

pub struct Hyper {
    pub a0: f64,
    pub b0: f64,
    pub alpha0: f64,
    pub beta0: f64,
    pub sigma_zeta: f64,
}

/// Retained beliefs for all coordinates.
pub struct Beliefs {
    /// gamma[i*K + k]
    pub gamma: Vec<f64>,
    /// rho[p*K + k] = q_ijk(1,1)
    pub rho: Vec<f64>,
    /// q[p*K + k] = [(0,0),(0,1),(1,0),(1,1)]
    pub q: Vec<[f64; 4]>,
}

impl Beliefs {
    pub fn product(d: &Data, pi: &[f64]) -> Self {
        let (n, k, np) = (d.n, d.k, d.pairs.len());
        let mut gamma = vec![0.0; n * k];
        for i in 0..n {
            for c in 0..k {
                gamma[i * k + c] = pi[c];
            }
        }
        let mut rho = vec![0.0; np * k];
        let mut q = vec![[0.0; 4]; np * k];
        for p in 0..np {
            for c in 0..k {
                let x = pi[c];
                rho[p * k + c] = x * x;
                q[p * k + c] = [(1.0 - x) * (1.0 - x), (1.0 - x) * x, x * (1.0 - x), x * x];
            }
        }
        Beliefs { gamma, rho, q }
    }
}

/// Residual interaction r_ij^{(-k)} (51).
pub fn residual_score(d: &Data, b: &Beliefs, lam: &[f64], zeta: f64, p: usize, k: usize) -> f64 {
    let mut r = zeta;
    for l in 0..d.k {
        if l != k {
            r += lam[l] * b.rho[p * d.k + l];
        }
    }
    r
}

pub struct MsgCfg {
    pub damping: f64,
    pub tol: f64,
    pub cap: usize,
}

/// Build the coordinate-k surrogate (52)–(54), run damped Bethe messages, and
/// write back gamma_{.k}, rho_{.k}, q_{.k}. Returns the final residual r_k.
pub fn coordinate_update(
    d: &Data,
    g: &Graph,
    b: &mut Beliefs,
    u: &mut Vec<f64>,
    k: usize,
    pi_k: f64,
    lam: &[f64],
    zeta: f64,
    cfg: &MsgCfg,
) -> f64 {
    let np = d.pairs.len();
    let mut lpsi = Vec::with_capacity(np);
    for p in 0..np {
        let r = residual_score(d, b, lam, zeta, p, k);
        // exp{ell(r + lam_k a c)}; the common factor exp{ell(r)} is removed
        let l11 = d.ell(p, r + lam[k]) - d.ell(p, r);
        lpsi.push([0.0, 0.0, 0.0, l11]);
    }
    let h = (pi_k / (1.0 - pi_k)).ln();
    let mrf = Mrf { g, h0: vec![h; d.n], lpsi };
    let (_, r) = mrf.run(u, cfg.damping, cfg.tol, cfg.cap, |_, _, _| {});
    let s = mrf.singletons(u);
    let pq = mrf.pairs(u);
    for i in 0..d.n {
        b.gamma[i * d.k + k] = s[i];
    }
    for p in 0..np {
        b.rho[p * d.k + k] = pq[p][3];
        b.q[p * d.k + k] = pq[p];
    }
    r
}

pub struct FixedResult {
    pub b: Beliefs,
    pub sweeps: usize,
    pub max_resid: f64,
    pub converged: bool,
}

/// Pairwise FSDB with continuous fields frozen at given values (Section 13.2).
pub fn fsdb_fixed(
    d: &Data,
    pi: &[f64],
    lam: &[f64],
    zeta: f64,
    cfg: &MsgCfg,
    coord_tol: f64,
    coord_cap: usize,
) -> FixedResult {
    let g = Graph::new(d.n, d.pairs.clone());
    let mut b = Beliefs::product(d, pi);
    let mut us = vec![vec![0.0; 2 * d.pairs.len()]; d.k];
    let mut last_r = 0.0;
    for sweep in 1..=coord_cap {
        let old = b.gamma.clone();
        last_r = 0.0f64;
        for k in 0..d.k {
            let r = coordinate_update(d, &g, &mut b, &mut us[k], k, pi[k], lam, zeta, cfg);
            last_r = last_r.max(r);
        }
        let delta = old.iter().zip(&b.gamma).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
        if delta < coord_tol {
            return FixedResult { b, sweeps: sweep, max_resid: last_r, converged: true };
        }
    }
    FixedResult { b, sweeps: coord_cap, max_resid: last_r, converged: false }
}

/// Product mean field on the exact Bernoulli–Gamma binary posterior with fixed
/// continuous fields. Expectations of ell(s_ij) under the product law are
/// computed exactly by enumerating the K shared-membership indicators.
pub fn mean_field(
    d: &Data,
    pi: &[f64],
    lam: &[f64],
    zeta: f64,
    damping: f64,
    tol: f64,
    cap: usize,
) -> (Vec<f64>, Vec<f64>, usize, bool) {
    let (n, kk) = (d.n, d.k);
    let mut gamma = vec![0.0; n * kk];
    for i in 0..n {
        for c in 0..kk {
            gamma[i * kk + c] = pi[c];
        }
    }
    let mut nbr: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (p, &(i, j)) in d.pairs.iter().enumerate() {
        nbr[i].push((j, p));
        nbr[j].push((i, p));
    }
    let mut sweeps = cap;
    let mut conv = false;
    for sweep in 1..=cap {
        let mut delta: f64 = 0.0;
        for k in 0..kk {
            for i in 0..n {
                let mut diff = 0.0;
                for &(j, p) in &nbr[i] {
                    // E[ell | Z_ik = z] for z = 1 and z = 0
                    let mut e1 = 0.0;
                    let mut e0 = 0.0;
                    let pk = gamma[j * kk + k];
                    for mask in 0u32..(1 << kk) {
                        if (mask >> k) & 1 == 1 {
                            continue;
                        }
                        let mut w = 1.0;
                        let mut s = zeta;
                        for l in 0..kk {
                            if l == k {
                                continue;
                            }
                            let pr = gamma[i * kk + l] * gamma[j * kk + l];
                            if (mask >> l) & 1 == 1 {
                                w *= pr;
                                s += lam[l];
                            } else {
                                w *= 1.0 - pr;
                            }
                        }
                        if w == 0.0 {
                            continue;
                        }
                        let v0 = d.ell(p, s);
                        let v1 = d.ell(p, s + lam[k]);
                        e1 += w * (pk * v1 + (1.0 - pk) * v0);
                        e0 += w * v0;
                    }
                    diff += e1 - e0;
                }
                let target = sigmoid((pi[k] / (1.0 - pi[k])).ln() + diff);
                let old = gamma[i * kk + k];
                let new = (1.0 - damping) * old + damping * target;
                delta = delta.max((new - old).abs());
                gamma[i * kk + k] = new;
            }
        }
        if delta < tol {
            sweeps = sweep;
            conv = true;
            break;
        }
    }
    let mut rho = vec![0.0; d.pairs.len() * kk];
    for (p, &(i, j)) in d.pairs.iter().enumerate() {
        for k in 0..kk {
            rho[p * kk + k] = gamma[i * kk + k] * gamma[j * kk + k];
        }
    }
    (gamma, rho, sweeps, conv)
}

/// Exact enumeration of p(Z | A, W, pi, lam, zeta). Returns (gamma, rho).
pub fn exact_fixed(d: &Data, pi: &[f64], lam: &[f64], zeta: f64) -> (Vec<f64>, Vec<f64>) {
    let (n, kk, np) = (d.n, d.k, d.pairs.len());
    let nv = n * kk;
    assert!(nv <= 26);
    let lpi1: Vec<f64> = pi.iter().map(|x| x.ln()).collect();
    let lpi0: Vec<f64> = pi.iter().map(|x| (1.0 - x).ln()).collect();
    let total = 1usize << nv;
    let mut lw = vec![0.0; total];
    let mut mx = f64::NEG_INFINITY;
    for x in 0..total {
        let mut s = 0.0;
        for i in 0..n {
            for k in 0..kk {
                s += if (x >> (i * kk + k)) & 1 == 1 { lpi1[k] } else { lpi0[k] };
            }
        }
        for (p, &(i, j)) in d.pairs.iter().enumerate() {
            let mut sc = zeta;
            for k in 0..kk {
                if (x >> (i * kk + k)) & 1 == 1 && (x >> (j * kk + k)) & 1 == 1 {
                    sc += lam[k];
                }
            }
            s += d.ell(p, sc);
        }
        lw[x] = s;
        mx = mx.max(s);
    }
    let mut z = 0.0;
    let mut gamma = vec![0.0; nv];
    let mut rho = vec![0.0; np * kk];
    for x in 0..total {
        let w = (lw[x] - mx).exp();
        z += w;
        for v in 0..nv {
            if (x >> v) & 1 == 1 {
                gamma[v] += w;
            }
        }
        for (p, &(i, j)) in d.pairs.iter().enumerate() {
            for k in 0..kk {
                if (x >> (i * kk + k)) & 1 == 1 && (x >> (j * kk + k)) & 1 == 1 {
                    rho[p * kk + k] += w;
                }
            }
        }
    }
    gamma.iter_mut().for_each(|v| *v /= z);
    rho.iter_mut().for_each(|v| *v /= z);
    (gamma, rho)
}

/// Constrained Bayes action (81)–(84): smallest prefix of sorted marginals
/// whose inclusion mass reaches alpha * M_i. Returns membership bits n x K.
pub fn decide(gamma: &[f64], n: usize, k: usize, alpha: f64) -> Vec<u8> {
    let mut out = vec![0u8; n * k];
    for i in 0..n {
        let row = &gamma[i * k..(i + 1) * k];
        let m: f64 = row.iter().sum();
        let mut idx: Vec<usize> = (0..k).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        let mut acc = 0.0;
        for &c in &idx {
            out[i * k + c] = 1;
            acc += row[c];
            if acc >= alpha * m {
                break;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Algorithm 1: complete alternating solver with point continuous fields.
// ---------------------------------------------------------------------------

pub struct Alg1Cfg {
    pub msg: MsgCfg,
    pub newton_damping: f64,
    pub root_tol: f64,
    pub root_cap: usize,
    pub outer_tol: f64,
    pub outer_cap: usize,
}

#[allow(dead_code)]
pub struct Alg1Result {
    pub gamma: Vec<f64>,
    pub rho: Vec<f64>,
    pub pi: Vec<f64>,
    pub lam: Vec<f64>,
    pub zeta: f64,
    pub resid: f64,
    pub sweeps: usize,
    pub converged: bool,
}

/// Safeguarded expanding-bracket damped Newton–bisection for a strictly
/// decreasing scalar function. `f(x)` returns (g, g'). Returns the root if
/// |g| <= tol within `cap` evaluations.
pub fn newton_bisect<F: FnMut(f64) -> (f64, f64)>(
    mut f: F,
    x0: f64,
    damping: f64,
    tol: f64,
    cap: usize,
) -> Option<f64> {
    let mut evals = 0usize;
    let (g0, d0) = f(x0);
    evals += 1;
    if g0.abs() <= tol {
        return Some(x0);
    }
    // expanding bracket
    let (mut lo, mut hi);
    let mut step = 1.0;
    if g0 > 0.0 {
        lo = x0;
        hi = x0 + step;
        loop {
            if evals >= cap {
                return None;
            }
            let (g, _) = f(hi);
            evals += 1;
            if g.abs() <= tol {
                return Some(hi);
            }
            if g < 0.0 {
                break;
            }
            lo = hi;
            step *= 2.0;
            hi += step;
        }
    } else {
        hi = x0;
        lo = x0 - step;
        loop {
            if evals >= cap {
                return None;
            }
            let (g, _) = f(lo);
            evals += 1;
            if g.abs() <= tol {
                return Some(lo);
            }
            if g > 0.0 {
                break;
            }
            hi = lo;
            step *= 2.0;
            lo -= step;
        }
    }
    let (mut x, mut g, mut dg) = (x0, g0, d0);
    if !(x > lo && x < hi) {
        x = 0.5 * (lo + hi);
        let r = f(x);
        evals += 1;
        g = r.0;
        dg = r.1;
        if g.abs() <= tol {
            return Some(x);
        }
        if g > 0.0 {
            lo = x;
        } else {
            hi = x;
        }
    }
    while evals < cap {
        let mut xn = if dg != 0.0 { x - damping * g / dg } else { f64::NAN };
        if !(xn > lo && xn < hi) {
            xn = 0.5 * (lo + hi);
        }
        let r = f(xn);
        evals += 1;
        x = xn;
        g = r.0;
        dg = r.1;
        if g.abs() <= tol {
            return Some(x);
        }
        if g > 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        if hi - lo <= f64::EPSILON * x.abs().max(1.0) {
            return None;
        }
    }
    None
}

pub fn algorithm1(d: &Data, h: &Hyper, cfg: &Alg1Cfg) -> Alg1Result {
    let (n, kk, np) = (d.n, d.k, d.pairs.len());
    let g = Graph::new(n, d.pairs.clone());
    // prior central values; tied interaction scales get log-scale perturbations
    let mut pi = vec![h.a0 / (h.a0 + h.b0); kk];
    let mut xi: Vec<f64> = (0..kk)
        .map(|k| {
            let off = if kk > 1 { 5e-4 * (2.0 * k as f64 / (kk - 1) as f64 - 1.0) } else { 0.0 };
            (h.alpha0 / h.beta0).ln() + off
        })
        .collect();
    let mut zeta = 0.0;
    let mut b = Beliefs::product(d, &pi);
    let mut us = vec![vec![0.0; 2 * np]; kk];
    let s2 = h.sigma_zeta * h.sigma_zeta;
    let mut resid = 0.0;
    let mut sweeps = cfg.outer_cap;
    let mut converged = false;
    for tout in 1..=cfg.outer_cap {
        let gamma_old = b.gamma.clone();
        let xi_old = xi.clone();
        let zeta_old = zeta;
        let lam: Vec<f64> = xi.iter().map(|x| x.exp()).collect();
        let mut rmax: f64 = 0.0;
        let mut sd_ok = true;
        for k in 0..kk {
            let r = coordinate_update(d, &g, &mut b, &mut us[k], k, pi[k], &lam, zeta, &cfg.msg);
            rmax = rmax.max(r);
            if r > cfg.msg.tol {
                sd_ok = false;
                break;
            }
        }
        resid = rmax;
        if !sd_ok {
            sweeps = tout;
            break;
        }
        // prevalence (64)
        for k in 0..kk {
            let sg: f64 = (0..n).map(|i| b.gamma[i * kk + k]).sum();
            pi[k] = (h.a0 + sg) / (h.a0 + h.b0 + n as f64);
        }
        // interaction scales (65), solved in xi = log lambda
        let mut root_ok = true;
        for k in 0..kk {
            let lam_now: Vec<f64> = xi.iter().map(|x| x.exp()).collect();
            let rk: Vec<f64> = (0..np).map(|p| residual_score(d, &b, &lam_now, zeta, p, k)).collect();
            let rho_k: Vec<f64> = (0..np).map(|p| b.rho[p * kk + k]).collect();
            let f = |x: f64| {
                let l = x.exp();
                let mut gv = h.alpha0 / l - h.beta0;
                let mut dv = -h.alpha0 / (l * l);
                for p in 0..np {
                    gv += rho_k[p] * d.d1(p, rk[p] + l);
                    dv += rho_k[p] * d.d2(p, rk[p] + l);
                }
                (gv, l * dv)
            };
            match newton_bisect(f, xi[k], cfg.newton_damping, cfg.root_tol, cfg.root_cap) {
                Some(x) => xi[k] = x,
                None => {
                    root_ok = false;
                    break;
                }
            }
        }
        if root_ok {
            // bias (66)
            let lam_now: Vec<f64> = xi.iter().map(|x| x.exp()).collect();
            // r^{(-k)} - zeta is independent of zeta
            let base: Vec<f64> = (0..np * kk)
                .map(|pk| {
                    let (p, k) = (pk / kk, pk % kk);
                    residual_score(d, &b, &lam_now, 0.0, p, k)
                })
                .collect();
            let f = |z: f64| {
                let mut gv = 0.0;
                let mut dv = 0.0;
                for p in 0..np {
                    for k in 0..kk {
                        let q = &b.q[p * kk + k];
                        let r = base[p * kk + k] + z;
                        let (d10, d20) = (d.d1(p, r), d.d2(p, r));
                        let (d11, d21) = (d.d1(p, r + lam_now[k]), d.d2(p, r + lam_now[k]));
                        let q0 = q[0] + q[1] + q[2];
                        gv += q0 * d10 + q[3] * d11;
                        dv += q0 * d20 + q[3] * d21;
                    }
                }
                (gv / kk as f64 - z / s2, dv / kk as f64 - 1.0 / s2)
            };
            match newton_bisect(f, zeta, cfg.newton_damping, cfg.root_tol, cfg.root_cap) {
                Some(z) => zeta = z,
                None => root_ok = false,
            }
        }
        if !root_ok {
            sweeps = tout;
            break;
        }
        let dg = gamma_old.iter().zip(&b.gamma).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
        let dx = xi_old.iter().zip(&xi).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
        let dz = (zeta - zeta_old).abs();
        if dg.max(dx).max(dz) < cfg.outer_tol {
            sweeps = tout;
            converged = true;
            break;
        }
    }
    // label gauge: decreasing lambda
    let lam: Vec<f64> = xi.iter().map(|x| x.exp()).collect();
    let mut order: Vec<usize> = (0..kk).collect();
    order.sort_by(|&a, &c| lam[c].partial_cmp(&lam[a]).unwrap());
    let mut gamma = vec![0.0; n * kk];
    let mut rho = vec![0.0; np * kk];
    for (newk, &oldk) in order.iter().enumerate() {
        for i in 0..n {
            gamma[i * kk + newk] = b.gamma[i * kk + oldk];
        }
        for p in 0..np {
            rho[p * kk + newk] = b.rho[p * kk + oldk];
        }
    }
    Alg1Result {
        gamma,
        rho,
        pi: order.iter().map(|&k| pi[k]).collect(),
        lam: order.iter().map(|&k| lam[k]).collect(),
        zeta,
        resid,
        sweeps,
        converged,
    }
}
