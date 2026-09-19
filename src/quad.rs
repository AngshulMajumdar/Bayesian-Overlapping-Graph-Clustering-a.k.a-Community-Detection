//! Gauss quadrature by Golub–Welsch (Jacobi eigenvalue iteration on the
//! symmetric tridiagonal Jacobi matrix), and the integrated Bayesian
//! reference for the end-to-end experiment (Section 13.3).

use crate::fsdb::Hyper;
use crate::model::Data;

fn sym_eig(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..200 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

/// Nodes and probability-normalized weights from a Jacobi matrix.
fn golub_welsch(diag: &[f64], off: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = diag.len();
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        a[i][i] = diag[i];
        if i + 1 < n {
            a[i][i + 1] = off[i];
            a[i + 1][i] = off[i];
        }
    }
    let (ev, v) = sym_eig(a);
    let mut w: Vec<f64> = (0..n).map(|j| v[0][j] * v[0][j]).collect();
    let s: f64 = w.iter().sum();
    w.iter_mut().for_each(|x| *x /= s);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| ev[i].partial_cmp(&ev[j]).unwrap());
    (idx.iter().map(|&i| ev[i]).collect(), idx.iter().map(|&i| w[i]).collect())
}

/// Generalized Gauss–Laguerre with weight x^alpha e^{-x}.
pub fn gen_laguerre(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let diag: Vec<f64> = (0..n).map(|k| 2.0 * k as f64 + alpha + 1.0).collect();
    let off: Vec<f64> = (1..n).map(|k| (k as f64 * (k as f64 + alpha)).sqrt()).collect();
    golub_welsch(&diag, &off)
}

/// Probabilists' Gauss–Hermite, weight e^{-x^2/2}.
pub fn hermite_prob(n: usize) -> (Vec<f64>, Vec<f64>) {
    let diag = vec![0.0; n];
    let off: Vec<f64> = (1..n).map(|k| (k as f64).sqrt()).collect();
    golub_welsch(&diag, &off)
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos log-gamma.
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Integrated Bayesian binary marginals for K = 2: Z enumerated, prevalences
/// integrated analytically (Beta), lambda_k by generalized Gauss–Laguerre and
/// zeta by Gauss–Hermite, both of the given order. Each quadrature point is
/// mapped to the decreasing-lambda label gauge (ties split evenly).
pub fn reference(d: &Data, h: &Hyper, order: usize) -> (Vec<f64>, Vec<f64>) {
    let (n, kk, np) = (d.n, d.k, d.pairs.len());
    assert_eq!(kk, 2);
    let (xl, wl) = gen_laguerre(order, h.alpha0 - 1.0);
    let lam_nodes: Vec<f64> = xl.iter().map(|x| x / h.beta0).collect();
    let (xh, wh) = hermite_prob(order);
    let zeta_nodes: Vec<f64> = xh.iter().map(|x| h.sigma_zeta * x).collect();
    let nv = n * kk;
    let total = 1usize << nv;
    // log prior of Z with pi integrated out
    let mut lpz = vec![0.0; total];
    for x in 0..total {
        let mut s = 0.0;
        for k in 0..kk {
            let m = (0..n).filter(|i| (x >> (i * kk + k)) & 1 == 1).count() as f64;
            s += ln_beta(h.a0 + m, h.b0 + n as f64 - m) - ln_beta(h.a0, h.b0);
        }
        lpz[x] = s;
    }
    // log weight of every (Z, lambda_1, lambda_2, zeta) quadrature point
    let mut lw_all: Vec<f64> = Vec::with_capacity(total * order * order * order);
    let mut mx = f64::NEG_INFINITY;
    for x in 0..total {
        let pat: Vec<[bool; 2]> = d
            .pairs
            .iter()
            .map(|&(i, j)| {
                [
                    (x >> (i * kk)) & 1 == 1 && (x >> (j * kk)) & 1 == 1,
                    (x >> (i * kk + 1)) & 1 == 1 && (x >> (j * kk + 1)) & 1 == 1,
                ]
            })
            .collect();
        for a in 0..order {
            for b in 0..order {
                for c in 0..order {
                    let mut s = lpz[x] + wl[a].ln() + wl[b].ln() + wh[c].ln();
                    for p in 0..np {
                        let mut sc = zeta_nodes[c];
                        if pat[p][0] {
                            sc += lam_nodes[a];
                        }
                        if pat[p][1] {
                            sc += lam_nodes[b];
                        }
                        s += d.ell(p, sc);
                    }
                    mx = mx.max(s);
                    lw_all.push(s);
                }
            }
        }
    }
    let mut z = 0.0;
    let mut gamma = vec![0.0; nv];
    let mut rho = vec![0.0; np * kk];
    let mut idx = 0;
    for x in 0..total {
        for a in 0..order {
            for b in 0..order {
                for _c in 0..order {
                    let w = (lw_all[idx] - mx).exp();
                    idx += 1;
                    z += w;
                    // gauge weights: (identity, swapped)
                    let (wid, wsw) = if lam_nodes[a] > lam_nodes[b] {
                        (w, 0.0)
                    } else if lam_nodes[a] < lam_nodes[b] {
                        (0.0, w)
                    } else {
                        (0.5 * w, 0.5 * w)
                    };
                    for i in 0..n {
                        for k in 0..kk {
                            if (x >> (i * kk + k)) & 1 == 1 {
                                gamma[i * kk + k] += wid;
                                gamma[i * kk + (1 - k)] += wsw;
                            }
                        }
                    }
                    for (p, &(i, j)) in d.pairs.iter().enumerate() {
                        for k in 0..kk {
                            if (x >> (i * kk + k)) & 1 == 1 && (x >> (j * kk + k)) & 1 == 1 {
                                rho[p * kk + k] += wid;
                                rho[p * kk + (1 - k)] += wsw;
                            }
                        }
                    }
                }
            }
        }
    }
    gamma.iter_mut().for_each(|v| *v /= z);
    rho.iter_mut().for_each(|v| *v /= z);
    (gamma, rho)
}
