//! Bernoulli–Gamma hurdle model (Section 3): score, likelihood, derivatives,
//! and the synthetic data generator.

use crate::rng::Rng;

pub fn sigmoid(s: f64) -> f64 {
    if s >= 0.0 {
        1.0 / (1.0 + (-s).exp())
    } else {
        let e = s.exp();
        e / (1.0 + e)
    }
}

/// log sigma(s), numerically stable.
pub fn log_sigmoid(s: f64) -> f64 {
    if s >= 0.0 {
        -(-s).exp().ln_1p()
    } else {
        s - s.exp().ln_1p()
    }
}

#[derive(Clone)]
pub struct Data {
    pub n: usize,
    pub k: usize,
    pub pairs: Vec<(usize, usize)>,
    pub a: Vec<u8>,
    pub w: Vec<f64>,
    pub kappa: f64,
    /// Ground-truth memberships, row-major n x K.
    pub z: Vec<u8>,
}

impl Data {
    /// Pair log likelihood (7) up to s-independent constants.
    #[inline]
    pub fn ell(&self, p: usize, s: f64) -> f64 {
        if self.a[p] == 1 {
            log_sigmoid(s) - self.kappa * s - self.kappa * self.w[p] * (-s).exp()
        } else {
            log_sigmoid(-s)
        }
    }

    /// Score derivative D_ij(s) (8).
    #[inline]
    pub fn d1(&self, p: usize, s: f64) -> f64 {
        let a = self.a[p] as f64;
        a - sigmoid(s) + a * self.kappa * (self.w[p] * (-s).exp() - 1.0)
    }

    /// Second derivative (9).
    #[inline]
    pub fn d2(&self, p: usize, s: f64) -> f64 {
        let sg = sigmoid(s);
        -sg * (1.0 - sg) - (self.a[p] as f64) * self.kappa * self.w[p] * (-s).exp()
    }
}

/// Sample memberships from Bernoulli(pi_k), enforce nonempty sets, then sample
/// the hurdle observations on the given pair set.
pub fn generate(
    rng: &mut Rng,
    n: usize,
    pi: &[f64],
    lam: &[f64],
    zeta: f64,
    kappa: f64,
    pairs: Vec<(usize, usize)>,
) -> Data {
    let k = pi.len();
    let mut z = vec![0u8; n * k];
    for i in 0..n {
        let mut any = false;
        for c in 0..k {
            if rng.bernoulli(pi[c]) {
                z[i * k + c] = 1;
                any = true;
            }
        }
        if !any {
            let c = rng.below(k);
            z[i * k + c] = 1;
        }
    }
    let mut a = Vec::with_capacity(pairs.len());
    let mut w = Vec::with_capacity(pairs.len());
    for &(i, j) in &pairs {
        let mut s = zeta;
        for c in 0..k {
            s += lam[c] * (z[i * k + c] * z[j * k + c]) as f64;
        }
        if rng.bernoulli(sigmoid(s)) {
            a.push(1u8);
            let mut x = rng.gamma(kappa, kappa * (-s).exp());
            if x <= 0.0 {
                x = f64::MIN_POSITIVE;
            }
            w.push(x);
        } else {
            a.push(0u8);
            w.push(0.0);
        }
    }
    Data { n, k, pairs, a, w, kappa, z }
}

pub fn complete_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            v.push((i, j));
        }
    }
    v
}
