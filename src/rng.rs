//! Deterministic pseudo-random generator (xoshiro256**, seeded by splitmix64)
//! with the samplers needed by the synthetic generators.

pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let s = [
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
        ];
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, m).
    pub fn below(&mut self, m: usize) -> usize {
        (self.uniform() * m as f64) as usize % m
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    pub fn normal(&mut self) -> f64 {
        loop {
            let u1 = self.uniform();
            if u1 > 0.0 {
                let u2 = self.uniform();
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }

    /// Gamma(shape, rate) via Marsaglia–Tsang.
    pub fn gamma(&mut self, shape: f64, rate: f64) -> f64 {
        if shape < 1.0 {
            let u = loop {
                let u = self.uniform();
                if u > 0.0 {
                    break u;
                }
            };
            return self.gamma(shape + 1.0, rate) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v / rate;
            }
        }
    }
}
