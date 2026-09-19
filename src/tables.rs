//! Drivers for Tables 4–8. Each writes one CSV into the output directory.

use crate::bethe::Graph;
use crate::fsdb::{self, Alg1Cfg, Hyper, MsgCfg};
use crate::micro;
use crate::model::{self, Data};
use crate::quad;
use crate::rng::Rng;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = if v.len() > 1 { v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    (m, var.sqrt())
}

fn mae(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

fn e(x: f64) -> String {
    format!("{:.6e}", x)
}

// ---------------------------------------------------------------- Tables 4, 5

pub fn tables_4_5(out: &Path) -> std::io::Result<()> {
    let mut t4 = File::create(out.join("table4.csv"))?;
    writeln!(t4, "graph,closure,singleton_mae,edge_pair_mae")?;
    let mut t5 = File::create(out.join("table5.csv"))?;
    writeln!(t5, "graph,iteration,residual_R,singleton_mae")?;
    for (name, edges, checkpoints) in [
        ("Path", micro::path_edges(), vec![10usize, 30]),
        ("2-tree", micro::two_tree_edges(), vec![10usize, 27]),
    ] {
        let g = Graph::new(micro::N, edges.clone());
        let ex = micro::exact(&edges);
        let mf = micro::mean_field(&g);
        let m = micro::mrf(&g);
        let mut u = vec![0.0; 2 * edges.len()];
        let mut rows: Vec<(String, f64, f64)> = Vec::new();
        let (it, r) = m.run(&mut u, 0.3, 1e-12, 10_000, |t, uu, r| {
            if checkpoints.contains(&t) {
                let b = micro::bethe_marg(&m, uu);
                rows.push((t.to_string(), r, micro::single_mae(&b, &ex)));
            }
        });
        let b = micro::bethe_marg(&m, &u);
        rows.push((format!("converged ({} iterations)", it), r, micro::single_mae(&b, &ex)));
        let r3 = micro::region3(&edges);
        for (cl, mg) in [("Product mean field", &mf), ("Bethe pair closure", &b), ("Region-3", &r3)] {
            writeln!(t4, "{},{},{},{}", name, cl, e(micro::single_mae(mg, &ex)), e(micro::pair_mae(mg, &ex)))?;
        }
        for (t, r, s) in rows {
            writeln!(t5, "{},{},{},{}", name, t, e(r), e(s))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- Table 6

pub fn table6(out: &Path) -> std::io::Result<()> {
    let (n, pi, lam, zeta, kappa) = (8usize, [0.35, 0.30], [1.4, 1.0], -1.8, 3.0);
    let alpha = 0.80;
    let k = 2usize;
    let cfg = MsgCfg { damping: 0.5, tol: 1e-10, cap: 500 };
    // per method: singleton, pair, agreement, hamming, time
    let mut acc = vec![vec![Vec::new(); 5]; 3];
    let mut max_sweeps = [0usize; 2];
    let mut max_resid: f64 = 0.0;
    let mut all_conv = true;
    for seed in 260918u64..=260937 {
        let mut rng = Rng::new(seed);
        let d = model::generate(&mut rng, n, &pi, &lam, zeta, kappa, model::complete_pairs(n));

        let t0 = Instant::now();
        let (g_ex, r_ex) = fsdb::exact_fixed(&d, &pi, &lam, zeta);
        let t_ex = t0.elapsed().as_secs_f64();
        let s_ex = fsdb::decide(&g_ex, n, k, alpha);

        let t0 = Instant::now();
        let (g_mf, r_mf, sw_mf, c_mf) = fsdb::mean_field(&d, &pi, &lam, zeta, 0.5, 1e-8, 500);
        let t_mf = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let fr = fsdb::fsdb_fixed(&d, &pi, &lam, zeta, &cfg, 1e-8, 500);
        let t_fs = t0.elapsed().as_secs_f64();

        max_sweeps[0] = max_sweeps[0].max(fr.sweeps);
        max_sweeps[1] = max_sweeps[1].max(sw_mf);
        max_resid = max_resid.max(fr.max_resid);
        all_conv &= fr.converged && c_mf;

        let results = [
            (g_ex.clone(), r_ex.clone(), t_ex),
            (g_mf, r_mf, t_mf),
            (fr.b.gamma.clone(), fr.b.rho.clone(), t_fs),
        ];
        for (m, (g, r, t)) in results.iter().enumerate() {
            let s = fsdb::decide(g, n, k, alpha);
            let agree = (0..n).filter(|&i| s[i * k..(i + 1) * k] == s_ex[i * k..(i + 1) * k]).count() as f64 / n as f64;
            let ham = s.iter().zip(&s_ex).filter(|(a, b)| a != b).count() as f64 / (n * k) as f64;
            acc[m][0].push(mae(g, &g_ex));
            acc[m][1].push(mae(r, &r_ex));
            acc[m][2].push(agree);
            acc[m][3].push(ham);
            acc[m][4].push(*t);
        }
    }
    let mut f = File::create(out.join("table6.csv"))?;
    writeln!(
        f,
        "method,singleton_mae_mean,singleton_mae_sd,pair_mae_mean,pair_mae_sd,set_agreement_mean,set_agreement_sd,hamming_mean,hamming_sd,time_s_mean,time_s_sd"
    )?;
    for (m, name) in ["Exact enumeration", "Product mean field", "Pairwise FSDB"].iter().enumerate() {
        let mut line = name.to_string();
        for c in 0..5 {
            let (mu, sd) = mean_sd(&acc[m][c]);
            line += &format!(",{},{}", e(mu), e(sd));
        }
        writeln!(f, "{}", line)?;
    }
    eprintln!(
        "table6: all converged = {}, max FSDB sweeps = {}, max MF sweeps = {}, max FSDB residual = {:.3e}",
        all_conv, max_sweeps[0], max_sweeps[1], max_resid
    );
    Ok(())
}

// ---------------------------------------------------------------- Table 7

pub fn alg1_cfg(outer_cap: usize) -> Alg1Cfg {
    Alg1Cfg {
        msg: MsgCfg { damping: 0.5, tol: 1e-10, cap: 1000 },
        newton_damping: 0.8,
        root_tol: 1e-10,
        root_cap: 200,
        outer_tol: 1e-8,
        outer_cap,
    }
}

pub const HYPER: Hyper = Hyper { a0: 2.0, b0: 2.0, alpha0: 6.0, beta0: 4.0, sigma_zeta: 0.8 };

pub fn table7(out: &Path) -> std::io::Result<()> {
    let (n, pi, lam, zeta, kappa) = (4usize, [0.55, 0.30], [1.6, 0.8], -1.2, 3.0);
    let cfg = alg1_cfg(500);
    let (mut s1, mut s2, mut sw, mut rr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut ok = 0;
    let runs = 10;
    for seed in 260913u64..=260922 {
        let mut rng = Rng::new(seed);
        let d = model::generate(&mut rng, n, &pi, &lam, zeta, kappa, model::complete_pairs(n));
        let res = fsdb::algorithm1(&d, &HYPER, &cfg);
        let (g_ref, r_ref) = quad::reference(&d, &HYPER, 26);
        if res.converged {
            ok += 1;
        }
        s1.push(mae(&res.gamma, &g_ref));
        s2.push(mae(&res.rho, &r_ref));
        sw.push(res.sweeps as f64);
        rr.push(res.resid);
    }
    let mut f = File::create(out.join("table7.csv"))?;
    writeln!(f, "quantity,mean,sd")?;
    writeln!(f, "Successful outer runs,{}/{},", ok, runs)?;
    for (name, v) in [
        ("Singleton-marginal MAE", &s1),
        ("Pair-marginal MAE", &s2),
        ("Outer sweeps", &sw),
        ("Maximum message residual", &rr),
    ] {
        let (m, s) = mean_sd(v);
        writeln!(f, "{},{},{}", name, e(m), e(s))?;
    }
    Ok(())
}

// ---------------------------------------------------------------- Table 8

pub const T8_K: usize = 5;
pub const T8_PI: f64 = 0.25;
pub const T8_LAMBDA: f64 = 2.0;

fn random_pairs(rng: &mut Rng, n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut set = std::collections::HashSet::new();
    let mut v = Vec::with_capacity(m);
    while v.len() < m {
        let i = rng.below(n);
        let j = rng.below(n);
        if i == j {
            continue;
        }
        let p = (i.min(j), i.max(j));
        if set.insert(p) {
            v.push(p);
        }
    }
    v.sort();
    v
}

fn peak_rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    f64::NAN
}

/// Best label permutation of predicted communities against the truth.
fn align(pred: &[u8], truth: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut perm: Vec<usize> = (0..k).collect();
    let mut best = perm.clone();
    let mut best_score = -1i64;
    fn next_perm(p: &mut [usize]) -> bool {
        let n = p.len();
        if n < 2 {
            return false;
        }
        let mut i = n - 1;
        while i > 0 && p[i - 1] >= p[i] {
            i -= 1;
        }
        if i == 0 {
            return false;
        }
        let mut j = n - 1;
        while p[j] <= p[i - 1] {
            j -= 1;
        }
        p.swap(i - 1, j);
        p[i..].reverse();
        true
    }
    loop {
        // predicted community perm[c] is matched to true community c
        let mut sc = 0i64;
        for i in 0..n {
            for c in 0..k {
                if pred[i * k + perm[c]] == 1 && truth[i * k + c] == 1 {
                    sc += 1;
                }
            }
        }
        if sc > best_score {
            best_score = sc;
            best = perm.clone();
        }
        if !next_perm(&mut perm) {
            break;
        }
    }
    let mut out = vec![0u8; n * k];
    for i in 0..n {
        for c in 0..k {
            out[i * k + c] = pred[i * k + best[c]];
        }
    }
    out
}

/// One Table 8 run in the current process. Returns
/// (runtime_s, peak_rss_mb, node_f1, jaccard, converged).
pub fn table8_single(n: usize, seed: u64) -> (f64, f64, f64, f64, bool) {
    let k = T8_K;
    let mut rng = Rng::new(seed);
    let pairs = random_pairs(&mut rng, n, 6 * n);
    let pi = vec![T8_PI; k];
    let lam = vec![T8_LAMBDA; k];
    let d: Data = model::generate(&mut rng, n, &pi, &lam, -2.2, 3.0, pairs);
    let cfg = alg1_cfg(35);
    let t0 = Instant::now();
    let res = fsdb::algorithm1(&d, &HYPER, &cfg);
    let s = fsdb::decide(&res.gamma, n, k, 0.80);
    let rt = t0.elapsed().as_secs_f64();
    let s = align(&s, &d.z, n, k);
    let (mut f1, mut jac) = (0.0, 0.0);
    for i in 0..n {
        let (mut inter, mut a, mut b) = (0.0, 0.0, 0.0);
        for c in 0..k {
            let (x, y) = (s[i * k + c] == 1, d.z[i * k + c] == 1);
            if x && y {
                inter += 1.0;
            }
            if x {
                a += 1.0;
            }
            if y {
                b += 1.0;
            }
        }
        f1 += 2.0 * inter / (a + b);
        jac += inter / (a + b - inter);
    }
    (rt, peak_rss_mb(), f1 / n as f64, jac / n as f64, res.converged)
}

pub fn table8(out: &Path) -> std::io::Result<()> {
    let exe = std::env::current_exe()?;
    let mut f = File::create(out.join("table8.csv"))?;
    writeln!(
        f,
        "n,P,runtime_s_mean,runtime_s_sd,peak_rss_mb_mean,peak_rss_mb_sd,node_f1_mean,node_f1_sd,jaccard_mean,jaccard_sd,converged_runs"
    )?;
    for n in [100usize, 500, 1000] {
        let mut cols = vec![Vec::new(); 4];
        let mut conv = 0;
        for seed in 260913u64..=260922 {
            // separate process so that peak RSS is per run
            let o = std::process::Command::new(&exe)
                .args(["table8-run", &n.to_string(), &seed.to_string()])
                .output()?;
            let line = String::from_utf8_lossy(&o.stdout);
            let v: Vec<&str> = line.trim().split(',').collect();
            for c in 0..4 {
                cols[c].push(v[c].parse::<f64>().unwrap());
            }
            if v[4] == "true" {
                conv += 1;
            }
        }
        let mut line = format!("{},{}", n, 6 * n);
        for c in 0..4 {
            let (m, s) = mean_sd(&cols[c]);
            line += &format!(",{},{}", e(m), e(s));
        }
        line += &format!(",{}/10", conv);
        writeln!(f, "{}", line)?;
    }
    Ok(())
}

