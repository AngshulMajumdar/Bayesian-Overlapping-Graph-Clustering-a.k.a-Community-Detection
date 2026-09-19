mod bethe;
mod fsdb;
mod micro;
mod model;
mod quad;
mod rng;
mod tables;

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 4 && args[1] == "table8-run" {
        let n: usize = args[2].parse().unwrap();
        let seed: u64 = args[3].parse().unwrap();
        let (rt, rss, f1, jac, conv) = tables::table8_single(n, seed);
        println!("{},{},{},{},{}", rt, rss, f1, jac, conv);
        return Ok(());
    }
    let out = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| "results".to_string()));
    std::fs::create_dir_all(&out)?;
    tables::tables_4_5(&out)?;
    eprintln!("tables 4, 5 done");
    tables::table6(&out)?;
    eprintln!("table 6 done");
    tables::table7(&out)?;
    eprintln!("table 7 done");
    tables::table8(&out)?;
    eprintln!("table 8 done");
    Ok(())
}
