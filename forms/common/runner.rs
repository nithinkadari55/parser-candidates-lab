#![allow(dead_code)]
use serde_json::{json, Value};
use std::time::Instant;

pub type Extract = fn(&str) -> Result<Vec<Value>, String>;

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(s) = e.downcast_ref::<&str>() {
        return s.to_string();
    }
    String::from("panic")
}

pub fn rect(r: [f64; 4]) -> Value {
    let q = |v: f64| (v * 100.0).round() / 100.0;
    json!([q(r[0].min(r[2])), q(r[1].min(r[3])), q(r[0].max(r[2])), q(r[1].max(r[3]))])
}

pub fn widget(page: Option<u32>, r: Option<[f64; 4]>, as_state: Option<String>) -> Value {
    json!({"page": page, "rect": r.map(rect), "as": as_state})
}

pub fn run(tool: &str, version: &str, f: Extract) {
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let files: Vec<String> = std::env::args().skip(1).collect();
    let mut results = Vec::new();
    for p in &files {
        let name = std::path::Path::new(p).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let t0 = Instant::now();
        let r = std::panic::catch_unwind(|| f(p));
        let first_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let entry = match r {
            Err(e) => json!({"file": name, "status": "panic", "error": panic_msg(e), "ms_first": first_ms, "fields": []}),
            Ok(Err(e)) => json!({"file": name, "status": "error", "error": e, "ms_first": first_ms, "fields": []}),
            Ok(Ok(fields)) => {
                let mut times = Vec::new();
                for _ in 0..iters {
                    let t = Instant::now();
                    let _ = std::panic::catch_unwind(|| f(p));
                    times.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = if times.is_empty() { first_ms } else { times[times.len() / 2] };
                json!({"file": name, "status": "ok", "ms_first": first_ms, "ms_median": med, "fields": fields})
            }
        };
        results.push(entry);
    }
    let out = json!({"tool": tool, "version": version, "os": std::env::consts::OS, "results": results});
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
