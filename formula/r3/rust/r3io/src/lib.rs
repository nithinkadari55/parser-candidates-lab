use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

#[derive(Debug, Clone, Deserialize)]
pub struct Region {
    pub id: String,
    pub pdf: String,
    pub page_index: i64,
    pub bbox: [f64; 4],
    pub mediabox: [f64; 4],
}

#[derive(Debug, Serialize)]
struct OutputLine<'a> {
    id: &'a str,
    tool: &'a str,
    latex: Option<String>,
    confidence: Option<f64>,
    status: &'a str,
    error: Option<String>,
    ms: f64,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub latex: Option<String>,
    pub confidence: Option<f64>,
    pub status: &'static str,
    pub error: Option<String>,
    pub ms: Option<f64>,
}

impl Outcome {
    pub fn error(msg: impl Into<String>) -> Self {
        Outcome {
            latex: None,
            confidence: None,
            status: "error",
            error: Some(msg.into()),
            ms: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DownRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl DownRect {
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    pub fn overlaps(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
        x0 < self.x1 && self.x0 < x1 && y0 < self.y1 && self.y0 < y1
    }

    pub fn selects(&self, bx0: f32, by0: f32, bx1: f32, by1: f32, ox: f32, oy: f32) -> bool {
        self.contains_point((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)
            || (self.contains_point(ox, oy) && self.overlaps(bx0, by0, bx1, by1))
    }
}

pub fn to_top_left(bbox: &[f64; 4], mediabox: &[f64; 4], pad: f64) -> DownRect {
    let mx0 = mediabox[0].min(mediabox[2]);
    let my1 = mediabox[1].max(mediabox[3]);
    let bx0 = bbox[0].min(bbox[2]);
    let bx1 = bbox[0].max(bbox[2]);
    let by0 = bbox[1].min(bbox[3]);
    let by1 = bbox[1].max(bbox[3]);
    DownRect {
        x0: (bx0 - mx0 - pad) as f32,
        y0: (my1 - by1 - pad) as f32,
        x1: (bx1 - mx0 + pad) as f32,
        y1: (my1 - by0 + pad) as f32,
    }
}

pub struct Args {
    pub input: String,
    pub output: String,
    pub pad: f64,
}

pub fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut pad = 0.0f64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next(),
            "--output" => output = it.next(),
            "--pad" => {
                pad = it
                    .next()
                    .ok_or("--pad needs a value")?
                    .parse::<f64>()
                    .map_err(|e| format!("--pad: {e}"))?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
        pad,
    })
}

pub fn validate(region: &Region) -> Result<(), String> {
    if region.page_index < 0 {
        return Err(format!("negative page_index {}", region.page_index));
    }
    if region.bbox.iter().chain(region.mediabox.iter()).any(|v| !v.is_finite()) {
        return Err("non-finite bbox or mediabox".into());
    }
    Ok(())
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

pub fn run<F>(tool: &'static str, mut handle: F) -> i32
where
    F: FnMut(&Region, f64) -> Outcome,
{
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\nusage: {tool}-region --input regions.jsonl --output out.jsonl [--pad PT]");
            return 2;
        }
    };
    let reader = match File::open(&args.input) {
        Ok(f) => BufReader::new(f),
        Err(e) => {
            eprintln!("cannot open {}: {e}", args.input);
            return 1;
        }
    };
    let mut writer = match File::create(&args.output) {
        Ok(f) => BufWriter::new(f),
        Err(e) => {
            eprintln!("cannot create {}: {e}", args.output);
            return 1;
        }
    };
    let (mut total, mut ok) = (0usize, 0usize);
    for (n, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("read error at line {}: {e}", n + 1);
                return 1;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        let started = Instant::now();
        let (id, outcome) = match serde_json::from_str::<Region>(&line) {
            Ok(region) => {
                let outcome = match validate(&region) {
                    Err(e) => Outcome::error(e),
                    Ok(()) => match catch_unwind(AssertUnwindSafe(|| handle(&region, args.pad))) {
                        Ok(o) => o,
                        Err(p) => Outcome::error(format!("panic: {}", panic_text(p.as_ref()))),
                    },
                };
                (region.id, outcome)
            }
            Err(e) => {
                let id = serde_json::from_str::<serde_json::Value>(&line)
                    .ok()
                    .and_then(|v| v.get("id").and_then(|i| i.as_str()).map(str::to_string))
                    .unwrap_or_else(|| format!("line:{}", n + 1));
                (id, Outcome::error(format!("bad input line: {e}")))
            }
        };
        let wall = started.elapsed().as_secs_f64() * 1000.0;
        let ms = outcome.ms.unwrap_or(wall);
        if outcome.status == "ok" {
            ok += 1;
        }
        let out = OutputLine {
            id: &id,
            tool,
            latex: outcome.latex,
            confidence: outcome.confidence,
            status: outcome.status,
            error: outcome.error,
            ms: (ms * 1000.0).round() / 1000.0,
        };
        let text = serde_json::to_string(&out).expect("serialisable");
        if writeln!(writer, "{text}").and_then(|_| writer.flush()).is_err() {
            eprintln!("write error on {}", args.output);
            return 1;
        }
    }
    eprintln!("{tool}: {total} regions, {ok} ok");
    0
}
