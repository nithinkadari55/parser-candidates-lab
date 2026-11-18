use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::time::Instant;

type CalXlsx = calamine::Xlsx<BufReader<File>>;
type Out<'a> = &'a mut dyn Write;

fn col_letters(mut c: u32) -> String {
    let mut s = Vec::new();
    c += 1;
    while c > 0 {
        let r = ((c - 1) % 26) as u8;
        s.push(b'A' + r);
        c = (c - 1) / 26;
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

fn a1(row0: u32, col0: u32) -> String {
    format!("{}{}", col_letters(col0), row0 + 1)
}

fn range_str(r0: u32, c0: u32, r1: u32, c1: u32) -> String {
    format!("{}:{}", a1(r0, c0), a1(r1, c1))
}

struct Merges(Vec<(u32, u32, u32, u32)>);

impl Merges {
    fn find(&self, r: u32, c: u32) -> Option<String> {
        self.0
            .iter()
            .find(|m| r >= m.0 && r <= m.2 && c >= m.1 && c <= m.3)
            .map(|m| range_str(m.0, m.1, m.2, m.3))
    }
    fn list(&self) -> Vec<String> {
        self.0.iter().map(|m| range_str(m.0, m.1, m.2, m.3)).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn emit(
    out: Out,
    krate: &str,
    file: &str,
    sheet: &str,
    r: u32,
    c: u32,
    t: &str,
    v: Value,
    serial: Option<f64>,
    iso: Option<String>,
    fmt: Option<String>,
    merged: Option<String>,
    formula: Option<String>,
) {
    let rec = json!({
        "kind": "cell", "crate": krate, "file": file, "sheet": sheet,
        "addr": a1(r, c), "t": t, "v": v, "serial": serial, "iso": iso,
        "fmt": fmt.unwrap_or_else(|| "General".to_string()),
        "merged": merged, "formula": formula,
    });
    writeln!(out, "{rec}").unwrap();
}

fn emit_sheet(out: Out, krate: &str, file: &str, sheet: &str, merges: &[String], err: Option<String>, extra: Value) {
    let rec = json!({"kind": "sheet", "crate": krate, "file": file, "sheet": sheet, "merges": merges, "error": err, "extra": extra});
    writeln!(out, "{rec}").unwrap();
}

fn cal_value(d: &calamine::DataRef) -> Option<(&'static str, Value, Option<f64>, Option<String>)> {
    use calamine::DataRef as D;
    Some(match d {
        D::Int(i) => ("int", json!(i), None, None),
        D::Float(f) => ("float", json!(f), None, None),
        D::String(s) => ("string", json!(s), None, None),
        D::SharedString(s) => ("string", json!(s), None, None),
        D::Bool(b) => ("bool", json!(b), None, None),
        D::DateTime(dt) => {
            if dt.is_duration() {
                let iso = dt.as_duration().map(|x| format!("{}s", x.num_milliseconds() as f64 / 1000.0));
                ("duration", json!(dt.as_f64()), Some(dt.as_f64()), iso)
            } else {
                let iso = dt.as_datetime().map(|x| x.format("%Y-%m-%dT%H:%M:%S%.3f").to_string());
                ("datetime", json!(dt.as_f64()), Some(dt.as_f64()), iso)
            }
        }
        D::DateTimeIso(s) => ("datetime_iso", json!(s), None, Some(s.clone())),
        D::DurationIso(s) => ("duration_iso", json!(s), None, Some(s.clone())),
        D::Error(e) => ("error", json!(e.to_string()), None, None),
        D::Empty => return None,
        #[allow(unreachable_patterns)]
        _ => ("unknown", json!(format!("{d:?}")), None, None),
    })
}

fn cal_value_owned(d: &calamine::Data) -> Option<(&'static str, Value, Option<f64>, Option<String>)> {
    use calamine::Data as D;
    use calamine::DataRef as R;
    let r = match d {
        D::Int(i) => R::Int(*i),
        D::Float(f) => R::Float(*f),
        D::String(s) => R::String(s.clone()),
        D::Bool(b) => R::Bool(*b),
        D::DateTime(x) => R::DateTime(*x),
        D::DateTimeIso(s) => R::DateTimeIso(s.clone()),
        D::DurationIso(s) => R::DurationIso(s.clone()),
        D::Error(e) => R::Error(e.clone()),
        D::Empty => return None,
        #[allow(unreachable_patterns)]
        _ => return Some(("unknown", json!(format!("{d:?}")), None, None)),
    };
    cal_value(&r)
}

fn calamine_prep(wb: &mut CalXlsx, name: &str) -> Result<(Merges, HashMap<(u32, u32), String>), String> {
    let merges = wb
        .merge_cells_by_sheet_name(name)
        .map_err(|e| format!("merge: {e}"))?
        .iter()
        .map(|d| (d.start.0, d.start.1, d.end.0, d.end.1))
        .collect();
    let mut formulas = HashMap::new();
    use calamine::Reader;
    match wb.worksheet_formula(name) {
        Ok(fr) => {
            if let Some((sr, sc)) = fr.start() {
                for (r, c, f) in fr.used_cells() {
                    if !f.is_empty() {
                        formulas.insert((sr + r as u32, sc + c as u32), f.clone());
                    }
                }
            }
        }
        Err(e) => return Err(format!("formula: {e}")),
    }
    Ok((Merges(merges), formulas))
}

fn dump_calamine_stream(path: &Path, file: &str, out: Out) -> Result<(), String> {
    use calamine::Reader;
    let k = "calamine_stream";
    let mut wb: CalXlsx = calamine::open_workbook::<CalXlsx, _>(path).map_err(|e| e.to_string())?;
    let metas: Vec<(String, String)> = wb.sheets_metadata().iter().map(|s| (s.name.clone(), format!("{:?}", s.typ))).collect();
    let is_1904 = wb.has_1904_epoch();
    for (name, typ) in metas {
        let (merges, formulas) = match calamine_prep(&mut wb, &name) {
            Ok(x) => x,
            Err(e) => {
                emit_sheet(out, k, file, &name, &[], Some(e), json!({"type": typ, "is_1904": is_1904}));
                continue;
            }
        };
        let mut reader = match wb.worksheet_cells_reader(&name) {
            Ok(r) => r,
            Err(e) => {
                emit_sheet(out, k, file, &name, &merges.list(), Some(e.to_string()), json!({"type": typ, "is_1904": is_1904}));
                continue;
            }
        };
        let mut err = None;
        loop {
            match reader.next_cell_with_style_id() {
                Ok(Some((cell, sid))) => {
                    let (r, c) = cell.get_position();
                    let Some((t, v, serial, iso)) = cal_value(cell.get_value()) else { continue };
                    let fmt = reader.styles().get(sid).and_then(|s| s.number_format.as_ref()).map(|n| n.format_code.clone());
                    emit(out, k, file, &name, r, c, t, v, serial, iso, fmt, merges.find(r, c), formulas.get(&(r, c)).cloned());
                }
                Ok(None) => break,
                Err(e) => {
                    err = Some(e.to_string());
                    break;
                }
            }
        }
        drop(reader);
        emit_sheet(out, k, file, &name, &merges.list(), err, json!({"type": typ, "is_1904": is_1904}));
    }
    Ok(())
}

fn dump_calamine_range(path: &Path, file: &str, out: Out) -> Result<(), String> {
    use calamine::Reader;
    let k = "calamine_range";
    let mut wb: CalXlsx = calamine::open_workbook::<CalXlsx, _>(path).map_err(|e| e.to_string())?;
    let names = wb.sheet_names();
    for name in names {
        let (merges, formulas) = match calamine_prep(&mut wb, &name) {
            Ok(x) => x,
            Err(e) => {
                emit_sheet(out, k, file, &name, &[], Some(e), json!({}));
                continue;
            }
        };
        let range = match wb.worksheet_range(&name) {
            Ok(r) => r,
            Err(e) => {
                emit_sheet(out, k, file, &name, &merges.list(), Some(e.to_string()), json!({}));
                continue;
            }
        };
        let styles = match wb.worksheet_style(&name) {
            Ok(s) => s,
            Err(e) => {
                emit_sheet(out, k, file, &name, &merges.list(), Some(format!("style: {e}")), json!({}));
                continue;
            }
        };
        if let Some((sr, sc)) = range.start() {
            for (rr, cc, d) in range.used_cells() {
                let (r, c) = (sr + rr as u32, sc + cc as u32);
                let Some((t, v, serial, iso)) = cal_value_owned(d) else { continue };
                let fmt = styles.get((r, c)).and_then(|s| s.number_format.as_ref()).map(|n| n.format_code.clone());
                emit(out, k, file, &name, r, c, t, v, serial, iso, fmt, merges.find(r, c), formulas.get(&(r, c)).cloned());
            }
        }
        emit_sheet(out, k, file, &name, &merges.list(), None, json!({}));
    }
    Ok(())
}

fn parse_a1_range(s: &str) -> Option<(u32, u32, u32, u32)> {
    fn cell(x: &str) -> Option<(u32, u32)> {
        let x = x.replace('$', "");
        let letters: String = x.chars().take_while(|ch| ch.is_ascii_alphabetic()).collect();
        let digits = &x[letters.len()..];
        let mut col = 0u32;
        for ch in letters.to_ascii_uppercase().bytes() {
            col = col * 26 + (ch - b'A' + 1) as u32;
        }
        let row: u32 = digits.parse().ok()?;
        Some((row - 1, col - 1))
    }
    let mut it = s.split(':');
    let a = cell(it.next()?)?;
    let b = match it.next() {
        Some(x) => cell(x)?,
        None => a,
    };
    Some((a.0, a.1, b.0, b.1))
}

fn dump_umya(path: &Path, file: &str, out: Out) -> Result<(), String> {
    let k = "umya";
    let book = umya_spreadsheet::reader::xlsx::read(path).map_err(|e| e.to_string())?;
    let date1904 = book.date1904();
    for ws in book.sheet_collection() {
        let name = ws.name().to_string();
        let merges = Merges(ws.merge_cells().iter().filter_map(|r| parse_a1_range(&r.range())).collect());
        let mut cells = ws.cells();
        cells.sort_by_key(|c| (c.coordinate().row_num(), c.coordinate().col_num()));
        for cell in cells {
            let r = cell.coordinate().row_num() - 1;
            let c = cell.coordinate().col_num() - 1;
            use umya_spreadsheet::CellRawValue as V;
            let (t, v) = match cell.raw_value() {
                V::String(s) => ("string", json!(s.to_string())),
                V::RichText(rt) => ("string", json!(rt.text())),
                V::Lazy(s) => ("string", json!(s.to_string())),
                V::Numeric(f) => ("float", json!(f)),
                V::Bool(b) => ("bool", json!(b)),
                V::Error(e) => ("error", json!(e.to_string())),
                V::Empty => {
                    if cell.is_formula() {
                        ("empty_formula", Value::Null)
                    } else {
                        continue;
                    }
                }
            };
            let fmt = cell.style().number_format().map(|n| n.format_code().to_string());
            let formula = if cell.is_formula() { Some(cell.formula().to_string()) } else { None };
            emit(out, k, file, &name, r, c, t, v, None, None, fmt, merges.find(r, c), formula);
        }
        emit_sheet(out, k, file, &name, &merges.list(), None, json!({"date1904": date1904}));
    }
    Ok(())
}

fn rxls_value(cell: &rxls::Cell, date1904: bool) -> Option<(&'static str, Value, Option<f64>, Option<String>)> {
    use rxls::Cell as C;
    Some(match cell {
        C::Text(s) => ("string", json!(s), None, None),
        C::Number(f) => ("float", json!(f), None, None),
        C::Date(f) => {
            let iso = rxls::excel_serial_to_naive_datetime(*f, date1904).map(|x| x.format("%Y-%m-%dT%H:%M:%S%.3f").to_string());
            ("datetime", json!(f), Some(*f), iso)
        }
        C::Bool(b) => ("bool", json!(b), None, None),
        C::Error(e) => ("error", json!(e), None, None),
        C::Formula { cached, .. } => return rxls_value(cached, date1904),
        #[allow(unreachable_patterns)]
        _ => ("unknown", json!(format!("{cell:?}")), None, None),
    })
}

fn dump_rxls(path: &Path, file: &str, out: Out) -> Result<(), String> {
    let k = "rxls";
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let wb = rxls::Workbook::open(&bytes).map_err(|e| e.to_string())?;
    let d1904 = wb.date1904;
    for sheet in &wb.sheets {
        let name = sheet.name.clone();
        let merges = Merges(sheet.merged_ranges().iter().map(|m| (m.0, m.1 as u32, m.2, m.3 as u32)).collect());
        for dc in sheet.display_cells() {
            let (r, c) = (dc.row, dc.col as u32);
            let formula = match dc.value {
                rxls::Cell::Formula { formula, .. } => Some(formula.clone()),
                _ => None,
            };
            let Some((t, v, serial, iso)) = rxls_value(dc.value, d1904) else { continue };
            let fmt = sheet.cell_style(dc.row, dc.col).and_then(|s| s.num_fmt.clone());
            emit(out, k, file, &name, r, c, t, v, serial, iso, fmt, merges.find(r, c), formula);
        }
        emit_sheet(out, k, file, &name, &merges.list(), None, json!({"is_worksheet": sheet.is_worksheet, "date1904": d1904, "text_truncated": wb.text_truncated}));
    }
    Ok(())
}

type DumpFn = fn(&Path, &str, Out) -> Result<(), String>;

const CRATES: [(&str, DumpFn); 4] = [
    ("calamine_stream", dump_calamine_stream),
    ("calamine_range", dump_calamine_range),
    ("umya", dump_umya),
    ("rxls", dump_rxls),
];

fn run_guarded(f: DumpFn, path: &Path, file: &str, out: Out) -> Result<(), String> {
    match catch_unwind(AssertUnwindSafe(|| f(path, file, out))) {
        Ok(r) => r,
        Err(p) => Err(format!(
            "PANIC: {}",
            p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
        )),
    }
}

fn cmd_dump(outdir: &str, files: &[String]) {
    std::fs::create_dir_all(outdir).unwrap();
    for (k, f) in CRATES {
        let mut w = BufWriter::new(File::create(Path::new(outdir).join(format!("{k}.jsonl"))).unwrap());
        for p in files {
            let path = Path::new(p);
            let file = path.file_name().unwrap().to_string_lossy().to_string();
            let t0 = Instant::now();
            let res = run_guarded(f, path, &file, &mut w);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let rec = json!({"kind": "file", "crate": k, "file": file, "ok": res.is_ok(), "error": res.err(), "ms": ms});
            println!("{rec}");
            writeln!(w, "{rec}").unwrap();
        }
    }
}

fn count_calamine_stream(path: &Path) -> Result<usize, String> {
    use calamine::Reader;
    let mut wb: CalXlsx = calamine::open_workbook::<CalXlsx, _>(path).map_err(|e| e.to_string())?;
    let mut n = 0;
    for name in wb.sheet_names() {
        let mut reader = wb.worksheet_cells_reader(&name).map_err(|e| e.to_string())?;
        while let Some((cell, sid)) = reader.next_cell_with_style_id().map_err(|e| e.to_string())? {
            if cal_value(cell.get_value()).is_some() {
                let f = reader.styles().get(sid).and_then(|s| s.number_format.as_ref()).map(|x| x.format_code.len()).unwrap_or(0);
                std::hint::black_box(f);
                n += 1;
            }
        }
    }
    Ok(n)
}

fn count_calamine_range(path: &Path) -> Result<usize, String> {
    use calamine::Reader;
    let mut wb: CalXlsx = calamine::open_workbook::<CalXlsx, _>(path).map_err(|e| e.to_string())?;
    let mut n = 0;
    for name in wb.sheet_names() {
        let range = wb.worksheet_range(&name).map_err(|e| e.to_string())?;
        let styles = wb.worksheet_style(&name).map_err(|e| e.to_string())?;
        let (sr, sc) = range.start().unwrap_or((0, 0));
        for (r, c, d) in range.used_cells() {
            if cal_value_owned(d).is_some() {
                let f = styles.get((sr + r as u32, sc + c as u32)).and_then(|s| s.number_format.as_ref()).map(|x| x.format_code.len()).unwrap_or(0);
                std::hint::black_box(f);
                n += 1;
            }
        }
    }
    Ok(n)
}

fn count_calamine_values_only(path: &Path) -> Result<usize, String> {
    use calamine::Reader;
    let mut wb: CalXlsx = calamine::open_workbook::<CalXlsx, _>(path).map_err(|e| e.to_string())?;
    let mut n = 0;
    for name in wb.sheet_names() {
        let range = wb.worksheet_range(&name).map_err(|e| e.to_string())?;
        n += range.used_cells().count();
    }
    Ok(n)
}

fn count_umya(path: &Path) -> Result<usize, String> {
    let book = umya_spreadsheet::reader::xlsx::read(path).map_err(|e| e.to_string())?;
    let mut n = 0;
    for ws in book.sheet_collection() {
        for cell in ws.cells() {
            if !matches!(cell.raw_value(), umya_spreadsheet::CellRawValue::Empty) {
                let f = cell.style().number_format().map(|x| x.format_code().len()).unwrap_or(0);
                std::hint::black_box(f);
                n += 1;
            }
        }
    }
    Ok(n)
}

fn count_rxls(path: &Path) -> Result<usize, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let wb = rxls::Workbook::open(&bytes).map_err(|e| e.to_string())?;
    let mut n = 0;
    for sheet in &wb.sheets {
        for dc in sheet.display_cells() {
            let f = sheet.cell_style(dc.row, dc.col).and_then(|s| s.num_fmt.as_ref()).map(|x| x.len()).unwrap_or(0);
            std::hint::black_box(f);
            n += 1;
        }
    }
    Ok(n)
}

type CountFn = fn(&Path) -> Result<usize, String>;

fn cmd_bench(file: &str, reps: usize) {
    let path = Path::new(file);
    let fname = path.file_name().unwrap().to_string_lossy().to_string();
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let benches: [(&str, CountFn); 5] = [
        ("calamine_values_only", count_calamine_values_only),
        ("calamine_stream", count_calamine_stream),
        ("calamine_range", count_calamine_range),
        ("umya", count_umya),
        ("rxls", count_rxls),
    ];
    for (k, f) in benches {
        let mut times = Vec::new();
        let mut cells = 0;
        let mut err = None;
        for _ in 0..reps {
            let t0 = Instant::now();
            let r = catch_unwind(AssertUnwindSafe(|| f(path))).unwrap_or_else(|_| Err("PANIC".into()));
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            match r {
                Ok(n) => {
                    cells = n;
                    times.push(ms);
                }
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rec = json!({"kind": "bench", "crate": k, "file": fname, "bytes": size, "cells": cells,
            "min_ms": times.first(), "median_ms": times.get(times.len() / 2), "runs": times, "error": err,
            "os": std::env::consts::OS});
        println!("BENCH {rec}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("dump") => cmd_dump(&args[2], &args[3..]),
        Some("bench") => {
            let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
            cmd_bench(&args[2], reps)
        }
        _ => {
            eprintln!("usage: excel-probe dump <outdir> <files...> | bench <file> [reps]");
            std::process::exit(2);
        }
    }
}
