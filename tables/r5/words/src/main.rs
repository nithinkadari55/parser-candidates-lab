//! usage: r5words JOBS.jsonl OUT_DIR
//! Each job line: {"set", "id", "pdf", "page", "crop_pt": [l, t, r, b]} (page points, top-left
//! origin of the rendered area). Writes OUT_DIR/<set>/<id>.json with docling.rs's word cells
//! inside the crop (centre inside, boxes in crop pixels at the page render scale) and
//! OUT_DIR/<set>/<id>.png, the crop of docling.rs's own page render.
use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use docling_pdf::pdfium_backend::{walk_pages, BitmapRender};
use docling_pdf::{PdfError, TextOptions};
use serde_json::{json, Value};

struct Page {
    scale: f32,
    width: f32,
    height: f32,
    words: Vec<(String, [f32; 4])>,
    image: image::RgbImage,
}

fn load(path: &str, page: usize) -> Result<Page, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut out = None;
    walk_pages::<PdfError, _>(
        &bytes,
        None,
        true,
        true,
        Some((page, page)),
        BitmapRender::OnDemand,
        TextOptions::default(),
        |_, _, mut p| {
            p.resolve_image()?;
            let words = p
                .word_cells
                .iter()
                .map(|c| (c.text.clone(), [c.l, c.t, c.r, c.b]))
                .collect();
            out = Some(Page {
                scale: p.scale,
                width: p.width,
                height: p.height,
                words,
                image: p.image.clone(),
            });
            Ok(())
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    out.ok_or_else(|| "page not found".to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jobs = std::fs::File::open(&args[1]).expect("jobs file");
    let out_dir = std::path::PathBuf::from(&args[2]);
    let mut by_pdf: BTreeMap<(String, usize), Vec<Value>> = BTreeMap::new();
    for line in std::io::BufReader::new(jobs).lines() {
        let line = line.expect("line");
        if line.trim().is_empty() {
            continue;
        }
        let job: Value = serde_json::from_str(&line).expect("json");
        let pdf = job["pdf"].as_str().unwrap().to_string();
        let page = job["page"].as_u64().unwrap_or(0) as usize;
        by_pdf.entry((pdf, page)).or_default().push(job);
    }
    let (mut ok, mut failed) = (0usize, 0usize);
    for ((pdf, page), jobs) in by_pdf {
        let t0 = std::time::Instant::now();
        let loaded = load(&pdf, page);
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / jobs.len() as f64;
        for job in jobs {
            let set = job["set"].as_str().unwrap();
            let id = job["id"].as_str().unwrap();
            let dir = out_dir.join(set);
            std::fs::create_dir_all(&dir).unwrap();
            let page = match &loaded {
                Ok(p) => p,
                Err(e) => {
                    failed += 1;
                    let rec = json!({"id": id, "error": e});
                    std::fs::write(dir.join(format!("{id}.json")), rec.to_string()).unwrap();
                    continue;
                }
            };
            let c: Vec<f32> = job["crop_pt"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let (l, t, r, b) = (c[0], c[1], c[2], c[3]);
            let s = page.scale;
            let (iw, ih) = page.image.dimensions();
            let px = |v: f32, max: u32| ((v * s).round_ties_even().max(0.0) as u32).min(max);
            let (x0, y0, x1, y1) = (px(l, iw), px(t, ih), px(r, iw), px(b, ih));
            let words: Vec<Value> = page
                .words
                .iter()
                .filter(|(text, _)| !text.trim().is_empty())
                .filter(|(_, w)| {
                    let (cx, cy) = ((w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0);
                    l <= cx && cx <= r && t <= cy && cy <= b
                })
                .map(|(text, w)| {
                    let bbox = [(w[0] - l) * s, (w[1] - t) * s, (w[2] - l) * s, (w[3] - t) * s];
                    json!({"text": text, "bbox": bbox})
                })
                .collect();
            if x1 <= x0 || y1 <= y0 {
                failed += 1;
                let rec = json!({"id": id, "error": "empty crop"});
                std::fs::write(dir.join(format!("{id}.json")), rec.to_string()).unwrap();
                continue;
            }
            let crop = image::imageops::crop_imm(&page.image, x0, y0, x1 - x0, y1 - y0).to_image();
            crop.save(dir.join(format!("{id}.png"))).unwrap();
            let rec = json!({
                "id": id,
                "words": words,
                "crop_px": [x0, y0, x1, y1],
                "scale": s,
                "page_pt": [page.width, page.height],
                "image_px": [iw, ih],
                "ms_page": ms,
                "words_src": "docling-pdf 2.84.0 PdfPage::word_cells (walk_pages, TextOptions::default)",
            });
            std::fs::write(dir.join(format!("{id}.json")), rec.to_string()).unwrap();
            ok += 1;
        }
    }
    println!("r5words ok {ok} failed {failed}");
    std::io::stdout().flush().ok();
}
