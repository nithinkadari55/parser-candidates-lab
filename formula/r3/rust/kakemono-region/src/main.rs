mod kakemono_latex;

use kakemono::analysis::lines::{join_math_letters, take_ruby};
use kakemono::analysis::math::{formula_from_glyphs, MathConfig};
use kakemono::analysis::page::PageData;
use kakemono::backend::{FontInfo, PdfFile, RawPage};
use kakemono::common::geom::Rect;
use kakemono::config::LayoutConfig;
use kakemono::model::LineGlyph;
use r3io::{to_top_left, Outcome, Region};
use std::collections::HashMap;
use std::time::Instant;

struct Doc {
    path: String,
    loaded: Result<(Vec<Option<RawPage>>, Vec<FontInfo>), String>,
    pages: HashMap<usize, PageData>,
}

fn open(path: &str) -> Doc {
    let started = Instant::now();
    let loaded = std::fs::read(path)
        .map_err(|e| format!("open failed: {e}"))
        .and_then(|data| PdfFile::open(data).map_err(|e| format!("open failed: {e}")))
        .map(|pdf| {
            let (raw, fonts) = pdf.extract(0);
            (raw.into_iter().map(Some).collect::<Vec<_>>(), fonts)
        });
    eprintln!(
        "kakemono: opened and extracted {path} in {:.1} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    Doc {
        path: path.to_string(),
        loaded,
        pages: HashMap::new(),
    }
}

fn has_drawing(page: &PageData, region: &Rect, cfg: &MathConfig) -> bool {
    let rule = |bbox: &Rect| bbox.height() <= cfg.rule_max_thickness_pt && bbox.width() >= cfg.rule_min_width_pt;
    let drawn = |bbox: &Rect| bbox.intersects(region) && !bbox.contains(region);
    page.paths
        .iter()
        .filter(|path| !path.is_white())
        .any(|path| drawn(&path.bbox) && !rule(&path.bbox))
        || page.images.iter().any(|image| drawn(&image.bbox))
}

fn region(doc: &mut Doc, r: &Region, pad: f64, layout: &LayoutConfig) -> Outcome {
    let (raw, fonts) = match &mut doc.loaded {
        Ok(l) => l,
        Err(e) => return Outcome::error(e.clone()),
    };
    let index = r.page_index as usize;
    if index >= raw.len() {
        return Outcome::error(format!("page_index {index} out of range ({} pages)", raw.len()));
    }
    if !doc.pages.contains_key(&index) {
        let Some(raw_page) = raw[index].take() else {
            return Outcome::error("page data unavailable");
        };
        let mut page = PageData::from_raw(raw_page, &layout.lines);
        join_math_letters(&mut page.lines, fonts, &layout.lines);
        let _ = take_ruby(&mut page.lines, &layout.lines);
        doc.pages.insert(index, page);
    }
    let page = &doc.pages[&index];
    let mw = (r.mediabox[2] - r.mediabox[0]).abs() as f32;
    let mh = (r.mediabox[3] - r.mediabox[1]).abs() as f32;
    if (page.width - mw).abs() > 0.5 || (page.height - mh).abs() > 0.5 {
        eprintln!(
            "kakemono: warning: {} page {index}: hayro page box {}x{} differs from mediabox {}x{} (crop box or /Rotate?)",
            doc.path, page.width, page.height, mw, mh
        );
    }

    let started = Instant::now();
    let d = to_top_left(&r.bbox, &r.mediabox, pad);
    let rect = Rect::new(d.x0, d.y0, d.x1, d.y1);
    let cfg = &layout.math;
    if std::env::var_os("R3_DEBUG").is_some() {
        let n = to_top_left(&r.bbox, &r.mediabox, pad + 6.0);
        for g in page.lines.iter().flat_map(|line| line.glyphs.iter()) {
            if n.selects(g.ink.x0, g.ink.y0, g.ink.x1, g.ink.y1, g.ink.cx(), g.y) {
                let inside = d.selects(g.ink.x0, g.ink.y0, g.ink.x1, g.ink.y1, g.ink.cx(), g.y);
                let font = fonts.get(g.font as usize).map(|f| f.name.as_str()).unwrap_or("?");
                eprintln!(
                    "R3_DEBUG kakemono {} {} {:?} ink=[{:.2},{:.2},{:.2},{:.2}] y={:.2} size={:.2} font={}",
                    r.id, if inside { "IN " } else { "OUT" }, g.text, g.ink.x0, g.ink.y0, g.ink.x1, g.ink.y1, g.y, g.size, font
                );
            }
        }
        for p in &page.paths {
            if n.contains_point(p.bbox.cx(), p.bbox.cy()) {
                eprintln!(
                    "R3_DEBUG kakemono {} path white={} bbox=[{:.2},{:.2},{:.2},{:.2}]",
                    r.id, p.is_white(), p.bbox.x0, p.bbox.y0, p.bbox.x1, p.bbox.y1
                );
            }
        }
    }
    let glyphs: Vec<LineGlyph> = page
        .lines
        .iter()
        .flat_map(|line| line.glyphs.iter())
        .filter(|g| d.selects(g.ink.x0, g.ink.y0, g.ink.x1, g.ink.y1, g.ink.cx(), g.y))
        .cloned()
        .collect();
    let elapsed = |s: &Instant| Some(s.elapsed().as_secs_f64() * 1000.0);
    if glyphs.is_empty() {
        return Outcome {
            latex: None,
            confidence: None,
            status: "no_math",
            error: None,
            ms: elapsed(&started),
        };
    }
    if has_drawing(page, &rect, cfg) {
        return Outcome {
            latex: None,
            confidence: None,
            status: "picture",
            error: None,
            ms: elapsed(&started),
        };
    }
    let rules: Vec<Rect> = page
        .paths
        .iter()
        .filter(|path| !path.is_white())
        .map(|path| path.bbox)
        .filter(|b| b.height() <= cfg.rule_max_thickness_pt && b.width() >= cfg.rule_min_width_pt)
        .filter(|b| d.contains_point(b.cx(), b.cy()))
        .collect();
    let node = formula_from_glyphs(glyphs, &rules, fonts);
    let latex = node.as_ref().map(kakemono_latex::latex);
    let ms = elapsed(&started);
    match latex {
        Some(tex) if !tex.trim().is_empty() => Outcome {
            latex: Some(tex),
            confidence: None,
            status: "ok",
            error: None,
            ms,
        },
        Some(_) => Outcome {
            latex: None,
            confidence: None,
            status: "no_math",
            error: None,
            ms,
        },
        None => Outcome {
            latex: None,
            confidence: None,
            status: "picture",
            error: None,
            ms,
        },
    }
}

fn main() {
    let layout = LayoutConfig::default();
    let mut current: Option<Doc> = None;
    let code = r3io::run("kakemono", |r, pad| {
        if current.as_ref().map(|d| d.path.as_str()) != Some(r.pdf.as_str()) {
            current = None;
            current = Some(open(&r.pdf));
        }
        let doc = current.as_mut().expect("document just opened");
        region(doc, r, pad, &layout)
    });
    std::process::exit(code);
}
