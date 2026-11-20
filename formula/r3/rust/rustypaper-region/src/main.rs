use r3io::{to_top_left, Outcome, Region};
use rustypaper::backend::{self, Backend, PageSource};
use rustypaper::ir::{FontTable, PageRaw};
use rustypaper::math::{build, latex, MIN_CONFIDENCE};
use std::collections::HashMap;
use std::time::Instant;

struct Doc {
    path: String,
    backend: Result<Backend, String>,
    fonts: FontTable,
    pages: HashMap<usize, Result<PageRaw, String>>,
}

fn open(path: &str) -> Doc {
    let started = Instant::now();
    let backend = backend::open(path).map_err(|e| format!("open failed: {e}"));
    eprintln!(
        "rustypaper: opened {path} in {:.1} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    Doc {
        path: path.to_string(),
        backend,
        fonts: FontTable::new(),
        pages: HashMap::new(),
    }
}

fn region(doc: &mut Doc, r: &Region, pad: f64) -> Outcome {
    let backend = match &doc.backend {
        Ok(b) => b,
        Err(e) => return Outcome::error(e.clone()),
    };
    let index = r.page_index as usize;
    let count = backend.page_count();
    if index >= count {
        return Outcome::error(format!("page_index {index} out of range ({count} pages)"));
    }
    let fonts = &mut doc.fonts;
    let page = doc.pages.entry(index).or_insert_with(|| {
        let started = Instant::now();
        let p = backend.page(index, fonts).map_err(|e| format!("page extraction failed: {e}"));
        eprintln!(
            "rustypaper: extracted page {index} in {:.1} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
        p
    });
    let page = match page {
        Ok(p) => p,
        Err(e) => return Outcome::error(e.clone()),
    };
    if page.rotation != 0 {
        return Outcome::error(format!(
            "page has /Rotate {}; mediabox-based conversion to rustypaper page space is not defined",
            page.rotation
        ));
    }
    let mw = (r.mediabox[2] - r.mediabox[0]).abs() as f32;
    let mh = (r.mediabox[3] - r.mediabox[1]).abs() as f32;
    if (page.width - mw).abs() > 0.5 || (page.height - mh).abs() > 0.5 {
        eprintln!(
            "rustypaper: warning: {} page {index}: rustium page box {}x{} differs from mediabox {}x{} (crop box?)",
            doc.path, page.width, page.height, mw, mh
        );
    }

    let started = Instant::now();
    let rect = to_top_left(&r.bbox, &r.mediabox, pad);
    if std::env::var_os("R3_DEBUG").is_some() {
        let near = to_top_left(&r.bbox, &r.mediabox, pad + 6.0);
        let mut buf = [0u8; 4];
        for g in &page.glyphs {
            if near.contains_point(g.bbox.center_x(), g.bbox.center_y()) {
                let inside = rect.contains_point(g.bbox.center_x(), g.bbox.center_y());
                eprintln!(
                    "R3_DEBUG rustypaper {} {} {:?} bbox=[{:.2},{:.2},{:.2},{:.2}] origin_y={:.2} size={:.2} font={}",
                    r.id,
                    if inside { "IN " } else { "OUT" },
                    page.glyph_str(g, &mut buf),
                    g.bbox.x0, g.bbox.y0, g.bbox.x1, g.bbox.y1, g.origin.y, g.size,
                    doc.fonts.name(g.font)
                );
            }
        }
        for p in &page.paths {
            if near.contains_point(p.bbox.center_x(), p.bbox.center_y()) {
                eprintln!(
                    "R3_DEBUG rustypaper {} path {:?} bbox=[{:.2},{:.2},{:.2},{:.2}]",
                    r.id, p.kind, p.bbox.x0, p.bbox.y0, p.bbox.x1, p.bbox.y1
                );
            }
        }
    }
    let glyphs: Vec<_> = page
        .glyphs
        .iter()
        .filter(|g| rect.contains_point(g.bbox.center_x(), g.bbox.center_y()))
        .copied()
        .collect();
    let paths: Vec<_> = page
        .paths
        .iter()
        .filter(|p| rect.contains_point(p.bbox.center_x(), p.bbox.center_y()))
        .copied()
        .collect();
    if glyphs.is_empty() {
        return Outcome {
            latex: None,
            confidence: None,
            status: "no_math",
            error: None,
            ms: Some(started.elapsed().as_secs_f64() * 1000.0),
        };
    }
    let sub = PageRaw {
        index: page.index,
        width: page.width,
        height: page.height,
        rotation: page.rotation,
        glyphs,
        paths,
        images: Vec::new(),
        expansions: page.expansions.clone(),
    };
    let indices: Vec<usize> = (0..sub.glyphs.len()).collect();
    let formula = build::build(&sub, &indices);
    let tex = latex::render(&formula.root);
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    let confidence = Some(formula.confidence as f64);
    if tex.trim().is_empty() {
        return Outcome {
            latex: None,
            confidence,
            status: "no_math",
            error: None,
            ms: Some(ms),
        };
    }
    let status = if formula.confidence < MIN_CONFIDENCE {
        "picture"
    } else {
        "ok"
    };
    Outcome {
        latex: Some(tex),
        confidence,
        status,
        error: None,
        ms: Some(ms),
    }
}

fn main() {
    let mut current: Option<Doc> = None;
    let code = r3io::run("rustypaper", |r, pad| {
        if current.as_ref().map(|d| d.path.as_str()) != Some(r.pdf.as_str()) {
            current = None;
            current = Some(open(&r.pdf));
        }
        let doc = current.as_mut().expect("document just opened");
        region(doc, r, pad)
    });
    std::process::exit(code);
}
