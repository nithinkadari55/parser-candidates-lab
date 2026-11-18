#[path = "../../../common/runner.rs"]
mod runner;

use pdfium_render::prelude::*;
use serde_json::{json, Value};
use std::path::Path;

thread_local! {
    static PDFIUM: Pdfium = {
        let dir = std::env::var("PDFIUM_LIB_DIR").unwrap_or_else(|_| ".".to_string());
        Pdfium::new(Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(&dir)).expect("bind pdfium"))
    };
}

fn extract(path: &str) -> Result<Vec<Value>, String> {
    PDFIUM.with(|pdfium| {
        let doc = pdfium.load_pdf_from_file(path, None).map_err(|e| format!("load: {e:?}"))?;
        let mut out: Vec<Value> = Vec::new();
        for (i, page) in doc.pages().iter().enumerate() {
            for annot in page.annotations().iter() {
                let Some(field) = annot.as_form_field() else { continue };
                let name = field.name().unwrap_or_default();
                let (ty, value, checked) = match field.field_type() {
                    PdfFormFieldType::Text => ("Tx", field.as_text_field().and_then(|f| f.value()).map(Value::String).unwrap_or(Value::Null), None),
                    PdfFormFieldType::Checkbox => {
                        let f = field.as_checkbox_field().unwrap();
                        ("Btn", f.group_value().map(Value::String).unwrap_or(Value::Null), f.is_checked().ok())
                    }
                    PdfFormFieldType::RadioButton => {
                        let f = field.as_radio_button_field().unwrap();
                        ("Btn", f.group_value().map(Value::String).unwrap_or(Value::Null), f.is_checked().ok())
                    }
                    PdfFormFieldType::ComboBox => ("Ch", field.as_combo_box_field().and_then(|f| f.value()).map(Value::String).unwrap_or(Value::Null), None),
                    PdfFormFieldType::ListBox => ("Ch", field.as_list_box_field().and_then(|f| f.value()).map(Value::String).unwrap_or(Value::Null), None),
                    PdfFormFieldType::PushButton => ("Btn", Value::Null, None),
                    PdfFormFieldType::Signature => ("Sig", Value::Null, None),
                    PdfFormFieldType::Unknown => ("", Value::Null, None),
                };
                let rect = annot.bounds().ok().map(|r| [r.left().value as f64, r.bottom().value as f64, r.right().value as f64, r.top().value as f64]);
                let mut w = runner::widget(Some(i as u32 + 1), rect, field.appearance_stream());
                w["checked"] = json!(checked);
                if let Some(existing) = out.iter_mut().find(|x| x["name"] == json!(name)) {
                    existing["widgets"].as_array_mut().unwrap().push(w);
                    if existing["value"].is_null() {
                        existing["value"] = value;
                    }
                } else {
                    out.push(json!({"name": name, "type": ty, "value": value, "widgets": [w]}));
                }
            }
        }
        Ok(out)
    })
}

fn render(dir: &str, files: &[String]) {
    PDFIUM.with(|pdfium| {
        for p in files {
            let stem = Path::new(p).file_stem().unwrap().to_string_lossy().into_owned();
            let Ok(doc) = pdfium.load_pdf_from_file(p, None) else {
                eprintln!("render: cannot load {p}");
                continue;
            };
            for (i, page) in doc.pages().iter().enumerate().take(2) {
                for (tag, form) in [("form", true), ("noform", false)] {
                    let cfg = PdfRenderConfig::new().set_target_width(1224).render_form_data(form);
                    match page.render_with_config(&cfg).and_then(|b| b.as_image()) {
                        Ok(img) => {
                            let out = format!("{dir}/{stem}_p{}_{tag}.png", i + 1);
                            if let Err(e) = img.save(&out) {
                                eprintln!("render save {out}: {e}");
                            }
                        }
                        Err(e) => eprintln!("render {stem} p{}: {e:?}", i + 1),
                    }
                }
            }
        }
    });
}

fn main() {
    runner::run("pdfium-render", "0.9.5+pdfium-8125", extract);
    if let Ok(dir) = std::env::var("RENDER_DIR") {
        std::fs::create_dir_all(&dir).ok();
        let files: Vec<String> = std::env::args().skip(1).collect();
        render(&dir, &files);
    }
}
