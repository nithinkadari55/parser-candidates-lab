#[path = "../../../common/runner.rs"]
mod runner;

use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::forms::{FieldType, FieldValue, FormExtractor};
use serde_json::{json, Value};

fn val(v: &FieldValue) -> Value {
    match v {
        FieldValue::Text(s) => json!(s),
        FieldValue::Name(s) => json!(s.trim_start_matches('/')),
        FieldValue::Boolean(b) => json!(b),
        FieldValue::Array(a) => json!(a),
        FieldValue::None => Value::Null,
    }
}

fn extract(path: &str) -> Result<Vec<Value>, String> {
    let doc = PdfDocument::open(path).map_err(|e| format!("open: {e:?}"))?;
    let fields = FormExtractor::extract_fields(&doc).map_err(|e| format!("extract: {e:?}"))?;
    Ok(fields
        .iter()
        .map(|f| {
            let ty = match &f.field_type {
                FieldType::Button => Some("Btn".to_string()),
                FieldType::Text => Some("Tx".to_string()),
                FieldType::Choice => Some("Ch".to_string()),
                FieldType::Signature => Some("Sig".to_string()),
                FieldType::Unknown(s) if s.is_empty() => None,
                FieldType::Unknown(s) => Some(s.clone()),
            };
            let widgets: Vec<Value> = f.bounds.map(|b| vec![runner::widget(None, Some(b), None)]).unwrap_or_default();
            json!({"name": f.full_name, "type": ty, "value": val(&f.value), "widgets": widgets})
        })
        .collect())
}

fn main() {
    runner::run("pdf_oxide", "0.3.78", extract);
}
