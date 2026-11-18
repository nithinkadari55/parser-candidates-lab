#[path = "../../../common/runner.rs"]
mod runner;

use pdfer_forms::{FieldValue, PdfReaderCompat};
use serde_json::{json, Value};

fn val(v: &Option<FieldValue>) -> Value {
    match v {
        Some(FieldValue::Text(s)) => json!(s),
        Some(FieldValue::Name(s)) => json!(s.trim_start_matches('/')),
        Some(FieldValue::List(a)) => json!(a.iter().map(|s| s.trim_start_matches('/').to_string()).collect::<Vec<_>>()),
        Some(FieldValue::Null) | None => Value::Null,
    }
}

fn extract(path: &str) -> Result<Vec<Value>, String> {
    let reader = PdfReaderCompat::load(path).map_err(|e| format!("load: {e:?}"))?;
    let fields = reader.get_fields().map_err(|e| format!("get_fields: {e:?}"))?.unwrap_or_default();
    Ok(fields
        .iter()
        .map(|(name, f)| {
            json!({"name": name, "type": f.field_type.as_ref().map(|t| t.trim_start_matches('/').to_string()), "value": val(&f.value), "states": f.states, "widgets": []})
        })
        .collect())
}

fn main() {
    runner::run("pdfer_forms", "0.3.4", extract);
}
