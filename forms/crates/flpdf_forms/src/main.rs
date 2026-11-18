#[path = "../../../common/runner.rs"]
mod runner;

use flpdf::{ObjectHandle, PageDocumentHelper, Pdf};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;

fn strip(b: &[u8]) -> String {
    String::from_utf8_lossy(b).trim_start_matches('/').to_string()
}

fn hv(h: &ObjectHandle) -> Value {
    if let Ok(Some(n)) = h.try_get_value_as_name() {
        return json!(strip(&n));
    }
    if let Ok(Some(s)) = h.try_get_value_as_utf8() {
        return json!(String::from_utf8_lossy(&s).into_owned());
    }
    if let Ok(true) = h.try_is_array() {
        if let Ok(items) = h.try_get_array_as_vector() {
            return Value::Array(items.iter().map(hv).collect());
        }
    }
    Value::Null
}

fn extract(path: &str) -> Result<Vec<Value>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut pdf = Pdf::open(BufReader::new(file)).map_err(|e| format!("open: {e:?}"))?;
    let infos = pdf.acroform().map_err(|e| format!("acroform: {e:?}"))?.field_infos().map_err(|e| format!("field_infos: {e:?}"))?;
    let map = pdf
        .acroform()
        .map_err(|e| format!("acroform: {e:?}"))?
        .annotation_to_field_map()
        .map_err(|e| format!("annotation_to_field_map: {e:?}"))?;
    let pages = PageDocumentHelper::new(&mut pdf).get_all_pages().map_err(|e| format!("pages: {e:?}"))?;
    let mut widgets: BTreeMap<(u32, u16), Vec<Value>> = BTreeMap::new();
    for (i, pg) in pages.iter().enumerate() {
        let Ok(annots) = pg.try_get_key(b"/Annots") else { continue };
        let Ok(list) = annots.try_get_array_as_vector() else { continue };
        for a in list {
            let Some(aref) = a.object_ref() else { continue };
            let Some(fref) = map.get(&aref) else { continue };
            let rect = a.try_get_key(b"/Rect").ok().and_then(|r| r.try_get_array_as_vector().ok()).and_then(|v| {
                let n: Vec<f64> = v.iter().filter_map(|x| x.try_get_numeric_value().ok()).collect();
                (n.len() == 4).then(|| [n[0], n[1], n[2], n[3]])
            });
            let as_state = a.try_get_key(b"/AS").ok().and_then(|x| x.try_get_value_as_name().ok().flatten()).map(|n| strip(&n));
            widgets.entry((fref.number, fref.generation)).or_default().push(runner::widget(Some(i as u32 + 1), rect, as_state));
        }
    }
    Ok(infos
        .iter()
        .map(|info| {
            let ty = info.field_type.as_deref().map(strip);
            let value = info.value.as_ref().map(hv).unwrap_or(Value::Null);
            let ws = widgets.get(&(info.object_ref.number, info.object_ref.generation)).cloned().unwrap_or_default();
            json!({"name": info.full_name, "type": ty, "value": value, "widgets": ws})
        })
        .collect())
}

fn main() {
    runner::run("flpdf", "0.10.0", extract);
}
