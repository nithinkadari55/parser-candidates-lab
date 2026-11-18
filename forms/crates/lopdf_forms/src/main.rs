#[path = "../../../common/runner.rs"]
mod runner;

use lopdf::{decode_text_string, Document, Object, ObjectId};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

const FF_RADIO: i64 = 1 << 15;
const FF_PUSHBUTTON: i64 = 1 << 16;

fn deref<'a>(doc: &'a Document, o: &'a Object) -> &'a Object {
    doc.dereference(o).map(|(_, x)| x).unwrap_or(o)
}

fn text_of(o: &Object) -> Option<String> {
    match o {
        Object::String(bytes, _) => Some(match decode_text_string(o) {
            Ok(s) => s.trim_start_matches('\u{feff}').to_string(),
            Err(_) => bytes.iter().map(|&b| b as char).collect(),
        }),
        Object::Name(n) => Some(String::from_utf8_lossy(n).into_owned()),
        _ => None,
    }
}

fn value_of(doc: &Document, o: &Object) -> Value {
    match deref(doc, o) {
        Object::Array(items) => Value::Array(items.iter().map(|i| value_of(doc, i)).collect()),
        Object::Null => Value::Null,
        Object::Boolean(b) => json!(b.to_string()),
        Object::Integer(i) => json!(i.to_string()),
        Object::Real(r) => json!(r.to_string()),
        x => text_of(x).map(Value::String).unwrap_or(Value::Null),
    }
}

#[derive(Clone, Copy, Default)]
struct Inh<'a> {
    ft: Option<&'a Object>,
    v: Option<&'a Object>,
    ff: Option<i64>,
}

struct Ctx<'a> {
    doc: &'a Document,
    annot_page: HashMap<ObjectId, u32>,
    page_of: HashMap<ObjectId, u32>,
    seen: HashSet<ObjectId>,
    out: Vec<Value>,
}

fn opt_list<'a>(doc: &'a Document, d: &'a lopdf::Dictionary) -> Vec<(String, String)> {
    let Ok(o) = d.get(b"Opt") else { return Vec::new() };
    let Ok(arr) = deref(doc, o).as_array() else { return Vec::new() };
    arr.iter()
        .filter_map(|e| match deref(doc, e) {
            Object::Array(pair) if pair.len() >= 2 => {
                Some((text_of(deref(doc, &pair[0]))?, text_of(deref(doc, &pair[1]))?))
            }
            x => text_of(x).map(|s| (s.clone(), s)),
        })
        .collect()
}

fn walk<'a>(cx: &mut Ctx<'a>, node: &'a Object, parent: &str, inh: Inh<'a>, depth: u32) {
    let doc = cx.doc;
    if depth > 64 {
        return;
    }
    if let Object::Reference(id) = node {
        if !cx.seen.insert(*id) {
            return;
        }
    }
    let Ok(d) = deref(doc, node).as_dict() else { return };
    let partial = d.get(b"T").ok().and_then(|t| text_of(deref(doc, t)));
    let name = match partial {
        Some(t) if parent.is_empty() => t,
        Some(t) => format!("{parent}.{t}"),
        None => parent.to_string(),
    };
    let mut inh = inh;
    if let Ok(x) = d.get(b"FT") {
        inh.ft = Some(deref(doc, x));
    }
    if let Ok(x) = d.get(b"V") {
        inh.v = Some(deref(doc, x));
    }
    if let Ok(x) = d.get(b"Ff") {
        inh.ff = deref(doc, x).as_i64().ok();
    }
    let kids: Vec<&'a Object> = d
        .get(b"Kids")
        .ok()
        .and_then(|k| deref(doc, k).as_array().ok())
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let field_kids: Vec<&'a Object> = kids
        .iter()
        .copied()
        .filter(|k| deref(doc, k).as_dict().map(|kd| kd.has(b"T") || kd.has(b"Kids")).unwrap_or(false))
        .collect();
    if !field_kids.is_empty() {
        for k in field_kids {
            walk(cx, k, &name, inh, depth + 1);
        }
        return;
    }
    let widgets: Vec<&'a Object> = if kids.is_empty() { vec![node] } else { kids };
    let ft = inh.ft.and_then(|o| o.as_name().ok()).map(|n| String::from_utf8_lossy(n).into_owned());
    let ff = inh.ff.unwrap_or(0);
    let value = inh.v.map(|v| value_of(doc, v)).unwrap_or(Value::Null);

    let mut ws = Vec::new();
    let mut on_states = Vec::new();
    for w in &widgets {
        let wid = if let Object::Reference(id) = w { Some(*id) } else { None };
        let wd = deref(doc, w).as_dict().ok();
        let page = wid.and_then(|id| cx.annot_page.get(&id).copied()).or_else(|| {
            wd.and_then(|wd| wd.get(b"P").ok())
                .and_then(|p| p.as_reference().ok())
                .and_then(|pid| cx.page_of.get(&pid).copied())
        });
        let rect = wd
            .and_then(|wd| wd.get(b"Rect").ok())
            .and_then(|r| deref(doc, r).as_array().ok())
            .and_then(|a| {
                let v: Vec<f64> = a.iter().filter_map(|x| deref(doc, x).as_float().ok().map(|f| f as f64)).collect();
                (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
            });
        let as_state = wd
            .and_then(|wd| wd.get(b"AS").ok())
            .and_then(|x| deref(doc, x).as_name().ok())
            .map(|n| String::from_utf8_lossy(n).into_owned());
        if let Some(s) = &as_state {
            if s != "Off" {
                on_states.push(s.clone());
            }
        }
        ws.push(runner::widget(page, rect, as_state));
    }

    let opts = opt_list(doc, d);
    let mut export = Value::Null;
    let display: Value = match ft.as_deref() {
        Some("Btn") if ff & FF_PUSHBUTTON != 0 => Value::Null,
        Some("Btn") => {
            let state = match &value {
                Value::String(s) => Some(s.clone()),
                _ => on_states.first().cloned(),
            };
            match state {
                Some(s) if s != "Off" => {
                    let shown = s
                        .parse::<usize>()
                        .ok()
                        .and_then(|i| opts.get(i).map(|o| o.1.clone()));
                    if let Some(e) = &shown {
                        if matches!(value, Value::String(_)) {
                            export = json!(e);
                        }
                    }
                    json!(shown.unwrap_or(s))
                }
                _ => json!("Off"),
            }
        }
        Some("Ch") => {
            let map = |s: &str| opts.iter().find(|o| o.0 == s).map(|o| o.1.clone()).unwrap_or_else(|| s.to_string());
            let has_pairs = opts.iter().any(|o| o.0 != o.1);
            match &value {
                Value::String(s) => {
                    if has_pairs {
                        export = json!(map(s));
                    }
                    json!(map(s))
                }
                Value::Array(a) => {
                    let v: Vec<String> = a.iter().filter_map(|x| x.as_str()).map(map).collect();
                    if has_pairs {
                        export = json!(v);
                    }
                    json!(v.join(", "))
                }
                _ => Value::Null,
            }
        }
        _ => value.clone(),
    };
    let kind = match ft.as_deref() {
        Some("Btn") if ff & FF_PUSHBUTTON != 0 => "pushbutton",
        Some("Btn") if ff & FF_RADIO != 0 => "radio",
        Some("Btn") => "checkbox",
        Some("Tx") => "text",
        Some("Ch") => "choice",
        Some("Sig") => "signature",
        _ => "unknown",
    };
    cx.out.push(json!({"name": name, "type": ft, "kind": kind, "value": value, "export": export, "display": display, "widgets": ws}));
}

fn extract(path: &str) -> Result<Vec<Value>, String> {
    let doc = Document::load(path).map_err(|e| format!("load: {e}"))?;
    let mut annot_page = HashMap::new();
    let mut page_of = HashMap::new();
    for (pno, pid) in doc.get_pages() {
        page_of.insert(pid, pno);
        if let Ok(pd) = doc.get_dictionary(pid) {
            if let Ok(a) = pd.get(b"Annots") {
                if let Ok(arr) = deref(&doc, a).as_array() {
                    for x in arr {
                        if let Object::Reference(id) = x {
                            annot_page.entry(*id).or_insert(pno);
                        }
                    }
                }
            }
        }
    }
    let catalog = doc.catalog().map_err(|e| format!("catalog: {e}"))?;
    let Some(af) = catalog.get(b"AcroForm").ok().and_then(|a| deref(&doc, a).as_dict().ok()) else {
        return Ok(Vec::new());
    };
    let fields: Vec<&Object> = af
        .get(b"Fields")
        .ok()
        .and_then(|f| deref(&doc, f).as_array().ok())
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let mut cx = Ctx { doc: &doc, annot_page, page_of, seen: HashSet::new(), out: Vec::new() };
    for f in fields {
        walk(&mut cx, f, "", Inh::default(), 0);
    }
    Ok(cx.out)
}

fn main() {
    runner::run("lopdf", "0.45.0", extract);
}
