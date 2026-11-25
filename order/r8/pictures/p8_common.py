"""Round 8, test A (picture classifier): shared label set, source table, manifest loading with
cross-source de-duplication, and the image preprocessing used by every arm."""
import glob
import json
import os

import numpy as np

# Shared label set: classes that docling's DocumentFigureClassifier-v2.5 (26 classes, docling.rs
# crates/docling-pdf/src/enrich.rs PICTURE_CLASSES) and the trained heads both support.
CLASSES = ["chart", "diagram", "photo", "screenshot", "logo", "signature", "stamp", "table_image", "map"]
D26 = ["logo", "photograph", "icon", "engineering_drawing", "line_chart", "bar_chart", "other", "table", "flow_chart",
       "screenshot_from_computer", "signature", "screenshot_from_manual", "geographical_map", "pie_chart",
       "page_thumbnail", "stamp", "music", "calendar", "qr_code", "bar_code", "full_page_image", "scatter_plot",
       "chemistry_structure", "topographical_map", "crossword_puzzle", "box_plot"]
D26_TO_SHARED = {"logo": "logo", "photograph": "photo", "engineering_drawing": "diagram", "line_chart": "chart",
                 "bar_chart": "chart", "table": "table_image", "flow_chart": "diagram",
                 "screenshot_from_computer": "screenshot", "signature": "signature",
                 "screenshot_from_manual": "screenshot", "geographical_map": "map", "pie_chart": "chart",
                 "stamp": "stamp", "scatter_plot": "chart", "chemistry_structure": "diagram",
                 "topographical_map": "map", "box_plot": "chart"}
# docling classes outside the shared set: icon, other, page_thumbnail, music, calendar, qr_code, bar_code,
# full_page_image, crossword_puzzle
SUBTYPES = ["bar", "line", "pie", "scatter", "box"]
D26_SUB = {"bar_chart": "bar", "line_chart": "line", "pie_chart": "pie", "scatter_plot": "scatter", "box_plot": "box"}

# name: (data group, licence as stated by the files we read, permissive for shipping, role)
# permissive = CC-BY, CC0 / public domain, Apache, MIT, CDLA (the user's list); role "pool" = used in the
# leave-one-source-out evaluation and for training, "test" = test only (never trained on).
SOURCES = {
    "chartx": ("charts", "Apache-2.0 (HF card InternScience/ChartX)", True, "pool"),
    "plotqa": ("charts", "CC-BY-4.0 (PlotQA; HF achang/plot_qa card 'cc')", True, "pool"),
    "chartqa": ("charts", "GPL-3.0 (HF HuggingFaceM4/ChartQA card)", False, "pool"),
    "ubpmc": ("ubpmc", "CC-BY (UB-PMC CHART-Info 2022 test, PMC open-access CC-BY figures)", True, "pool"),
    "scicap": ("scicap", "CC-BY-NC-SA-4.0 (HF CrowdAILab/scicap card)", False, "pool"),
    "ai2d": ("diagrams", "CC-BY-SA (AI2D, Allen Institute)", False, "pool"),
    "qzhou_flowchart": ("diagrams", "Apache-2.0 (HF Kingsoft-LLM/QZhou-Flowchart-QA card)", True, "pool"),
    "openimages": ("photos", "CC-BY-4.0 annotations, images CC-BY-2.0 (HF finedet/openimages card)", True, "pool"),
    "coco_ccby": ("photos", "COCO val2017 images with licence id 4/7/8 only (CC-BY, no known restriction, US gov)",
                  True, "pool"),
    "imagenette": ("photos", "ImageNet subset (non-commercial research terms)", False, "pool"),
    "rico": ("screens", "CC-BY-4.0 (HF bevaya/RICO-Screen2Words card)", True, "pool"),
    "screenspot": ("screens", "Apache-2.0 (HF OS-Copilot/ScreenSpot-v2 card)", True, "pool"),
    "websight": ("screens", "CC-BY-4.0 (HF HuggingFaceM4/WebSight card)", True, "pool"),
    "logodet": ("logosig", "MIT (HF axonstan/LogoDet-3K card; brand logos are trademarks)", True, "pool"),
    "signverod": ("logosig", "Apache-2.0 (HF ondrs/signverod card)", True, "pool"),
    "rf100_signatures": ("logosig", "CC-BY-4.0 (Roboflow 100 signatures-xc8up, HF LibreYOLO)", True, "pool"),
    "cedar_signatures": ("logosig", "MIT on HF T0KII card, but file names match CEDAR (research terms): treated "
                         "as non-permissive", False, "pool"),
    "stamps_jsam": ("stamps", "CC-BY-4.0 (HF mapo80/stamps, sub-source stamp_detection_jsam)", True, "pool"),
    "stamps_yolo": ("stamps", "CC-BY-4.0 (HF mapo80/stamps, sub-source yolo_stamp_classify)", True, "pool"),
    "stamps_marcos": ("stamps", "CC-BY-4.0 (HF mapo80/stamps, sub-source stamp_detectation_marcos)", True, "pool"),
    "stamps_stampa": ("stamps", "CC-BY-4.0 (HF mapo80/stamps, sub-source stamp_detection_stampa)", True, "pool"),
    "hyperbots_stamps": ("stamps", "CC-BY-4.0 (HF hyperbots-inc/synthetic-stamped-documents)", True, "pool"),
    "doclaynet_tables": ("tables", "CDLA-Permissive-1.0 (DocLayNet v1.1 val Table boxes)", True, "pool"),
    "pubtabnet": ("tables", "CDLA-Permissive-1.0 (HF apoidea/pubtabnet-html validation)", True, "pool"),
    "publaynet_tables": ("tables", "CDLA-Permissive-1.0 (PubLayNet validation Table boxes)", True, "pool"),
    "realdoc_doclaynet": ("realdoc", "DocLayNet v1.1 test Picture boxes, hand-labelled (CDLA-Permissive-1.0)",
                          False, "test"),
    "realdoc_omnidocbench": ("realdoc", "OmniDocBench figure boxes, hand-labelled (licence not stated)", False,
                             "test"),
}
GROUPS = sorted({v[0] for v in SOURCES.values()})
CAP = 600  # crops per (source, class)

MEAN_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DOCLING_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DOCLING_STD = np.array([0.47853944, 0.4732864, 0.47434163], dtype=np.float32)


def fingerprint(pil):
    """dHash (64 bit, 9x8 grey), whether the thumbnail has enough contrast for the dHash to mean
    anything, and an md5 of the 32x32 grey thumbnail (exact duplicates)."""
    import hashlib
    g = pil.convert("L")
    a = np.asarray(g.resize((9, 8)), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    t = np.asarray(g.resize((32, 32)), dtype=np.uint8)
    return {"dh": "%016x" % int("".join("1" if b else "0" for b in bits), 2), "inf": bool(a.std() > 12),
            "md": hashlib.md5(t.tobytes()).hexdigest()[:16]}


def load_manifest(root, verbose=True):
    """All items of all data groups under ROOT (ROOT/<group>/<source>/items.jsonl), de-duplicated:
    an item is dropped when its 32x32 grey thumbnail equals one already kept, or when both have a
    contrasty thumbnail and their 64-bit dHashes differ in at most 2 bits (sources are processed in a
    fixed order, so the first copy wins)."""
    items = []
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "items.jsonl"))):
        base = os.path.dirname(f)
        for line in open(f, encoding="utf-8"):
            if line.strip():
                m = json.loads(line)
                m["path"] = os.path.join(base, "img", os.path.basename(m["path"]))
                items.append(m)
    items.sort(key=lambda m: (SOURCES.get(m["src"], ("", "", False, "z"))[3] == "test", m["src"], m["id"]))
    kept = []
    dropped = {}
    seen_md = set()
    hv = np.zeros(len(items), dtype=np.uint64)
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    k = 0
    for m in items:
        if m["label"] not in CLASSES:
            continue
        dup = m["md"] in seen_md
        h = np.uint64(int(m["dh"], 16))
        if not dup and m["inf"] and k:
            x = np.bitwise_xor(hv[:k], h)
            d = pop[x.view(np.uint8)].reshape(-1, 8).sum(1)
            dup = bool(d.min() <= 2)
        if dup:
            dropped[m["src"]] = dropped.get(m["src"], 0) + 1
            continue
        seen_md.add(m["md"])
        if m["inf"]:
            hv[k] = h
            k += 1
        kept.append(m)
    if verbose:
        print("manifest:", len(items), "items,", len(kept), "kept; duplicates dropped per source:", dropped,
              flush=True)
    return kept, dropped


def squash(pil, side=224):
    from PIL import Image
    return pil.convert("RGB").resize((side, side), Image.BILINEAR)


def to_tensor(pil, mean, std, side=224):
    x = np.asarray(squash(pil, side), dtype=np.float32) / 255.0
    return ((x - mean) / std).transpose(2, 0, 1)
