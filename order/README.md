# order: layout models and reading order (logic vs docling vs PP-DocLayoutV3)

Runs only in `.github/workflows/order.yml` (ubuntu-latest). No dataset, model weight or third-party
source is committed: jobs download them and keep them in `actions/cache` (DocLayNet parquet files are
fetched per job and not cached, to stay under the 10 GB cache limit).

## Part 1: layout models (`order/layout/`)

| key | model | source |
|---|---|---|
| `heron` | docling-layout-heron fp32 ONNX (RT-DETRv2) as shipped by docling.rs 2.84.0 (`models-v1` release) | Apache-2.0 |
| `heron_int8` | its int8 variant (what docling.rs runs on CPU by default) | Apache-2.0 |
| `egret` | docling-layout-egret-medium (D-FINE) ONNX, `ningpp/docling-layout-egret-medium-ONNX@06b27d8` | Apache-2.0 |
| `ppl` | PP-DocLayout-L with int8-stored weights, `Ralutet/PP-DocLayout-L-onnx-int8@361fe8f` | Apache-2.0 |
| `ppv3` | PP-DocLayoutV3, `PaddlePaddle/PP-DocLayoutV3_onnx@46bbdf1` | Apache-2.0 |
| `dlrs` | docling-rs 2.84.0 end to end on the page image (heron + docling postprocessing + OCR); no scores | MIT |

Data: DocLayNet test split (4,999 pages, `docling-project/DocLayNet-v1.1`, CDLA-Permissive-1.0) and the
OmniDocBench English pages. Every native label set is mapped onto DocLayNet's classes with List-item folded
into Text (`layout/models.py`: `MAP_DOCLING`, `MAP_PP`, `MAP_DLRS`; PP `number` becomes page header or footer
by position, `formula_number` and `inline_formula` are dropped). On OmniDocBench, Title and Section-header are
merged (its `title` covers both) and masked/abandoned regions are ignore zones.

Metrics: COCO AP@[.5:.95] and AP50 per class (pycocotools, detections with score >= 0.05, top 150 per page);
table recall at IoU 0.5 and 0.7 on pages with a table (score >= 0.5); how often a page header, page footer,
caption or footnote is labelled as a heading and what share of predicted headings are such boxes; time per
page and peak RSS in a dedicated job (one process per model).

## Part 2: reading order (`order/scripts/`)

Data: OmniDocBench English pages and DrDocBench dev (`2077AIDataFoundation/DrDocBench@7a2bc38`, annotations
CC0-1.0, same format as OmniDocBench, English). Region-level order ground truth; headers, footers, page
numbers, page footnotes, abandoned regions and equation numbers are not scored. Pages are split dev/test by
a hash of the page name; rules were tuned on dev only.

| method | what |
|---|---|
| `docling_rb` | docling's rule-based `ReadingOrderPredictor` (MIT), fetched at docling commit `5e600e8` at run time |
| `xycut_*` | plain recursive XY-cut |
| `xycutpp` | XY-Cut++-style ordering written from the paper (arXiv 2504.10258), geometry only (`ro.py`) |
| `+fig` | PDFFigures2-style caption/figure grouping (`addons.py`) |
| `+deco` | PdfPig DecorationTextBlockClassifier-style removal of repeated page decorations (run with all boxes, `full:` columns) |
| `+box` | Burdoc-style boxed sections: drawn rectangles and shaded areas found in the page image enclose a section |
| `+num` | item numbering as an order constraint (27 before 28) |
| `+tiny` | equation numbers, "(n marks)" and small side words attached to their line; margin words attached to the nearest block |
| `ppv3:own_order` | PP-DocLayoutV3's predicted order |
| `<model>:xycutpp` | XY-Cut++ on the boxes of each layout model (real pipeline) |

Metrics: Kendall tau, normalised edit distance and % of pages fully correct, per bucket (simple, medium,
complex), edge tags and layout type; dev and test splits; sign tests for each add-on. With detected boxes,
each GT region is matched to the detected box covering it and the order is scored over matched regions.

Other reading-order sets checked and not used: Comp-HRDoc (images only from HRDoc-Hard on Google Drive,
order over text lines), ROOR/EC-FUNSD (form-entity relations, no page order), DocBench-100 (not released),
DocGenome (no order field), ReadingBank (word level).
