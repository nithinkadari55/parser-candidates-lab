# compare: docling.rs 2.84.0 vs PaddleOCR PP-StructureV3

Question: which stack fits our Rust document harness (structured tree per file, English, CPU, Windows + Linux,
MIT/Apache/BSD), and are parts of one worth combining with the other?

Everything runs in `.github/workflows/compare.yml` (manual dispatch). No dataset, model weight or third-party
source is committed; jobs download them and keep them in `actions/cache`.

## Stacks

| name | what |
|---|---|
| `dlrs` | docling-rs 2.84.0 PyPI wheel, our harness options: fp32 layout (`DOCLING_RS_FP32=1`), TableFormer accurate fp32, OCR (PP-OCRv6 small det + medium rec), OCR scale 3, heading hierarchy, broken-text-layer repair, picture classification, picture OCR, text-line orientation. Models: `dlrs_models.json`. |
| `dlrs_formula` | `dlrs` plus `do_formula_enrichment` (CodeFormulaV2 fp32), re-run only on the selected pages that have display formulas. |
| `pp` | paddleocr 3.7.0 / paddlex 3.7.2 / paddlepaddle 3.3.1 CPU, `PPStructureV3(device="cpu")`, everything else default. PaddleX ships one PP-StructureV3 config (no light / accurate presets in code). |
| `pp_onnx` | same pipeline with `engine="onnxruntime"` in an environment without paddlepaddle (with and without formula recognition). |

## Data

- docling.rs test corpus at commit `cfb8df2`: `tests/data/pdf/sources` + `tests/data/scanned/sources` PDFs; docx/pptx/xlsx sources.
- Synthetic Office files (`scripts/make_office.py`): one unique marker per feature (Word comment, header, footer;
  speaker notes, hidden slide, chart data; hidden sheet, merged cells, sheet names).
- OmniDocBench (`opendatalab/OmniDocBench` @ `aa1ee96`, research use only, not redistributed): 319 English pages,
  80 per bucket (simple has 79). Buckets, first match wins: edge = watermark / fuzzy scan / colourful background /
  deformation / handwriting / layout_hard subset / landscape; complex = 3-column, mixed or other layout, newspaper,
  magazine, slides; simple = single column, no table, v1.5 subset; medium = the rest. Inside each bucket, pages with
  tables, with display formulas and other pages are interleaved (sha1 order). Pages are images, so both stacks OCR.
- `data/order_expected.json`: hand-checked snippet order on public docling.rs PDFs (copied from `order/data`).

## Metrics

- OmniDocBench official end2end evaluator (`opendatalab/OmniDocBench` @ `7ddb8df`, quick_match): text edit
  distance, table TEDS / TEDS structure / edit distance, display-formula edit distance, reading-order edit distance,
  split per bucket through an added `bucket` page attribute. CDM is not run (needs TeX Live + ImageMagick); formula
  BLEU per bucket is computed with `evaluate`'s bleu on the official matched, normalised formula pairs.
- docling.rs tables are written as HTML (`rowspan`/`colspan`/`<th>`) from its JSON so spans are scored.
- Corpus PDFs: bag-of-words F1 of each stack's text against the PDF text layer (pypdfium2), structure counts from
  each stack's JSON, reading order on the hand-checked pages.
- Timing job: one runner, both stacks in their own venv and process, same 12 ODB pages + 3 PDFs, one warm-up page,
  peak RSS from `getrusage`, each stack run twice to check byte-identical output.
