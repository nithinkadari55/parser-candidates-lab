# tables: table-structure model benchmark

Compares three table-structure models on CPU:

| Model | Files (downloaded in CI) | Licence |
|---|---|---|
| TATR v1.1-all, INT8 ONNX | `Xenova/table-transformer-structure-recognition-v1.1-all` `onnx/model_quantized.onnx` (30 MB, sha256 checked); weights from `microsoft/table-transformer-structure-recognition-v1.1-all` (MIT) | MIT (code and weights) |
| SLANet-1M, ONNX | `bdatdo0601/slanet-1m-onnx` `slanet_1m.onnx` (7.7 MB, sha256 checked), export of `dimtri009/SLANet-1M` | MIT (both repos) |
| docling TableFormer | `docling-rs==2.84.0` from PyPI, models from its `models-v1` release (`docling_rs.download_models()`) | see docling.rs |

## Data (never committed; downloaded and cached inside the job)

| Set | Source | Licence | Sample |
|---|---|---|---|
| `ptn` | PubTabNet val, via `docling-project/PubTabNet_OTSL` (original HTML + cell tokens/boxes) | CDLA-Permissive-1.0 (annotations); images under PMC OA terms | 600, seed 42, stratified |
| `ftn` | FinTabNet test, via `docling-project/FinTabNet_OTSL` | CDLA-Permissive (IBM, as cited by `bsmock/FinTabNet.c`) | 600, seed 42, stratified |
| `pt1m` | PubTables-1M test shard 0, via `docling-project/PubTables-1M_OTSL` | CDLA-Permissive-2.0 | 400, seed 42, stratified |
| `icdar` | ICDAR 2013 table competition (67 EU/US government PDFs, structure GT) from tamirhassan.com | no explicit licence stated; used for local evaluation only | all regions on unrotated pages |
| `ours` | 7 tables from docling.rs test PDFs; `truth.json` is the hand-checked grid | public test files | 7 |

Benchmark data is used for evaluation only and is not redistributed: artifacts contain only predicted
cell boxes and scores keyed by table id.

## Difficulty buckets

Assigned from the ground truth (and image rules for "ruled"), first match wins:
`complex` (>=3 span cells, nested multi-row header, hierarchical first column, >=10 columns or >=30 rows) >
`edge` (multi-line cells, >=15% empty cells, footnote marks, <=3 rows) >
`simple` (fully ruled grid, single header row) > `medium` (everything else: borderless or multi-row header).
Each set is sampled with an equal quota per bucket; ruled grids are rare in these sets, so `simple` is small.

## Method

- Every model gets the same image (dataset crop + 20 px white pad; ICDAR: page render at 144 dpi around the GT region).
- Cell text: each predicted cell is filled from the dataset's own text boxes (PubTabNet/FinTabNet/PubTables: GT cell content boxes; ICDAR: PDF words from pdfplumber) by largest overlap, the same for all models.
- TableFormer runs through the full docling-rs PDF pipeline: dataset tables are wrapped in one PDF page each (text layer = dataset text, image on top); ICDAR runs on the original PDF pages and tables are matched to the GT region by IoU.
- Metrics: TEDS and TEDS-Struct with PubTabNet's `metric.py` (thead/tbody ignored; `distance.levenshtein` replaced by the identical `rapidfuzz` implementation for speed), GriTS-Top/Con with TATR's `grits.py` (tables up to 600 grid cells), exact rows+cols, merged-row errors (a predicted cell holding text of two GT rows that are not row-spanning), header rows exactly right, failures (TEDS-Struct < 0.5).

## Layout

`src/` scripts, `third_party/` (`postprocess.py` MIT, `metric.py` Apache-2.0, headers kept), `truth.json`,
workflow `.github/workflows/tables.yml` (prep -> model x shard matrix -> timing -> aggregate).
