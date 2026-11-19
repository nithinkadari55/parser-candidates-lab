# tables: table-structure model benchmark

All models get the same table crops and the same words (dataset text boxes, or PDF words for real PDFs).

## Models (downloaded inside the jobs, never committed)

| Name | What | Source | Licence |
|---|---|---|---|
| `tatr32` / `tatr16` / `tatr` / `tatr4` | TATR v1.1-all ONNX: FP32, FP16, INT8 (dynamic), INT4 weight-only (our `MatMulNBitsQuantizer`, block 32, from FP32) | `Xenova/table-transformer-structure-recognition-v1.1-all` (`onnx/model.onnx`, `model_fp16.onnx`, `model_quantized.onnx`), weights from `microsoft/table-transformer-structure-recognition-v1.1-all` | MIT |
| `potatr` / `potatr8` | POTATR v1.1 ONNX (FP32) and our INT8 dynamic quantisation of it; postprocessing from `kensho-technologies/POTATR` | `opticsWolf/potatr-onnx` | MIT |
| `tcn` | TableCenterNet ONNX, run through OnnxTR's table predictor with the words | `Felix92/onnxtr-tablecenternet`, `onnxtr==0.9.0` | Apache-2.0 |
| `tfv2` / `tfv2q` | TableFormerV2 with PyTorch on CPU, and our ONNX export (encoder, uncached decoder, box head) with INT8 dynamic quantisation | `docling-project/TableFormerV2`, `docling-ibm-models==4.0.4` | Apache-2.0 weights, MIT code |
| `tableformer` | docling TableFormer through the full `docling-rs==2.84.0` PDF pipeline (layout + TableFormer) | PyPI, `models-v1` release | see docling.rs |
| `slanet1m` | SLANet-1M ONNX (reference row) | `bdatdo0601/slanet-1m-onnx` | MIT |

## Data (downloaded and processed inside the jobs; only derived crops are cached, never committed or uploaded)

| Set | Source | Licence | Sample |
|---|---|---|---|
| `ptn` | PubTabNet val (`docling-project/PubTabNet_OTSL`) | CDLA-Permissive-1.0 annotations; PMC OA images | 2000 |
| `ftn` | FinTabNet test (`docling-project/FinTabNet_OTSL`) | CDLA-Permissive | 2000 |
| `pt1m` | PubTables-1M test shards 0-1 (`docling-project/PubTables-1M_OTSL`), cropped to the table box | CDLA-Permissive-2.0 | 2000 |
| `pt2` | PubTables-v2 Cropped Tables test (`kensho/PubTables-v2`), all long or wide | CDLA-Permissive-2.0 | 1000 |
| `icdar` | ICDAR 2013 table competition, EU/US government PDFs, all regions incl. rotated pages and multi-page tables | no explicit licence; evaluation only | all |
| `scitsr` | SciTSR test, licence-filtered subsets `bevaya/SciTSR-cc-by-nc-sa` + `bevaya/SciTSR-pd` (arXiv table PDFs) | CC BY-NC-SA 4.0 / public domain | all |
| `tabrec` | TabRecSet English (scanned and camera images), all-line / no-line / three-line | CC BY 4.0 | 600 |
| `rot` | 150 PubTabNet + 150 FinTabNet tables of the sample, rotated 90 degrees | as sources | 300 |

Sampling is seeded and stratified: 15% simple, 20% medium, 35% complex, 30% edge (missing simple/medium go to complex/edge).
Edge: very long (>=50 rows), very wide (>=15 columns), tiny (<=3 rows), rotated, split across pages, then multi-line cells,
empty cells and footnote marks. Complex: >=3 span cells, nested header, hierarchical first column, >=10 columns or >=30 rows.

## Metrics

TEDS and TEDS-Struct (PubTabNet `metric.py`, thead/tbody ignored, tables up to 1200 grid cells), GriTS-Top/Con (TATR `grits.py`,
up to 1200 cells), exact rows+cols, merged-row errors (a predicted cell holding words of two GT rows that are not row-spanning),
header rows exact, failures (TEDS-Struct < 0.5, or GriTS-Top when TEDS is capped). Paired Wilcoxon tests with bootstrap CIs.

## Layout

`src/` scripts, `third_party/` (`postprocess.py` MIT, `metric.py` Apache-2.0, headers kept), `truth.json`,
workflow `.github/workflows/tables.yml` (prep -> model x shard matrix -> timing (3 clean jobs) -> aggregate).
