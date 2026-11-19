# formula: formula-recognition benchmark (image to LaTeX, CPU, onnxruntime)

Nothing in this folder contains model weights or benchmark data: the workflow downloads them inside the jobs
(Hugging Face and arXiv, pinned revisions) and keeps them only in the Actions cache.

## Models

| id | what |
|---|---|
| `p2t15_fp32` | Pix2Text MFR 1.5 (`breezedeus/pix2text-mfr-1.5`, MIT), original ONNX |
| `p2t15_fp16` | the same, `onnxconverter-common` float16 with `keep_io_types=True` |
| `p2t15_int16` | onnxruntime QDQ static quantisation, QInt16 weights and activations (MinMax on the 12 own crops) |
| `p2t15_int8` | onnxruntime `quantize_dynamic` QInt8 (about 31 MB) |
| `p2t15_int8_split` | the INT8 model with the decoder split so the cross-attention K/V is computed once per image |
| `p2t15_int4_b32`, `p2t15_int4_b128` | `MatMulNBitsQuantizer` 4-bit weight-only, block 32 / 128 |
| `p2t15_int4_b128_int8rest` | int4 MatMul plus INT8 for Gather and Conv |
| `pp_formulanet_plus_s` | reference only: community ONNX of PP-FormulaNet_plus-S (`x3zvawq/paddleocr-js-onnx`), PaddleX pre/post-processing |
| `unirec_0_1b` | reference only: `topdu/unirec_0_1b_onnx` (Apache-2.0), OpenOCR greedy decoding with KV cache |
| `doctr_formulanet`, `doctr_formulanet_int8` | docTR FormulaNet via OnnxTR (`Felix92/onnxtr-formulanet`, Apache-2.0) and its `quantize_dynamic` INT8 |

## Test sets

- UniMER-Test (`wanderkid/UniMER_Dataset`, Apache-2.0): all SPE, CPE and SCE samples; HWE only for docTR FormulaNet,
  to compare with its published numbers.
- arXiv: the papers in `papers.json`. Display and inline formulas from the LaTeX source, located through the PDF text
  layer and cropped at 3x; some displays also at 1.5x and 1x. A crop is kept only when the characters inside it match the
  source formula. Complex displays are taken first.
- own12: 12 crops from docling.rs test PDFs with `gt.json`.

## Metrics

Exact match after `scripts/texnorm.py`, edit score (1 - Levenshtein / max length), corpus BLEU-4 on normalised tokens,
runaway (no end token within the model's cap) and empty outputs, CDM (UniMERNet `cdm`) on a fixed subset. Buckets and
tags: `scripts/taxonomy.py`.

## Workflow

`.github/workflows/formula.yml`, `workflow_dispatch` with `mode=probe` (prep and a short probe of every model) or
`mode=full` (all shards from `plan.json`, one timing job per model, CDM, aggregate report in the job summary).
