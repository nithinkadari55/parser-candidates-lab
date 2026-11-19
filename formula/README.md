# formula: small formula-recognition models

Benchmark of three ONNX formula-recognition (image to LaTeX) models on CPU:

| id | model | files |
|---|---|---|
| `p2t15_fp32` | Pix2Text MFR 1.5, fp32 (`breezedeus/pix2text-mfr-1.5`, MIT) | encoder 87.5 MB + decoder 32.0 MB |
| `p2t10_int8_mathorium` | INT8 pair from `samirahmed007/mathorium` (MIT), an INT8 build of Pix2Text MFR **1.0** (`Brian314/pix2text-mfr-quantized`, MIT) | encoder 23.1 MB + `decoder_model_int8.onnx` 8.0 MB |
| `p2t15_int8_own` | `onnxruntime.quantization.quantize_dynamic` (QInt8, per-tensor) of MFR 1.5, built in CI | about 22.9 MB + 8.4 MB |

Nothing in this folder contains model weights or benchmark data. The workflow downloads them inside the job
(Hugging Face, GitHub raw at a pinned commit, arXiv) and keeps them only in the Actions cache.

## Test sets

- **UniMER-Test** (`wanderkid/UniMER_Dataset`, Apache-2.0): 300 SPE + 300 CPE + 100 SCE images, random sample, fixed seed. HWE is not used.
- **arXiv**: the 44 papers listed in `papers.json` (November 2026 submissions, many categories). Display equations and inline
  formulas are taken from the LaTeX source (macros expanded, `\label`/`\tag` removed, `align`-type environments wrapped in
  `aligned`), located on the rendered PDF page through the text layer and cropped at 3x (some displays also at 1.5x as
  a low-resolution case). A crop is kept only when the characters inside it match the source formula exactly.
- **own12**: 12 crops from docling.rs test PDFs with `gt.json`.

## Metrics

Exact match after normalisation (`scripts/texnorm.py`), normalised edit score (1 - Levenshtein / max length),
corpus BLEU-4 over normalised LaTeX tokens, runaway outputs (512-token cap reached) and empty outputs.
Difficulty buckets (`scripts/evaluate.py`): complex = environments, line breaks, nested fractions or more than 100
tokens; medium = fractions, roots, big operators, limits; simple = the rest. Edge slices: very wide, tall,
low-resolution, `\text`, calligraphic or fraktur, Greek-heavy, inline tight crops.

## Workflow

`.github/workflows/formula.yml`: `prep` (download and build, cached) then `acc` (3 models x 6 shards, results as
JSONL artifacts holding predictions only), `timing` (one dedicated job, 4 threads, one model process at a time) and
`aggregate` (metrics per set, bucket and edge slice in the job summary and the `formula-report` artifact).
