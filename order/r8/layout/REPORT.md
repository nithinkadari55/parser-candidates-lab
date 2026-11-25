# Round 8, test B: heron fp32 vs heron int8 on layout sets not covered before

**Runs:**
- 45672090398 (`heron-r8.yml`): ICDAR-2017 POD and DocBank. All 16 jobs succeeded.
- 45678061782: real pages without ground truth. Rerun after a fix that adds the RVL-CDIP form pages. All jobs succeeded.

The real-page set of the first run is superseded. Its World Bank and slide pages are identical in both runs.

**Models:** docling.rs `layout_heron.onnx` / `layout_heron_int8.onnx`, same preprocessing. The round-7 code is reused unchanged (`order/layout/r7/r7_run.py`, `r7_eval.py` metrics). Scoring: `order/r8/layout/r8_eval.py`. Data: `r8_data.py`. Box-count check: `r8_boxcounts.py`.

The numbers below were recomputed locally from the per-page detection files of both runs. They match the workflow reports.

## Ground-truth sets

| set | pages | scored classes | fp32 mAP | int8 mAP | paired page F1, int8 - fp32 [95% CI] | Wilcoxon p | int8 better / worse pages |
|---|---|---|---|---|---|---|---|
| ICDAR-2017 POD (scientific papers) | 792 | Table | 0.950 | 0.948 | +0.0003 [-0.0032, +0.0042] | 0.95 | 7 / 7 |
| DocBank test sample (arXiv) | 2,000 | Table, Picture, Caption | 0.583 (mean) | 0.583 (mean) | -0.0015 [-0.0047, +0.0016] | 0.41 | 19 / 27 |

DocBank per class (fp32 / int8 mAP):
- Table 0.566 / 0.564
- Picture 0.531 / 0.535
- Caption 0.653 / 0.651

POD Table mAP50: 0.979 / 0.980.

Notes on the ground truth:
- **POD:** the official Train/Test downloads are gone (404). The re-release (github `cndplab-founder/ICDAR2017_POD_dataset_supplement`) holds the 792 table pages with full Table boxes only, so only Table is scored.
- **DocBank:** boxes are built from token labels. Text and equation boxes do not follow paragraph / display-formula granularity (inline math counts as "equation"), so fp32 and int8 both score near zero on them:
  - Text 0.073 / 0.074
  - Formula 0.015 / 0.015

  Only Table, Picture and Caption are scored. Images were read from the 50 GB split archive by HTTP range requests (2,000 random test pages).

## Real pages without ground truth (1,500 pages)

| source | pages | mean agreement | identical pages | pages below 0.9 | pages below 0.75 |
|---|---|---|---|---|---|
| World Bank documents (reports, working papers, appraisal docs, briefs) | 525 | 0.9325 | 48.2% | 26.3% | 5.3% |
| Zenodo slide decks (CC-BY presentations, PDF) | 525 | 0.9126 | 32.2% | 34.9% | 5.9% |
| scanned forms, FUNSD | 199 | 0.9179 | 5.0% | 29.6% | 2.0% |
| scanned forms, RVL-CDIP "form" class | 251 | 0.9015 | 7.2% | 37.5% | 4.0% |
| all | 1,500 | 0.9184 | 30.0% | 31.6% | 4.9% |

Agreement = F1 between the two models' boxes (score >= 0.5, same coarse class, IoU >= 0.5).

**Are the differences one-sided?** Box counts at score >= 0.5, all 1,500 pages ("only" = no same-class match in the other model):

| class | fp32 boxes | int8 boxes | fp32 only | int8 only |
|---|---|---|---|---|
| Table | 436 | 436 | 18 | 18 |
| Picture | 2,992 | 3,015 | 127 | 150 |
| Text | 26,241 | 26,375 | 933 | 1,064 |
| Caption | 421 | 430 | 28 | 37 |
| Formula | 68 | 65 | 4 | 1 |
| Page-header | 1,198 | 1,207 | 36 | 45 |
| Page-footer | 1,732 | 1,765 | 38 | 71 |

**Scanned forms (RVL-CDIP):**
- Table 65 vs 60 boxes (6 fp32-only, 1 int8-only).
- Text 8,411 vs 8,462.

**How to read the agreement:**
- The disagreement is two-sided and small: int8 keeps about 0.6% more boxes, spread over all classes. It is the score-threshold flicker seen before.
- Round 7 had ground-truth sets with the same agreement level: D4LA 0.908 and M6Doc 0.914. There, paired page F1 showed no difference (-0.0006 and +0.0016).
- Scans have the lowest share of identical pages (5-7%). That matches D4LA scans in round 7 (11%), where accuracy was still equal.

**Speed (CPU, GitHub runner, ms per page, p50):**

| set | fp32 | int8 |
|---|---|---|
| POD | 623 | 311 |
| DocBank | 608 | 538 |
| real pages | 810 | 324 |

## Verdict

**Rule:** heron int8 is chosen if it ties or beats fp32 overall and is not clearly worse on any document type.

- **Scientific papers:**
  - POD tables: tie (+0.0003, CI spans 0).
  - DocBank Table / Picture / Caption: tie (-0.0015, CI spans 0; mAP 0.583 vs 0.583).
- **Reports, slide decks, scanned forms:** no ground truth. Agreement is 0.90-0.93, and the box differences are two-sided and small.
- With round 7, int8 has now tied fp32 on DocLayNet, OmniDocBench English, D4LA, M6Doc, PubLayNet, POD and DocBank. It was not clearly worse on any type, and it is 1.1-2.5x faster.

**Decision: heron int8 is chosen** (keep it as the harness default).

**Remaining limit:** for slide decks and scanned forms we only know that the two models agree on about 91% of boxes, not which one is right where they differ.

## Sets skipped

- **ICDAR-2017 POD full set** (figure and formula boxes): official archives return 404. Only the table subset was used.
- **PRImA layout analysis dataset:** the download requires a registered account (primaresearch.org redirects to /register). Gated, skipped.
- **IIIT-AR-13K (annual reports):** cvit.iiit.ac.in does not resolve (DNS) from the laptop or from GitHub runners.
- **HJDataset:** skipped. It is historical Japanese, with nested page / row / text-region boxes, and not our target.
- **SEC EDGAR filings as PDF:**
  - EDGAR full-text search answers.
  - www.sec.gov/Archives returns 403 to a client without a contact e-mail in its User-Agent, which we do not send.
  - World Bank PDFs were used for the report type instead.
- **IMF reports:** imf.org returns 403 (Akamai bot block).
- **multimodal-annual-reports (HF):** not used. Its RIGHTS.md limits the source PDFs to personal use.
