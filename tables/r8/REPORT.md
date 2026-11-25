# Tables round 8: TATR alone vs the TableFormer + TATR combo (broad benchmark)

Run **45676430603** (workflow `tables-r8`, all jobs green; artifacts `r8-report`, `r8pub-0..15`, `r8tf-0..19`). Probes: 45669760702, 45670228052, 45673907192 (`limit=6`); dataset probe 45666488109; cTDaR orientation check 45675705203.

## Summary

- **Decision (rule fixed in advance): keep the combo.** All three conditions fail, with tight CIs:
  - overall (test split, 10,945 tables): combo − TATR+logic = **+1.31 TEDS-Struct points** [+1.13, +1.53] (limit < 0.5) and **+3.98 pp exact shape** [+3.42, +4.66] (limit < 1);
  - by type the candidate is more than 2 points worse on financial (+2.37), government/business (+4.79) and scans/photos (+2.23).
  - Robust: without SynFinTabs +0.92 / +2.30; real source PDFs only +1.15 / +2.54. With the retrained selector the gap grows to +2.14 / +5.99.
- **TATR alone vs TableFormer:** TATR +1.63 TEDS-Struct points but **−3.85 pp exact shape** (it gets the grid size right less often).
- **TATR + logic:** only one repair survived dev tuning (column merge when ≥ 50% of a column pair's words straddle both). Test gain +0.03 points (p = 0.06), +0.51 pp shape. The gated wrapped-row join hurt in all 19 settings (−0.12 to −12.5 points on dev), as round 5 found. Row split, empty rows/columns and column split did not help either.
- **Selector:** the committed harness selector earns its place (+1.35 points over TATR, +2.98 over TableFormer). Retrained on this broader data with document-grouped CV it adds **+0.82 points** [+0.70, +0.94] and +2.0 pp shape, so retraining the harness selector on this data is worth doing.
- **PP-LCNet for scans: adopt only with a confidence gate.**
  - On 2,355 turned scan tables it fixes 89.5%: +19.0 points [+17.8, +20.2].
  - Ungated, it turns 9.4% of upright scan tables (37% historical ledgers, 16% WTW, 1.2% TabRecSet) and costs −0.98 points on upright answer-key crops [−1.26, −0.69].
  - Gate picked on the dev split (turn only when confidence ≥ 0.8; computed afterwards from the per-item files): on the test split upright scans +0.20 points (1.9% turned), rotated scans +16.8 points (80% fixed).
  - On text PDFs the glyph-angle vote stays better: 100% right on 590 rotated pages, 0 of 12,994 upright tables turned. PP-LCNet turns 0.3–3.8% of upright digital tables (FinTabNet.c 3.5%).

## Datasets (20,046 answer-key tables; 15,589 found by docling's layout and compared; 15,537 scorable)

| set | source and words | licence (from the files) | tables | docling found |
|---|---|---|---|---|
| r4pt | PubTables-1M test, PMC source PDFs (KDD bundle); docling words | bundle card MIT; PubTables-1M CDLA-Permissive-2.0 | 3000 | 2977 |
| r4scitsr | SciTSR test (incl. COMP), arXiv source PDFs; docling words | as round 4 | 3000 | 2896 |
| ptn | PubTabNet val: image page + **answer-key cell text** as the text layer (optimistic words) | CDLA-Permissive-1.0 | 2500 | 2179 |
| pt2 | PubTables-v2 cropped test: image + its PDF words as the text layer | CDLA-Permissive-2.0 | 1500 | 1332 |
| ftnc | FinTabNet.c test: image + its PDF words as the text layer; structure from its PDF annotations (**no page PDFs exist**) | CDLA-Permissive-2.0 | 2500 | 1663 |
| synfin | SynFinTabs test: image + generator word boxes | MIT (card) | 1500 | 1468 |
| r4icdar | ICDAR-2013 corrected, real government PDFs (KDD bundle) | no explicit licence; evaluation only | 151 | 151 |
| r4brgm | Table-BRGM real FR/EN reports (KDD bundle) | bundle card MIT | 328 | 328 |
| ctdarm | ICDAR-2019 cTDaR modern; image-only, OCR words; no answer-key text | no licence file; evaluation only | 145 | 134 |
| ctdarh | cTDaR historical (archival ledgers) | same | 925 | 132 |
| tabrec | TabRecSet English all-/no-/three-line photos and scans | CC BY 4.0 | 1800 | 1019 |
| wtw | WTW test, ModelScope repack `yichao666/WTW_test` (crops + HTML) | CC BY-NC 4.0 (WTW License file) | 1997 | 719 |
| rotpdf | round-4 pages turned 90/180/270 (/Rotate or content) | as sources | 700 | 591 |

**Skipped:**
- WTW official download: gated behind a Tianchi login; the ModelScope repack was used instead.
- PubTables-v2 multi-page tables: one grid per whole multi-page table and no per-page structure, so page fragments cannot be scored.
- FinTabNet page PDFs: no source with a licence found.

All datasets stay in Actions caches only; artifacts carry no table text.

## Method and caveats

- **Arms on the same tables:** tables docling's fp32 layout found (IoU ≥ 0.3 to the answer key), with the harness crop rule (layout box + 4 pt).
- **TableFormer:** docling-rs 2.84.0. Text pages ran with OCR off; scans with the harness OCR settings (PP-OCRv6 small, ocr_scale 3).
- **TATR:** v1.1 fp16 with docling-pdf `word_cells` (`tables/r5/words`), or with PP-OCRv6 small det + rec on the crop for scans; the glyph-angle turn is applied as in the harness.
- **Combo:** the committed harness `table_selector.json` (sha256 c27d7295…, checked), with the harness rules: TableFormer when TATR is empty, TATR on rotated tables.
- **Logic:** tuned on the dev split (30% of documents); everything is reported on the test split.
- **Model files:** TableFormer as the harness loads it ≈ **263.7 MB** (encoder 107.8 + decoder_kv 116.0 + layers 0.2 + bbox 39.7); TATR fp16 **58.3 MB**; PP-LCNet 6.8 MB; OCR det 9.9 + rec 21.2 MB.
- **Docling coverage on scans is low:** historical 14%, WTW 36%, TabRecSet 57%; FinTabNet.c pages 66%.
- **Scan numbers are hypothetical for the harness:** it currently keeps TableFormer on scans, so TATR and the combo there assume OCR words are fed to them.
- **CPU times** are Python on mixed runner CPUs (TATR and the selector run in Rust in the harness).
- **Excluded tables:** 51 answer keys over 1,200 cells (unscorable for every arm) and 1 TATR error.

---

## Generated report (analyze8.py)


All numbers below are on the **test split** (70% of documents, fixed by a hash of the document id) unless marked; the logic repairs were tuned on the dev split only.
Value per table: TEDS-Struct (GriTS-Top where the answer key or the predicted grid exceeds the TEDS cap). Exact shape: rows and columns both right. CIs: 95% bootstrap over documents; p: Wilcoxon.

### Overall, test split (10945 tables, 8858 documents)

| arm | TEDS-Struct | GriTS-Top | GriTS-Con | exact shape % |
|---|---|---|---|---|
| TableFormer (docling-rs 2.84.0) | 0.8855 | 0.9026 | 0.8546 | 72.6 |
| TATR v1.1 fp16 alone | 0.9018 | 0.9211 | 0.8873 | 68.7 |
| TATR + logic repairs | 0.9022 | 0.9209 | 0.8877 | 69.2 |
| Combo: TF + TATR + committed selector | 0.9153 | 0.9312 | 0.8991 | 73.2 |
| Combo, selector retrained (CV) | 0.9235 | 0.9383 | 0.9065 | 75.2 |
| Combo, retrained, TF + TATR+logic | 0.9235 | 0.9382 | 0.9064 | 75.4 |
| Best of TF/TATR per table | 0.9348 | 0.9479 | 0.9175 | 79.9 |

### Overall, all tables (dev + test; logic numbers on dev are in-sample) (15537 tables, 12545 documents)

| arm | TEDS-Struct | GriTS-Top | GriTS-Con | exact shape % |
|---|---|---|---|---|
| TableFormer (docling-rs 2.84.0) | 0.8840 | 0.9013 | 0.8533 | 72.1 |
| TATR v1.1 fp16 alone | 0.9002 | 0.9197 | 0.8859 | 68.0 |
| TATR + logic repairs | 0.9006 | 0.9196 | 0.8863 | 68.6 |
| Combo: TF + TATR + committed selector | 0.9138 | 0.9300 | 0.8980 | 72.5 |
| Combo, selector retrained (CV) | 0.9223 | 0.9374 | 0.9058 | 74.6 |
| Combo, retrained, TF + TATR+logic | 0.9224 | 0.9373 | 0.9058 | 74.8 |
| Best of TF/TATR per table | 0.9336 | 0.9469 | 0.9167 | 79.2 |

### By document type (test split): TEDS-Struct / exact shape

| group | tables | TableFormer (docling-rs 2.84.0) | TATR v1.1 fp16 alone | TATR + logic repairs | Combo: TF + TATR + committed selector | Combo, selector retrained (CV) | combo − TATR(+logic) TEDS-S pts [95% CI] |
|---|---|---|---|---|---|---|---|
| scientific | 6623 | 0.9263 / 80.7% | 0.9504 / 80.1% | 0.9515 / 81.1% | 0.9580 / 83.0% | 0.9579 / 82.4% | +0.65 [+0.55, +0.77] p=7.1e-17 (n=6623, +530/−159) |
| financial | 2220 | 0.9084 / 72.4% | 0.8859 / 50.2% | 0.8859 / 50.1% | 0.9096 / 60.0% | 0.9371 / 71.4% | +2.37 [+2.05, +2.71] p=4.4e-31 (n=2220, +372/−52) |
| government/business | 426 | 0.9694 / 90.1% | 0.9225 / 77.2% | 0.9205 / 76.8% | 0.9684 / 89.0% | 0.9706 / 89.7% | +4.79 [+1.50, +8.17] p=2.7e-13 (n=426, +103/−9) |
| scans/photos | 1278 | 0.6708 / 38.3% | 0.6572 / 36.1% | 0.6545 / 34.9% | 0.6768 / 38.6% | 0.6991 / 38.7% | +2.23 [+1.56, +2.86] p=7.7e-09 (n=1278, +389/−255) |
| rotated | 398 | 0.6795 / 30.2% | 0.9454 / 78.6% | 0.9482 / 79.9% | 0.9454 / 78.6% | 0.9454 / 78.6% | -0.28 [-0.48, -0.11] p=0.28 (n=398, +0/−11) |

### By dataset (test split): TEDS-Struct / exact shape

| group | tables | TableFormer (docling-rs 2.84.0) | TATR v1.1 fp16 alone | TATR + logic repairs | Combo: TF + TATR + committed selector | Combo, selector retrained (CV) | combo − TATR(+logic) TEDS-S pts [95% CI] |
|---|---|---|---|---|---|---|---|
| r4pt: PubTables-1M test, PMC source PDFs (KDD bundle) | 2052 | 0.9059 / 80.7% | 0.9653 / 81.9% | 0.9664 / 83.1% | 0.9684 / 83.7% | 0.9630 / 83.2% | +0.20 [+0.07, +0.36] p=0.1 (n=2052, +79/−40) |
| r4scitsr: SciTSR test, arXiv source PDFs (incl. COMP) | 2026 | 0.9403 / 80.7% | 0.9489 / 82.1% | 0.9496 / 82.5% | 0.9635 / 85.1% | 0.9598 / 84.1% | +1.39 [+1.14, +1.67] p=2.3e-22 (n=2026, +276/−26) |
| ptn: PubTabNet val (image + answer-key cell text as the text layer) | 1572 | 0.9601 / 85.8% | 0.9449 / 87.3% | 0.9445 / 87.4% | 0.9488 / 89.1% | 0.9602 / 88.4% | +0.43 [+0.27, +0.62] p=0.0028 (n=1572, +84/−21) |
| pt2: PubTables-v2 cropped test (image + PDF words as the text layer) | 973 | 0.8854 / 72.4% | 0.9311 / 60.5% | 0.9351 / 64.1% | 0.9394 / 67.3% | 0.9396 / 67.8% | +0.44 [+0.12, +0.77] p=0.24 (n=973, +91/−72) |
| ftnc: FinTabNet.c test (image + PDF words as the text layer) | 1190 | 0.9057 / 83.9% | 0.9540 / 80.5% | 0.9541 / 80.5% | 0.9538 / 81.4% | 0.9528 / 81.4% | -0.02 [-0.17, +0.12] p=0.66 (n=1190, +23/−31) |
| synfin: SynFinTabs test (image + generator words as the text layer) | 1030 | 0.9114 / 59.0% | 0.8073 / 15.1% | 0.8072 / 15.0% | 0.8585 / 35.2% | 0.9191 / 59.7% | +5.13 [+4.58, +5.70] p=3.6e-52 (n=1030, +349/−21) |
| r4icdar: ICDAR-2013 corrected, EU/US government PDFs (KDD bundle) | 105 | 0.9606 / 93.3% | 0.9663 / 92.4% | 0.9663 / 92.4% | 0.9777 / 96.2% | 0.9649 / 94.3% | +1.15 [+0.44, +1.98] p=0.016 (n=105, +14/−0) |
| r4brgm: Table-BRGM, FR/EN reports (KDD bundle) | 228 | 0.9838 / 93.9% | 0.9099 / 70.6% | 0.9109 / 72.8% | 0.9862 / 91.7% | 0.9854 / 92.1% | +7.53 [+1.71, +12.51] p=7.4e-12 (n=228, +71/−3) |
| ctdarm: ICDAR-2019 cTDaR modern (images, OCR words) | 93 | 0.9439 / 77.4% | 0.9039 / 76.3% | 0.8925 / 68.8% | 0.9141 / 74.2% | 0.9410 / 78.5% | +2.16 [+0.52, +3.98] p=0.037 (n=93, +18/−6) |
| ctdarh: ICDAR-2019 cTDaR historical (images, OCR words) | 63 | 0.7102 / 46.0% | 0.7474 / 44.4% | 0.7214 / 42.9% | 0.7314 / 42.9% | 0.7809 / 46.0% | +1.00 [-3.32, +5.71] p=0.46 (n=63, +16/−12) |
| tabrec: TabRecSet English (scans and photos, incl. curved) | 723 | 0.8124 / 49.9% | 0.7802 / 46.2% | 0.7756 / 45.1% | 0.8008 / 50.5% | 0.8261 / 50.8% | +2.52 [+1.70, +3.37] p=1e-08 (n=723, +203/−100) |
| wtw: WTW test (photos, scans, web; ModelScope repack) | 492 | 0.4578 / 20.1% | 0.4650 / 20.1% | 0.4679 / 18.9% | 0.4875 / 20.5% | 0.5019 / 20.1% | +1.96 [+1.04, +2.93] p=0.039 (n=492, +170/−143) |
| rotpdf: Rotated text-PDF pages (90/180/270; /Rotate and content) | 398 | 0.6795 / 30.2% | 0.9454 / 78.6% | 0.9482 / 79.9% | 0.9454 / 78.6% | 0.9454 / 78.6% | -0.28 [-0.48, -0.11] p=0.28 (n=398, +0/−11) |

### Slices (test split): TEDS-Struct / exact shape

| group | tables | TableFormer (docling-rs 2.84.0) | TATR v1.1 fp16 alone | TATR + logic repairs | Combo: TF + TATR + committed selector | Combo, selector retrained (CV) | combo − TATR(+logic) TEDS-S pts [95% CI] |
|---|---|---|---|---|---|---|---|
| complex: >= 3 spanning cells | 3500 | 0.7778 / 58.5% | 0.8536 / 57.5% | 0.8549 / 58.5% | 0.8600 / 59.7% | 0.8588 / 60.1% | +0.51 [+0.33, +0.68] p=0.00023 (n=3500, +435/−308) |
| nested header | 1226 | 0.8888 / 74.6% | 0.9224 / 68.2% | 0.9245 / 70.2% | 0.9260 / 71.7% | 0.9323 / 71.9% | +0.15 [-0.06, +0.36] p=0.74 (n=1226, +85/−80) |
| long (>= 30 rows) | 1332 | 0.8656 / 62.3% | 0.9143 / 54.2% | 0.9146 / 54.4% | 0.9238 / 59.7% | 0.9268 / 60.4% | +0.92 [+0.58, +1.30] p=3.7e-07 (n=1332, +161/−56) |
| wide (>= 10 columns) | 1366 | 0.8051 / 53.5% | 0.8382 / 48.9% | 0.8404 / 53.1% | 0.8477 / 54.6% | 0.8507 / 54.7% | +0.73 [+0.37, +1.09] p=0.0029 (n=1366, +243/−176) |
| multi-line cells | 1441 | 0.8940 / 69.6% | 0.8955 / 63.3% | 0.8959 / 64.2% | 0.9078 / 68.1% | 0.9140 / 67.8% | +1.20 [+0.87, +1.53] p=1e-06 (n=1441, +202/−95) |
| SciTSR COMP (complicated) | 514 | 0.8937 / 68.5% | 0.9234 / 74.9% | 0.9237 / 75.3% | 0.9439 / 78.0% | 0.9403 / 76.8% | +2.02 [+1.60, +2.46] p=1.4e-23 (n=514, +176/−20) |
| bucket simple | 1007 | 0.9487 / 84.7% | 0.9626 / 87.5% | 0.9627 / 87.6% | 0.9754 / 90.9% | 0.9728 / 90.1% | +1.27 [+0.92, +1.65] p=1.5e-06 (n=1007, +96/−12) |
| bucket medium | 1777 | 0.9369 / 83.8% | 0.9619 / 86.7% | 0.9617 / 86.6% | 0.9690 / 88.9% | 0.9702 / 88.2% | +0.74 [+0.50, +1.00] p=1.4e-06 (n=1777, +137/−26) |
| bucket complex | 4910 | 0.8474 / 66.9% | 0.8651 / 59.4% | 0.8651 / 59.3% | 0.8812 / 64.8% | 0.8982 / 69.8% | +1.62 [+1.43, +1.82] p=6.2e-39 (n=4910, +782/−245) |
| bucket edge | 3251 | 0.8955 / 71.2% | 0.9056 / 67.1% | 0.9069 / 69.1% | 0.9187 / 71.8% | 0.9211 / 71.7% | +1.18 [+0.70, +1.81] p=4.3e-08 (n=3251, +379/−203) |

### Paired differences (test split; TEDS-Struct in points, exact shape in percentage points)

| comparison | TEDS-Struct | exact shape |
|---|---|---|
| TATR − TableFormer | +1.63 [+1.28, +1.96] p=1.1e-15 (n=10945, +3462/−2885) | -3.85 [-4.87, -2.92] p=9.7e-12 (n=10945, +799/−1220) |
| TATR+logic − TATR | +0.03 [-0.01, +0.08] p=0.057 (n=10945, +323/−218) | +0.51 [+0.27, +0.75] p=0.29 (n=10945, +122/−66) |
| combo − TATR | +1.35 [+1.16, +1.57] p=4.2e-65 (n=10945, +1360/−345) | +4.50 [+3.91, +5.17] p=6.8e-19 (n=10945, +573/−81) |
| combo − TATR+logic | +1.31 [+1.13, +1.53] p=4.4e-52 (n=10945, +1394/−486) | +3.98 [+3.42, +4.66] p=4.2e-15 (n=10945, +560/−124) |
| combo − TableFormer | +2.98 [+2.71, +3.23] p=2.2e-110 (n=10945, +3117/−1525) | +0.65 [-0.07, +1.35] p=0.23 (n=10945, +718/−647) |
| retrained − committed selector | +0.82 [+0.70, +0.94] p=4.7e-24 (n=10945, +1115/−520) | +2.01 [+1.58, +2.43] p=6e-05 (n=10945, +374/−154) |
| retrained combo with TATR+logic − retrained combo | -0.01 [-0.06, +0.05] p=0.3 (n=10945, +279/−222) | +0.17 [-0.02, +0.36] p=0.72 (n=10945, +67/−48) |

## Decision (rule fixed in advance)

- TATR candidate: **TATR + logic repairs** (TATR+logic counts only if it beats TATR alone on the test split).
- Combo (committed selector) − candidate, overall: TEDS-Struct **+1.31 points** (limit < 0.5), exact shape **+3.98 pp** (limit < 1).
- Combo − candidate by type (TEDS-Struct points; limit: candidate not more than 2 worse): scientific +0.65, financial +2.37, government/business +4.79, scans/photos +2.23, rotated -0.28
- Same by type, exact shape pp (for information): scientific +1.86, financial +9.86, government/business +12.21, scans/photos +3.68, rotated -1.26
- **Decision: keep the combo.** With the retrained selector as the combo: TEDS +2.14, shape +5.99, worst type +5.12 → combo.

## Logic repairs (arm 3), tuned on the dev split

Chosen: `{"CM": {"share": 0.5}}`; dev gain TEDS-Struct +0.06 points, shape +0.61 pp. Tables changed (all splits): 769.

| rule | setting | dev Δ TEDS-S (pts) | dev Δ shape (pp) | worst type Δ (pts) | tables changed |
|---|---|---|---|---|---|
| E | `{"rows": "off", "cols": "thin"}` | +0.004 | +0.065 | +0.000 | 5 |
| E | `{"rows": "off", "cols": "all"}` | +0.005 | +0.065 | +0.000 | 6 |
| E | `{"rows": "thin", "cols": "off"}` | -0.000 | +0.000 | -0.001 | 4 |
| E | `{"rows": "thin", "cols": "thin"}` | +0.004 | +0.065 | +0.000 | 9 |
| E | `{"rows": "thin", "cols": "all"}` | +0.005 | +0.065 | +0.000 | 10 |
| E | `{"rows": "all", "cols": "off"}` | -0.077 | -0.523 | -0.121 | 160 |
| E | `{"rows": "all", "cols": "thin"}` | -0.072 | -0.457 | -0.115 | 165 |
| E | `{"rows": "all", "cols": "all"}` | -0.072 | -0.457 | -0.115 | 166 |
| CM | `{"share": 0.15}` | -0.176 | +0.218 | -1.368 | 333 |
| CM | `{"share": 0.3}` | -0.074 | +0.457 | -0.779 | 293 |
| CM | `{"share": 0.5}` | +0.058 | +0.610 | -0.004 | 223 |
| CS | `{"gap": 1.0, "frac": 0.5}` | -0.189 | -0.806 | -0.960 | 49 |
| CS | `{"gap": 1.0, "frac": 0.8}` | -0.139 | -0.566 | -0.705 | 36 |
| CS | `{"gap": 2.0, "frac": 0.5}` | -0.022 | -0.087 | -0.112 | 4 |
| CS | `{"gap": 2.0, "frac": 0.8}` | -0.005 | -0.022 | -0.024 | 1 |
| RS | `{"m": 2, "numeric": true}` | +0.040 | +0.544 | -0.217 | 260 |
| RS | `{"m": 2, "numeric": false}` | -2.108 | -10.366 | -5.650 | 1253 |
| RS | `{"m": 3, "numeric": true}` | -0.055 | +0.436 | -0.137 | 149 |
| RS | `{"m": 3, "numeric": false}` | -1.064 | -5.836 | -3.769 | 682 |
| J | `{"G": 0.35, "R": 0.6, "cue": true, "header": false}` | -0.120 | -0.174 | -0.805 | 114 |
| J | `{"G": 0.35, "R": 0.6, "cue": false, "header": false}` | -1.197 | -2.003 | -5.473 | 439 |
| J | `{"G": 0.35, "R": 0.85, "cue": true, "header": false}` | -0.154 | -0.174 | -1.435 | 138 |
| J | `{"G": 0.35, "R": 0.85, "cue": false, "header": false}` | -1.462 | -2.809 | -5.695 | 532 |
| J | `{"G": 0.35, "R": Infinity, "cue": true, "header": false}` | -0.476 | -0.828 | -3.995 | 221 |
| J | `{"G": 0.35, "R": Infinity, "cue": false, "header": false}` | -4.261 | -7.295 | -11.691 | 895 |
| J | `{"G": 0.6, "R": 0.6, "cue": true, "header": false}` | -0.143 | -0.174 | -0.805 | 122 |
| J | `{"G": 0.6, "R": 0.6, "cue": false, "header": false}` | -1.323 | -2.199 | -5.697 | 475 |
| J | `{"G": 0.6, "R": 0.85, "cue": true, "header": false}` | -0.202 | -0.240 | -1.461 | 158 |
| J | `{"G": 0.6, "R": 0.85, "cue": false, "header": false}` | -1.811 | -4.138 | -5.981 | 649 |
| J | `{"G": 0.6, "R": Infinity, "cue": true, "header": false}` | -0.874 | -1.829 | -4.003 | 304 |
| J | `{"G": 0.6, "R": Infinity, "cue": false, "header": false}` | -7.797 | -13.741 | -12.952 | 1365 |
| J | `{"G": 1.0, "R": 0.6, "cue": true, "header": false}` | -0.152 | -0.196 | -0.805 | 126 |
| J | `{"G": 1.0, "R": 0.6, "cue": false, "header": false}` | -1.434 | -2.308 | -5.979 | 487 |
| J | `{"G": 1.0, "R": 0.85, "cue": true, "header": false}` | -0.226 | -0.414 | -1.461 | 171 |
| J | `{"G": 1.0, "R": 0.85, "cue": false, "header": false}` | -2.025 | -4.834 | -6.394 | 693 |
| J | `{"G": 1.0, "R": Infinity, "cue": true, "header": false}` | -1.279 | -3.092 | -4.003 | 381 |
| J | `{"G": 1.0, "R": Infinity, "cue": false, "header": false}` | -12.509 | -22.234 | -14.929 | 1844 |
| CM | `{"CM": {"share": 0.5}}` | +0.058 | +0.610 | -0.004 | 223 |

## Selectors

- combo: picks TATR on 4743 of 10945 test tables (43.3%).
- combo_retrained: picks TATR on 3603 of 10945 test tables (32.9%).
- combo_retrained_logic: picks TATR on 3683 of 10945 test tables (33.7%).
- Retrained selector: XGBoost (round-5 settings), 5-fold GroupKFold by document over 14936 unrotated tables; out-of-fold picks.
- Selector feature time (Python): p50 94.1 ms, p90 1144.2 ms per table.

## Orientation

**PP-LCNet on upright tables (label ≠ 0 = it would turn an upright table):**

| set | tables | wrongly turned % |
|---|---|---|
| r4pt | 2977 | 1.4 |
| r4scitsr | 2896 | 0.3 |
| ptn | 2179 | 1.7 |
| pt2 | 1332 | 3.8 |
| ftnc | 1663 | 3.5 |
| synfin | 1468 | 0.7 |
| r4icdar | 151 | 1.3 |
| r4brgm | 328 | 0.6 |
| ctdarm | 134 | 3.0 |
| ctdarh | 132 | 37.1 |
| tabrec | 1019 | 1.2 |
| wtw | 719 | 16.0 |
| type: scientific | 9384 | 1.5 |
| type: financial | 3131 | 2.2 |
| type: government/business | 613 | 1.3 |
| type: scans/photos | 1870 | 9.4 |

**Scans on the answer-key crops** (every scan table, found by docling or not; OCR words; no TableFormer here):

| set | tables | TATR + OCR TEDS-Struct | exact shape % | PP-LCNet turns an upright crop % |
|---|---|---|---|---|
| ctdarm | 145 | 0.9241 | 78.6 | 0.7 |
| ctdarh | 925 | 0.5460 | 7.6 | 21.8 |
| tabrec | 1800 | 0.6967 | 32.1 | 1.9 |
| wtw | 1997 | 0.4378 | 13.6 | 12.0 |

Answer-key crop errors: 0.

Upright scans, answer-key crops (4773): with PP-LCNet − without: TEDS-Struct -0.98 [-1.26, -0.69] p=0.0013 (n=4773, +138/−257); shape -0.25 pp [-0.44, -0.08] p=0.73 (n=4773, +3/−15).

**Rotated scans** (2355 scan tables turned 90/180/270, OCR + TATR on the turned crop): PP-LCNet right on 89.5% (the opposite-direction reading would be right on 33.2%, a convention check).

| arm | TEDS-Struct | exact shape % |
|---|---|---|
| no orientation fix | 0.3669 | 4.9 |
| PP-LCNet then OCR + TATR | 0.5566 | 21.1 |
| upright (oracle) | 0.5660 | 21.4 |

PP-LCNet − no fix on rotated scans: +18.97 [+17.81, +20.19] p=6.5e-165 (n=2355, +1610/−527)

Upright scans on the run-time crops (1956, all splits): with PP-LCNet − without: TEDS-Struct -0.23 [-0.51, +0.04] p=0.16 (n=1956, +48/−81); shape -0.31 pp [-0.61, -0.05] p=0.79 (n=1956, +1/−7).

Rotated text-PDF tables docling found: 590; glyph-angle vote right on 100.0% (PP-LCNet 98.5%). Upright text tables wrongly turned by the vote: 0.00% of 12994.

## Coverage (tables docling's layout found; only those are compared)

| set | answer-key tables | docling found | conversion errors | TATR-job errors |
|---|---|---|---|---|
| r4pt | 3000 | 2977 (99.2%) | 0 | 0 |
| r4scitsr | 3000 | 2896 (96.5%) | 0 | 0 |
| ptn | 2500 | 2179 (87.2%) | 0 | 0 |
| pt2 | 1500 | 1332 (88.8%) | 0 | 0 |
| ftnc | 2500 | 1663 (66.5%) | 0 | 0 |
| synfin | 1500 | 1468 (97.9%) | 0 | 0 |
| r4icdar | 151 | 151 (100.0%) | 0 | 0 |
| r4brgm | 328 | 328 (100.0%) | 0 | 0 |
| ctdarm | 145 | 134 (92.4%) | 0 | 0 |
| ctdarh | 925 | 132 (14.3%) | 0 | 0 |
| tabrec | 1800 | 1019 (56.6%) | 0 | 0 |
| wtw | 1997 | 719 (36.0%) | 0 | 0 |
| rotpdf | 700 | 591 (84.4%) | 0 | 1 |

Excluded from the comparison: {'answer key over 1,200 cells (unscorable)': 51, 'tatr job error': 1}.

## CPU time per table (GitHub-hosted runners, 4 vCPU; medians, p90)

| step | p50 ms | p90 ms |
|---|---|---|
| TableFormer (docling stage, per table) | 772 | 1584 |
| TATR fp16 (pre + model + post, Python) | 304 | 686 |
| TATR model only | 229 | 468 |
| PP-OCRv6 small on the crop (scans) | 698 | 1656 |
| PP-LCNet | 7 | 12 |
| glyph-angle vote | 14 | 70 |
| selector features (Python) | 94 | 1144 |
| logic repairs (Python) | 2 | 7 |

CPUs seen in the TATR jobs: {'AMD EPYC 9V74 80-Core Processor': 4995, 'AMD EPYC 9V45 96-Core Processor': 3736, 'INTEL(R) XEON(R) PLATINUM 8573C': 1218, 'AMD EPYC 7763 64-Core Processor': 6304, 'Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz': 1275, 'Intel(R) Xeon(R) 6975P-C': 2518}

**Model files (MB):** layout_heron.onnx 172.2, layout_heron_int8.onnx 68.7, ocr_det 9.9, ocr_rec_v6 21.2, pplcnet 6.8, tableformer/bbox.onnx 0.1, tableformer/bbox.onnx.data 39.6, tableformer/decoder.onnx 0.4, tableformer/decoder.onnx.data 77.8, tableformer/decoder_int8.onnx 49.9, tableformer/decoder_kv.onnx 0.4, tableformer/decoder_kv.onnx.data 115.6, tableformer/decoder_kv_layers.onnx 0.2, tableformer/encoder.onnx 107.8, tableformer/encoder_fp16.onnx 54.0, tatr_fp16 58.3
