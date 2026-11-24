# Tables round 5: TATR and the selector on docling.rs words (harness port)

Run 45317894543 (workflow tables-r5, artifacts r5tatr-0..4 and r5-sel). Words and crops: docling-pdf 2.84.0 `PdfPage::word_cells` and its 2x page render (tool in `words/`), the source the harness reads at run time. TableFormer predictions reused from round-4 run 44879805191 (they do not depend on our words).

Value per table: TEDS-Struct (GriTS-Top where the grid exceeds 1200 cells). Tables: 6476, documents (CV groups): 5509. TableFormer = round-4 predictions (docling-rs 2.84.0, unchanged). TATR = v1.1 fp16 on docling.rs crops with docling.rs words. Selector = round-3 features on docling.rs words + XGBoost, 5-fold GroupKFold by document. CI = 95% bootstrap over documents.

| dataset | tables | docs | TableFormer | TATR fp16 (docling words) | selector (CV) | oracle | gain vs best single [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| all | 6476 | 5509 | 0.9160 | 0.9514 | 0.9613 | 0.9685 | +0.0099 [+0.0062, +0.0143] | 6.2e-23 |
| KDD PubTables (PMC PDFs) | 2999 | 2895 | 0.9002 | 0.9636 | 0.9634 | 0.9705 | -0.0003 [-0.0015, +0.0009] | 0.71 |
| KDD Table-BRGM (FR/EN reports) | 328 | 25 | 0.9783 | 0.8936 | 0.9833 | 0.9870 | +0.0051 [-0.0016, +0.0124] | 0.13 |
| KDD ICDAR-2013 (corrected) | 151 | 66 | 0.9556 | 0.9735 | 0.9690 | 0.9912 | -0.0045 [-0.0195, +0.0064] | 0.89 |
| SciTSR test (original PDFs) | 2998 | 2523 | 0.9230 | 0.9443 | 0.9564 | 0.9634 | +0.0121 [+0.0098, +0.0143] | 4.3e-22 |

**Round 4 (pdfplumber words): TableFormer 0.916, TATR 0.951, selector 0.961, oracle 0.968. Round 5 (docling.rs words): TableFormer 0.9160, TATR 0.9514, selector 0.9613, oracle 0.9685. Selector vs round 4: +0.0003.**

Selector picks TATR on 2080 of 6476 tables (32.1%).
Feature time (Python) p50 17.8 ms, p95 520.3 ms.
XGBoost feature importance (final model, all tables): agree 0.249, d_spans 0.096, d_rows 0.076, b_split 0.038, d_multiline 0.028, b_rows 0.023, a_rowh_cv 0.022, b_rowh_cv 0.019, d_empty 0.019, d_n_cells 0.018, d_cam_acc 0.018, a_cam_score 0.016
Exported 300 trees; export check max |margin difference| 6.68e-06.

Exported trees: `table_selector.json` in the r5-sel artifact, committed to the harness as `crates/contextual-worker/src/tables/table_selector.json`.

## Round 5e: the harness's crop rule (run 45379814159)

Value per table: TEDS-Struct (GriTS-Top where the grid exceeds 1200 cells). Tables: 6350, documents (CV groups): 5417. TableFormer = round-4 predictions (docling-rs 2.84.0, unchanged). TATR = v1.1 fp16 on docling.rs crops with docling.rs words. Selector = round-3 features on docling.rs words + XGBoost, 5-fold GroupKFold by document. CI = 95% bootstrap over documents.

| dataset | tables | docs | TableFormer | TATR fp16 (docling words) | selector (CV) | oracle | gain vs best single [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| all | 6350 | 5417 | 0.9259 | 0.9532 | 0.9627 | 0.9697 | +0.0095 [+0.0058, +0.0139] | 8.1e-21 |
| KDD PubTables (PMC PDFs) | 2976 | 2873 | 0.9061 | 0.9639 | 0.9638 | 0.9715 | -0.0002 [-0.0014, +0.0011] | 0.68 |
| KDD Table-BRGM (FR/EN reports) | 328 | 25 | 0.9811 | 0.8956 | 0.9841 | 0.9875 | +0.0030 [-0.0010, +0.0059] | 0.15 |
| KDD ICDAR-2013 (corrected) | 151 | 66 | 0.9552 | 0.9648 | 0.9680 | 0.9829 | +0.0032 [-0.0034, +0.0106] | 0.15 |
| SciTSR test (original PDFs) | 2895 | 2453 | 0.9384 | 0.9480 | 0.9589 | 0.9651 | +0.0109 [+0.0091, +0.0131] | 8e-20 |

**Round 4 (pdfplumber words): TableFormer 0.916, TATR 0.951, selector 0.961, oracle 0.968. Round 5 (docling.rs words): TableFormer 0.9259, TATR 0.9532, selector 0.9627, oracle 0.9697. Selector vs round 4: +0.0017.**

**Committed selector (round-5 trees, trained on answer-key crops) applied unchanged to these tables:**
- all: TEDS-Struct 0.9660, vs TableFormer +0.0402 [+0.0372, +0.0432]
- KDD PubTables (PMC PDFs): TEDS-Struct 0.9669, vs TableFormer +0.0608 [+0.0569, +0.0648]
- KDD Table-BRGM (FR/EN reports): TEDS-Struct 0.9850, vs TableFormer +0.0039 [+0.0005, +0.0071]
- KDD ICDAR-2013 (corrected): TEDS-Struct 0.9779, vs TableFormer +0.0227 [+0.0076, +0.0397]
- SciTSR test (original PDFs): TEDS-Struct 0.9624, vs TableFormer +0.0240 [+0.0196, +0.0284]

Selector picks TATR on 1965 of 6350 tables (30.9%).
Feature time (Python) p50 29.9 ms, p95 857.3 ms.
XGBoost feature importance (final model, all tables): agree 0.272, d_spans 0.090, d_rows 0.076, b_split 0.028, d_multiline 0.023, b_rows 0.022, a_rowh_cv 0.021, d_split 0.020, b_n_cells 0.019, d_cols 0.017, b_wpc 0.017, b_spans 0.015
Exported 300 trees; export check max |margin difference| 6.20e-06.

## Round 5d: exact table shape (rows and columns both right), committed selector vs single models
Tables: 6350 (the round-5 selector rows). Selector = committed round-5 trees' 5-fold CV picks. 95% CI: bootstrap over documents.

| dataset | tables | TableFormer [CI] | TATR [CI] | selector [CI] | selector - TableFormer [CI] | selector - TATR [CI] | oracle |
|---|---|---|---|---|---|---|---|
| all | 6350 | 0.8109 [0.8006, 0.8209] | 0.8049 [0.7903, 0.8192] | 0.8373 [0.8277, 0.8464] | +0.0265 [+0.0200, +0.0333] | +0.0324 [+0.0206, +0.0454] | 0.8635 |
| ... committed selector | | | | 0.8417 [0.8322, 0.8509] | +0.0309 [+0.0242, +0.0379] | | |
| KDD PubTables (PMC PDFs) | 2976 | 0.8017 [0.7872, 0.8160] | 0.8095 [0.7957, 0.8232] | 0.8243 [0.8101, 0.8374] | +0.0225 [+0.0121, +0.0330] | +0.0148 [+0.0087, +0.0214] | 0.8636 |
| ... committed selector | | | | 0.8266 [0.8130, 0.8403] | +0.0249 [+0.0141, +0.0357] | | |
| KDD Table-BRGM (FR/EN reports) | 328 | 0.9329 [0.9091, 0.9652] | 0.6616 [0.5290, 0.8429] | 0.9177 [0.8896, 0.9558] | -0.0152 [-0.0349, +0.0031] | +0.2561 [+0.0913, +0.3732] | 0.9421 |
| ... committed selector | | | | 0.9177 [0.8896, 0.9540] | -0.0152 [-0.0380, +0.0043] | | |
| KDD ICDAR-2013 (corrected) | 151 | 0.9007 [0.8397, 0.9533] | 0.8344 [0.7517, 0.9067] | 0.8675 [0.7902, 0.9333] | -0.0331 [-0.0722, +0.0000] | +0.0331 [-0.0061, +0.0802] | 0.9272 |
| ... committed selector | | | | 0.8940 [0.8188, 0.9554] | -0.0066 [-0.0473, +0.0312] | | |
| SciTSR test (original PDFs) | 2895 | 0.8017 [0.7851, 0.8169] | 0.8149 [0.7994, 0.8299] | 0.8401 [0.8250, 0.8541] | +0.0383 [+0.0300, +0.0466] | +0.0252 [+0.0185, +0.0326] | 0.8511 |
| ... committed selector | | | | 0.8459 [0.8312, 0.8595] | +0.0442 [+0.0360, +0.0528] | | |
