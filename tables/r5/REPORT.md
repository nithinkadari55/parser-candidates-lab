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
