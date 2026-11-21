# Round 4 table tests (gaps 15, 14, 6, RF-DETR-Large and PP-DocLayout_plus-L layout)

Runs (nithinkadari55/parser-candidates-lab, workflow tables-r4.yml): gap 15 and gap 6 from run 44879805191; gap 14 and RF-DETR-Large from run 44880415246 (the gap 14 DocLayNet shards of 44879805191 crashed on a parser bug, fixed in 1f9e629 and rerun); PP-DocLayout_plus-L from run 44905325277 (detections), 44905597422 (ONNX vs native parity) and 44906925432 (evaluation); the PP section was assembled by report run 44907206361. All model work ran on 4-vCPU ubuntu-latest runners; datasets were downloaded inside the jobs and never committed or uploaded.

Notes: (1) heron INT8 counts 'Table' and 'Document Index' as tables here, so its DocLayNet recall is 0.862 with 331 misses (round 3 counted 'Table' only: 0.839, 385 misses). (2) RF-DETR-Large was fine-tuned on DocLayNet v1.2 train, so DocLayNet test is in-domain for it; OmniDocBench is the out-of-domain check. (3) RF-DETR ran in torch fp32 with torch's default 2 threads; heron INT8 ran in ONNX Runtime with its default threads, so the ms/page are not like for like. (4) Gap 14 bar: precision of added regions >= 0.5 with <= 5 false tables / 100 pages. No train-free proposer (SPARTAN, vector rules) meets it. RF-DETR-Large >= 0.5 in uncovered regions meets it on DocLayNet only (0.561 precision, 0.9 false / 100 pages, 55 of 331 misses), not on OmniDocBench (0.043). (5) PP-DocLayout_plus-L verdict: do not adopt. On OmniDocBench (the deciding set) its shared-class mAP50-95 ties heron INT8 (0.453 vs 0.458) but its paired per-page F1 is lower (-0.067 [-0.085, -0.049]), mostly from formula false positives (precision 0.14) and exam pages (-0.31); it wins on colorful textbooks (+0.17) and magazines (+0.05). On DocLayNet it is far behind (0.320 vs 0.678; F1 -0.176), partly from label-convention differences. Its missed-table arm on OmniDocBench recovers 10 of 43 heron misses but at precision 0.27. (6) The DocLayNet ms/page of PP (999) and the OmniDocBench one (407) come from different runner hosts for a fixed 800x800 input; the parity job measured 446 ms ONNX vs 672 ms native Paddle on one host. Use about 0.4-0.45 s/page, vs about 0.33 s for heron INT8.

## Verdicts

- **Gap 15 (selector on real PDF words): does the selector still win? yes.** Retrained selector vs the better single model per dataset: all: wins (+0.0103, CI [+0.0069, +0.0142] vs best single = TATR); r4pt: no significant difference (-0.0005, CI [-0.0015, +0.0005] vs best single = TATR); r4brgm: no significant difference (+0.0043, CI [-0.0016, +0.0118] vs best single = TableFormer); r4icdar: no significant difference (+0.0008, CI [-0.0063, +0.0082] vs best single = TATR); r4scitsr: wins (+0.0138, CI [+0.0116, +0.0163] vs best single = TATR).
- **Gap 14 (missed tables): no proposer reaches precision >= 0.5 with <= 5 false tables / 100 pages; do not add it to the pipeline as is** (see the per-variant table).
- **Gap 6 (cross-page tables):** our rule set link P 0.991 / R 0.476, false-merge rate 0.0033, exact multi-part assembly 0.475; Table Stitcher port P 0.974 / R 0.625; Textract rules P 0.975 / R 0.134. Region splitting: {"neg": 3536, "pos": 137, "pos_split": 125, "pos_exact": 62, "pos_wrongcut": 0, "neg_split": 241}.
- **Layout: RF-DETR-Large (test only, not a replacement):** dln: mean-class mAP50-95 RF-DETR 0.750 vs heron INT8 0.688, paired page F1 diff +0.0286 [+0.0245, +0.0330], 805 vs 317 ms/page; missed tables (uncovered regions, >= 0.5): RF-DETR recovers 55 of 331 (precision 0.561, 0.9 false/100 pages), YOLOX-L 36 (precision 0.371, 1.2 false/100 pages); odb: mean-class mAP50-95 RF-DETR 0.355 vs heron INT8 0.458, paired page F1 diff -0.0536 [-0.0696, -0.0388], 816 vs 334 ms/page; missed tables (uncovered regions, >= 0.5): RF-DETR recovers 1 of 43 (precision 0.043, 3.0 false/100 pages), YOLOX-L 2 (precision 0.091, 2.7 false/100 pages).
- **Layout: PP-DocLayout_plus-L (test only, OmniDocBench decides):** dln: shared-class mAP50-95 PP 0.320 vs heron 0.678, paired page F1 diff -0.1758 [-0.1829, -0.1683], 999 vs 317 ms/page; missed tables (uncovered, >= 0.5): 16 of 331 recovered, precision 0.195, 1.3 false/100 pages; odb: shared-class mAP50-95 PP 0.453 vs heron 0.458, paired page F1 diff -0.0668 [-0.0849, -0.0492], 407 vs 334 ms/page; missed tables (uncovered, >= 0.5): 10 of 43 recovered, precision 0.270, 3.6 false/100 pages.

## Sources checked from the actual files

- SPARTAN: the real repository is github.com/shlok-n/spartan (MIT, commit 66ada02, v2.2.0). github.com/IvoLeist/spartan-table-extraction is a GitHub fork of it with the same commits. Called unmodified (detect_tables + extract_pdf_words, 300 DPI, OCR off).
- pdf-inspector: github.com/firecrawl/pdf-inspector (MIT, v1.27.0). Its Python wheel exposes only pages_with_tables, not table boxes, so the line/rect proposal idea (detect_lines.rs, detect_rects.rs constants) was ported to Python on pdfplumber paths.
- Table Stitcher: github.com/PebbleRoad/table-stitcher (MIT). AWS rules: aws-samples/amazon-textract-response-parser t_tables.py (Apache-2.0).
- KDD 2026 bundle: HF marijanic/table-extraction-scientific-datasets (card MIT; source PDFs keep their own terms). PubTables subset: 23,567 PDFs (35 GB tar, streamed); Table-BRGM 26 PDFs; ICDAR-2013 (corrected) 67 PDFs.
- SciTSR: Academic-Hammer/SciTSR (MIT code); data from the README Google Drive link (SciTSR.tar.gz, 802 MB), test/pdf + test/structure; arXiv PDFs keep their own terms.
- FinTabNet.c: HF bsmock/FinTabNet.c (CDLA-Permissive-2.0) ships only JSON annotations (no page PDFs); the original IBM FinTabNet download (dax-cdn.cdn.appdomain.cloud) did not respond, so FinTabNet was not tested.
- DocLayNet v1.2 test (CDLA-Permissive-1.0) carries the page PDFs; heron INT8 detections and GT from order run 44477997398 (joined by page hash).
- OmniDocBench: original PDF pages only on the v1_0 branch (ori_pdfs, 981 pages); README: research purposes only, not commercial; used for internal testing only.
- SCI-3000: Zenodo 8165783, open access; record licence CC BY 4.0 while the LICENSE file inside the zip says CC BY-SA 4.0. Both allow research use. Read by HTTP range requests (no full download).
- PubTables-v2 Full Documents test: HF kensho/PubTables-v2, CDLA-Permissive-2.0.
- RF-DETR-Large DocLayNet: HF nie3e/rf-detr-large-doclay @fecf537, model card licence apache-2.0 (no separate LICENSE file in the model repo), base Roboflow/rf-detr-large card apache-2.0, roboflow/rf-detr code Apache-2.0; trained on DocLayNet v1.2 (CDLA-Permissive-1.0). Run through transformers 5.22.2 RfDetrForObjectDetection (code read: sigmoid + top-k post-processing, no NMS).
- PP-DocLayout_plus-L: HF PaddlePaddle/PP-DocLayout_plus-L_onnx @feb7461 (inference.onnx) and PaddlePaddle/PP-DocLayout_plus-L @aa52b85 (native weights): model card tag apache-2.0, no LICENSE file in either repo; PaddleOCR and PaddleX code Apache-2.0. Pre/post-processing from inference.yml and PaddleX 3.7.2 DetPostProcess; ONNX parity with native paddlepaddle 3.2.2 / paddleocr 3.7.0 / paddlex 3.7.2: 1058/1058 boxes.

## Gap 15: table selector on real PDF words (pdfplumber words re-extracted from the source PDFs)
Value per table: TEDS-Struct (GriTS-Top where the grid exceeds 1200 cells). Selector = round-3 features + XGBoost; 'retrained' = 5-fold GroupKFold by document on these 6476 tables only; 'frozen r3' = the round-3 selector trained on the 8250 round-3 tables (answer-key words for PubTabNet/FinTabNet/PT1M) and applied unchanged. CI = 95% bootstrap over documents.

| dataset | tables | docs | TableFormer | TATR fp16 | rule: token fit | LR retrained | XGB retrained | XGB frozen r3 | oracle | best single | gain XGB-retrained vs best [95% CI] | Wilcoxon p | gain frozen-r3 vs best [95% CI] | regret vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 6476 | 5509 | 0.9160 | 0.9509 | 0.9523 | 0.9546 | 0.9612 | 0.9441 | 0.9681 | TATR | +0.0103 [+0.0069, +0.0142] | 1.8e-28 | -0.0068 [-0.0108, -0.0022] | 0.0070 |
| KDD PubTables (PMC PDFs) | 2999 | 2895 | 0.9002 | 0.9629 | 0.9631 | 0.9549 | 0.9624 | 0.9277 | 0.9698 | TATR | -0.0005 [-0.0015, +0.0005] | 0.89 | -0.0352 [-0.0381, -0.0324] | 0.0074 |
| KDD Table-BRGM (FR/EN reports) | 328 | 25 | 0.9783 | 0.9019 | 0.9235 | 0.9453 | 0.9826 | 0.9788 | 0.9864 | TableFormer | +0.0043 [-0.0016, +0.0118] | 0.36 | +0.0005 [-0.0060, +0.0074] | 0.0038 |
| KDD ICDAR-2013 (corrected) | 151 | 66 | 0.9556 | 0.9721 | 0.9721 | 0.9688 | 0.9729 | 0.9691 | 0.9924 | TATR | +0.0008 [-0.0063, +0.0082] | 0.55 | -0.0030 [-0.0194, +0.0137] | 0.0195 |
| SciTSR test (original PDFs) | 2998 | 2523 | 0.9230 | 0.9431 | 0.9438 | 0.9546 | 0.9570 | 0.9554 | 0.9632 | TATR | +0.0138 [+0.0116, +0.0163] | 3.1e-30 | +0.0123 [+0.0101, +0.0146] | 0.0063 |

### By complexity bucket (TEDS-Struct means)
| dataset/bucket | n | TableFormer | TATR fp16 | XGB retrained | XGB frozen r3 | oracle | gain vs best [95% CI] |
|---|---|---|---|---|---|---|---|
| r4pt/simple | 61 | 0.9937 | 0.9945 | 0.9978 | 0.9943 | 1.0000 | +0.0033 [+0.0000, +0.0098] |
| r4pt/medium | 507 | 0.9585 | 0.9843 | 0.9807 | 0.9763 | 0.9908 | -0.0036 [-0.0084, -0.0003] |
| r4pt/complex | 1301 | 0.8555 | 0.9545 | 0.9542 | 0.8962 | 0.9596 | -0.0003 [-0.0013, +0.0006] |
| r4pt/edge | 1130 | 0.9206 | 0.9612 | 0.9617 | 0.9386 | 0.9705 | +0.0005 [-0.0013, +0.0022] |
| r4brgm/simple | 66 | 0.9901 | 0.9594 | 0.9866 | 0.9889 | 0.9960 | -0.0035 [-0.0106, +0.0000] |
| r4brgm/complex | 37 | 0.9679 | 0.9664 | 0.9777 | 0.9679 | 0.9793 | +0.0098 [+0.0011, +0.0171] |
| r4brgm/edge | 221 | 0.9766 | 0.8725 | 0.9824 | 0.9777 | 0.9849 | +0.0058 [-0.0027, +0.0189] |
| r4icdar/simple | 52 | 0.9917 | 0.9891 | 0.9936 | 0.9908 | 0.9973 | +0.0019 [-0.0082, +0.0117] |
| r4icdar/medium | 19 | 0.9380 | 0.9565 | 0.9565 | 0.9898 | 0.9898 | +0.0000 [+0.0000, +0.0000] |
| r4icdar/complex | 52 | 0.9133 | 0.9575 | 0.9660 | 0.9299 | 0.9848 | +0.0084 [+0.0003, +0.0260] |
| r4icdar/edge | 28 | 0.9791 | 0.9782 | 0.9586 | 0.9876 | 0.9995 | -0.0204 [-0.0419, -0.0006] |
| r4scitsr/simple | 985 | 0.9680 | 0.9644 | 0.9787 | 0.9764 | 0.9817 | +0.0106 [+0.0053, +0.0167] |
| r4scitsr/medium | 991 | 0.9081 | 0.9506 | 0.9642 | 0.9619 | 0.9704 | +0.0136 [+0.0090, +0.0187] |
| r4scitsr/complex | 492 | 0.8703 | 0.8934 | 0.9060 | 0.9082 | 0.9185 | +0.0127 [+0.0076, +0.0181] |
| r4scitsr/edge | 530 | 0.9161 | 0.9357 | 0.9504 | 0.9481 | 0.9570 | +0.0146 [+0.0087, +0.0219] |
| r4scitsr/SciTSR-COMP | 714 | 0.9056 | 0.9092 | 0.9317 | 0.9321 | 0.9430 | +0.0225 [+0.0178, +0.0273] |

### TEDS with content (cells filled with our PDF words)
| dataset | n | TableFormer | TATR fp16 | XGB retrained pick |
|---|---|---|---|---|
| r4pt | 2999 | 0.8597 | 0.9243 | 0.9236 |
| r4brgm | 328 | 0.9383 | 0.8654 | 0.9428 |
| r4icdar | 151 | 0.9432 | 0.9563 | 0.9527 |
| r4scitsr | 2998 | 0.8972 | 0.9203 | 0.9345 |

### Failures and TableFormer table detection on the real page
| dataset | TF found table on full page | TF needed crop fallback | TF no table | TF empty | TATR empty | TF fail (<0.5) | TATR fail | XGB-retrained fail |
|---|---|---|---|---|---|---|---|---|
| r4pt | 2974 | 9 | 16 | 17 | 0 | 85 | 30 | 31 |
| r4brgm | 327 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| r4icdar | 151 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| r4scitsr | 2906 | 44 | 48 | 50 | 2 | 175 | 60 | 53 |

### Speed (ms per table on a 4-vCPU GitHub runner, default ONNX threads)
| dataset | TableFormer page pipeline p50 / p95 | TATR fp16 p50 / p95 | selector features p50 / p95 |
|---|---|---|---|
| r4pt | 1515 / 2929 | 236 / 770 | 50.1 / 950.0 |
| r4brgm | 990 / 1892 | 195 / 559 | 9.3 / 475.9 |
| r4icdar | 1210 / 2785 | 277 / 1065 | 42.0 / 1357.8 |
| r4scitsr | 1143 / 1886 | 206 / 587 | 12.7 / 229.5 |

XGBoost feature importance (fitted on all r4 tables): agree 0.257, d_spans 0.086, d_rows 0.071, a_cam_score 0.032, a_cam_acc 0.031, a_tatr_fit 0.030, b_split 0.029, d_multiline 0.024, d_tatr_fit 0.022, b_wpc 0.019, b_rowh_cv 0.016, b_rows 0.016


## Gap 14: missed tables - train-free proposals where heron INT8 found no table
Added region = proposal with < 30% of its area covered by heron INT8 tables (score >= 0.5). 'with figures' keeps proposals inside heron Picture boxes (they always carry aligned words or ruling lines by construction); 'no figures' drops them. A heron miss = GT table without a heron table at IoU >= 0.5. False table = added region with IoU < 0.5 to every GT table.

### DocLayNet test (CDLA-Permissive-1.0): 4999 pages (0 errors), 2394 GT tables, heron INT8 recall@.5 0.862, precision 0.857, misses 331
| proposer | figures | added / 100 pages | heron misses recovered @.5 | inc. recall @.5 | inc. recall @.3 | precision of added @.5 | precision @.3 | false tables / 100 pages | heron+added recall | heron+added precision |
|---|---|---|---|---|---|---|---|---|---|---|
| SPARTAN (boundary + text) | with | 44.9 | 70 of 331 | 0.211 | 0.242 | 0.031 | 0.036 | 43.5 | 0.891 | 0.458 |
| SPARTAN (boundary + text) | no | 23.3 | 35 of 331 | 0.106 | 0.121 | 0.030 | 0.034 | 22.6 | 0.876 | 0.587 |
| SPARTAN boundary only | with | 24.4 | 8 of 331 | 0.024 | 0.027 | 0.007 | 0.007 | 24.3 | 0.865 | 0.571 |
| SPARTAN boundary only | no | 7.4 | 4 of 331 | 0.012 | 0.015 | 0.011 | 0.013 | 7.3 | 0.863 | 0.744 |
| SPARTAN text (borderless) only | with | 20.5 | 62 of 331 | 0.187 | 0.215 | 0.060 | 0.069 | 19.3 | 0.888 | 0.619 |
| SPARTAN text (borderless) only | no | 15.9 | 31 of 331 | 0.094 | 0.106 | 0.039 | 0.044 | 15.3 | 0.875 | 0.654 |
| vector rules (pdf-inspector idea) | with | 9.6 | 12 of 331 | 0.036 | 0.045 | 0.025 | 0.031 | 9.4 | 0.867 | 0.718 |
| vector rules (pdf-inspector idea) | no | 3.8 | 5 of 331 | 0.015 | 0.018 | 0.026 | 0.031 | 3.7 | 0.864 | 0.796 |
| union SPARTAN + vector | with | 50.3 | 78 of 331 | 0.236 | 0.272 | 0.031 | 0.036 | 48.7 | 0.894 | 0.435 |
| union SPARTAN + vector | no | 25.4 | 38 of 331 | 0.115 | 0.133 | 0.030 | 0.035 | 24.7 | 0.878 | 0.571 |
| agreement SPARTAN and vector | with | 4.9 | 4 of 331 | 0.012 | 0.012 | 0.016 | 0.016 | 4.8 | 0.863 | 0.780 |
| agreement SPARTAN and vector | no | 1.9 | 2 of 331 | 0.006 | 0.006 | 0.022 | 0.022 | 1.8 | 0.863 | 0.826 |

**Miss taxonomy (dln)** - tags overlap; recovered = union SPARTAN + vector, figures kept
| tag | GT tables | heron miss rate | misses | recovered @.5 | recovered @.3 |
|---|---|---|---|---|---|
| borderless | 1206 | 0.214 | 258 | 65 | 75 |
| tiny or key-value | 455 | 0.365 | 166 | 35 | 41 |
| in-figure | 121 | 0.851 | 103 | 39 | 46 |
| heron box partial (IoU .1-.5) | 98 | 1.000 | 98 | 0 | 0 |
| horizontal rules only | 888 | 0.073 | 65 | 7 | 9 |
| heron label: Key-Value Region | 56 | 0.411 | 23 | 12 | 13 |
| heron label: Picture | 36 | 0.528 | 19 | 7 | 9 |
| low-contrast | 30 | 0.400 | 12 | 3 | 4 |
| heron label: Text | 14 | 0.857 | 12 | 2 | 3 |
| heron label: Code | 10 | 0.900 | 9 | 2 | 2 |
| ruled grid | 300 | 0.027 | 8 | 6 | 6 |
| heron label: Form | 9 | 0.778 | 7 | 5 | 5 |
| heron label: List-item | 5 | 0.800 | 4 | 2 | 2 |
| rotated | 9 | 0.222 | 2 | 1 | 1 |
| no text layer | 6 | 0.333 | 2 | 0 | 0 |
| heron label: Page-header | 1 | 1.000 | 1 | 0 | 0 |

Speed per page: SPARTAN 300 DPI (render + words + detect) p50 153 ms / p95 289 ms; vector proposer p50 62 / p95 349 ms; heron INT8 p50 317 ms. Mappings used: {'stretch': 4955, 'fit-topleft': 22, 'fit-center': 22}; vector notes: {'rotated page: no vector proposals': 14}

### OmniDocBench EN pages with original PDFs (research-only): 286 pages (0 errors), 127 GT tables, heron INT8 recall@.5 0.937, precision 0.952, misses 8
| proposer | figures | added / 100 pages | heron misses recovered @.5 | inc. recall @.5 | inc. recall @.3 | precision of added @.5 | precision @.3 | false tables / 100 pages | heron+added recall | heron+added precision |
|---|---|---|---|---|---|---|---|---|---|---|
| SPARTAN (boundary + text) | with | 63.3 | 2 of 8 | 0.250 | 0.250 | 0.011 | 0.011 | 62.6 | 0.953 | 0.395 |
| SPARTAN (boundary + text) | no | 35.3 | 2 of 8 | 0.250 | 0.250 | 0.020 | 0.020 | 34.6 | 0.953 | 0.535 |
| SPARTAN boundary only | with | 34.3 | 1 of 8 | 0.125 | 0.125 | 0.010 | 0.010 | 33.9 | 0.945 | 0.538 |
| SPARTAN boundary only | no | 9.4 | 1 of 8 | 0.125 | 0.125 | 0.037 | 0.037 | 9.1 | 0.945 | 0.789 |
| SPARTAN text (borderless) only | with | 29.0 | 1 of 8 | 0.125 | 0.125 | 0.012 | 0.012 | 28.7 | 0.945 | 0.577 |
| SPARTAN text (borderless) only | no | 25.9 | 1 of 8 | 0.125 | 0.125 | 0.014 | 0.014 | 25.5 | 0.945 | 0.603 |
| vector rules (pdf-inspector idea) | with | 17.8 | 0 of 8 | 0.000 | 0.125 | 0.000 | 0.020 | 17.8 | 0.937 | 0.676 |
| vector rules (pdf-inspector idea) | no | 4.9 | 0 of 8 | 0.000 | 0.125 | 0.000 | 0.071 | 4.9 | 0.937 | 0.856 |
| union SPARTAN + vector | with | 75.9 | 2 of 8 | 0.250 | 0.250 | 0.009 | 0.009 | 75.2 | 0.953 | 0.354 |
| union SPARTAN + vector | no | 38.1 | 2 of 8 | 0.250 | 0.250 | 0.018 | 0.018 | 37.4 | 0.953 | 0.517 |
| agreement SPARTAN and vector | with | 6.6 | 1 of 8 | 0.125 | 0.125 | 0.053 | 0.053 | 6.3 | 0.945 | 0.833 |
| agreement SPARTAN and vector | no | 2.8 | 1 of 8 | 0.125 | 0.125 | 0.125 | 0.125 | 2.4 | 0.945 | 0.902 |

**Miss taxonomy (odb)** - tags overlap; recovered = union SPARTAN + vector, figures kept
| tag | GT tables | heron miss rate | misses | recovered @.5 | recovered @.3 |
|---|---|---|---|---|---|
| borderless | 38 | 0.158 | 6 | 1 | 1 |
| tiny or key-value | 34 | 0.118 | 4 | 0 | 0 |
| horizontal rules only | 71 | 0.028 | 2 | 1 | 1 |
| heron box partial (IoU .1-.5) | 1 | 1.000 | 1 | 0 | 0 |
| ruled grid | 18 | 0.000 | 0 | 0 | 0 |
| in-figure | 3 | 0.000 | 0 | 0 | 0 |
| no text layer | 9 | 0.000 | 0 | 0 | 0 |
| heron label: Picture | 2 | 0.000 | 0 | 0 | 0 |
| low-contrast | 2 | 0.000 | 0 | 0 | 0 |
| rotated | 4 | 0.000 | 0 | 0 | 0 |

Speed per page: SPARTAN 300 DPI (render + words + detect) p50 158 ms / p95 302 ms; vector proposer p50 82 / p95 388 ms; heron INT8 p50 336 ms. Mappings used: {'stretch': 286}; vector notes: {}

### SCI-3000 sample (CC BY 4.0 / BY-SA 4.0): 4093 pages (0 errors), 813 GT tables, heron INT8 recall@.5 0.977, precision 0.928, misses 19
| proposer | figures | added / 100 pages | heron misses recovered @.5 | inc. recall @.5 | inc. recall @.3 | precision of added @.5 | precision @.3 | false tables / 100 pages | heron+added recall | heron+added precision |
|---|---|---|---|---|---|---|---|---|---|---|
| SPARTAN (boundary + text) | with | 57.8 | 4 of 19 | 0.211 | 0.211 | 0.002 | 0.002 | 57.7 | 0.982 | 0.248 |
| SPARTAN (boundary + text) | no | 15.2 | 2 of 19 | 0.105 | 0.105 | 0.003 | 0.003 | 15.2 | 0.979 | 0.538 |
| SPARTAN boundary only | with | 42.6 | 3 of 19 | 0.158 | 0.158 | 0.002 | 0.002 | 42.5 | 0.980 | 0.307 |
| SPARTAN boundary only | no | 2.7 | 1 of 19 | 0.053 | 0.053 | 0.009 | 0.009 | 2.7 | 0.978 | 0.823 |
| SPARTAN text (borderless) only | with | 15.2 | 1 of 19 | 0.053 | 0.053 | 0.002 | 0.002 | 15.2 | 0.978 | 0.538 |
| SPARTAN text (borderless) only | no | 12.6 | 1 of 19 | 0.053 | 0.053 | 0.002 | 0.002 | 12.5 | 0.978 | 0.580 |
| vector rules (pdf-inspector idea) | with | 17.7 | 0 of 19 | 0.000 | 0.000 | 0.000 | 0.000 | 17.7 | 0.977 | 0.503 |
| vector rules (pdf-inspector idea) | no | 3.6 | 0 of 19 | 0.000 | 0.000 | 0.000 | 0.000 | 3.6 | 0.977 | 0.791 |
| union SPARTAN + vector | with | 67.8 | 4 of 19 | 0.211 | 0.211 | 0.001 | 0.001 | 67.7 | 0.982 | 0.220 |
| union SPARTAN + vector | no | 18.4 | 2 of 19 | 0.105 | 0.105 | 0.003 | 0.003 | 18.3 | 0.979 | 0.495 |
| agreement SPARTAN and vector | with | 9.4 | 0 of 19 | 0.000 | 0.000 | 0.000 | 0.000 | 9.4 | 0.977 | 0.641 |
| agreement SPARTAN and vector | no | 0.8 | 0 of 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.8 | 0.977 | 0.895 |

**Miss taxonomy (sci)** - tags overlap; recovered = union SPARTAN + vector, figures kept
| tag | GT tables | heron miss rate | misses | recovered @.5 | recovered @.3 |
|---|---|---|---|---|---|
| heron box partial (IoU .1-.5) | 13 | 1.000 | 13 | 0 | 0 |
| horizontal rules only | 632 | 0.014 | 9 | 0 | 0 |
| borderless | 71 | 0.127 | 9 | 4 | 4 |
| tiny or key-value | 94 | 0.053 | 5 | 3 | 3 |
| in-figure | 11 | 0.273 | 3 | 2 | 2 |
| heron label: Picture | 11 | 0.273 | 3 | 2 | 2 |
| no text layer | 13 | 0.231 | 3 | 2 | 2 |
| ruled grid | 110 | 0.009 | 1 | 0 | 0 |
| heron label: Text | 1 | 1.000 | 1 | 1 | 1 |
| heron label: Key-Value Region | 1 | 1.000 | 1 | 0 | 0 |
| heron label: Form | 1 | 1.000 | 1 | 1 | 1 |
| rotated | 2 | 0.000 | 0 | 0 | 0 |
| low-contrast | 1 | 0.000 | 0 | 0 | 0 |

Speed per page: SPARTAN 300 DPI (render + words + detect) p50 231 ms / p95 348 ms; vector proposer p50 116 / p95 364 ms; heron INT8 p50 545 ms. Mappings used: {'stretch': 4093}; vector notes: {'rotated page: no vector proposals': 14}

### all datasets: 9378 pages (0 errors), 3334 GT tables, heron INT8 recall@.5 0.893, precision 0.878, misses 358
| proposer | figures | added / 100 pages | heron misses recovered @.5 | inc. recall @.5 | inc. recall @.3 | precision of added @.5 | precision @.3 | false tables / 100 pages | heron+added recall | heron+added precision |
|---|---|---|---|---|---|---|---|---|---|---|
| SPARTAN (boundary + text) | with | 51.1 | 76 of 358 | 0.212 | 0.240 | 0.016 | 0.018 | 50.3 | 0.915 | 0.373 |
| SPARTAN (boundary + text) | no | 20.2 | 39 of 358 | 0.109 | 0.123 | 0.021 | 0.023 | 19.8 | 0.904 | 0.571 |
| SPARTAN boundary only | with | 32.7 | 12 of 358 | 0.034 | 0.036 | 0.004 | 0.004 | 32.5 | 0.896 | 0.463 |
| SPARTAN boundary only | no | 5.4 | 6 of 358 | 0.017 | 0.020 | 0.012 | 0.014 | 5.4 | 0.894 | 0.765 |
| SPARTAN text (borderless) only | with | 18.5 | 64 of 358 | 0.179 | 0.204 | 0.037 | 0.042 | 17.8 | 0.912 | 0.594 |
| SPARTAN text (borderless) only | no | 14.8 | 33 of 358 | 0.092 | 0.103 | 0.024 | 0.027 | 14.4 | 0.903 | 0.631 |
| vector rules (pdf-inspector idea) | with | 13.4 | 12 of 358 | 0.034 | 0.045 | 0.010 | 0.013 | 13.3 | 0.896 | 0.644 |
| vector rules (pdf-inspector idea) | no | 3.8 | 5 of 358 | 0.014 | 0.020 | 0.014 | 0.020 | 3.7 | 0.894 | 0.797 |
| union SPARTAN + vector | with | 58.7 | 84 of 358 | 0.235 | 0.268 | 0.015 | 0.017 | 57.8 | 0.918 | 0.344 |
| union SPARTAN + vector | no | 22.7 | 42 of 358 | 0.117 | 0.134 | 0.020 | 0.023 | 22.3 | 0.905 | 0.547 |
| agreement SPARTAN and vector | with | 6.9 | 5 of 358 | 0.014 | 0.014 | 0.008 | 0.008 | 6.8 | 0.894 | 0.739 |
| agreement SPARTAN and vector | no | 1.4 | 3 of 358 | 0.008 | 0.008 | 0.023 | 0.023 | 1.4 | 0.894 | 0.846 |

**Miss taxonomy (all)** - tags overlap; recovered = union SPARTAN + vector, figures kept
| tag | GT tables | heron miss rate | misses | recovered @.5 | recovered @.3 |
|---|---|---|---|---|---|
| borderless | 1315 | 0.208 | 273 | 70 | 80 |
| tiny or key-value | 583 | 0.300 | 175 | 38 | 44 |
| heron box partial (IoU .1-.5) | 112 | 1.000 | 112 | 0 | 0 |
| in-figure | 135 | 0.785 | 106 | 41 | 48 |
| horizontal rules only | 1591 | 0.048 | 76 | 8 | 10 |
| heron label: Key-Value Region | 57 | 0.421 | 24 | 12 | 13 |
| heron label: Picture | 49 | 0.449 | 22 | 9 | 11 |
| heron label: Text | 15 | 0.867 | 13 | 3 | 4 |
| low-contrast | 33 | 0.364 | 12 | 3 | 4 |
| ruled grid | 428 | 0.021 | 9 | 6 | 6 |
| heron label: Code | 10 | 0.900 | 9 | 2 | 2 |
| heron label: Form | 10 | 0.800 | 8 | 6 | 6 |
| no text layer | 28 | 0.179 | 5 | 2 | 2 |
| heron label: List-item | 5 | 0.800 | 4 | 2 | 2 |
| rotated | 15 | 0.133 | 2 | 1 | 1 |
| heron label: Page-header | 1 | 1.000 | 1 | 0 | 0 |

Speed per page: SPARTAN 300 DPI (render + words + detect) p50 192 ms / p95 337 ms; vector proposer p50 91 / p95 355 ms; heron INT8 p50 424 ms. Mappings used: {'stretch': 9334, 'fit-topleft': 22, 'fit-center': 22}; vector notes: {'rotated page: no vector proposals': 28}


## Gap 6: cross-page tables (PubTables-v2 Full Documents test, CDLA-Permissive-2.0): 872 documents
Fragments = ground-truth table parts; decisions use only page words around the fragments. Link = consecutive parts of one logical table. False-merge rate = linked adjacent pairs that belong to different tables / all adjacent different-table pairs. Hard negatives = adjacent different-table pairs with column counts within 1 and (header similarity >= 0.6 or bottom-of-page -> top-of-page position).

| rule | link precision | link recall | F1 | cross-page P / R | same-page (column) P / R | false-merge rate | hard-negative false merges | exact logical tables (all) | exact (multi-part) | clusters mixing tables |
|---|---|---|---|---|---|---|---|---|---|---|
| textract (accuracy 98, default) | 0.975 | 0.134 | 0.235 | 0.975 / 0.143 | 0.000 / 0.000 | 0.0026 (4/1518) | 4/458 | 0.638 | 0.110 (106/967) | 4 |
| textract (accuracy 90) | 0.972 | 0.176 | 0.299 | 0.972 / 0.189 | 0.000 / 0.000 | 0.0040 (6/1518) | 6/458 | 0.652 | 0.147 (142/967) | 6 |
| stitcher | 0.974 | 0.625 | 0.762 | 0.974 / 0.670 | 0.000 / 0.000 | 0.0132 (20/1518) | 19/458 | 0.832 | 0.615 (595/967) | 20 |
| r4 | 0.991 | 0.476 | 0.644 | 0.992 / 0.471 | 0.978 / 0.550 | 0.0033 (5/1518) | 5/458 | 0.785 | 0.475 (459/967) | 5 |

**Official page-pair labels** (cross_page_table_pairs: does a table on page A continue on page B):
| rule | precision | recall | F1 | pairs |
|---|---|---|---|---|
| textract (accuracy 98, default) | 0.969 | 0.143 | 0.249 | 1862 |
| textract (accuracy 90) | 0.968 | 0.189 | 0.316 | 1862 |
| stitcher | 0.973 | 0.671 | 0.794 | 1862 |
| r4 | 0.992 | 0.472 | 0.640 | 1862 |

Decision reasons (adjacent pairs): stitcher: page_gap_out_of_range 846, content_between_tables 698, header_similarity_strict 633, header_similarity_too_low 386, headerless_width_match 76, header_similarity_loose_layout 30, headerless_width_drift_layout 23, width_difference_too_large 10; r4: page_gap 423, content_between 396, same_page_not_column_flow 349, not_bottom_to_top 346, continued_marker 277, no_header_evidence 277, new_caption 275, repeated_header 222

**Splitting one region that holds two logical tables** (synthetic over-merge: union box of two vertically adjacent different tables in one column; negatives = every single table part with >= 4 text lines)
| positives | split at the right place | exactly one correct cut | wrong cut only | negatives | false split rate |
|---|---|---|---|---|---|
| 137 | 0.912 | 0.453 | 0 | 3536 | 0.0682 |


## Layout: RF-DETR-Large (DocLayNet fine-tune) vs heron INT8
Model nie3e/rf-detr-large-doclay @fecf537 (card: Apache-2.0, base Roboflow/rf-detr-large Apache-2.0, trained on DocLayNet v1.2 train; no LICENSE file in the model repo), transformers 5.22.2 + torch 2.15.0 CPU fp32; heron INT8 = docling layout_heron_int8.onnx detections from order run 44477997398. Same pages, same GT; COCO-style AP (101-point, max 100 detections per page), recall/precision at score >= 0.5 and IoU 0.5. OmniDocBench: Section-header -> Title and List-item -> Text (ODB has no such split); ODB categories not mapped: {'page_number': 537, 'abandon': 251, 'equation_semantic': 142, 'list_group': 17, 'chart_mask': 12, 'algorithm_mask': 1, 'text_mask': 383, 'need_mask': 6, 'equation_explanation': 16, 'unknown_mask': 10}.

### DocLayNet test: 4999 pages
| class | GT | RF-DETR mAP50-95 | heron mAP50-95 | RF-DETR mAP50 | heron mAP50 | RF-DETR R / P | heron R / P |
|---|---|---|---|---|---|---|---|
| Table | 2394 | 0.851 | 0.843 | 0.905 | 0.884 | 0.855 / 0.913 | 0.862 / 0.858 |
| Picture | 3534 | 0.736 | 0.781 | 0.828 | 0.856 | 0.803 / 0.793 | 0.896 / 0.692 |
| Text | 29940 | 0.854 | 0.836 | 0.943 | 0.935 | 0.901 / 0.924 | 0.917 / 0.780 |
| Title | 335 | 0.818 | 0.341 | 0.911 | 0.424 | 0.860 / 0.855 | 0.304 / 0.699 |
| List-item | 10522 | 0.818 | 0.789 | 0.932 | 0.919 | 0.891 / 0.910 | 0.908 / 0.790 |
| Formula | 1966 | 0.646 | 0.645 | 0.935 | 0.916 | 0.928 / 0.896 | 0.959 / 0.842 |
| Caption | 1543 | 0.721 | 0.734 | 0.862 | 0.903 | 0.782 / 0.854 | 0.892 / 0.759 |
| Footnote | 387 | 0.723 | 0.726 | 0.870 | 0.877 | 0.786 / 0.921 | 0.858 / 0.695 |
| Page-header | 3366 | 0.752 | 0.714 | 0.935 | 0.940 | 0.863 / 0.893 | 0.958 / 0.778 |
| Page-footer | 3994 | 0.626 | 0.529 | 0.930 | 0.882 | 0.915 / 0.921 | 0.916 / 0.738 |
| Section-header | 8550 | 0.705 | 0.630 | 0.915 | 0.918 | 0.870 / 0.884 | 0.917 / 0.775 |
| **mean over classes** | | **0.750** | **0.688** | **0.906** | **0.860** | 0.859 / 0.888 | 0.853 / 0.764 |

Paired per-page F1 (all classes, score >= 0.5, IoU 0.5): RF-DETR 0.8833 vs heron 0.8546; difference +0.0286 [95% bootstrap CI +0.0245, +0.0330], Wilcoxon p 9.2e-47; RF-DETR better on 1893 pages, worse on 1160.
CPU ms per page (4-vCPU runner): RF-DETR p50 805 / mean 744; heron INT8 p50 317 / mean 349.

**Missed-table arm (dln)**: 4999 pages with heron, RF-DETR and YOLOX-L (round 3, run 44689850800); heron recall@.5 0.862, misses 331
| second detector | where | added / 100 pages | misses recovered @.5 | inc. recall | precision of added | false tables / 100 pages | heron+added recall |
|---|---|---|---|---|---|---|---|
| RF-DETR-Large >= 0.5 | uncovered regions | 2.0 | 55 of 331 | 0.166 | 0.561 | 0.9 | 0.885 |
| RF-DETR-Large >= 0.5 | pages without a heron table | 1.1 | 32 of 331 | 0.097 | 0.571 | 0.5 | 0.875 |
| RF-DETR-Large >= 0.7 | uncovered regions | 1.3 | 43 of 331 | 0.130 | 0.662 | 0.4 | 0.880 |
| RF-DETR-Large >= 0.7 | pages without a heron table | 0.8 | 26 of 331 | 0.079 | 0.684 | 0.2 | 0.873 |
| YOLOX-L >= 0.5 | uncovered regions | 1.9 | 36 of 331 | 0.109 | 0.371 | 1.2 | 0.877 |
| YOLOX-L >= 0.5 | pages without a heron table | 1.3 | 22 of 331 | 0.066 | 0.344 | 0.8 | 0.871 |
| YOLOX-L >= 0.7 | uncovered regions | 0.8 | 15 of 331 | 0.045 | 0.357 | 0.5 | 0.868 |
| YOLOX-L >= 0.7 | pages without a heron table | 0.5 | 12 of 331 | 0.036 | 0.444 | 0.3 | 0.867 |

### OmniDocBench English: 740 pages
| class | GT | RF-DETR mAP50-95 | heron mAP50-95 | RF-DETR mAP50 | heron mAP50 | RF-DETR R / P | heron R / P |
|---|---|---|---|---|---|---|---|
| Table | 244 | 0.617 | 0.802 | 0.670 | 0.840 | 0.590 / 0.847 | 0.820 / 0.901 |
| Picture | 487 | 0.594 | 0.727 | 0.674 | 0.788 | 0.702 / 0.612 | 0.858 / 0.458 |
| Text | 7309 | 0.568 | 0.721 | 0.721 | 0.831 | 0.693 / 0.777 | 0.902 / 0.648 |
| Title | 1567 | 0.344 | 0.449 | 0.586 | 0.702 | 0.530 / 0.724 | 0.736 / 0.643 |
| Formula | 1660 | 0.301 | 0.365 | 0.510 | 0.581 | 0.453 / 0.776 | 0.564 / 0.716 |
| Caption | 727 | 0.171 | 0.299 | 0.262 | 0.413 | 0.164 / 0.726 | 0.377 / 0.755 |
| Footnote | 223 | 0.095 | 0.137 | 0.151 | 0.209 | 0.103 / 0.523 | 0.184 / 0.373 |
| Page-header | 699 | 0.251 | 0.336 | 0.585 | 0.710 | 0.625 / 0.643 | 0.810 / 0.590 |
| Page-footer | 264 | 0.253 | 0.288 | 0.519 | 0.528 | 0.697 / 0.520 | 0.845 / 0.413 |
| **mean over classes** | | **0.355** | **0.458** | **0.520** | **0.622** | 0.506 / 0.683 | 0.677 / 0.611 |

Paired per-page F1 (all classes, score >= 0.5, IoU 0.5): RF-DETR 0.6395 vs heron 0.6931; difference -0.0536 [95% bootstrap CI -0.0696, -0.0388], Wilcoxon p 5.7e-09; RF-DETR better on 268 pages, worse on 389.
CPU ms per page (4-vCPU runner): RF-DETR p50 816 / mean 690; heron INT8 p50 334 / mean 367.

**Missed-table arm (odb)**: 740 pages with heron, RF-DETR and YOLOX-L (round 3, run 44689850800); heron recall@.5 0.824, misses 43
| second detector | where | added / 100 pages | misses recovered @.5 | inc. recall | precision of added | false tables / 100 pages | heron+added recall |
|---|---|---|---|---|---|---|---|
| RF-DETR-Large >= 0.5 | uncovered regions | 3.1 | 1 of 43 | 0.023 | 0.043 | 3.0 | 0.828 |
| RF-DETR-Large >= 0.5 | pages without a heron table | 2.3 | 1 of 43 | 0.023 | 0.059 | 2.2 | 0.828 |
| RF-DETR-Large >= 0.7 | uncovered regions | 1.1 | 1 of 43 | 0.023 | 0.125 | 0.9 | 0.828 |
| RF-DETR-Large >= 0.7 | pages without a heron table | 0.8 | 1 of 43 | 0.023 | 0.167 | 0.7 | 0.828 |
| YOLOX-L >= 0.5 | uncovered regions | 3.0 | 2 of 43 | 0.047 | 0.091 | 2.7 | 0.832 |
| YOLOX-L >= 0.5 | pages without a heron table | 2.4 | 2 of 43 | 0.047 | 0.111 | 2.2 | 0.832 |
| YOLOX-L >= 0.7 | uncovered regions | 1.5 | 1 of 43 | 0.023 | 0.091 | 1.4 | 0.828 |
| YOLOX-L >= 0.7 | pages without a heron table | 1.1 | 1 of 43 | 0.023 | 0.125 | 0.9 | 0.828 |

## Layout: PP-DocLayout_plus-L vs heron INT8
Model PaddlePaddle/PP-DocLayout_plus-L_onnx @feb7461 (inference.onnx, 130 MB; model card tag apache-2.0, no LICENSE file in either HF repo; PaddleOCR / PaddleX code Apache-2.0), ONNX Runtime CPU with the same default thread settings as heron INT8; pre/post-processing per inference.yml and PaddleX 3.7.2 DetPostProcess (800x800 cubic resize, /255, threshold, no layout NMS, large-image filter). Same pages, GT and scoring as the RF-DETR-Large section. Fixed label map: doc_title->Title, paragraph_title->Section-header, image+chart->Picture, formula->Formula, table->Table, header->Page-header, footer->Page-footer, footnote->Footnote, figure_title->Caption, text->Text; all other PP labels dropped (abstract, content, reference, reference_content, aside_text, algorithm, number, formula_number, seal). PP has no list class: List-item is not scored for it; 'Text+List-item' merges both classes for every model and GT.

**ONNX vs native Paddle parity** (paddlepaddle 3.2.2, paddleocr 3.7.0, paddlex 3.7.2, same weights, threshold 0.3, 40 pages): 1058 of 1058 native boxes matched by an ONNX box of the same label at IoU >= 0.95 (100.0%; 100.0% of 1058 ONNX boxes), mean |score difference| 7.778744165748639e-07, max 0.00012236833572387695, mean IoU 0.9999774233345032; native 672 ms/page vs ONNX 446 ms/page.

### DocLayNet test: 4999 pages
| class | GT | PP-DocLayout_plus-L mAP50-95 | heron INT8 mAP50-95 | RF-DETR-Large mAP50-95 | PP-DocLayout_plus-L mAP50 | heron INT8 mAP50 | RF-DETR-Large mAP50 | PP-DocLayout_plus-L R / P | heron INT8 R / P | RF-DETR-Large R / P |
|---|---|---|---|---|---|---|---|---|---|---|
| Table | 2394 | 0.764 | 0.843 | 0.851 | 0.831 | 0.884 | 0.905 | 0.787 / 0.860 | 0.862 / 0.858 | 0.855 / 0.913 |
| Picture | 3534 | 0.507 | 0.781 | 0.736 | 0.591 | 0.856 | 0.828 | 0.526 / 0.685 | 0.896 / 0.692 | 0.803 / 0.793 |
| Text | 29940 | 0.611 | 0.836 | 0.854 | 0.816 | 0.935 | 0.943 | 0.857 / 0.697 | 0.917 / 0.780 | 0.901 / 0.924 |
| Title | 335 | 0.153 | 0.341 | 0.818 | 0.293 | 0.424 | 0.911 | 0.409 / 0.223 | 0.304 / 0.699 | 0.860 / 0.855 |
| Section-header | 8550 | 0.248 | 0.630 | 0.705 | 0.805 | 0.918 | 0.915 | 0.713 / 0.803 | 0.917 / 0.775 | 0.870 / 0.884 |
| Formula | 1966 | 0.181 | 0.645 | 0.646 | 0.454 | 0.916 | 0.935 | 0.616 / 0.089 | 0.959 / 0.842 | 0.928 / 0.896 |
| Caption | 1543 | 0.374 | 0.734 | 0.721 | 0.749 | 0.903 | 0.862 | 0.778 / 0.490 | 0.892 / 0.759 | 0.782 / 0.854 |
| Footnote | 387 | 0.175 | 0.726 | 0.723 | 0.364 | 0.877 | 0.870 | 0.305 / 0.534 | 0.858 / 0.695 | 0.786 / 0.921 |
| Page-header | 3366 | 0.116 | 0.714 | 0.752 | 0.429 | 0.940 | 0.935 | 0.508 / 0.604 | 0.958 / 0.778 | 0.863 / 0.893 |
| Page-footer | 3994 | 0.068 | 0.529 | 0.626 | 0.293 | 0.882 | 0.930 | 0.292 / 0.533 | 0.916 / 0.738 | 0.915 / 0.921 |
| Text+List-item | 40462 | 0.615 | 0.837 | 0.853 | 0.877 | 0.949 | 0.950 | 0.808 / 0.887 | 0.938 / 0.803 | 0.916 / 0.938 |
| Text (PP extended map) | 29940 | 0.600 | - | - | 0.798 | - | - | 0.864 / 0.649 | - | - |
| **shared-class mean** | | **0.320** | **0.678** | **0.743** | **0.563** | **0.854** | **0.903** |  |  |  |

Coverage: PP-DocLayout_plus-L covers 10 of the 11 DocLayNet classes; List-item (10522 GT boxes) has no PP class and is excluded from its shared-class mean (PP predicts list items as text, which counts against its Text precision; see the Text+List-item row).
PP detections (score >= 0.5) under dropped labels: {'number': 4409, 'reference_content': 2121, 'formula_number': 1584, 'aside_text': 491, 'reference': 173, 'abstract': 86, 'content': 80, 'algorithm': 52, 'seal': 1}.

Paired per-page F1 (shared classes, Text+List-item merged, score >= 0.5, IoU 0.5): PP 0.6876 vs heron 0.8635; difference -0.1758 [95% bootstrap CI -0.1829, -0.1683], Wilcoxon p 0; PP better on 900 pages, worse on 3714. RF-DETR-Large on the same basis: 0.8917.
CPU ms per page (4-vCPU runner): PP-DocLayout_plus-L p50 999 / mean 958; heron INT8 p50 317 / mean 349; RF-DETR-Large p50 805 / mean 744.

**Missed-table arm (dln)**: 4999 pages; heron recall@.5 0.862, misses 331
| second detector | where | added / 100 pages | misses recovered @.5 | inc. recall | precision of added | false tables / 100 pages | heron+added recall |
|---|---|---|---|---|---|---|---|
| PP-DocLayout_plus-L >= 0.5 | uncovered regions | 1.6 | 16 of 331 | 0.048 | 0.195 | 1.3 | 0.868 |
| PP-DocLayout_plus-L >= 0.5 | pages without a heron table | 1.1 | 8 of 331 | 0.024 | 0.148 | 0.9 | 0.865 |
| PP-DocLayout_plus-L >= 0.7 | uncovered regions | 1.1 | 12 of 331 | 0.036 | 0.222 | 0.8 | 0.867 |
| PP-DocLayout_plus-L >= 0.7 | pages without a heron table | 0.7 | 5 of 331 | 0.015 | 0.147 | 0.6 | 0.864 |
| RF-DETR-Large >= 0.5 | uncovered regions | 2.0 | 55 of 331 | 0.166 | 0.561 | 0.9 | 0.885 |
| RF-DETR-Large >= 0.5 | pages without a heron table | 1.1 | 32 of 331 | 0.097 | 0.571 | 0.5 | 0.875 |
| RF-DETR-Large >= 0.7 | uncovered regions | 1.3 | 43 of 331 | 0.130 | 0.662 | 0.4 | 0.880 |
| RF-DETR-Large >= 0.7 | pages without a heron table | 0.8 | 26 of 331 | 0.079 | 0.684 | 0.2 | 0.873 |
| YOLOX-L >= 0.5 | uncovered regions | 1.9 | 36 of 331 | 0.109 | 0.371 | 1.2 | 0.877 |
| YOLOX-L >= 0.5 | pages without a heron table | 1.3 | 22 of 331 | 0.066 | 0.344 | 0.8 | 0.871 |
| YOLOX-L >= 0.7 | uncovered regions | 0.8 | 15 of 331 | 0.045 | 0.357 | 0.5 | 0.868 |
| YOLOX-L >= 0.7 | pages without a heron table | 0.5 | 12 of 331 | 0.036 | 0.444 | 0.3 | 0.867 |

### OmniDocBench English: 740 pages
| class | GT | PP-DocLayout_plus-L mAP50-95 | heron INT8 mAP50-95 | RF-DETR-Large mAP50-95 | PP-DocLayout_plus-L mAP50 | heron INT8 mAP50 | RF-DETR-Large mAP50 | PP-DocLayout_plus-L R / P | heron INT8 R / P | RF-DETR-Large R / P |
|---|---|---|---|---|---|---|---|---|---|---|
| Table | 244 | 0.785 | 0.802 | 0.617 | 0.822 | 0.840 | 0.670 | 0.795 / 0.833 | 0.820 / 0.901 | 0.590 / 0.847 |
| Picture | 487 | 0.730 | 0.727 | 0.594 | 0.805 | 0.788 | 0.674 | 0.795 / 0.680 | 0.858 / 0.458 | 0.702 / 0.612 |
| Text | 7309 | 0.710 | 0.721 | 0.568 | 0.848 | 0.831 | 0.721 | 0.827 / 0.800 | 0.902 / 0.648 | 0.693 / 0.777 |
| Title | 1567 | 0.361 | 0.449 | 0.344 | 0.661 | 0.702 | 0.586 | 0.595 / 0.765 | 0.736 / 0.643 | 0.530 / 0.724 |
| Formula | 1660 | 0.278 | 0.365 | 0.301 | 0.436 | 0.581 | 0.510 | 0.578 / 0.143 | 0.564 / 0.716 | 0.453 / 0.776 |
| Caption | 727 | 0.264 | 0.299 | 0.171 | 0.424 | 0.413 | 0.262 | 0.395 / 0.723 | 0.377 / 0.755 | 0.164 / 0.726 |
| Footnote | 223 | 0.205 | 0.137 | 0.095 | 0.311 | 0.209 | 0.151 | 0.143 / 0.780 | 0.184 / 0.373 | 0.103 / 0.523 |
| Page-header | 699 | 0.351 | 0.336 | 0.251 | 0.835 | 0.710 | 0.585 | 0.763 / 0.864 | 0.810 / 0.590 | 0.625 / 0.643 |
| Page-footer | 264 | 0.392 | 0.288 | 0.253 | 0.837 | 0.528 | 0.519 | 0.875 / 0.740 | 0.845 / 0.413 | 0.697 / 0.520 |
| Text (PP extended map) | 7309 | 0.728 | - | - | 0.866 | - | - | 0.866 / 0.775 | - | - |
| **shared-class mean** | | **0.453** | **0.458** | **0.355** | **0.664** | **0.622** | **0.520** |  |  |  |
PP detections (score >= 0.5) under dropped labels: {'reference_content': 614, 'number': 545, 'formula_number': 192, 'reference': 35, 'aside_text': 30, 'abstract': 17, 'content': 5, 'algorithm': 2}.

Paired per-page F1 (shared classes, Text+List-item merged, score >= 0.5, IoU 0.5): PP 0.6263 vs heron 0.6931; difference -0.0668 [95% bootstrap CI -0.0849, -0.0492], Wilcoxon p 2.3e-10; PP better on 295 pages, worse on 402. RF-DETR-Large on the same basis: 0.6395.
CPU ms per page (4-vCPU runner): PP-DocLayout_plus-L p50 407 / mean 430; heron INT8 p50 334 / mean 367; RF-DETR-Large p50 816 / mean 690.

**Per genre (OmniDocBench data_source)** - shared-class mean mAP50-95 and paired page F1 (PP - heron)
| genre | pages | PP-DocLayout_plus-L mAP50-95 | heron INT8 mAP50-95 | RF-DETR-Large mAP50-95 | PP - heron page F1 [95% CI] |
|---|---|---|---|---|---|
| paper | 205 | 0.492 | 0.551 | 0.460 | -0.0978 [-0.1277, -0.0684] |
| book | 126 | 0.460 | 0.440 | 0.391 | -0.1091 [-0.1466, -0.0705] |
| exam | 82 | 0.459 | 0.423 | 0.284 | -0.3105 [-0.3646, -0.2555] |
| slides | 114 | 0.363 | 0.363 | 0.266 | -0.0287 [-0.0771, +0.0113] |
| newspaper | 77 | 0.398 | 0.427 | 0.230 | +0.0083 [-0.0137, +0.0291] |
| magazine | 84 | 0.498 | 0.512 | 0.446 | +0.0507 [+0.0251, +0.0793] |
| colorful_textbook | 51 | 0.480 | 0.378 | 0.318 | +0.1663 [+0.0807, +0.2514] |

**Missed-table arm (odb)**: 740 pages; heron recall@.5 0.824, misses 43
| second detector | where | added / 100 pages | misses recovered @.5 | inc. recall | precision of added | false tables / 100 pages | heron+added recall |
|---|---|---|---|---|---|---|---|
| PP-DocLayout_plus-L >= 0.5 | uncovered regions | 5.0 | 10 of 43 | 0.233 | 0.270 | 3.6 | 0.865 |
| PP-DocLayout_plus-L >= 0.5 | pages without a heron table | 3.2 | 6 of 43 | 0.140 | 0.250 | 2.4 | 0.848 |
| PP-DocLayout_plus-L >= 0.7 | uncovered regions | 2.6 | 6 of 43 | 0.140 | 0.316 | 1.8 | 0.848 |
| PP-DocLayout_plus-L >= 0.7 | pages without a heron table | 1.9 | 5 of 43 | 0.116 | 0.357 | 1.2 | 0.844 |
| RF-DETR-Large >= 0.5 | uncovered regions | 3.1 | 1 of 43 | 0.023 | 0.043 | 3.0 | 0.828 |
| RF-DETR-Large >= 0.5 | pages without a heron table | 2.3 | 1 of 43 | 0.023 | 0.059 | 2.2 | 0.828 |
| RF-DETR-Large >= 0.7 | uncovered regions | 1.1 | 1 of 43 | 0.023 | 0.125 | 0.9 | 0.828 |
| RF-DETR-Large >= 0.7 | pages without a heron table | 0.8 | 1 of 43 | 0.023 | 0.167 | 0.7 | 0.828 |
| YOLOX-L >= 0.5 | uncovered regions | 3.0 | 2 of 43 | 0.047 | 0.091 | 2.7 | 0.832 |
| YOLOX-L >= 0.5 | pages without a heron table | 2.4 | 2 of 43 | 0.047 | 0.111 | 2.2 | 0.832 |
| YOLOX-L >= 0.7 | uncovered regions | 1.5 | 1 of 43 | 0.023 | 0.091 | 1.4 | 0.828 |
| YOLOX-L >= 0.7 | pages without a heron table | 1.1 | 1 of 43 | 0.023 | 0.125 | 0.9 | 0.828 |
