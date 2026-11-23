# Round 4 (repo 1): gap 13 reading order and gap 10 picture classification

## Gap 13. Newspaper, magazine and sidebar reading order

Run 44881508619 (`visual-r4.yml`, part=order). Four shards and the report job succeeded.

**Setup:**
- 1,166 pages: OmniDocBench English plus DrDocBench dev.
- Test split: 573 pages, document-hashed, the same split as rounds 1-3. Continuity margins and sidebar policies were chosen on the dev split only.
- Baseline = the current pick:
  - GT boxes: XY-Cut++ + repeated-decoration removal (80.3% of test pages fully correct, as in round 3);
  - heron int8 boxes: XY-Cut++ (76.3%).
- Add-ons, all our own code (`order/r4/r4order.py`):
  - text continuity (unfinished sentence, line-end hyphen, lowercase start, rare shared terms) choosing among geometric candidate orders;
  - sidebar masks from tinted or ruled boxes found in the page image, with policies inline / after / end;
  - exercise-number ordering;
  - "continued on/from page" lines.

### Fully correct pages, test split

| box source | slice | pages | current pick | best add-on (chosen on dev) | change | sign test |
|---|---|---|---|---|---|---|
| GT boxes | hard: newspaper, magazine, colourful textbook, exam | 191 | 63.9% | side_inline 64.9% | +1.0 | 3 vs 1, p = 0.62 |
| GT boxes | newspaper + magazine | 104 | 56.7% | side_inline 57.7% | +1.0 | - |
| GT boxes | boxed or shaded region | 34 | 29.4% | side_inline 32.4% | +2.9 | 1 vs 0, p = 1 |
| GT boxes | ordinary | 382 | 88.5% | side_inline 88.5% | 0.0 | 0 vs 0 |
| heron int8 | hard | 191 | 59.2% | side_inline 60.7% | +1.6 | 3 vs 0, p = 0.25 |
| heron int8 | newspaper + magazine | 104 | 55.8% | side_inline 57.7% | +1.9 | - |
| heron int8 | ordinary | 382 | 84.8% | side_inline 84.8% | 0.0 | 0 vs 0 |

**Other add-ons:**
- **Continuity:** with the margin chosen on dev (3.0, heron boxes only; on GT boxes dev chose to never apply it), it changed 11 test pages and gave -1.0 on the hard slice (0 vs 2).
- **Sidebar placement:**
  - "after": -1.0 hard, -0.8 ordinary;
  - "end": -2.1 hard, -1.6 ordinary, p = 0.031 on ordinary pages.
- **Rarely triggered:** grid numbering on 2-3 test pages, continuation lines on 2; neither changed any result.
- **Cost:** the full stack is about 1 ms per page (median), p95 under 5 ms.

**American Stories:** licence cc-by-4.0. The release holds article text per scan. The 10 hand-labelled evaluation scans are not released, so there is no page-level reading-order ground truth, and it was not used. (The probe job checked the actual release.)

**Verdict, gap 13: no add-on meets the bar** (round-4 research target: +10 points on the hard slice with <= 1 point loss on ordinary pages).
- Best: sidebar units placed inline, +1.0 / +1.6 points on the hard slice, with no loss on ordinary pages. Not significant (3 pages vs 0-1).
- Keep XY-Cut++ + decoration removal. `side_inline` is a harmless optional add-on (0.3 ms per page, it only changes pages with detected boxes) if sidebars matter to users.
- Text-continuity re-ranking and exercise / continuation rules give no measurable gain on this data.
- What remains: newspaper and magazine pages (about 56% fully correct) need article segmentation that boxes plus text cues alone do not supply.

Full tables: artifact `visual-r4-report` (order_r4.md) of run 44881508619.

## Gap 10. Picture classification

**Run:** 44897815693 (`visual-r4.yml`, part=pics). All jobs succeeded.
- 9,180 training and 3,793 test crops, split by group (paper / chart / company / document / page).
- The DocFigure download failed again within its 25-minute cap (the CVIT server is too slow), so DocFigure is not in the data. The classes come from: ChartX (chart), AI2D (diagram), Imagenette (photo), WebSight (screenshot), LogoDet-3K (logo), SignverOD (signature; its redaction boxes are "other"), mapo80/stamps (stamp), DocLayNet tables (table_image).
- Heads: logistic regression on frozen features (timm mobilenetv3_small_100.lamb_in1k; facebook/dinov2-small CLS + mean tokens), C chosen by grouped 3-fold CV. Exported to ONNX as backbone + head in one graph, then dynamic INT8.
- docling's DocumentFigureClassifier-v2.5 (picture_classifier.onnx from docling.rs models-v1) was run with docling.rs preprocessing, and its 26 classes mapped to these 9.

Data sources (crops kept per split and class): {"domain:unlabeled": 748, "test:chart": 496, "test:diagram": 463, "test:logo": 500, "test:other": 182, "test:photo": 500, "test:screenshot": 500, "test:signature": 481, "test:stamp": 476, "test:table_image": 195, "train:chart": 1200, "train:diagram": 1081, "train:logo": 1178, "train:other": 518, "train:photo": 1173, "train:screenshot": 1185, "train:signature": 1135, "train:stamp": 1200, "train:table_image": 510}

Source status: docfigure: failed: RuntimeError: download failed after 8 tries: https://cvit.iiit.ac.in/usodi/img/projects/classification/Docfig/DocFigure_dataset.zip; chartx: 1700; ai2d: 1544; imagenette: 1700; websight: 1700; logos: 1699; signatures: 2316; stamps: 1699; doclaynet: 1453

| model | macro-F1 | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | other | MB | ms/crop (1 thread) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnv3 head (sklearn) | 0.977 | 0.99 | 1.00 | 0.99 | 0.98 | 0.96 | 0.98 | 0.96 | 0.96 | 0.98 | - | - |
| mnv3 head (onnx_fp32) | 0.977 | 0.99 | 1.00 | 0.99 | 0.98 | 0.96 | 0.98 | 0.96 | 0.96 | 0.98 | 6.1 | 3.1 |
| mnv3 head (onnx_int8) | 0.033 | 0.00 | 0.00 | 0.24 | 0.00 | 0.05 | 0.00 | 0.01 | 0.00 | 0.00 | 1.7 | 25.3 |
| dinov2 head (sklearn) | 0.989 | 0.99 | 0.99 | 1.00 | 0.99 | 0.98 | 0.99 | 0.98 | 0.98 | 0.99 | - | - |
| dinov2 head (onnx_fp32) | 0.989 | 0.99 | 0.99 | 1.00 | 0.99 | 0.98 | 0.99 | 0.98 | 0.98 | 0.99 | 88.5 | 69.6 |
| dinov2 head (onnx_int8) | 0.965 | 0.95 | 0.99 | 1.00 | 0.91 | 0.98 | 0.99 | 0.97 | 0.93 | 0.97 | 24.2 | 40.5 |
| docling DocumentFigureClassifier-v2.5 | 0.689 | 0.84 | 0.80 | 0.86 | 0.47 | 0.75 | 0.81 | 0.76 | 0.59 | 0.33 | 16.9 | 5.6 |

- mnv3 onnx_fp32 vs docling on 3793 crops: ours right / docling wrong 989, reverse 38, sign test p = 3.8e-240
- mnv3 onnx_int8 vs docling on 3793 crops: ours right / docling wrong 25, reverse 2274, sign test p = 0
- dinov2 onnx_fp32 vs docling on 3793 crops: ours right / docling wrong 1014, reverse 18, sign test p = 1e-272
- dinov2 onnx_int8 vs docling on 3793 crops: ours right / docling wrong 943, reverse 31, sign test p = 4.3e-235

Test crops by source: imagenette 500, websight 500, logodet 500, chartx 496, signverod 481, stamps 476, ai2d 463, doclaynet 195, signverod_redaction 182

DocLayNet picture crops (no labels; domain check only) - predicted class shares:

| model | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | other |
|---|---|---|---|---|---|---|---|---|---|
| docling | 0.34 | 0.06 | 0.37 | 0.00 | 0.12 | 0.05 | 0.00 | 0.01 | 0.04 |
| mnv3 int8 | 0.00 | 0.00 | 0.95 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 |
| dinov2 int8 | 0.24 | 0.10 | 0.32 | 0.04 | 0.20 | 0.05 | 0.03 | 0.03 | 0.00 |

docling raw 26-class predictions on the test crops: {"flow_chart": 227, "photograph": 603, "page_thumbnail": 38, "bar_code": 1, "logo": 481, "other": 39, "pie_chart": 69, "engineering_drawing": 279, "calendar": 3, "bar_chart": 281, "stamp": 291, "full_page_image": 9, "screenshot_from_manual": 13, "table": 440, "geographical_map": 3, "box_plot": 31, "screenshot_from_computer": 149, "line_chart": 218, "music": 8, "icon": 216, "signature": 363, "scatter_plot": 27, "chemistry_structure": 4}

**Caveat:** each class comes from its own source, so a head can partly learn the source style. The near-perfect macro-F1 overstates real-document accuracy. On unlabelled DocLayNet picture crops the class mix looks plausible for DINOv2 INT8 and docling. The MobileNet INT8 export labels 95% of them "photo".

**Verdict, gap 10:**
- **DINOv2-small + linear head:** 0.989 macro-F1 in fp32 (88.5 MB, 70 ms/crop) and 0.965 in INT8 (24.2 MB, 41 ms/crop). It beats docling's classifier on 943 vs 31 crops (p = 4e-235).
- **MobileNetV3-small + head:** 0.977 in fp32 (6.1 MB, 3 ms/crop). Its dynamic INT8 export is broken (0.033), so INT8 needs static quantization or must stay fp32.
- **docling's classifier on these labels:** 0.689. It is weak on screenshots (0.47), table images (0.59) and "other" (0.33), partly because its taxonomy differs.
- **Pick:** MobileNetV3-small fp32 (6 MB, 3 ms) as the cheap default, or DINOv2-small INT8 (24 MB) where accuracy matters.
- **Before shipping:**
  - retrain on permissive document crops (DocFigure is non-commercial and was not used anyway);
  - validate on owned, mixed-source document crops to remove the source-style effect.
