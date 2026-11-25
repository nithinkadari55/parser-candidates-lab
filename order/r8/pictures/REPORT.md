# Round 8, test A: picture classifier on a broad, multi-source set

**Run:** 45678324039 (`pictures-r8.yml`).
- All jobs succeeded: 10 data groups, then the docling / mnv3 / dinov2 model jobs, then the report.
- The tables below were regenerated locally from the run's per-item files with the committed `p8_report.py`. The added table is macro-F1 over the classes that have at least two sources.
- Code: `order/r8/pictures/` (`p8_data.py`, `p8_model.py`, `p8_report.py`, `p8_common.py`). Hand labels: `realdoc_labels.jsonl`.

## What changed from round 4

Round 4 drew each class from one dataset. Its 0.977-0.989 macro-F1 mostly measured source style. This round fixes that in four ways:
- **Several sources per class.** 25 sources and 15,043 crops after cross-source de-duplication.
- **Leave-one-source-out scoring.** Every crop is predicted by a head that never saw its source.
- **Hand-labelled real-document set.** 465 crops of real pictures from DocLayNet test pages and OmniDocBench pages:
  - labelled by eye (by Claude) from contact sheets, before any model was run on them;
  - ambiguous, composite, decorative and clip-art crops (135 of 609) were dropped;
  - never trained on.
- **Shippable variant.** Every head is also trained only on permissive-licence sources ("perm": CC-BY, CC0 / PD, Apache, MIT, CDLA). That is the version that could ship.

**Shared label set:** chart, diagram, photo, screenshot, logo, signature, stamp, table_image, map. These are the classes docling's 26-class DocumentFigureClassifier-v2.5 and the heads both cover; the class list comes from docling.rs `crates/docling-pdf/src/enrich.rs`.

**How docling is scored:** at its best, "restricted" = argmax of the summed probabilities of the shared classes. Its raw top-1 is also shown.

**How every model sees the crop:** the same 224x224 squash resize that docling.rs uses.

## Key numbers

macro-F1 [95% cluster-bootstrap CI]:

| arm | held-out sources, all 9 classes | held-out sources, 7 classes with at least 2 sources | real-document crops (6 classes with at least 10 crops) |
|---|---|---|---|
| **docling DocumentFigureClassifier-v2.5 (restricted)** | **0.798 [0.791, 0.805]** | **0.811 [0.804, 0.818]** | **0.899 [0.830, 0.948]** |
| docling (raw top-1) | 0.794 | 0.805 | 0.896 |
| MobileNetV3-small head, fp32, all data | 0.615 | 0.791 | 0.700 |
| MobileNetV3-small head, fp32, **permissive only** | 0.540 [0.534, 0.546] | 0.694 [0.686, 0.702] | 0.571 [0.487, 0.627] |
| DINOv2-small head, fp32, all data | 0.661 | **0.850 [0.844, 0.856]** | 0.836 [0.758, 0.888] |
| DINOv2-small head, fp32, **permissive only** | 0.594 [0.589, 0.599] | 0.764 [0.757, 0.770] | 0.719 [0.628, 0.781] |
| DINOv2-small head, int8, all data | 0.654 | 0.841 | 0.785 |
| DINOv2-small head, int8, **permissive only** | 0.595 [0.590, 0.600] | 0.765 [0.758, 0.772] | 0.675 [0.582, 0.741] |

**Why the heads score 0 on two classes in the first column:** logo and map each have only one pool source, LogoDet-3K and UB-PMC. Leaving that source out leaves the head no examples of the class, so both score 0. The second column drops those two classes. The real-document set still covers logos (26) and maps (13), using heads trained on all pool sources.

**Real-document crops, per class.** F1 for docling (restricted) / DINOv2 fp32 perm:

| class | docling | DINOv2 fp32 perm |
|---|---|---|
| chart | 0.92 | 0.77 |
| diagram | 0.92 | 0.57 |
| photo | 0.95 | 0.89 |
| screenshot | 0.94 | 0.63 |
| logo | 0.89 | 0.77 |
| map | 0.77 | 0.69 |

The permissive diagram data is only synthetic flowcharts (QZhou). The heads cannot recognise chemistry structures, patent / engineering drawings or science diagrams from them.

**Size and speed** (CPU, 1 thread, per crop, GitHub runner):

| model | MB | ms |
|---|---|---|
| docling | 16.9 | 5.4 |
| MobileNetV3 + head | 6.1 | 0.9 |
| DINOv2 fp32 + head | 88.5 | 104 |
| DINOv2 int8 + head | 24.2 | 52 |

The int8 graph keeps the head in fp32. It agrees with the feature path on 98.5% of checked crops.

**Chart subtype** (bar / line / pie / scatter / box; 981 crops from ChartX, SciCap, UB-PMC):
- docling: 0.867 restricted to its chart classes, 0.855 raw.
- DINOv2 fp32 head: 0.836.
- MobileNetV3 head: 0.783.

## Decision

**Rule:** replace docling's classifier only if the shippable head beats it clearly. Its macro-F1 CI must lie above docling's, both on the shared classes over held-out sources and on the real-document crops.

**Result:** no permissive-only head beats docling on either condition. All of them are clearly below:
- Held-out sources, 7 multi-source classes: DINOv2 perm -0.047 [-0.056, -0.039].
- Real-document crops: DINOv2 fp32 perm -0.180 [-0.262, -0.110].

**Verdict: keep docling's DocumentFigureClassifier-v2.5.** This also withdraws the round-4 pick ("MobileNetV3 head as the cheap default"). Once source style is removed, the MobileNetV3 head is well below docling.

Only the non-shippable DINOv2 head trained on all data, including GPL / NC / SA sources, beats docling, and only on the 7 multi-source classes: +0.039 [+0.031, +0.047]. On real-document crops it is level or below (-0.063 [-0.138, +0.004]).

## Caveats

- **Hand labels:**
  - The real-document labels were made by one annotator (Claude) from 300 px contact sheets. There was no second annotator.
  - The 135 dropped crops (icons, clip art, cartoons, composites, QR / bar codes, text) fall outside the shared classes. docling has dedicated classes for several of them; the heads have none.
- **Rare classes:** the real-document set has few tables (4), signatures (1) and stamps (0). Those classes are judged on the held-out-source sets only.
- **docling's training data:** it is not documented here. Some public sources (charts, tables) may overlap with it, which would favour docling on held-out sources. The real-document crops are the cleaner comparison, and docling leads there too.
- **Stamp sub-sources:** two mapo80 stamp sub-sources (`stamp_detectation_marcos`, `stamp_detection_stampa`) are text rubber-stamps and noisy crops. docling labels them logo / photograph (accuracy 0.07 / 0.12), which drags its stamp F1 down to 0.58. A spot check of 12 crops confirmed they are real document stamps, with some noisy crops.
- **Shipping gap:** a shippable replacement would also need docling's other classes (icon, QR / bar code, page thumbnail and the chart subtypes), which the harness uses today.

## Data sources skipped

- **DocFigure and IIIT-AR-13K:** cvit.iiit.ac.in does not resolve (DNS) from the laptop or from GitHub runners.
- **ACL-Fig** (HF `citeseerx/ACL-fig`) **and tech4humans/signature-detection:** gated on Hugging Face (terms must be accepted).
- **UB-PMC:** used. Its test set (CC-BY PMC figures) also supplied maps.
- **Wikimedia Commons** (planned as a permissive second source for maps, logos, signatures, diagrams): every image download from the runners failed after retries. It worked from the laptop. Dropped.
- **cedar_signatures** (HF T0KII, non-permissive): hit HF rate limit 429 part-way, so 128 crops were kept.
- **DocLayNet / OmniDocBench picture-type metadata:** neither dataset has any, hence the hand labels.
- **COCO:** only val2017 images with licence id 4 / 7 / 8 were used.

---

# Full tables (generated by p8_report.py from the run's per-item files)

Shared label set (classes that docling's DocumentFigureClassifier-v2.5 and the heads both support): chart, diagram, photo, screenshot, logo, signature, stamp, table_image, map.
docling's 26 classes map as: line/bar/pie/scatter/box chart -> chart; flow_chart, engineering_drawing, chemistry_structure -> diagram; photograph -> photo; screenshot_from_computer/manual -> screenshot; geographical/topographical_map -> map; table -> table_image; logo, signature, stamp as is. **docling (restricted)** = argmax of the summed probabilities of the shared classes (its best case on a closed set, used for the decision); **docling (raw)** = its top-1 class, where icon, other, page_thumbnail, qr_code, bar_code, music, calendar, full_page_image and crossword_puzzle count as wrong.

Heads: logistic regression (class-balanced, C by grouped CV inside each fold) on frozen features. All arms see the same crop resized to 224x224 (squash, bilinear), the way docling.rs feeds its classifier. **Held-out sources:** every source is predicted by a head that never saw it (leave-one-source-out). *_all = trained on all other sources; *_perm = trained only on the other permissively licensed sources (CC-BY, CC0/PD, Apache, MIT, CDLA) - the shippable candidate. Hand-labelled real-document crops are never trained on. CIs: 95% bootstrap over groups (page, document, paper or chart), 2,000 resamples.

## Data

| source | role | permissive | licence (as read from the files) | crops kept | per class | status |
|---|---|---|---|---|---|---|
| chartx | pool | yes | Apache-2.0 (HF card InternScience/ChartX) | 594 | chart 594 | ok |
| plotqa | pool | yes | CC-BY-4.0 (PlotQA; HF achang/plot_qa card 'cc') | 600 | chart 600 | ok |
| chartqa | pool | no | GPL-3.0 (HF HuggingFaceM4/ChartQA card) | 592 | chart 592 | ok |
| ubpmc | pool | yes | CC-BY (UB-PMC CHART-Info 2022 test, PMC open-access CC-BY figures) | 968 | chart 598, map 370 | ok |
| scicap | pool | no | CC-BY-NC-SA-4.0 (HF CrowdAILab/scicap card) | 1199 | chart 599, diagram 600 | ok |
| ai2d | pool | no | CC-BY-SA (AI2D, Allen Institute) | 599 | diagram 599 | ok |
| qzhou_flowchart | pool | yes | Apache-2.0 (HF Kingsoft-LLM/QZhou-Flowchart-QA card) | 600 | diagram 600 | ok |
| openimages | pool | yes | CC-BY-4.0 annotations, images CC-BY-2.0 (HF finedet/openimages card) | 600 | photo 600 | ok |
| coco_ccby | pool | yes | COCO val2017 images with licence id 4/7/8 only (CC-BY, no known restriction, US gov) | 600 | photo 600 | ok |
| imagenette | pool | no | ImageNet subset (non-commercial research terms) | 600 | photo 600 | ok |
| rico | pool | yes | CC-BY-4.0 (HF bevaya/RICO-Screen2Words card) | 595 | screenshot 595 | ok |
| screenspot | pool | yes | Apache-2.0 (HF OS-Copilot/ScreenSpot-v2 card) | 588 | screenshot 588 | ok |
| websight | pool | yes | CC-BY-4.0 (HF HuggingFaceM4/WebSight card) | 588 | screenshot 588 | ok |
| logodet | pool | yes | MIT (HF axonstan/LogoDet-3K card; brand logos are trademarks) | 600 | logo 600 | ok |
| signverod | pool | yes | Apache-2.0 (HF ondrs/signverod card) | 600 | signature 600 | ok |
| rf100_signatures | pool | yes | CC-BY-4.0 (Roboflow 100 signatures-xc8up, HF LibreYOLO) | 443 | signature 443 | ok |
| cedar_signatures | pool | no | MIT on HF T0KII card, but file names match CEDAR (research terms): treated as non-permissive | 128 | signature 128 | failed: HfHubHTTPError: 429 Client Error: Too Many Requests for url: https://huggingface.co/api/datasets/T0KII/signature |
| stamps_jsam | pool | yes | CC-BY-4.0 (HF mapo80/stamps, sub-source stamp_detection_jsam) | 600 | stamp 600 | ok |
| stamps_yolo | pool | yes | CC-BY-4.0 (HF mapo80/stamps, sub-source yolo_stamp_classify) | 599 | stamp 599 | ok |
| stamps_marcos | pool | yes | CC-BY-4.0 (HF mapo80/stamps, sub-source stamp_detectation_marcos) | 412 | stamp 412 | ok |
| stamps_stampa | pool | yes | CC-BY-4.0 (HF mapo80/stamps, sub-source stamp_detection_stampa) | 600 | stamp 600 | ok |
| hyperbots_stamps | pool | yes | CC-BY-4.0 (HF hyperbots-inc/synthetic-stamped-documents) | 162 | stamp 162 | ok |
| doclaynet_tables | pool | yes | CDLA-Permissive-1.0 (DocLayNet v1.1 val Table boxes) | 517 | table_image 517 | ok |
| pubtabnet | pool | yes | CDLA-Permissive-1.0 (HF apoidea/pubtabnet-html validation) | 599 | table_image 599 | ok |
| publaynet_tables | pool | yes | CDLA-Permissive-1.0 (PubLayNet validation Table boxes) | 595 | table_image 595 | ok |
| realdoc_doclaynet | test | no | DocLayNet v1.1 test Picture boxes, hand-labelled (CDLA-Permissive-1.0) | 240 | chart 69, diagram 75, logo 24, map 11, photo 40, screenshot 18, signature 1, table_image 2 | ok |
| realdoc_omnidocbench | test | no | OmniDocBench figure boxes, hand-labelled (licence not stated) | 225 | chart 72, diagram 67, logo 2, map 2, photo 74, screenshot 6, table_image 2 | ok |

Near-duplicates removed across sources (first copy kept): {"ai2d": 1, "chartqa": 8, "chartx": 6, "doclaynet_tables": 10, "hyperbots_stamps": 11, "publaynet_tables": 5, "pubtabnet": 1, "rico": 5, "scicap": 1, "screenspot": 12, "stamps_marcos": 1, "stamps_yolo": 1, "ubpmc": 5, "websight": 12, "realdoc_doclaynet": 9}

Sources per class (pool): chart: 5 (3 permissive); diagram: 3 (1 permissive); photo: 3 (2 permissive); screenshot: 3 (3 permissive); logo: 1 (1 permissive); signature: 3 (2 permissive); stamp: 5 (5 permissive); table_image: 3 (3 permissive); map: 1 (1 permissive)


## Results


### Held-out sources (leave-one-source-out, all pool sources): 14578 crops, 12057 groups

Class counts: chart 2983, diagram 1799, photo 1800, screenshot 1771, logo 600, signature 1171, stamp 2373, table_image 1711, map 370

| arm | macro-F1 [95% CI] | accuracy | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | map |
|---|---|---|---|---|---|---|---|---|---|---|---|
| docling_restricted | 0.798 [0.791, 0.805] | 0.813 | 0.93 | 0.80 | 0.87 | 0.82 | 0.57 | 0.85 | 0.58 | 0.82 | 0.93 |
| docling_raw | 0.794 [0.787, 0.801] | 0.791 | 0.93 | 0.80 | 0.88 | 0.78 | 0.58 | 0.85 | 0.57 | 0.82 | 0.93 |
| mnv3_fp32_all | 0.615 [0.609, 0.620] | 0.761 | 0.83 | 0.67 | 0.91 | 0.78 | 0.00 | 0.66 | 0.77 | 0.92 | 0.00 |
| mnv3_fp32_perm | 0.540 [0.534, 0.546] | 0.692 | 0.77 | 0.23 | 0.89 | 0.74 | 0.00 | 0.61 | 0.73 | 0.89 | 0.00 |
| dinov2_fp32_all | 0.661 [0.657, 0.666] | 0.815 | 0.81 | 0.74 | 0.94 | 0.79 | 0.00 | 0.83 | 0.87 | 0.96 | 0.00 |
| dinov2_fp32_perm | 0.594 [0.589, 0.599] | 0.752 | 0.73 | 0.30 | 0.94 | 0.76 | 0.00 | 0.80 | 0.88 | 0.94 | 0.00 |
| dinov2_int8_all | 0.654 [0.650, 0.659] | 0.809 | 0.82 | 0.70 | 0.94 | 0.82 | 0.00 | 0.82 | 0.83 | 0.96 | 0.00 |
| dinov2_int8_perm | 0.595 [0.590, 0.600] | 0.751 | 0.77 | 0.37 | 0.91 | 0.78 | 0.00 | 0.80 | 0.79 | 0.95 | 0.00 |

Paired difference vs docling (restricted), same bootstrap resamples:

| arm | macro-F1 - docling [95% CI] |
|---|---|
| docling_raw | -0.004 [-0.006, -0.001] |
| mnv3_fp32_all | -0.183 [-0.191, -0.175] |
| mnv3_fp32_perm | -0.258 [-0.267, -0.250] |
| dinov2_fp32_all | -0.137 [-0.145, -0.128] |
| dinov2_fp32_perm | -0.204 [-0.212, -0.196] |
| dinov2_int8_all | -0.143 [-0.152, -0.135] |
| dinov2_int8_perm | -0.203 [-0.212, -0.194] |

### Held-out sources, classes with at least two sources (chart, diagram, photo, screenshot, signature, stamp, table_image): 14578 crops, 12057 groups

Class counts: chart 2983, diagram 1799, photo 1800, screenshot 1771, logo 600, signature 1171, stamp 2373, table_image 1711, map 370; macro-F1 over: chart, diagram, photo, screenshot, signature, stamp, table_image

| arm | macro-F1 [95% CI] | accuracy | chart | diagram | photo | screenshot | signature | stamp | table_image |
|---|---|---|---|---|---|---|---|---|---|
| docling_restricted | 0.811 [0.804, 0.818] | 0.813 | 0.93 | 0.80 | 0.87 | 0.82 | 0.85 | 0.58 | 0.82 |
| docling_raw | 0.805 [0.798, 0.812] | 0.791 | 0.93 | 0.80 | 0.88 | 0.78 | 0.85 | 0.57 | 0.82 |
| mnv3_fp32_all | 0.791 [0.783, 0.797] | 0.761 | 0.83 | 0.67 | 0.91 | 0.78 | 0.66 | 0.77 | 0.92 |
| mnv3_fp32_perm | 0.694 [0.686, 0.702] | 0.692 | 0.77 | 0.23 | 0.89 | 0.74 | 0.61 | 0.73 | 0.89 |
| dinov2_fp32_all | 0.850 [0.844, 0.856] | 0.815 | 0.81 | 0.74 | 0.94 | 0.79 | 0.83 | 0.87 | 0.96 |
| dinov2_fp32_perm | 0.764 [0.757, 0.770] | 0.752 | 0.73 | 0.30 | 0.94 | 0.76 | 0.80 | 0.88 | 0.94 |
| dinov2_int8_all | 0.841 [0.835, 0.848] | 0.809 | 0.82 | 0.70 | 0.94 | 0.82 | 0.82 | 0.83 | 0.96 |
| dinov2_int8_perm | 0.765 [0.758, 0.772] | 0.751 | 0.77 | 0.37 | 0.91 | 0.78 | 0.80 | 0.79 | 0.95 |

Paired difference vs docling (restricted), same bootstrap resamples:

| arm | macro-F1 - docling [95% CI] |
|---|---|
| docling_raw | -0.007 [-0.009, -0.004] |
| mnv3_fp32_all | -0.021 [-0.029, -0.012] |
| mnv3_fp32_perm | -0.117 [-0.126, -0.108] |
| dinov2_fp32_all | +0.039 [+0.031, +0.047] |
| dinov2_fp32_perm | -0.047 [-0.056, -0.039] |
| dinov2_int8_all | +0.030 [+0.022, +0.038] |
| dinov2_int8_perm | -0.046 [-0.055, -0.037] |

### Real-document picture crops (DocLayNet test + OmniDocBench, hand-labelled, never trained on): 465 crops, 262 groups

Class counts: chart 141, diagram 142, photo 114, screenshot 24, logo 26, signature 1, table_image 4, map 13; macro-F1 over: chart, diagram, photo, screenshot, logo, map

| arm | macro-F1 [95% CI] | accuracy | chart | diagram | photo | screenshot | logo | map |
|---|---|---|---|---|---|---|---|---|
| docling_restricted | 0.899 [0.830, 0.948] | 0.916 | 0.92 | 0.92 | 0.95 | 0.94 | 0.89 | 0.77 |
| docling_raw | 0.896 [0.827, 0.942] | 0.903 | 0.91 | 0.90 | 0.96 | 0.94 | 0.87 | 0.80 |
| mnv3_fp32_all | 0.700 [0.616, 0.760] | 0.729 | 0.81 | 0.73 | 0.90 | 0.62 | 0.50 | 0.64 |
| mnv3_fp32_perm | 0.571 [0.487, 0.627] | 0.555 | 0.75 | 0.24 | 0.88 | 0.63 | 0.44 | 0.48 |
| dinov2_fp32_all | 0.836 [0.758, 0.888] | 0.852 | 0.86 | 0.84 | 0.95 | 0.76 | 0.88 | 0.73 |
| dinov2_fp32_perm | 0.719 [0.628, 0.781] | 0.725 | 0.77 | 0.57 | 0.89 | 0.63 | 0.77 | 0.69 |
| dinov2_int8_all | 0.785 [0.700, 0.850] | 0.798 | 0.85 | 0.82 | 0.94 | 0.81 | 0.56 | 0.73 |
| dinov2_int8_perm | 0.675 [0.582, 0.741] | 0.682 | 0.84 | 0.60 | 0.86 | 0.62 | 0.47 | 0.67 |

Paired difference vs docling (restricted), same bootstrap resamples:

| arm | macro-F1 - docling [95% CI] |
|---|---|
| docling_raw | -0.003 [-0.018, +0.010] |
| mnv3_fp32_all | -0.199 [-0.277, -0.130] |
| mnv3_fp32_perm | -0.328 [-0.402, -0.272] |
| dinov2_fp32_all | -0.063 [-0.138, +0.004] |
| dinov2_fp32_perm | -0.180 [-0.262, -0.110] |
| dinov2_int8_all | -0.114 [-0.182, -0.060] |
| dinov2_int8_perm | -0.224 [-0.299, -0.165] |

###   of which DocLayNet test pictures: 240 crops, 95 groups

Class counts: chart 69, diagram 75, photo 40, screenshot 18, logo 24, signature 1, table_image 2, map 11; macro-F1 over: chart, diagram, photo, screenshot, logo, map

| arm | macro-F1 [95% CI] | accuracy | chart | diagram | photo | screenshot | logo | map |
|---|---|---|---|---|---|---|---|---|
| docling_restricted | 0.916 [0.830, 0.963] | 0.917 | 0.90 | 0.93 | 0.96 | 0.94 | 0.88 | 0.87 |
| docling_raw | 0.909 [0.826, 0.956] | 0.908 | 0.90 | 0.93 | 0.96 | 0.94 | 0.86 | 0.87 |
| mnv3_fp32_all | 0.727 [0.626, 0.800] | 0.721 | 0.82 | 0.78 | 0.89 | 0.68 | 0.56 | 0.62 |
| mnv3_fp32_perm | 0.631 [0.529, 0.694] | 0.554 | 0.79 | 0.31 | 0.83 | 0.71 | 0.57 | 0.57 |
| dinov2_fp32_all | 0.904 [0.837, 0.950] | 0.904 | 0.93 | 0.91 | 0.96 | 0.85 | 0.91 | 0.86 |
| dinov2_fp32_perm | 0.789 [0.688, 0.854] | 0.754 | 0.77 | 0.67 | 0.86 | 0.71 | 0.86 | 0.86 |
| dinov2_int8_all | 0.842 [0.752, 0.907] | 0.833 | 0.93 | 0.88 | 0.94 | 0.85 | 0.59 | 0.86 |
| dinov2_int8_perm | 0.736 [0.627, 0.812] | 0.713 | 0.90 | 0.73 | 0.78 | 0.70 | 0.51 | 0.80 |

Paired difference vs docling (restricted), same bootstrap resamples:

| arm | macro-F1 - docling [95% CI] |
|---|---|
| docling_raw | -0.007 [-0.022, +0.000] |
| mnv3_fp32_all | -0.189 [-0.276, -0.105] |
| mnv3_fp32_perm | -0.285 [-0.375, -0.211] |
| dinov2_fp32_all | -0.012 [-0.086, +0.075] |
| dinov2_fp32_perm | -0.127 [-0.221, -0.039] |
| dinov2_int8_all | -0.074 [-0.147, +0.001] |
| dinov2_int8_perm | -0.180 [-0.264, -0.103] |

###   of which OmniDocBench figures: 225 crops, 167 groups

Class counts: chart 72, diagram 67, photo 74, screenshot 6, logo 2, table_image 2, map 2; macro-F1 over: chart, diagram, photo

| arm | macro-F1 [95% CI] | accuracy | chart | diagram | photo |
|---|---|---|---|---|---|
| docling_restricted | 0.929 [0.886, 0.966] | 0.916 | 0.93 | 0.91 | 0.95 |
| docling_raw | 0.919 [0.874, 0.957] | 0.898 | 0.93 | 0.87 | 0.96 |
| mnv3_fp32_all | 0.795 [0.736, 0.852] | 0.738 | 0.80 | 0.68 | 0.91 |
| mnv3_fp32_perm | 0.592 [0.537, 0.646] | 0.556 | 0.71 | 0.16 | 0.91 |
| dinov2_fp32_all | 0.834 [0.776, 0.886] | 0.796 | 0.78 | 0.77 | 0.95 |
| dinov2_fp32_perm | 0.704 [0.636, 0.765] | 0.693 | 0.76 | 0.45 | 0.90 |
| dinov2_int8_all | 0.817 [0.761, 0.867] | 0.760 | 0.77 | 0.74 | 0.94 |
| dinov2_int8_perm | 0.706 [0.640, 0.764] | 0.649 | 0.78 | 0.44 | 0.90 |

Paired difference vs docling (restricted), same bootstrap resamples:

| arm | macro-F1 - docling [95% CI] |
|---|---|
| docling_raw | -0.009 [-0.028, +0.005] |
| mnv3_fp32_all | -0.133 [-0.185, -0.084] |
| mnv3_fp32_perm | -0.337 [-0.393, -0.279] |
| dinov2_fp32_all | -0.095 [-0.149, -0.046] |
| dinov2_fp32_perm | -0.225 [-0.294, -0.150] |
| dinov2_int8_all | -0.112 [-0.166, -0.063] |
| dinov2_int8_perm | -0.222 [-0.291, -0.159] |

### Figures cropped from real papers with source labels (UB-PMC, SciCap): 2167 crops, 2031 groups

Class counts: chart 1197, diagram 600, map 370

| arm | macro-F1 [95% CI] | accuracy | chart | diagram | map |
|---|---|---|---|---|---|
| docling_restricted | 0.966 [0.957, 0.973] | 0.960 | 0.97 | 0.95 | 0.98 |
| docling_raw | 0.962 [0.953, 0.970] | 0.948 | 0.96 | 0.95 | 0.98 |
| mnv3_fp32_all | 0.549 [0.538, 0.559] | 0.725 | 0.90 | 0.75 | 0.00 |
| mnv3_fp32_perm | 0.418 [0.402, 0.434] | 0.526 | 0.80 | 0.46 | 0.00 |
| dinov2_fp32_all | 0.575 [0.565, 0.585] | 0.766 | 0.87 | 0.86 | 0.00 |
| dinov2_fp32_perm | 0.468 [0.452, 0.483] | 0.622 | 0.79 | 0.61 | 0.00 |
| dinov2_int8_all | 0.564 [0.554, 0.574] | 0.746 | 0.85 | 0.84 | 0.00 |
| dinov2_int8_perm | 0.502 [0.488, 0.516] | 0.619 | 0.80 | 0.71 | 0.00 |

Paired difference vs docling (restricted), same bootstrap resamples:

| arm | macro-F1 - docling [95% CI] |
|---|---|
| docling_raw | -0.004 [-0.007, -0.001] |
| mnv3_fp32_all | -0.417 [-0.429, -0.405] |
| mnv3_fp32_perm | -0.547 [-0.565, -0.530] |
| dinov2_fp32_all | -0.391 [-0.402, -0.379] |
| dinov2_fp32_perm | -0.498 [-0.515, -0.481] |
| dinov2_int8_all | -0.402 [-0.413, -0.389] |
| dinov2_int8_perm | -0.463 [-0.479, -0.448] |

## Accuracy per held-out source

| source | crops | docling_restricted | docling_raw | mnv3_fp32_all | mnv3_fp32_perm | dinov2_fp32_all | dinov2_fp32_perm | dinov2_int8_all | dinov2_int8_perm |
|---|---|---|---|---|---|---|---|---|---|
| ai2d | 599 | 0.90 | 0.75 | 0.55 | 0.07 | 0.52 | 0.05 | 0.42 | 0.10 |
| cedar_signatures | 128 | 0.97 | 0.95 | 0.97 | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 |
| chartqa | 592 | 0.98 | 0.98 | 0.76 | 0.77 | 0.67 | 0.60 | 0.73 | 0.65 |
| chartx | 594 | 0.95 | 0.93 | 0.85 | 0.84 | 0.90 | 0.90 | 0.91 | 0.88 |
| coco_ccby | 600 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| doclaynet_tables | 517 | 0.92 | 0.92 | 0.85 | 0.86 | 0.92 | 0.92 | 0.88 | 0.88 |
| hyperbots_stamps | 162 | 0.76 | 0.75 | 0.93 | 0.93 | 0.94 | 0.92 | 0.98 | 0.98 |
| imagenette | 600 | 0.98 | 0.96 | 0.88 | 0.89 | 0.98 | 0.96 | 0.98 | 0.96 |
| logodet | 600 | 0.75 | 0.69 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| openimages | 600 | 0.97 | 0.96 | 0.96 | 0.95 | 0.98 | 0.96 | 0.97 | 0.96 |
| plotqa | 600 | 1.00 | 1.00 | 0.95 | 0.94 | 1.00 | 0.99 | 1.00 | 0.99 |
| publaynet_tables | 595 | 0.99 | 0.99 | 0.94 | 0.94 | 0.99 | 0.99 | 0.99 | 0.98 |
| pubtabnet | 599 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 0.97 | 0.98 |
| qzhou_flowchart | 600 | 0.67 | 0.65 | 0.48 | 0.00 | 0.55 | 0.00 | 0.51 | 0.00 |
| realdoc_doclaynet | 240 | 0.92 | 0.91 | 0.72 | 0.55 | 0.90 | 0.75 | 0.83 | 0.71 |
| realdoc_omnidocbench | 225 | 0.92 | 0.90 | 0.74 | 0.56 | 0.80 | 0.69 | 0.76 | 0.65 |
| rf100_signatures | 443 | 0.73 | 0.72 | 0.46 | 0.45 | 0.80 | 0.78 | 0.80 | 0.78 |
| rico | 595 | 0.85 | 0.78 | 0.92 | 0.93 | 0.95 | 0.95 | 0.94 | 0.96 |
| scicap | 1199 | 0.97 | 0.96 | 0.88 | 0.63 | 0.92 | 0.74 | 0.89 | 0.78 |
| screenspot | 588 | 0.89 | 0.87 | 0.67 | 0.73 | 0.88 | 0.86 | 0.76 | 0.71 |
| signverod | 600 | 0.85 | 0.84 | 0.81 | 0.73 | 0.97 | 0.98 | 0.93 | 0.93 |
| stamps_jsam | 600 | 0.64 | 0.63 | 0.91 | 0.91 | 0.95 | 0.94 | 0.96 | 0.96 |
| stamps_marcos | 412 | 0.07 | 0.05 | 0.29 | 0.32 | 0.65 | 0.70 | 0.70 | 0.75 |
| stamps_stampa | 600 | 0.12 | 0.11 | 0.78 | 0.76 | 0.95 | 0.95 | 0.95 | 0.95 |
| stamps_yolo | 599 | 0.63 | 0.62 | 0.95 | 0.96 | 0.98 | 0.98 | 0.99 | 0.99 |
| ubpmc | 968 | 0.95 | 0.93 | 0.53 | 0.40 | 0.57 | 0.47 | 0.57 | 0.42 |
| websight | 588 | 0.39 | 0.32 | 0.79 | 0.80 | 0.53 | 0.66 | 0.71 | 0.75 |

**Real-document crops, confusion of docling_restricted** (rows = label, columns = prediction):

| label | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | map |
|---|---|---|---|---|---|---|---|---|---|
| chart (141) | 131 | 8 |  |  |  |  |  | 1 | 1 |
| diagram (142) | 8 | 131 |  |  |  | 1 |  | 1 | 1 |
| photo (114) | 1 | 3 | 105 | 1 |  |  |  |  | 4 |
| screenshot (24) |  |  |  | 22 |  |  |  | 2 |  |
| logo (26) | 5 |  |  |  | 21 |  |  |  |  |
| signature (1) |  |  |  |  |  | 1 |  |  |  |
| table_image (4) |  |  | 1 |  |  |  |  | 3 |  |
| map (13) |  | 1 |  |  |  |  |  |  | 12 |

**Real-document crops, confusion of docling_raw** (rows = label, columns = prediction):

| label | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | map | other:icon | other:other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chart (141) | 131 | 8 |  |  |  |  |  | 1 | 1 |  |  |
| diagram (142) | 9 | 125 |  |  |  | 1 |  |  | 1 | 5 | 1 |
| photo (114) | 1 | 2 | 106 | 1 |  |  |  |  | 3 | 1 |  |
| screenshot (24) |  |  |  | 22 |  |  |  | 2 |  |  |  |
| logo (26) | 5 |  |  |  | 20 |  |  |  |  | 1 |  |
| signature (1) |  |  |  |  |  | 1 |  |  |  |  |  |
| table_image (4) |  |  | 1 |  |  |  |  | 3 |  |  |  |
| map (13) |  | 1 |  |  |  |  |  |  | 12 |  |  |

**Real-document crops, confusion of mnv3_fp32_perm** (rows = label, columns = prediction):

| label | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | map |
|---|---|---|---|---|---|---|---|---|---|
| chart (141) | 106 | 2 |  | 2 | 1 | 12 | 7 | 8 | 3 |
| diagram (142) | 27 | 20 | 1 | 8 | 5 | 50 | 19 | 5 | 7 |
| photo (114) | 3 |  | 91 | 1 | 10 | 1 | 5 | 1 | 2 |
| screenshot (24) | 5 |  |  | 17 |  |  |  | 2 |  |
| logo (26) | 1 |  |  | 1 | 12 | 2 | 9 | 1 |  |
| signature (1) |  |  |  |  |  | 1 |  |  |  |
| table_image (4) |  |  |  |  |  | 1 |  | 3 |  |
| map (13) | 1 |  | 1 | 1 |  |  | 2 |  | 8 |

**Real-document crops, confusion of dinov2_fp32_perm** (rows = label, columns = prediction):

| label | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | map |
|---|---|---|---|---|---|---|---|---|---|
| chart (141) | 128 | 4 |  | 4 |  | 1 |  | 3 | 1 |
| diagram (142) | 58 | 59 | 3 | 6 | 1 | 1 | 10 | 3 | 1 |
| photo (114) | 2 |  | 94 | 4 | 8 |  |  |  | 6 |
| screenshot (24) | 3 | 1 |  | 18 |  |  |  | 2 |  |
| logo (26) | 2 |  |  | 1 | 22 |  | 1 |  |  |
| signature (1) |  |  |  |  |  | 1 |  |  |  |
| table_image (4) |  |  |  |  |  |  |  | 4 |  |
| map (13) |  | 1 | 1 |  |  |  |  |  | 11 |

**Real-document crops, confusion of dinov2_int8_perm** (rows = label, columns = prediction):

| label | chart | diagram | photo | screenshot | logo | signature | stamp | table_image | map |
|---|---|---|---|---|---|---|---|---|---|
| chart (141) | 120 | 4 |  | 1 |  | 1 | 11 | 3 | 1 |
| diagram (142) | 20 | 64 | 4 | 6 |  | 1 | 46 | 1 |  |
| photo (114) | 2 |  | 91 | 5 | 10 |  |  |  | 6 |
| screenshot (24) | 3 | 2 | 1 | 16 |  |  |  | 2 |  |
| logo (26) | 1 |  |  |  | 11 | 1 | 13 |  |  |
| signature (1) |  |  |  |  |  | 1 |  |  |  |
| table_image (4) |  |  |  |  |  |  |  | 4 |  |
| map (13) |  |  | 1 |  |  |  | 2 |  | 10 |

docling raw 26-class top-1 on the real-document crops: [["photograph", 107], ["line_chart", 96], ["engineering_drawing", 88], ["bar_chart", 33], ["flow_chart", 27], ["screenshot_from_computer", 23], ["chemistry_structure", 21], ["logo", 20], ["geographical_map", 16], ["pie_chart", 13], ["icon", 7], ["table", 6], ["scatter_plot", 3], ["signature", 2], ["topographical_map", 1], ["box_plot", 1], ["other", 1]]

## Chart subtype (bar / line / pie / scatter / box), chart crops whose source gives it

981 crops; by source {"chartx": 300, "scicap": 399, "ubpmc": 282}

| arm | accuracy | bar | line | pie | scatter | box |
|---|---|---|---|---|---|---|
| docling top-1 (any of 26) | 0.855 | 0.91 | 0.98 | 0.99 | 0.63 | 0.88 |
| docling restricted to its 5 chart classes | 0.867 | 0.92 | 0.98 | 1.00 | 0.66 | 0.88 |
| mnv3 fp32 head (leave-one-source-out) | 0.783 | 0.89 | 0.58 | 0.98 | 0.68 | 0.51 |
| dinov2 fp32 head (leave-one-source-out) | 0.836 | 0.92 | 0.61 | 1.00 | 0.79 | 0.63 |
| dinov2 int8 head (leave-one-source-out) | 0.796 | 0.87 | 0.57 | 0.98 | 0.81 | 0.43 |

## Size and speed (CPU, 1 thread, per crop, GitHub runner)

| model | MB | ms p50 | ms p95 | graph = feature path |
|---|---|---|---|---|
| docling DocumentFigureClassifier-v2.5 | 16.9 | 5.4 | 5.6 | - |
| mnv3 fp32 + head | 6.1 | 0.9 | 1.0 | 1.000 |
| dinov2 fp32 + head | 88.5 | 104.3 | 109.5 | 1.000 |
| dinov2 int8 + head | 24.2 | 52.5 | 55.3 | 0.985 |

## Decision

Rule: replace docling's classifier only if the shippable (permissive-data) head beats it clearly - its macro-F1 CI lies above docling's CI - on the shared classes over held-out sources AND on the real-document crops. docling is scored at its best (restricted).

- mnv3_fp32_perm: held-out sources no clear win; real-document crops no clear win.
- dinov2_fp32_perm: held-out sources no clear win; real-document crops no clear win.
- dinov2_int8_perm: held-out sources no clear win; real-document crops no clear win.

**Verdict: keep docling's classifier (no shippable head passes both conditions).**
