# order: reading-order benchmark (logic vs docling vs PP-DocLayoutV3)

Question: can rule-based ordering over layout boxes (no model) match PP-DocLayoutV3's learned
reading order and beat docling's rule-based order on English documents?

Everything runs in `.github/workflows/order.yml` (ubuntu-latest). No dataset, model weight or
third-party source is committed; jobs download them and keep them in `actions/cache`.

## Methods

| name | what |
|---|---|
| `docling_rb` | docling's rule-based `ReadingOrderPredictor` (`reading_order_rb.py`, MIT), fetched at docling commit `5e600e8` into `order/vendor/` at run time. docling.rs 2.84.0 is a port of it. |
| `xycut_yfirst`, `xycut_xfirst` | plain recursive XY-cut over region boxes (`scripts/ro.py`), cutting horizontally / vertically first. |
| `xycutpp` | XY-Cut++-style ordering written from the paper (arXiv 2504.10258): spanning blocks (no side neighbour, crossing a column gutter) split the page into bands; each band is cut into columns; a stacked-groups check prefers horizontal cuts when the column groups do not overlap vertically. Geometry only. |
| `xycutpp_lab` | `xycutpp` plus caption/footnote attachment to the nearest picture/table (needs labels). |
| `heron:docling_rs` | docling-rs 2.84.0 end to end on the page image (heron layout + docling.rs reading order); order = JSON body order. |
| `pp:ppdlv3_order` | PP-DocLayoutV3 ONNX (`PaddlePaddle/PP-DocLayoutV3_onnx@46bbdf1`, Apache-2.0) predicted order index. |
| `heron:*`, `pp:*` | the logic methods run on the boxes detected by heron / PP-DocLayoutV3. |
| reference tools (own cases only) | EdgeParse (pip wheel and source build), pdf_oxide, OpenDataLoader PDF (Java), tagged-PDF structure tree. |

## Data

- OmniDocBench (`opendatalab/OmniDocBench` on Hugging Face, research use only, not redistributed):
  all English pages with at least two ordered regions. Scored regions: every element with a
  reading-order index except header, footer, page number, page footnote, abandon and equation number.
- Buckets: simple = single column without figures/tables; medium = double column, or single column with
  figures/tables; complex = three/mixed/other columns, newspapers, magazines, slides. Edge tags
  (overlapping): full-width block interrupting columns, footnotes, caption beside figure, exam/numbered
  items, landscape or spread.
- `split` dev/test: XY-Cut++ rules were tuned by looking at dev pages only; test numbers are unbiased.
- `data/heron_subset.txt`: 139 stratified pages run through docling-rs (OCR on images).
- `data/own_expected.json`: hand-checked snippet order for public docling.rs test PDFs
  (`tests/data/pdf/sources`, fetched at docling.rs commit `cfb8df2`).

## Metrics

Kendall tau and normalised edit distance between predicted and true region order, and the share of
pages in fully correct order. With detected boxes each ground-truth region is matched to the detected
box covering it (intersection >= 50 % of the GT box, or >= 80 % of the detected box); order is scored
over matched regions and coverage is reported. Own cases: share of snippet pairs in the right order,
all snippets found, no interleaving of independent chains.
