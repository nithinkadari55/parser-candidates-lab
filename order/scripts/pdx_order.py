"""PaddleX `xycut_enhanced` (Apache-2.0, fetched at run time into order/vendor/pdx_lp by fetch.py paddlex).

PaddleX feeds it blocks whose text-line height, line width and line count come from OCR. We have no OCR
lines, so each non-graphic block gets an estimate: page line height L = 25th percentile of text-block
heights (clipped to 0.6 %-3 % of the page height), num_of_lines = round(height / L), line width = block
width, and horizontal direction (English pages). Graphic blocks keep PaddleX's bbox-based direction.
"""
import os, sys

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")

# OmniDocBench / DrDocBench categories -> PaddleX labels
FROM_ODB = {"title": "paragraph_title", "text_block": "text", "figure": "image", "table": "table", "figure_caption": "figure_title",
            "table_caption": "table_title", "figure_footnote": "vision_footnote", "table_footnote": "vision_footnote",
            "equation_isolated": "formula", "reference": "reference", "code_txt": "algorithm", "code_algorithm": "algorithm",
            "code_txt_caption": "figure_title", "equation_explanation": "text", "chart_mask": "chart", "table_mask": "table",
            "unknown_mask": "image", "algorithm_mask": "algorithm", "text_mask": "text", "need_mask": "text", "header": "header",
            "footer": "footer", "page_number": "number", "page_footnote": "footnote", "abandon": "aside_text",
            "equation_caption": "formula_number", "chart": "chart", "chart_caption": "chart_title"}
# docling / heron / egret native labels and docling-rs JSON labels -> PaddleX labels
FROM_DOCLING = {"Caption": "figure_title", "Footnote": "footnote", "Formula": "formula", "List-item": "text", "Page-footer": "footer",
                "Page-header": "header", "Picture": "image", "Section-header": "paragraph_title", "Table": "table", "Text": "text",
                "Title": "doc_title", "Document Index": "content", "Code": "algorithm", "Checkbox-Selected": "text",
                "Checkbox-Unselected": "text", "Form": "text", "Key-Value Region": "text",
                "caption": "figure_title", "footnote": "footnote", "formula": "formula", "list_item": "text", "page_footer": "footer",
                "page_header": "header", "picture": "image", "chart": "chart", "section_header": "paragraph_title", "table": "table",
                "text": "text", "title": "doc_title", "document_index": "content", "code": "algorithm", "checkbox_selected": "text",
                "checkbox_unselected": "text", "form": "text", "key_value_region": "text", "reference": "reference"}
GRAPHIC = {"image", "table", "chart", "seal", "figure", "flowchart", "header_image", "footer_image"}
_M = None


def _mods():
    global _M
    if _M is None:
        sys.path.insert(0, VENDOR)
        from pdx_lp.layout_objects import LayoutBlock, LayoutRegion
        from pdx_lp.xycut_enhanced import xycut_enhanced
        _M = (LayoutBlock, LayoutRegion, xycut_enhanced)
    return _M


def order(boxes, labels, texts, W, H):
    """boxes [x0,y0,x1,y1], labels in PaddleX vocabulary. Returns a permutation of range(len(boxes))."""
    LayoutBlock, LayoutRegion, xycut_enhanced = _mods()
    n = len(boxes)
    hs = sorted(b[3] - b[1] for b, l in zip(boxes, labels) if l not in GRAPHIC and b[3] > b[1])
    L = hs[len(hs) // 4] if hs else 0.015 * H
    L = max(0.006 * H, min(0.03 * H, L), 4.0)
    blocks = []
    for i, (b, l) in enumerate(zip(boxes, labels)):
        bb = [int(round(b[0])), int(round(b[1])), int(round(max(b[2], b[0] + 1))), int(round(max(b[3], b[1] + 1)))]
        blk = LayoutBlock(label=l, bbox=bb, content=(texts[i] if texts else "") or "")
        if l not in GRAPHIC:
            h = bb[3] - bb[1]
            blk.num_of_lines = max(1, int(round(h / L)))
            blk.text_line_height = float(min(h, L)) if h >= 2 else 2.0
            blk.text_line_width = float(max(2, bb[2] - bb[0]))
            blk.seg_start_coordinate = bb[0]
            blk.seg_end_coordinate = bb[2]
            blk.update_direction("horizontal")
        blk.pdx_index = i
        blocks.append(blk)
    region = LayoutRegion(bbox=[0, 0, int(W), int(H)], blocks=blocks)
    res = xycut_enhanced(region)
    out, seen = [], set()
    for blk in res:
        i = getattr(blk, "pdx_index", None)
        if i is not None and i not in seen:
            seen.add(i)
            out.append(i)
    missing = sorted((i for i in range(n) if i not in seen), key=lambda i: (boxes[i][1], boxes[i][0]))
    return out + missing, len(missing)
