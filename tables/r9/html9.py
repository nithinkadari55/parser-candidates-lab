"""Round 9 set html9: real tables from public-domain US sources, rendered by headless Chromium into text-layer PDF pages.

  DailyMed SPL drug labels (dailymed.nlm.nih.gov web services; HL7 SPL XML tables)   -> type pharma/medical
  Federal Register documents (federalregister.gov API; the documents' HTML tables)   -> type legal/regulatory
Table content and structure (rows, rowspan, colspan, header rows) are the source's; the page layout is ours: one of
three table styles (ruled grid, rules only, plain) and two fonts, chosen by a hash of the document. A table taller
than a page is split into page fragments (the header rows repeat, a row span cut by the page break continues as an
empty spanning cell), so long multi-page tables are tested page by page. The answer key is built from the same cell
model the page is drawn from, so it is exact; cell and table boxes come from the rendered DOM (1 CSS px = 0.75 pt).
usage (from prep9.py): html9.build(writer, scratch, target)"""
import os, re, json, html, time, random, zlib, collections, urllib.request

PAGE_W, PAGE_H, MARGIN = 816, 1056, 72          # CSS px (Letter at 96 dpi), 0.75 in margins
UA = {"User-Agent": "table-bench/1.0"}
STYLES = {
    "ruled": "table{border-collapse:collapse} td,th{border:1px solid #333;padding:3px 5px;vertical-align:top}",
    "rules": "table{border-collapse:collapse;border-top:1.5px solid #000;border-bottom:1.5px solid #000} th,td{padding:3px 6px;vertical-align:top}"
             " thead tr:last-child th, thead tr:last-child td{border-bottom:1px solid #000}",
    "plain": "table{border-collapse:separate;border-spacing:6px 2px} td,th{padding:1px 3px;vertical-align:top}",
}
FONTS = ["'Liberation Serif', 'DejaVu Serif', serif", "'Liberation Sans', 'DejaVu Sans', sans-serif"]


def fetch(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r: return r.read()
        except Exception as e:
            print("fetch retry", k, repr(e)[:120], flush=True); time.sleep(5 + 10 * k)
    return None


def norm(t): return re.sub(r"\s+", " ", t or "").strip()


# ---------- source tables -> cell model: rows of dict(text, rs, cs, th), header row count ----------

def table_model(tab, ns=""):
    """lxml table element -> (rows, n_header_rows) or None"""
    q = lambda n: f"{{{ns}}}{n}" if ns else n
    if tab.find(".//" + q("table")) is not None: return None
    head = tab.find(q("thead")); rows = []; nh = 0
    trs = []
    if head is not None:
        trs += [(tr, True) for tr in head.findall(q("tr"))]
    for part in tab:
        if part.tag in (q("tbody"), q("tfoot")): trs += [(tr, False) for tr in part.findall(q("tr"))]
        elif part.tag == q("tr"): trs.append((part, False))
    for tr, ish in trs:
        row = []
        for td in tr:
            if td.tag not in (q("td"), q("th")): continue
            cs = int(re.sub(r"\D", "", td.get("colspan") or "1") or 1); rs = int(re.sub(r"\D", "", td.get("rowspan") or "1") or 1)
            row.append(dict(text=norm("".join(td.itertext())), cs=max(1, min(cs, 50)), rs=max(1, min(rs, 200)), th=td.tag == q("th")))
        if row: rows.append(row)
        if ish: nh += 1
    if nh == 0:  # leading all-th rows count as header rows
        while nh < len(rows) and all(c["th"] for c in rows[nh]): nh += 1
    return rows, nh


def place(rows):
    """HTML placement: -> list of dict(r, c, rs, cs, text, th), R, C"""
    occ = set(); out = []
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in occ: c += 1
            for a in range(r, r + cell["rs"]):
                for b in range(c, c + cell["cs"]): occ.add((a, b))
            out.append(dict(cell, r=r, c=c)); c += cell["cs"]
    R = len(rows); C = max((b for _, b in occ), default=-1) + 1
    for x in out: x["rs"] = min(x["rs"], R - x["r"])
    return out, R, C


def usable(rows, nh):
    cells, R, C = place(rows)
    if R < 2 or C < 2 or R * C > 1200: return False
    filled = sum(1 for x in cells if x["text"])
    return filled >= 3 and filled >= 0.3 * len(cells) and nh < R


# ---------- sources ----------

def dailymed_tables(n_labels, rng):
    from lxml import etree
    ns = "urn:hl7-org:v3"; out = []
    pages = list(range(1, 60)); rng.shuffle(pages)
    setids = []
    for p in pages[:max(3, n_labels // 100 + 2)]:
        d = fetch(f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?page={p}&pagesize=100")
        if d: setids += [x["setid"] for x in json.loads(d).get("data", [])]
    rng.shuffle(setids)
    for sid in setids[:n_labels]:
        x = fetch(f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{sid}.xml")
        if not x: continue
        try:
            root = etree.fromstring(x)
        except Exception:
            continue
        title = norm("".join(root.find(f".//{{{ns}}}title").itertext())) if root.find(f".//{{{ns}}}title") is not None else ""
        for k, t in enumerate(root.iter(f"{{{ns}}}table")):
            m = table_model(t, ns)
            if m and usable(*m): out.append(dict(src="dailymed", doc="dailymed_" + sid, k=k, rows=m[0], nh=m[1], title=title[:200]))
        time.sleep(0.2)
    return out


def fedreg_tables(n_docs, rng):
    import lxml.html
    out = []; docs = []
    for p in range(1, max(3, n_docs // 50 + 3)):
        d = fetch(f"https://www.federalregister.gov/api/v1/documents.json?per_page=100&page={p}&order=newest&fields[]=document_number&fields[]=body_html_url&fields[]=title&conditions[type][]=RULE&conditions[type][]=PRORULE&conditions[type][]=NOTICE")
        if d: docs += json.loads(d).get("results", [])
    rng.shuffle(docs); used = 0
    for doc in docs:
        if used >= n_docs: break
        if not doc.get("body_html_url"): continue
        b = fetch(doc["body_html_url"])
        if not b: continue
        try:
            root = lxml.html.fromstring(b)
        except Exception:
            continue
        got = 0
        for k, t in enumerate(root.iter("table")):
            m = table_model(t)
            if m and usable(*m): out.append(dict(src="fedreg", doc="fedreg_" + doc["document_number"], k=k, rows=m[0], nh=m[1], title=norm(doc.get("title"))[:200])); got += 1
        used += 1 if got else 0
        time.sleep(0.2)
    return out


# ---------- rendering ----------

def cell_html(x, extra=""):
    a = (f' colspan="{x["cs"]}"' if x["cs"] > 1 else "") + (f' rowspan="{x["rs"]}"' if x["rs"] > 1 else "")
    tag = "th" if x["th"] else "td"
    return f'<{tag}{a} data-r="{x["r"]}" data-c="{x["c"]}"{extra}>{html.escape(x["text"])}</{tag}>'


def page_html(cells, R, C, nh, style, font, size, title, note):
    rows = collections.defaultdict(list)
    for x in sorted(cells, key=lambda x: (x["r"], x["c"])): rows[x["r"]].append(x)
    head = "".join("<tr>" + "".join(cell_html(x) for x in rows[r]) + "</tr>" for r in range(nh))
    body = "".join("<tr>" + "".join(cell_html(x) for x in rows[r]) + "</tr>" for r in range(nh, R))
    return (f"<html><head><meta charset='utf-8'><style>@page{{size:{PAGE_W}px {PAGE_H}px;margin:0}} html,body{{margin:0}}"
            f" body{{padding:{MARGIN}px;width:{PAGE_W - 2 * MARGIN}px;font-family:{font};font-size:{size}px;line-height:1.25}}"
            f" p{{margin:0 0 10px 0}} table{{max-width:100%;font-size:{size}px}} {STYLES[style]}</style></head><body>"
            f"<p><b>{html.escape(title)}</b></p><table id='t'>{('<thead>' + head + '</thead>') if nh else ''}<tbody>{body}</tbody></table>"
            f"<p>{html.escape(note)}</p></body></html>")


def fragments(cells, R, nh, heights, avail):
    """split body rows into pages: each page = header rows + a run of body rows; spans cut at a page edge continue
    as empty spanning cells on the next page. -> list of (cells, R_frag)"""
    hh = sum(heights[:nh]); runs = []; start = nh; acc = hh
    for r in range(nh, R):
        if acc + heights[r] > avail and r > start:
            runs.append((start, r)); start = r; acc = hh
        acc += heights[r]
    runs.append((start, R))
    out = []
    for a, b in runs:
        fc = []
        for x in cells:
            if x["r"] < nh:
                if x["r"] + x["rs"] <= nh: fc.append(dict(x))
                continue
            top, bot = max(x["r"], a), min(x["r"] + x["rs"], b)
            if top >= bot: continue
            y = dict(x); y["rs"] = bot - top; y["r"] = nh + (top - a)
            if top > x["r"]: y["text"] = ""
            fc.append(y)
        out.append((fc, nh + (b - a)))
    return out


def build(w, scratch, target, gt_record, grid_html, tags_of, check_html):
    from playwright.sync_api import sync_playwright
    rng = random.Random(9)
    tabs = dailymed_tables(max(20, target // 6), rng) + fedreg_tables(max(10, target // 10), rng)
    print("html9 source tables", collections.Counter(t["src"] for t in tabs), flush=True)
    rng.shuffle(tabs)
    per_doc = collections.Counter(); chosen = []
    for t in tabs:
        if per_doc[t["doc"]] >= 6: continue
        per_doc[t["doc"]] += 1; chosen.append(t)
        if len(chosen) >= target: break
    stats = collections.Counter()
    with sync_playwright() as p:
        br = p.chromium.launch(); pg = br.new_page(viewport={"width": PAGE_W, "height": PAGE_H})
        for t in chosen:
            h = zlib.crc32(t["doc"].encode())
            style = list(STYLES)[h % 3]; font = FONTS[(h // 3) % 2]; size = [11, 12, 13][(h // 6) % 3]
            cells, R, C = place(t["rows"])
            note = "Source: " + ("DailyMed (U.S. National Library of Medicine)" if t["src"] == "dailymed" else "Federal Register")
            try:
                pg.set_content(page_html(cells, R, C, t["nh"], style, font, size, t["title"], note))
                heights = pg.evaluate("() => Array.from(document.querySelectorAll('#t tr')).map(r => r.getBoundingClientRect().height)")
                top = pg.evaluate("() => document.getElementById('t').getBoundingClientRect().top")
                avail = PAGE_H - MARGIN - top - 30
                frs = fragments(cells, R, t["nh"], heights, avail) if sum(heights) > avail else [(cells, R)]
                if len(frs) > 6: frs = frs[:6]; stats["fragments_capped"] += 1
                for fi, (fc, fr) in enumerate(frs):
                    pg.set_content(page_html(fc, fr, C, t["nh"], style, font, size, t["title"] + (f" (continued)" if fi else ""), note))
                    fs = size
                    while pg.evaluate("() => document.getElementById('t').getBoundingClientRect().right") > PAGE_W - MARGIN / 2 and fs > 7:
                        fs -= 1; pg.evaluate(f"() => {{document.getElementById('t').style.fontSize = '{fs}px'}}")
                    tb = pg.evaluate("() => {const r = document.getElementById('t').getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]}")
                    boxes = pg.evaluate("() => Array.from(document.querySelectorAll('#t td, #t th')).map(e => {const r = e.getBoundingClientRect(); return [+e.dataset.r, +e.dataset.c, r.left, r.top, r.right, r.bottom]})")
                    if tb[2] > PAGE_W - 4: stats["too_wide"] += 1; continue
                    if tb[3] > PAGE_H - 10: stats["overflow"] += 1; continue
                    z = 1.0
                    bx = {(int(b[0]), int(b[1])): [v * 0.75 * z for v in b[2:]] for b in boxes}
                    tid = f"html9_{re.sub(r'[^A-Za-z0-9]', '', t['doc'])[:40]}_t{t['k']}_f{fi}"
                    fn = tid + ".pdf"
                    pg.pdf(path=os.path.join(w.dir, fn), width=f"{PAGE_W}px", height=f"{PAGE_H}px", print_background=True, prefer_css_page_size=True)
                    gc = [dict(r=x["r"], c=x["c"], rs=x["rs"], cs=x["cs"], text=x["text"], bbox=bx.get((x["r"], x["c"]), [0, 0, 0, 0])) for x in fc]
                    toks, out = grid_html(gc, t["nh"])
                    check_html(toks)
                    tbp = [tb[0] * 0.75 * z - 4, tb[1] * 0.75 * z - 4, tb[2] * 0.75 * z + 4, tb[3] * 0.75 * z + 4]
                    T = tags_of(toks, out, None, dict(split_page=len(frs) > 1))
                    w.add(gt_record(w.name, tid, t["doc"], fn, tbp, toks, out, T, "pdf",
                                    r9type="pharma/medical" if t["src"] == "dailymed" else "legal/regulatory",
                                    lang="en", src_meta=dict(src=t["src"], style=style, fragment=fi, n_fragments=len(frs), font_px=fs)))
                    stats["pages"] += 1
            except Exception as e:
                stats["error"] += 1; print("skip", t["doc"], t["k"], repr(e)[:160], flush=True)
        br.close()
    w.stats.update(stats)
    print("html9", dict(stats), flush=True)
