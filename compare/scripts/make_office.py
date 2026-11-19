"""Write small synthetic Office files with one unique marker per feature we need."""
import os, sys


def docx(path):
    import docx as D
    d = D.Document()
    d.sections[0].header.paragraphs[0].text = "HEADERMARK running head"
    d.sections[0].footer.paragraphs[0].text = "FOOTERMARK running foot"
    d.add_heading("HEADINGMARK chapter one", level=1)
    p = d.add_paragraph("BODYMARK the body paragraph with a commented word ")
    r = p.add_run("ANCHORMARK")
    d.add_comment(r, text="COMMENTMARK reviewer note", author="Tester", initials="T")
    d.save(path)


def pptx(path):
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches
    pr = Presentation()
    s1 = pr.slides.add_slide(pr.slide_layouts[1])
    s1.shapes.title.text = "SLIDE1MARK visible title"
    s1.placeholders[1].text = "VISIBLEBODYMARK bullet"
    s1.notes_slide.notes_text_frame.text = "NOTESMARK speaker notes text"
    s2 = pr.slides.add_slide(pr.slide_layouts[1])
    s2.shapes.title.text = "HIDDENSLIDEMARK hidden title"
    s2._element.set("show", "0")
    s3 = pr.slides.add_slide(pr.slide_layouts[5])
    s3.shapes.title.text = "SLIDE3MARK chart slide"
    cd = CategoryChartData()
    cd.categories = ["CatAlpha", "CatBeta"]
    cd.add_series("SeriesMark", (11.5, 22.75))
    gf = s3.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1.5), Inches(6), Inches(4), cd)
    gf.chart.has_title = True
    gf.chart.chart_title.text_frame.text = "CHARTTITLEMARK"
    pr.save(path)


def xlsx(path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "VisibleSheetMark"
    ws["A1"] = "MERGEDMARK header"
    ws.merge_cells("A1:C1")
    ws.append(["colA", "colB", "colC"])
    ws.append([1, 2, 3])
    ws.append([4, 5, 6])
    h = wb.create_sheet("HiddenSheetMark")
    h["A1"] = "HIDDENCELLMARK"
    h["A2"] = 7
    h.sheet_state = "hidden"
    wb.save(path)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    docx(os.path.join(out, "synthetic.docx"))
    pptx(os.path.join(out, "synthetic.pptx"))
    xlsx(os.path.join(out, "synthetic.xlsx"))
    print(sorted(os.listdir(out)))
