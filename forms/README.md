# forms — AcroForm field-value readers in Rust

Test programs that read filled PDF form fields (name, value, checkbox/radio state, page, widget rect)
with several Rust crates and compare them against a pypdf-based ground truth.

- `crates/*` — one binary per crate under test (lopdf, pdf_oxide, pdfer_forms, flpdf, pdfium-render).
  Each prints a JSON report for the PDFs given on the command line.
- `pdfs/` — test forms: a docling.rs fixture, synthetic forms built by `tools/gen_synth.py`
  (reportlab and low-level pypdf, no private data), and IRS W-9 / W-4 (public domain) filled with fake data.
- `ground_truth/` — expected fields per PDF, produced by `tools/ground_truth.py` (pypdf).
- `tools/compare.py` — scores the CI JSON reports against the ground truth.
- Workflow: `.github/workflows/forms.yml` (Linux + Windows; pdfium from bblanchon/pdfium-binaries).
