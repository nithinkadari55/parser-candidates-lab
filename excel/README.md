# excel

Probe for Rust crates that read Excel cells as typed values with number formats:
calamine 0.38.0, umya-spreadsheet 3.1.1, rxls 0.1.4.

- `src/main.rs`: `excel-probe dump <outdir> <files...>` writes one JSONL file per crate
  (A1 address, typed value, number format code, merged range, formula text with cached value);
  `excel-probe bench <file> [reps]` times a full read of values plus number formats.
- `gen_files.py`: writes synthetic test files (edge cases, a 1904-date-system copy, and a
  100k-row x 20-column file) using only the Python standard library.
- `testdata/`: public sample workbooks.

CI: `.github/workflows/excel.yml` (Ubuntu and Windows), results uploaded as artifacts.
