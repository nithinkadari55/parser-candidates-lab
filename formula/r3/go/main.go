package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/go-pdfkit/latex"
	"github.com/go-pdfkit/reader"
)

type region struct {
	ID        string     `json:"id"`
	PDF       string     `json:"pdf"`
	PageIndex *int       `json:"page_index"`
	BBox      []float64  `json:"bbox"`
	MediaBox  []float64  `json:"mediabox"`
}

type output struct {
	ID         string   `json:"id"`
	Tool       string   `json:"tool"`
	LaTeX      *string  `json:"latex"`
	Confidence *float64 `json:"confidence"`
	Status     string   `json:"status"`
	Error      *string  `json:"error"`
	MS         float64  `json:"ms"`
}

type pageResult struct {
	page *latex.HarnessPage
	err  error
}

type doc struct {
	path  string
	d     *reader.Document
	err   error
	pages map[int]pageResult
}

func openDoc(path string) *doc {
	start := time.Now()
	out := &doc{path: path, pages: map[int]pageResult{}}
	data, err := os.ReadFile(path)
	if err != nil {
		out.err = fmt.Errorf("open failed: %w", err)
	} else if out.d, err = reader.Open(data); err != nil {
		out.err = fmt.Errorf("open failed: %w", err)
	}
	fmt.Fprintf(os.Stderr, "gopdfkit: opened %s in %.1f ms\n", path, msSince(start))
	return out
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Nanoseconds()) / 1e6 }

func strp(s string) *string { return &s }

func validate(r region) error {
	if r.PageIndex == nil {
		return fmt.Errorf("missing page_index")
	}
	if *r.PageIndex < 0 {
		return fmt.Errorf("negative page_index %d", *r.PageIndex)
	}
	if len(r.BBox) != 4 || len(r.MediaBox) != 4 {
		return fmt.Errorf("bbox and mediabox must have 4 numbers")
	}
	for _, v := range append(append([]float64{}, r.BBox...), r.MediaBox...) {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("non-finite bbox or mediabox")
		}
	}
	return nil
}

func handle(cur **doc, r region, pad float64) (o output, ms *float64) {
	o.Status = "error"
	defer func() {
		if p := recover(); p != nil {
			o = output{Status: "error", Error: strp(fmt.Sprintf("panic: %v", p))}
			ms = nil
		}
	}()
	if err := validate(r); err != nil {
		o.Error = strp(err.Error())
		return o, nil
	}
	if *cur == nil || (*cur).path != r.PDF {
		*cur = nil
		*cur = openDoc(r.PDF)
	}
	dc := *cur
	if dc.err != nil {
		o.Error = strp(dc.err.Error())
		return o, nil
	}
	idx := *r.PageIndex
	if n := dc.d.PageCount(); idx >= n {
		o.Error = strp(fmt.Sprintf("page_index %d out of range (%d pages)", idx, n))
		return o, nil
	}
	pr, ok := dc.pages[idx]
	if !ok {
		start := time.Now()
		p, err := latex.HarnessReadPage(dc.d, idx+1)
		pr = pageResult{page: p, err: err}
		dc.pages[idx] = pr
		fmt.Fprintf(os.Stderr, "gopdfkit: read page %d in %.1f ms\n", idx, msSince(start))
	}
	if pr.err != nil {
		o.Error = strp(fmt.Sprintf("page read failed: %v", pr.err))
		return o, nil
	}
	box := pr.page.VisibleBox()
	mw, mh := math.Abs(r.MediaBox[2]-r.MediaBox[0]), math.Abs(r.MediaBox[3]-r.MediaBox[1])
	if math.Abs((box[2]-box[0])-mw) > 0.5 || math.Abs((box[3]-box[1])-mh) > 0.5 {
		fmt.Fprintf(os.Stderr, "gopdfkit: warning: %s page %d: visible box %v differs from mediabox %v\n", r.PDF, idx, box, r.MediaBox)
	}
	start := time.Now()
	x0, x1 := math.Min(r.BBox[0], r.BBox[2])-pad, math.Max(r.BBox[0], r.BBox[2])+pad
	y0, y1 := math.Min(r.BBox[1], r.BBox[3])-pad, math.Max(r.BBox[1], r.BBox[3])+pad
	tex, atoms, _ := pr.page.Region(r.ID, x0, y0, x1, y1)
	elapsed := msSince(start)
	o.Error = nil
	if atoms == 0 || tex == "" {
		o.Status = "no_math"
		return o, &elapsed
	}
	o.Status = "ok"
	o.LaTeX = strp(tex)
	return o, &elapsed
}

func main() {
	input := flag.String("input", "", "regions JSONL")
	outPath := flag.String("output", "", "output JSONL")
	pad := flag.Float64("pad", 0, "points added on every side of each bbox")
	flag.Parse()
	if *input == "" || *outPath == "" {
		fmt.Fprintln(os.Stderr, "usage: gopdfkit-region --input regions.jsonl --output out.jsonl [--pad PT]")
		os.Exit(2)
	}
	in, err := os.Open(*input)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer in.Close()
	outFile, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer outFile.Close()
	w := bufio.NewWriter(outFile)
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 1<<20), 1<<26)
	var cur *doc
	total, okCount, lineNo := 0, 0, 0
	for scanner.Scan() {
		lineNo++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		total++
		start := time.Now()
		var r region
		var o output
		var ms *float64
		if err := json.Unmarshal(line, &r); err != nil {
			var loose map[string]any
			id := fmt.Sprintf("line:%d", lineNo)
			if json.Unmarshal(line, &loose) == nil {
				if s, ok := loose["id"].(string); ok {
					id = s
				}
			}
			r.ID = id
			o = output{Status: "error", Error: strp("bad input line: " + err.Error())}
		} else {
			o, ms = handle(&cur, r, *pad)
		}
		o.ID = r.ID
		o.Tool = "gopdfkit"
		if ms != nil {
			o.MS = *ms
		} else {
			o.MS = msSince(start)
		}
		o.MS = math.Round(o.MS*1000) / 1000
		if o.Status == "ok" {
			okCount++
		}
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(o); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if _, err := w.Write(buf.Bytes()); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		w.Flush()
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := w.Flush(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "gopdfkit: %d regions, %d ok\n", total, okCount)
}
