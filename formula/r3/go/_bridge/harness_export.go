// Copyright (c) the go-pdfkit/latex authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Added by the parser-candidates-lab benchmark harness at build time to a copy of
// github.com/go-pdfkit/latex v0.4.1; it only exposes the package's own page reader
// and equation writer (readPage, visibleBox, mathSource) for one region.

package latex

import (
	"fmt"
	"os"
	"sort"

	"github.com/go-pdfkit/reader"
)

type HarnessPage struct {
	p   page
	box [4]float64
}

func HarnessReadPage(d *reader.Document, pageNumber int) (*HarnessPage, error) {
	p, err := readPage(d, pageNumber)
	if err != nil {
		return nil, err
	}
	dict, _ := d.Page(pageNumber)
	return &HarnessPage{p: p, box: visibleBox(d, dict)}, nil
}

func (h *HarnessPage) VisibleBox() [4]float64 { return h.box }

func (h *HarnessPage) Region(id string, x0, y0, x1, y1 float64) (string, int, int) {
	x0, x1 = x0-h.box[0], x1-h.box[0]
	y0, y1 = y0-h.box[1], y1-h.box[1]
	inside := func(x, y float64) bool { return x >= x0 && x <= x1 && y >= y0 && y <= y1 }
	var items []atom
	for _, a := range h.p.atoms {
		cy := a.y + 0.25*a.size
		if a.sh.math == mathExt || radicalOf(a) {
			cy = a.y - 0.6*a.size
		}
		in := inside(a.midX(), cy)
		if in {
			items = append(items, a)
		}
		if os.Getenv("R3_DEBUG") != "" && a.midX() >= x0-6 && a.midX() <= x1+6 && cy >= y0-6 && cy <= y1+6 {
			fmt.Fprintf(os.Stderr, "R3_DEBUG gopdfkit %s in=%v %q x=%.2f y=%.2f w=%.2f size=%.2f role=%d\n", id, in, a.text, a.x, a.y, a.width, a.size, a.sh.math)
		}
	}
	if len(items) == 0 {
		return "", 0, 0
	}
	sort.SliceStable(items, func(i, j int) bool { return items[i].x < items[j].x })
	var rs []rule
	for _, r := range h.p.rules {
		if inside(r.midX(), r.midY()) {
			rs = append(rs, r)
		}
	}
	return mathSource(items, mathRules(items, rs)), len(items), len(rs)
}
