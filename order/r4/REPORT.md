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

Not finished yet.
- First run 44881278893: the feature jobs failed (missing `ml_dtypes`).
- Second run 44884726600: data prep timed out; the DocFigure server delivered about 200 KB/s.
- Running now: 44897815693, with DocFigure capped at 25 minutes and skipped if not done, and SignverOD redactions added as "other".
- This section will be filled from that run.
