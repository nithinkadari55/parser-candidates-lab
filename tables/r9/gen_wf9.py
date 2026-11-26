"""Writes .github/workflows/tables-r9-base.yml and tables-r9-cand.yml (many repeated cache steps).
usage: python tables/r9/gen_wf9.py"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NSH = 30
R8H = "hashFiles('tables/r8/prep8.py', 'tables/src/prep_sets.py', 'tables/src/prep_pt2.py', 'tables/src/prep_tabrec.py', 'tables/src/r3_rotate.py')"
R4H = "hashFiles('tables/r4/r4_sel_prep.py', 'tables/r4/r4_common.py')"
# the round-9 set caches carry a fixed tag (the hashFiles value of the first base run, 45739658679), so later edits of
# prep9.py / common9.py do not orphan them; bump the tag by hand when a set's content must change
R9TAG = "f2f282aab5460467a300432f0182067f10fc74991a34634fd719a91dcc0d22df"
R9H = f"'{R9TAG}'"
R8SETS = {"ptn": ["ptn"], "pt2": ["pt2"], "ftnc": ["ftnc"], "synfin": ["synfin"], "tabrec": ["tabrec"], "wtw": ["wtw"], "ctdar": ["ctdarm", "ctdarh"], "rotpdf": ["rotpdf"]}
R4SETS = {"pt": "r4pt", "brgm": "r4brgm", "icdar": "r4icdar", "scitsr": "r4scitsr"}
R9SETS = ["pulse", "rdtb", "odb", "tabreczh", "mustard", "iflytab", "cisol"]
OPTIONAL = ["html9"]  # restored when present (prepared after the first base run)


def paths(lst, ind):
    if len(lst) == 1: return f"tables/work/data/sets/{lst[0]}"
    return "|\n" + "\n".join(f"{ind}  tables/work/data/sets/{x}" for x in lst)


def key_r8(n): return f"r8set-{n}-v1-L${{{{ inputs.limit }}}}-${{{{ {R8H} }}}}"
def key_r4(n): return f"r4sel-{n}-v1--${{{{ {R4H} }}}}"
def key_r9(n): return f"r9set-{n}-v1-L${{{{ inputs.limit }}}}-{R9TAG}"


def restore(path, key, ind="      ", fail=True):
    s = f"{ind}- uses: actions/cache/restore@v4\n{ind}  with:\n{ind}    path: {path}\n{ind}    key: {key}\n"
    if fail: s += f"{ind}    fail-on-cache-miss: true\n"
    return s


def restore_all(ind="      "):
    s = ""
    for n, p in R4SETS.items(): s += restore(f"tables/work/data/sets/{p}", key_r4(n), ind)
    for n, lst in R8SETS.items(): s += restore(paths(lst, ind + "    "), key_r8(n), ind)
    for n in R9SETS: s += restore(f"tables/work/data/sets/{n}", key_r9(n), ind)
    for n in OPTIONAL: s += restore(f"tables/work/data/sets/{n}", key_r9(n), ind, fail=False)
    return s


HEAD = """env:
  TB_WORK: ${{ github.workspace }}/tables/work
  TB_LIMIT: ${{ inputs.limit }}
  PYTHONUNBUFFERED: "1"
  HF_HUB_DISABLE_PROGRESS_BARS: "1"
  HF_HUB_DISABLE_XET: "1"
  DLRS_CACHE: /home/runner/dlrs
  R9_W: /home/runner/r9w
  R9_CODE: /home/runner/r9code
  R9_THREADS: "4"
"""

INSTALL = """      - name: Install
        run: |
          pip install -q -r tables/requirements.txt pypdf==6.1.3 gdown==5.2.0
          pip uninstall -y -q opencv-python opencv-python-headless || true
          pip install -q opencv-python-headless==5.1.0.95
"""

RUST = """      - name: Rust
        run: |
          rustup toolchain install stable --profile minimal
          rustup default stable
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: tables/r5/words
      - name: Build the docling.rs word tool
        working-directory: tables/r5/words
        run: cargo build --release
"""

DLRS = """      - id: dlrs
        uses: actions/cache@v4
        with:
          path: /home/runner/dlrs
          key: dlrs-2.84.0-models-v1
      - if: steps.dlrs.outputs.cache-hit != 'true'
        run: |
          pip install docling-rs==2.84.0
          python -c "from docling_rs import models as m; m.download_models(dest='/home/runner/dlrs', progress=False, enrichment=False)"
"""


def base_yml():
    shards = ", ".join(str(i) for i in range(NSH))
    prep_inc = "".join(f"          - name: {n}\n            kind: r8\n            path: {paths(l, '              ')}\n" for n, l in R8SETS.items())
    prep_inc += "".join(f"          - name: {n}\n            kind: r9\n            path: tables/work/data/sets/{n}\n" for n in R9SETS)
    pkey = f"${{{{ format('{{0}}set-{{1}}-v1-L{{2}}-{{3}}', matrix.kind, matrix.name, inputs.limit, matrix.kind == 'r8' && {R8H} || {R9H}) }}}}"
    r4 = "".join(restore(f"tables/work/data/sets/{p}", key_r4(n), "      ").replace("      - uses", "      - if: steps.have.outputs.cache-hit != 'true' && matrix.name == 'rotpdf'\n        uses", 1)
                 for n, p in R4SETS.items())
    return f"""name: tables-r9-base

# Round 9 baselines on the round-8 and round-9 sets (tables/r9): set preparation (Actions caches only), TableFormer
# through docling-rs (tf9.py), then words, orientation, TATR and selector features (base9.py). Public artifacts hold
# ids and numbers only; words and cells stay in Actions caches (r9priv-<run>-<shard>) for the candidate runs.

on:
  workflow_dispatch:
    inputs:
      limit:
        description: "probe: only the first N tables per set (empty = all)"
        default: ""
      sets:
        description: "comma-separated sets for TableFormer and the baselines (empty = all)"
        default: ""
      nshards:
        description: "number of shards"
        default: "{NSH}"
      shards:
        description: "JSON list of shard numbers (0 .. nshards-1)"
        default: "[{shards}]"

{HEAD}
jobs:
  prep:
    runs-on: ${{{{ matrix.name == 'html9' && 'ubuntu-24.04' || 'ubuntu-latest' }}}}
    continue-on-error: ${{{{ matrix.name == 'html9' }}}}
    timeout-minutes: 150
    strategy:
      fail-fast: false
      matrix:
        include:
{prep_inc}    defaults:
      run:
        shell: bash
    steps:
      - uses: actions/checkout@v4
      - id: have
        uses: actions/cache/restore@v4
        with:
          path: ${{{{ matrix.path }}}}
          key: {pkey}
          lookup-only: true
      - if: steps.have.outputs.cache-hit != 'true'
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
{r4}      - if: steps.have.outputs.cache-hit != 'true'
        name: Install
        run: |
          pip install -q -r tables/requirements.txt pypdf==6.1.3
          pip uninstall -y -q opencv-python opencv-python-headless || true
          pip install -q opencv-python-headless==5.1.0.95
          if [ "${{{{ matrix.name }}}}" = html9 ]; then pip install -q playwright==1.56.0 && python -m playwright install --with-deps chromium && sudo apt-get install -y -q fonts-liberation fonts-dejavu-core > /dev/null; fi
      - if: steps.have.outputs.cache-hit != 'true'
        name: Prepare
        env:
          R8_SCRATCH: ${{{{ runner.temp }}}}/r9scratch
        run: |
          if [ "${{{{ matrix.kind }}}}" = r8 ]; then python tables/r8/prep8.py ${{{{ matrix.name }}}}; else python tables/r9/prep9.py ${{{{ matrix.name }}}}; fi
          du -sh tables/work/data/sets/* | tail -n 20
      - if: steps.have.outputs.cache-hit != 'true'
        uses: actions/cache/save@v4
        with:
          path: ${{{{ matrix.path }}}}
          key: {pkey}

  tf:
    needs: prep
    runs-on: ubuntu-latest
    timeout-minutes: 340
    strategy:
      fail-fast: false
      max-parallel: 20
      matrix:
        shard: ${{{{ fromJSON(inputs.shards) }}}}
    defaults:
      run:
        shell: bash
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
{restore_all()}{INSTALL}      - run: pip install -q docling-rs==2.84.0
{DLRS}      - name: TableFormer (docling-rs 2.84.0, fp32 layout)
        run: nproc; grep -m1 "model name" /proc/cpuinfo; python tables/r9/tf9.py ${{{{ matrix.shard }}}} ${{{{ inputs.nshards }}}} "${{{{ inputs.sets }}}}"
      - name: docling stderr (tail)
        if: always()
        run: tail -n 30 tables/work/out/r9/tf/stderr_s${{{{ matrix.shard }}}}.log || true
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: r9tf-${{{{ matrix.shard }}}}
          path: tables/work/out/r9/tf/
          retention-days: 30

  base:
    needs: tf
    runs-on: ubuntu-latest
    timeout-minutes: 340
    strategy:
      fail-fast: false
      max-parallel: 20
      matrix:
        shard: ${{{{ fromJSON(inputs.shards) }}}}
    defaults:
      run:
        shell: bash
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
{RUST}{restore_all()}{INSTALL}{DLRS}      - name: Models
        run: |
          (cd tables/src && python fetch_models.py tatr16)
          python tables/r8/models8.py
      - uses: actions/download-artifact@v4
        with:
          pattern: r9tf-*
          path: tfout
      - name: Words, orientation, TATR, features, scoring
        run: nproc; grep -m1 "model name" /proc/cpuinfo; python tables/r9/base9.py ${{{{ matrix.shard }}}} ${{{{ inputs.nshards }}}} tfout tables/r5/words/target/release/r5words
      - uses: actions/cache/save@v4
        if: always()
        with:
          path: tables/work/out/r9/bundle/s${{{{ matrix.shard }}}}/rows.jsonl.gz
          key: r9priv-${{{{ github.run_id }}}}-${{{{ matrix.shard }}}}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: r9base-${{{{ matrix.shard }}}}
          path: tables/work/out/r9/pub/base/
          retention-days: 30
"""


def cand_yml():
    shards = ", ".join(str(i) for i in range(NSH))
    return f"""name: tables-r9-cand

# Round 9 candidates (tables/r9/cand9.py): each shard's crops are re-rendered from the set PDFs (mkbundle9.py, checked
# against base9's image hash); every model of the plan runs on them with the bundle's words and turn, and is scored.
# Public artifacts: ids, numbers, cell geometry without text, timings and memory.

on:
  workflow_dispatch:
    inputs:
      base_run:
        description: "tables-r9-base run id (its r9priv caches)"
        required: true
      plan:
        description: "group=model model;group=model ... (environments of tables/r9/env9.sh)"
        required: true
      parts:
        description: "JSON list of part numbers"
        default: "[0]"
      nparts:
        description: "parts per shard"
        default: "1"
      core:
        description: "core share for slow models (empty = every table)"
        default: ""
      shards:
        description: "JSON list of shards"
        default: "[{shards}]"
      limit:
        description: "the base run's limit input"
        default: ""
      os:
        description: "runner image"
        default: "ubuntu-latest"

{HEAD}  PLAN: ${{{{ inputs.plan }}}}

jobs:
  cand:
    runs-on: ${{{{ inputs.os }}}}
    timeout-minutes: 345
    strategy:
      fail-fast: false
      max-parallel: 20
      matrix:
        shard: ${{{{ fromJSON(inputs.shards) }}}}
        part: ${{{{ fromJSON(inputs.parts) }}}}
    defaults:
      run:
        shell: bash
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: |
            3.10
            3.11
{RUST}{restore_all()}      - uses: actions/cache/restore@v4
        with:
          path: tables/work/out/r9/bundle/s${{{{ matrix.shard }}}}/rows.jsonl.gz
          key: r9priv-${{{{ inputs.base_run }}}}-${{{{ matrix.shard }}}}
          fail-on-cache-miss: true
{INSTALL}          (cd tables/src && python fetch_models.py tatr16)
      - name: Bundle
        run: python tables/r9/mkbundle9.py ${{{{ matrix.shard }}}} tables/work/out/r9/bundle/s${{{{ matrix.shard }}}} tables/r5/words/target/release/r5words
      - uses: actions/cache@v4
        if: contains(inputs.plan, 'lore')
        with:
          path: |
            /home/runner/r9w/lore_wireless
            /home/runner/r9w/lore_ptn
            /home/runner/r9w/lore_wtw
          key: r9w-lore-v4
      - uses: actions/cache@v4
        if: contains(inputs.plan, 'trace')
        with:
          path: |
            /home/runner/r9w/trace_base
            /home/runner/r9w/trace_wtw
          key: r9w-trace-v4
      - name: Weights and model environments
        run: |
          IFS=';' read -ra GS <<< "$PLAN"
          for it in "${{GS[@]}}"; do g="${{it%%=*}}"; ms="${{it#*=}}"; python tables/r9/weights9.py $ms; bash tables/r9/env9.sh $g; done
      - name: Speed reference (TATR on 30 crops, this machine)
        run: python tables/r9/cand9.py ref tables/work/out/r9/bundle/s${{{{ matrix.shard }}}} /tmp/out/ref ${{{{ matrix.shard }}}}
      - name: Infer and score
        run: |
          B=tables/work/out/r9/bundle/s${{{{ matrix.shard }}}}; S=${{{{ matrix.shard }}}}; P=${{{{ matrix.part }}}}
          IFS=';' read -ra GS <<< "$PLAN"
          for it in "${{GS[@]}}"; do
            g="${{it%%=*}}"; ms="${{it#*=}}"; pyv="R9_PY_$g"; py="${{!pyv}}"
            for m in $ms; do
              $py tables/r9/cand9.py infer $m $S $P ${{{{ inputs.nparts }}}} $B /tmp/out "${{{{ inputs.core }}}}" || {{ echo "INFER FAILED $m"; FAILED=1; }}
              if [ -f /tmp/out/$m/cells_s${{S}}p${{P}}.jsonl.gz ]; then python tables/r9/cand9.py score $m $S $P $B /tmp/out; python tables/r9/cand9.py strip /tmp/out/$m $S $P; fi
            done
          done
          exit ${{FAILED:-0}}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: r9c-s${{{{ matrix.shard }}}}p${{{{ matrix.part }}}}
          path: |
            /tmp/out/*/s*.jsonl
            /tmp/out/*/run_s*.json
            /tmp/out/*/geom_s*.jsonl.gz
            /tmp/out/ref/
          retention-days: 30
"""


if __name__ == "__main__":
    d = os.path.join(ROOT, ".github", "workflows")
    open(os.path.join(d, "tables-r9-base.yml"), "w", newline="\n").write(base_yml())
    open(os.path.join(d, "tables-r9-cand.yml"), "w", newline="\n").write(cand_yml())
    print("written")
