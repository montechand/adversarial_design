# Pharma email adversarial-design pipeline

LangGraph-based pipeline that ingests an existing pharma HCP email (the
output of the **backend-server `message_generator_from_scratch` graph**) and
tries to beat it by generating k structurally diverse alternatives, then
judging all candidates (including the baseline) on creativity, HCP
relevance, layout, and "AI-likeness". The output is the **best prompt** plus
the **best HTML design**.

> This is the standalone runner. Eventually the backend-server pipeline will
> invoke `pipeline.graph.run_from_baseline` directly with the same three
> inputs. For now you can run it with the bundled example.

## Input contract

```
┌────────────────────────────────────────────────────────────────────┐
│  blueprint sections   (mirrors backend `content_blueprint`)        │
│  baseline HTML        (with [images_base64_N] placeholders)        │
│  baseline images map  ({placeholder_token: data URI})              │
│  baseline screenshot  slices (optional — auto-generated otherwise) │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  Stage 1  reverse_prompter   — extracts design intent + weaknesses │
│  Stage 2  creativizer        — k diverse design specs   [async k×] │
│  Stage 3  code_generator     — per-section MJML → compile → HTML   │
│                                [async candidate × section fan-out] │
│  Stage 4  screenshotter      — render → full-page PNG → slices     │
│  Stage 5a compliance_check   — MLR gate (text-only)     [async k×] │
│  Stage 5b quality_judge      — VLM scores incl. beats_baseline     │
│  Stage 5c anti_ai_judge      — VLM ranks vs. human references      │
│  Stage 6  aggregator         — weighted score → winner             │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                      winner.html + winner.prompt.txt
```

## Candidate generation — two MJML modes

Each non-baseline candidate is produced via one of two modes. Both end
in the same compile → mjml→html → validate tail (which mirrors the
production `compiler_mjml_node → mjml_to_html_node` flow).

Switch via `pipeline.candidate_mode` in `config/pipeline_config.yaml`.

### `full_mjml` (default — visual cohesion across sections)

```
for each candidate (parallel via asyncio.gather + Semaphore):
  ┌─── whole-email MJML designer (ONE LLM call per candidate) ─────┐
  │  Sees ALL blueprint sections at once.                          │
  │  → JSON { design_reasoning, sections: [ {sid, fragment, ...} ] }│
  │  Prompt carries: VERBATIM rule + self-check ritual +           │
  │                   Brand Rules (SUPREME) +                       │
  │                   data-section-id contract +                    │
  │                   CROSS-SECTION COHESION rules                  │
  │   (one color hierarchy, one typographic system, one spacing     │
  │   cadence, narrative arc, component reuse)                      │
  └────────────────────────────────────────────────────────────────┘
  ↓
  compile_mjml_document   ← pure Python; sorts by `order`,
                            wraps with <!-- Section:Start/End sid -->,
                            builds <mj-head> with brand body font
  ↓
  mjml_to_html (mjml-python)   ← with LLM-driven fix-retry loop on
                                  syntax failures
  ↓
  validate_all_sections   ← warns on missing data-section-id, <sup>
                            without data-claim-id, sentences from
                            copy_outline dropped from the fragment
                            (log only — production never retries on
                             prose drift)
```

`full_mjml` is the default because it produces visibly more cohesive
designs — color choices, typographic scale, spacing rhythm, and component
patterns carry across sections instead of being reinvented per section.
Per-candidate latency is higher (one big LLM call vs N small ones) but
the **k candidates still run in parallel**, so total wall-clock for the
pipeline is dominated by the slowest candidate, not by section count.

### `section_wise` (low latency, kept for comparison / fallback)

```
for each candidate (parallel):
  ┌─── per-section MJML designer (parallel fan-out per section) ───┐
  │  N independent LLM calls, gathered with asyncio.gather + Sem.  │
  │  Each call sees ONLY its own section.                          │
  └────────────────────────────────────────────────────────────────┘
  ↓ (same compile + render + validate tail as full_mjml)
```

Lower per-candidate latency but each section is designed in isolation,
so the assembled email often reads as N stitched-together panels. The
code is kept in `code_generator.py` (`_generate_one_candidate_section_wise`)
so we can A/B the modes easily.

In BOTH modes the mechanical checks mirror what the production summary
table lists: structural assembly is mechanical (iteration + ordering +
`data-section-id` contract + `<sup data-claim-id>` regex scan + MJML
syntax retry), prose-level fidelity is enforced via the prompt's
verbatim rule + the mandatory self-check ritual.

## Async parallelism

All four "loop over things" stages use `asyncio.gather` with bounded
`asyncio.Semaphore`s instead of sequential loops:

| Stage                          | What runs in parallel                 | Cap (config key)                    |
|--------------------------------|---------------------------------------|-------------------------------------|
| Stage 1 reverse_prompter       | baseline + every reference VLM call   | `concurrency.reverse_prompter` (4)  |
| Stage 2 creativizer            | k creativizer LLM calls               | `concurrency.creativizer` (4)       |
| Stage 3 code_generator (cand.) | k candidate flows (both modes)        | `concurrency.per_candidate` (3)     |
| Stage 3 code_generator (sect.) | N per-section calls (section_wise only) | `concurrency.per_section` (6)     |
| Stage 4 screenshotter          | Playwright renders (via to_thread)    | `concurrency.screenshotter` (2)     |
| Stage 5a compliance_check      | per-candidate LLM checks              | `concurrency.compliance` (4)        |

The graph is driven with `graph.ainvoke()`; the CLI wraps it in
`asyncio.run`. Tune caps in `config/pipeline_config.yaml` under the
`concurrency:` block — defaults are tuned for a laptop respecting the
Anthropic per-minute rate limits.

## Why the `[images_base64_N]` placeholders?

Backend-rendered emails embed every image as a `data:image/...;base64,XXX`
URI. A single email easily exceeds 50–200 KB of base64 per image, which
wastes context window on every LLM call. The pipeline:

- replaces each `data:image/...;base64,...` URI with a short
  `[images_base64_N]` token for all **text** LLM calls (reverse prompter
  narrative arm, creativizer, code generator context, compliance check),
- keeps the original data URIs in a side-car map and **rehydrates** the HTML
  for screenshotting and VLM judges.

Compression / rehydration is handled by `pipeline/utils/html_compression.py`
and is round-trip lossless (see `tests/test_html_compression.py`).

## Quick start

```bash
cd adversarial_design
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium

cp .env.example .env  # populate ANTHROPIC_API_KEY

# Run on the bundled synthetic Opdivo-style example (no backend needed)
python scripts/run_pipeline.py --from-example examples --iterations 1
```

Explicit invocation:

```bash
python scripts/run_pipeline.py \
  --blueprint-json    path/to/blueprint.json \
  --baseline-html     path/to/baseline_email.html \
  --baseline-images   path/to/baseline_images_b64.json \
  --baseline-screenshots path/to/slices_dir/ \
  --iterations 1
```

Outputs land in `storage/runs/<run_id>/`:

| File                   | What it is |
|------------------------|------------|
| `winner.html`          | best candidate design (rehydrated PNGs in screenshots/) |
| `winner.prompt.txt`    | the prompt that produced the winner — promoted to `storage/base_prompt.txt` if it beat the baseline |
| `winner.meta.json`     | candidate id, scores, axes, compliance reason |
| `scores.json`          | full leaderboard with weights, per-candidate breakdown |
| `candidates/*.html`    | every candidate's HTML (including the baseline) |
| `candidates/*_slice_*.png` | every candidate's slice PNGs (top→bottom) |
| `candidate_history/iteration_*/candidate_*.html` | per-iteration snapshot of every candidate HTML |
| `candidate_history/iteration_*/candidate*.png` | per-iteration snapshot of each candidate full PNG + slices |
| `descriptors/iteration_*.json` | per-iteration reverse-prompter output (one entry per baseline/reference) including `raw_description` and the exact `creativizer_prompt_summary` |
| `descriptors/iteration_*.creativizer_view.txt` | human-readable view of just what the creativizer sees per descriptor |
| `creativizer/iteration_*/candidate_*.prompt.txt` | per-iteration creativizer design spec — the exact string fed into the code generator for each candidate |
| `creativizer/iteration_*/candidate_*.metadata.json` | axes, compliance feedback, and image-token whitelist for that creativizer call |
| `creativizer/iteration_*/manifest.json` | index of all creativizer prompts for that iteration |
| `code_generator/iteration_*/candidate_*.raw_response.txt` | exact code-gen LLM response (or exception text if the call failed) |
| `code_generator/iteration_*/candidate_*.user_prompt.txt` | exact user prompt that was fed to the code-gen LLM |
| `code_generator/iteration_*/candidate_*.metadata.json` | `parse_status`, `llm_exception`, `sections_used_fallback`, `missing_section_ids`, MJML render result + attempts, per-section fallback flags — use this to explain why a candidate looks like the plain-text fallback template |
| `baseline/baseline.png`+ `baseline_slice_*.png` | baseline render + slices |

## Programmatic use

```python
import asyncio
from pipeline.graph import run_from_baseline, run_from_baseline_sync
from pipeline.utils.html_compression import compress_html_base64
from pipeline.utils.style_guide import load_style_guide

# 1. grab the backend's outputs
backend_state = ...   # whatever the backend graph returned
content_blueprint = backend_state["content_blueprint"]
rendered_html     = backend_state["html_document"]

# 2. compress images out of the HTML so LLM text calls stay cheap
compressed_html, images_b64 = compress_html_base64(rendered_html)

# 3. load brand rules (4-list dict — same shape as `_normalize_brand_tokens`)
brand_rules = load_style_guide("examples/style_guide_ruleset.json")

# 4. async — preferred (so the gathers actually parallelize)
final = asyncio.run(run_from_baseline(
    blueprint_sections=content_blueprint,           # list of dicts is fine
    base_prompt="",                                 # seed (optional)
    baseline_html_compressed=compressed_html,
    baseline_images_b64=images_b64,
    brand_rules=brand_rules,
    iterations=1,
))

# 4b. or sync wrapper (calls asyncio.run internally) for legacy callers
# final = run_from_baseline_sync(blueprint_sections=..., ...)

winner = final["winner"]
print("best HTML:", winner.html[:200])
print("best prompt:", winner.prompt_used)
```

## Brand rules / style guide

Provide brand constraints by populating `examples/style_guide_ruleset.json`
(or pointing `--style-guide` at any JSON of the same shape):

```json
{
  "design_bible": {
    "website": {
      "content_pattern_rules": ["..."],
      "color_scheme_rules":    ["..."],
      "design_pattern_rules":  ["..."],
      "other_rules":           ["..."]
    }
  }
}
```

This matches the production backend's `_normalize_brand_tokens` shape
exactly. The pipeline injects all four lists verbatim into the
creativizer + per-section MJML designer + compliance check prompts as the
**Brand Rules (SUPREME — NON-NEGOTIABLE)** block (lifted directly from
the production designer prompt). Compliance check additionally enforces
`other_rules` (REMS warnings, mandatory sign-offs, PI access lines, etc).

`--style-guide` is auto-loaded as `examples/style_guide_ruleset.json`
whenever `--from-example examples` is used.

## Configuration

- `config/pipeline_config.yaml` — k candidates, model choices, scoring
  weights, and **async concurrency caps** (`concurrency:` block).
- `config/design_axes.yaml`     — diversity axes the creativizer samples from

## Layout

```
adversarial_design/
├── pipeline/
│   ├── state.py                       # BlueprintSection, DesignDescriptor, Candidate, PipelineState
│   ├── graph.py                       # build_graph(), run_from_baseline()
│   ├── nodes/
│   │   ├── ingestion.py               # validates + auto-slices baseline
│   │   ├── reverse_prompter.py        # baseline + reference descriptors
│   │   ├── creativizer.py             # k design specs respecting blueprint
│   │   ├── code_generator.py          # HTML candidates (+ baseline as cand #0)
│   │   ├── screenshotter.py           # render + slice each candidate
│   │   ├── compliance_check.py        # MLR text gate
│   │   ├── quality_judge.py           # VLM scoring incl. beats_baseline
│   │   ├── anti_ai_judge.py           # VLM AI-likeness ranking
│   │   └── aggregator.py              # weighted score, winner persist
│   ├── prompts/                       # one module per stage
│   │   ├── mjml_full_designer.py      # whole-email MJML prompt (default)
│   │   └── mjml_section_designer.py   # per-section MJML prompt (alternate)
│   └── utils/
│       ├── html_compression.py        # data URI ⇄ [images_base64_N]
│       ├── mjml_utils.py              # compile fragments, mjml→html, validate
│       ├── style_guide.py             # 4-list brand rules loader + body-font extractor
│       ├── playwright_utils.py        # render + slice helpers
│       ├── image_utils.py
│       ├── llm_clients.py             # sync + AsyncAnthropic clients
│       └── storage.py
├── examples/                          # self-contained sample inputs
├── scripts/run_pipeline.py            # CLI entry point
├── tests/
│   ├── test_html_compression.py       # round-trip + example smoke tests
│   ├── test_mjml_utils.py             # compile / validate / find_missing_sentences
│   ├── test_style_guide.py            # loader + format_brand_rules_block + body font
│   └── test_code_generator.py         # full-MJML parser + fallback + mode dispatcher
└── config/*.yaml
```

## Tests

```bash
PYTHONPATH=. pytest tests/ -v
```
