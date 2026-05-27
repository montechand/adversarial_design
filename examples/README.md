# Example inputs for the adversarial-design pipeline

This directory contains a self-contained set of inputs so you can run the
adversarial-design pipeline **without** the backend-server. The pipeline
treats this trio exactly the way it will treat the real backend output once
the integration is wired up.

## Files

| File                          | Role |
|-------------------------------|------|
| `blueprint.json`              | Structured content blueprint for ULTOMIRIS gMG — matches the backend `SectionSpec` shape (see `Backend-Server/src/content_generation_new/application/Agentic_Workflows/message_generator_from_scratch_v0_1.py`, `class SectionSpec` at line 693). |
| `blueprint1.json`             | Structured content blueprint for adjuvant NSCLC (Opdivo-style), used for offline pipeline testing. |
| `baseline_email.html`         | Baseline HTML for `blueprint.json` (ULTOMIRIS gMG). Hand-authored to match the blueprint's section structure. No image placeholders (`use_uploaded_image: false` throughout). |
| `baseline_email1.html`        | Baseline HTML for `blueprint1.json` (adjuvant NSCLC). Image `<img src="...">` URIs are compressed to `[images_base64_N]` placeholder tokens. |
| `baseline_images_b64.json`    | The side-car `{placeholder_token: full data URI}` map needed to rehydrate the NSCLC baseline HTML for screenshotting and VLM judging. |
| `style_guide_ruleset.json`    | Generic **style guide template** — same four-list shape as production `design_bible.website`; placeholder entries show expected rule format. |
| `style_guide_ruleset_ultomiris.json` | **ULTOMIRIS HCP brand rules** — populated from Alexion brand guidelines + production HCP email references (`examples/references/US_ULT-N_*.html`). Use with `--style-guide examples/style_guide_ruleset_ultomiris.json` when running the gMG blueprint. |
| `references/`                 | **Reference creative ad gallery** — drop human-made HCP / editorial ads (`.html`, `.png`, `.jpg`) here with optional sidecar caption files (`<stem>.md` / `<stem>.txt`) or a `manifest.json`. Each is rendered (if HTML), downscaled to ≤1280px, and interleaved with its caption into the reverse prompter's **baseline VLM call** so the analyzer can ground the baseline's weaknesses against an explicit human standard. See `examples/references/README.md` for the format. |
| `baseline_screenshots/`       | *(absent)* If supplied as a directory of PNG slices (top→bottom), the pipeline skips re-rendering. If absent, the pipeline rehydrates + renders + slices automatically through Playwright on first run. |

## Run

```bash
cd adversarial_design
pip install -e ".[dev]"
playwright install chromium
export ANTHROPIC_API_KEY=...

# convenience: load everything in this directory at once
python scripts/run_pipeline.py --from-example examples
```

or explicitly:

```bash
python scripts/run_pipeline.py \
  --blueprint-json    examples/blueprint.json \
  --baseline-html     examples/baseline_email.html \
  --baseline-images   examples/baseline_images_b64.json \
  --iterations 1
```

Outputs land in `storage/runs/<run_id>/` (winner HTML, winning prompt, scores).

## Style guide ruleset

`style_guide_ruleset.json` holds the brand/design rules that are injected into the pipeline's
LLM prompts as **supreme constraints** (they override all generic design instructions). The
structure mirrors the four lists that `_normalize_brand_tokens` extracts from the production
`design_bible`:

| Key | What goes here |
|-----|----------------|
| `content_pattern_rules` | Typography hierarchy, copy length, citation format, ISI/references rendering, CTA copy style |
| `color_scheme_rules`    | Named hex codes + their allowed usage contexts (headline, CTA bg, panel fill, body text, warning panels) |
| `design_pattern_rules`  | Layout constraints (width, columns), data callout patterns, section spacing, image handling, inline-CSS-only requirement |
| `other_rules`           | Drug name/® rules, regulatory language, statistical notation, sign-off text, mandatory safety elements |

**To populate from a PDF:**

1. Paste the full PDF text (or a copy-paste of its text content) into `raw_pdf_dump`.
2. Run the extraction helper (when wired up):
   ```bash
   python scripts/extract_style_rules.py examples/style_guide_ruleset.json
   ```
   This uses Claude to distil the PDF into the four lists. Review and trim the output before committing.
3. Alternatively, manually read the PDF and write one declarative sentence per rule into the appropriate list.
   Be specific — include hex codes, px values, font names, exact claim text.

**To use it in a pipeline run:**

```bash
python scripts/run_pipeline.py \
  --blueprint-json   examples/blueprint.json \
  --baseline-html    examples/baseline_email.html \
  --style-guide      examples/style_guide_ruleset_ultomiris.json \
  --iterations 2
```

---

## Authoring your own example

The synthetic Opdivo-style blueprint here is offline test data. To author a
real one:

1. Run the backend-server email generator on a real brief; capture
   `state["content_blueprint"]` and the final `state["html_document"]`.
2. Save the blueprint as a list under `content_blueprint` in a JSON file.
3. Save the HTML as-is (it may contain raw `data:image/...;base64,...` URIs —
   the pipeline will auto-compress them).
4. Either pass the HTML straight to `--baseline-html` (the CLI auto-extracts
   the images into a sidecar) **or** pre-compress with
   `pipeline.utils.html_compression.compress_html_base64()` and save the map
   separately.
