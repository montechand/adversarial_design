# Reference creative examples

Drop human-made creative ads here. The reverse prompter interleaves each one
with its caption into the **baseline VLM call** as a comparison anchor, so
the analyzer can list the baseline's weaknesses against an explicit human
standard ("baseline lacks the asymmetric grid rhythm seen in reference #2",
etc.).

These are **NOT analyzed individually** — they exist purely to ground the
baseline's design analysis.

## Accepted file types (auto-discovered)

| File              | Behavior                                                  |
|-------------------|-----------------------------------------------------------|
| `*.html` / `*.htm`| Rendered to PNG via Playwright on first use, then cached. |
| `*.png` / `*.jpg` / `*.jpeg` / `*.webp` | Used as-is.                          |
| `<stem>.md` or `<stem>.txt` | Sidecar — body becomes the caption. Optional.    |
| `manifest.json`   | Optional curated index — overrides directory discovery.   |

All images are downscaled to a max longest-side of `1280px` before being
base64-encoded into the VLM call, so the prompt payload stays small. Drop
in arbitrarily large source PNGs / screenshots — the pipeline will
thumbnail them automatically (originals untouched).

## Captions

Captions are how you give each reference a name and a *why-it-matters*
sentence. Two options:

### 1. Sidecar files (simplest)

```
references/
├── pfizer_xeljanz_2023.png
├── pfizer_xeljanz_2023.md          ← caption for the PNG above
├── lilly_trulicity_editorial.html
└── lilly_trulicity_editorial.txt   ← caption for the HTML above
```

Sidecar body becomes the caption verbatim. Keep it 1-3 sentences —
something like:

> Pfizer XELJANZ HCP email, 2023. Notable for its asymmetric 2-column
> grid, oversized editorial type, and the deliberate use of a single
> dominant statistic per section. No tinted callout boxes anywhere.

### 2. `manifest.json` (curated)

When precise ordering or richer metadata matters:

```json
{
  "items": [
    {
      "name": "pfizer_xeljanz_2023",
      "image": "pfizer_xeljanz_2023.png",
      "description": "Pfizer XELJANZ HCP email, 2023. Asymmetric 2-column grid, oversized editorial type, single dominant statistic per section."
    },
    {
      "name": "lilly_trulicity_editorial",
      "html": "lilly_trulicity_editorial.html",
      "description": "Lilly TRULICITY editorial-style. Magazine-dispatch architecture with chapter-style section entries, near-black + electric accent palette."
    }
  ]
}
```

When `manifest.json` exists, only its listed items load (in order). Sidecar
files are ignored.

## How references appear in the prompt

The reverse prompter constructs a multimodal user message of the shape:

```
[text]   blueprint summary + baseline HTML + slice intro
[text]   "[baseline slice 1/N]"
[image]  baseline slice 1
…
[text]   "## REFERENCE CREATIVE EXAMPLES — comparison anchors only"
[text]   "REFERENCE EXAMPLE [1]: pfizer_xeljanz_2023 — Pfizer XELJANZ HCP email…"
[image]  pfizer_xeljanz_2023.png
[text]   "REFERENCE EXAMPLE [2]: lilly_trulicity_editorial — Lilly TRULICITY editorial-style…"
[image]  lilly_trulicity_editorial.png
[text]   "Now extract the BASELINE's design intent… ground weaknesses against the references above."
```

(The Anthropic API doesn't take captions on image blocks; labeling is purely
positional via adjacent `text` blocks.)

## Controlling how many references reach the prompt

`config/pipeline_config.yaml`:

```yaml
pipeline:
  max_references: 6   # alphabetical-first N reach the VLM call
```

Set to `0` to disable references without removing files from this directory.

## CLI

```bash
# auto-discovered when using --from-example examples
python scripts/run_pipeline.py --from-example examples

# or point at any directory
python scripts/run_pipeline.py \
    --blueprint-json    examples/blueprint.json \
    --baseline-html     examples/baseline_email.html \
    --reference-examples examples/references
```

## On-disk caching

Rendered HTML and downscaled images land in `<run_dir>/reference_examples/`.
Re-renders trigger when source files change (mtime check).
