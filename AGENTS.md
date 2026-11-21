# AGENTS.md

## What this is

Extracts invoice data from images using the GLM-OCR vision model, served locally by Ollama (`http://localhost:11434`). No cloud calls. Strict structured JSON output with a loose HTML/Markdown fallback.

## Architecture

Three modules, one runner:

| File | Role |
|------|------|
| `invoice_html.py` | Shared `generate()` (the Ollama `/api/generate` call, base64 image, temperature 0) + the loose HTML/Markdown path. Constants `OLLAMA_URL` and `MODEL` live here. |
| `invoice_json.py` | Import's `generate` from `invoice_html.py`. Defines the Pydantic `Invoice`/`Party`/`LineItem` schema; the JSON skeleton sent to the model is **generated from that schema** (`build_skeleton`), so prompt shape and validation shape can't drift apart. |
| `main.py` | Runner. **In-process** fallback: calls `invoice_json.extract_invoice_json()` and on `ValueError`/`JSONDecodeError`/`ValidationError` calls `invoice_html.extract_invoice()` + `render_markdown()`. Deliberately does not shell out to the CLIs, so it never depends on their exit codes or stdout. |

Fallback contract: JSON succeeds → `<image>.json`. JSON output fails schema validation → `<image>.html` + `<image>.md`. Ollama down → abort with exit 2 (the fallback uses the same model, so retrying is pointless).

Exit codes (`main.py` and `invoice_json.py`): `0` success · `1` bad args / file missing / (json CLI) schema failure · `2` Ollama API error · `3` (main) unexpected error.

## Invariants — do not break

1. **Prompt-limited model.** GLM-OCR only reliably understands its official task prompts (from the [model card](https://huggingface.co/zai-org/GLM-OCR) "Prompt Limited" section):
   - Text: `Text Recognition:`
   - Formula: `Formula Recognition:`
   - Table: `Table Recognition:`
   - Information extraction: the fixed Chinese instruction `请按下列JSON格式输出图中信息:` followed by a JSON skeleton with empty-string leaves.
   Free-form or paraphrased prompts silently degrade to unstructured text. Keep the strings in `invoice_json.py` / `invoice_html.py` verbatim.
2. **Schema = source of truth.** To change which fields are extracted, edit the Pydantic models in `invoice_json.py` — the prompt skeleton regenerates from them automatically. Never hand-edit the prompt skeleton.
3. **JSON stdout is pipeable.** `invoice_json.py` must keep stdout as pure JSON; diagnostics go to stderr.
4. **Validation must reject partial JSON.** Missing required fields → `ValidationError` → exit 1. Don't loosen `model_validate` into "best effort".

## Runtime environment & quirks

- Ollama base: `http://localhost:11434`, default model `glm-ocr:latest` (F16, ~2.65 GB); `glm-ocr:q8_0` commented in `invoice_html.py`.
- **Model eviction:** this box runs other Ollama models concurrently. Each GLM-OCR call evicts the resident model and vice versa. Batch GLM-OCR calls per turn; don't loop single-shot probing.
- The locally pulled `glm-ocr` tags come from the `dhiltgen` community mirror, not zai-org directly. After re-pulls, verify a known image still extracts correctly (golden values below) before trusting output.
- GLM-OCR can occasionally **drop array elements** (seen with `payment_methods`) and returns `currency: "$"` rather than `USD`. Downstream consumers should treat these as model limitations, not schema bugs.
- Dates come back ISO-normalized (`2025-07-29`) even when the invoice prints prose dates.
- Known CLI wart: `invoice_html.py`'s `__main__` catches API errors but exits 0 — fine for humans, unsuitable for scripting. `main.py` avoids it by calling library functions; fix the CLI if anything starts depending on its exit code.

## Testing

No test framework is set up; the test fixture is `assets/invoice.jpg`.

Regression golden values (from that image, cross-checked against the human-readable original):

- 4 line items; `40 × 150.00 = 6000.00`, `20 × 125.00 = 2500.00`, `8 × 100.00 = 800.00`, `1 × 250.00 = 250.00`
- Subtotal `9550.00`, tax `8.5%` → `811.75`, total `10361.75`
- Invoice `INV-2024-0847`; dated 2025-07-29, due 2025-08-28

Cheap offline checks (no model round-trip):

```python
# fake fenced response with currency strings — must validate + coerce
fake = """```json
{"invoice_number": "INV-2024-0847",
 "line_items": [{"description": "Web Dev", "quantity": "40", "unit_price": "$150.00", "amount": "$6,000.00"}],
 "subtotal": "$9,550.00", "tax_rate": 8.5, "tax_amount": "$811.75", "total": "$10,361.75"}
```"""
inv = invoice_json.extract_json(fake)
assert inv.total == 10361.75 and inv.subtotal == 9550.0
```

To exercise `main.process_image`'s success and fallback branches without touching the GPU, monkeypatch `invoice_json.generate` (return valid JSON / garbage) and `invoice_html.generate` (return text / `<table>...`), then point it at a throwaway copy of the image.
