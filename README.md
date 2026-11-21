# invoice-ocr

Local invoice OCR on [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), served through [Ollama](https://ollama.com). Give it an invoice image; it returns structured JSON — or the full text/table markup if the strict path isn't confident enough.

## How it works

`main.py` runs each image through two paths, in order:

1. **JSON (strict)** — `invoice_json.py` feeds the model a JSON skeleton generated from a Pydantic `Invoice` schema (invoice number, dates, bill from/to, line items, subtotal, tax, total, payment info) and validates the response against it. Output: `<image>.json`.
2. **HTML (loose fallback)** — `invoice_html.py` runs the model's official Text and Table Recognition prompts and saves `<image>.html` plus `<image>.md`. This is the safety net: if the model output doesn't fit the schema, you still get the full content.

## Requirements

- Python 3.10+ (developed on 3.14)
- [Ollama](https://ollama.com) running locally with the GLM-OCR model: `ollama pull glm-ocr` (or `glm-ocr:q8_0`)
- Python deps: `pip install requests pydantic markdownify`

## Usage

```bash
# runner: JSON first, HTML fallback on schema failure (default image: assets/invoice.jpg)
python main.py assets/invoice.jpg

# individual paths
python invoice_json.py assets/invoice.jpg   # strict; pipeable JSON on stdout
python invoice_html.py assets/invoice.jpg   # loose; markdown on stdout
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0    | success |
| 1    | bad invocation / file not found — or `invoice_json.py` alone: model output failed schema validation (the intended fall-back signal) |
| 2    | Ollama API error (model unreachable) |

## Example

`assets/invoice.jpg` → `assets/invoice.json`:

```json
{
  "invoice_number": "INV-2024-0847",
  "invoice_date": "2025-07-29",
  "due_date": "2025-08-28",
  "bill_to": {"name": "Tech Solutions Inc.", "address": "456 Innovation Drive\nSan Francisco, CA 94105"},
  "line_items": [
    {"description": "Web Development Services", "quantity": 40.0, "unit_price": 150.0, "amount": 6000.0}
  ],
  "subtotal": 9550.0,
  "tax_rate": 8.5,
  "tax_amount": 811.75,
  "total": 10361.75
}
```

## Notes

- GLM-OCR is **prompt-limited**: it only reliably understands its official task prompts and the information-extraction format from the [model card](https://huggingface.co/zai-org/GLM-OCR). The prompt wording lives in `invoice_json.py` — don't paraphrase it.
- Amounts come back as numbers even when the invoice prints `$1,234.56` (coerced during validation).
- Details for agent/developer context: see [AGENTS.md](AGENTS.md).
