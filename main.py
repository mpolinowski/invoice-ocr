"""Invoice extraction runner: strict JSON first, loose HTML as fallback.

For each image it tries invoice_json's schema-validated JSON extraction,
then falls back to invoice_html's text + table recognition if the model
output doesn't fit the Pydantic schema.

Calls the modules' library functions directly (not their CLIs), so the
fallback contract doesn't depend on parsing their output.

Exit codes (consistent with invoice_json.py's CLI):
    0  every image extracted (JSON, or HTML fallback)
    1  bad invocation or image file not found
    2  Ollama API error (model unreachable; both paths use it)
    3  unexpected error

Usage:
    python main.py [IMAGE ...]        # no args -> assets/invoice.jpg
    python main.py a.jpg b.jpg
"""

import json
import sys
from pathlib import Path

import requests
from pydantic import ValidationError

import invoice_html
import invoice_json

DEFAULT_IMAGE = Path("assets/invoice.jpg")


def save_json(image_path: Path) -> Path:
    """Strict, schema-validated extraction. Saves <image>.json."""
    invoice = invoice_json.extract_invoice_json(image_path)
    out = image_path.with_suffix(".json")
    out.write_text(invoice.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out


def save_html(image_path: Path) -> tuple[Path, Path]:
    """Loose text + table extraction. Saves <image>.html and <image>.md."""
    html = invoice_html.extract_invoice(image_path)
    out_html = image_path.with_suffix(".html")
    out_html.write_text(html, encoding="utf-8")
    out_md = image_path.with_suffix(".md")
    out_md.write_text(invoice_html.render_markdown(html), encoding="utf-8")
    return out_html, out_md


def process_image(image_path: Path) -> None:
    """JSON first; on schema/parse failure fall back to HTML.

    Ollama API errors propagate: the fallback path uses the same model,
    so retrying it can only fail the same way."""
    try:
        out = save_json(image_path)
        print(f"{image_path}: JSON extracted -> {out}")
        return
    except requests.RequestException:
        raise
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        print(
            f"{image_path}: JSON extraction failed ({type(e).__name__}: {e}); "
            "falling back to HTML",
            file=sys.stderr,
        )
    out_html, out_md = save_html(image_path)
    print(
        f"{image_path}: HTML fallback -> {out_html}, {out_md}",
        file=sys.stderr,
    )


def main(argv: list[str]) -> int:
    images = [Path(p) for p in argv] if argv else [DEFAULT_IMAGE]

    missing = [p for p in images if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        return 1

    for image in images:
        try:
            process_image(image)
        except requests.RequestException as e:
            print(f"Ollama API error: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Unexpected error for {image}: {e!r}", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
