"""Structured invoice extraction with GLM-OCR's information-extraction mode.

GLM-OCR is prompt-limited: for information extraction the prompt must be the
fixed instruction from the model card followed by a JSON skeleton that the
model fills in. The skeleton is generated from the Pydantic schema below, so
the prompt shape and the validation shape can never drift apart.

On success: prints the validated JSON and saves `<image>.json` next to the
input image (stdout is pure JSON, safe to pipe).
On failure: prints model errors to stderr and exits 1, so a pipeline can fall
back to the looser HTML path (invoice_html.py).
"""

import base64
import json
import re
import sys
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import requests
from pydantic import BaseModel, ValidationError, field_validator
from types import UnionType

from invoice_html import generate

# Exact instruction from the zai-org/GLM-OCR model card ("Prompt Limited"
# section). The model only recognizes this verbatim.
EXTRACTION_INSTRUCTION = "请按下列JSON格式输出图中信息:"


# --- schema ---------------------------------------------------------------

class Party(BaseModel):
    name: str
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    contact: str | None = None


class LineItem(BaseModel):
    description: str
    quantity: float
    unit: str | None = None
    unit_price: float
    amount: float

    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        # Be tolerant of "$150.00" / "1,234.56" coming back as strings.
        if isinstance(v, str):
            v = v.strip().replace("$", "").replace(",", "")
            try:
                return float(v)
            except ValueError:
                return v
        return v


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    bill_from: Party | None = None
    bill_to: Party | None = None
    line_items: list[LineItem]
    subtotal: float | None = None
    tax_rate: float | None = None  # percent, e.g. 8.5
    tax_amount: float | None = None
    total: float | None = None
    payment_terms: str | None = None
    payment_methods: list[str] | None = None
    notes: str | None = None

    @field_validator("subtotal", "tax_rate", "tax_amount", "total", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().replace("$", "").replace(",", "")
            try:
                return float(v)
            except ValueError:
                return v
        return v


# --- skeleton + prompt ------------------------------------------------------

def build_skeleton(model: type[BaseModel]) -> dict:
    """JSON skeleton in the shape of `model`, every leaf an empty string,
    matching the model card's example format."""

    def placeholder(t: Any) -> Any:
        args = get_args(t)
        origin = get_origin(t)
        if origin in (Union, UnionType):  # Optional[X] -> X
            non_none = [a for a in args if a is not type(None)]
            return placeholder(non_none[0]) if non_none else ""
        if origin is list:
            return [placeholder(args[0]) if args else ""]
        if isinstance(t, type) and issubclass(t, BaseModel):
            return {n: placeholder(f.annotation) for n, f in t.model_fields.items()}
        return ""

    return placeholder(model)


def extraction_prompt(schema_model: type[BaseModel] = Invoice) -> str:
    skeleton = json.dumps(
        build_skeleton(schema_model), indent=4, ensure_ascii=False
    )
    return f"{EXTRACTION_INSTRUCTION}\n{skeleton}"


# --- extraction + validation -----------------------------------------------

def extract_json(raw: str, schema_model: type[BaseModel] = Invoice) -> BaseModel:
    """Parse the model's raw output and validate it against the schema."""
    text = raw.strip()
    # Tolerate the model wrapping the JSON in markdown fences.
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("model output contains no JSON object")
    data = json.loads(text[start : end + 1])
    return schema_model.model_validate(data)


def extract_invoice_json(image_path) -> BaseModel:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    raw = generate(extraction_prompt(), image_base64)
    return extract_json(raw, Invoice)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python invoice_json.py assets/invoice.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    try:
        invoice = extract_invoice_json(image_path)
    except requests.RequestException as e:
        print(f"Ollama API error: {e}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        # Print the failure, keep stdout clean: a pipeline can catch exit 1
        # and fall back to invoice_html.py (HTML table mode).
        print(f"JSON extraction failed: {e}", file=sys.stderr)
        if image_path.exists():
            print(
                "Falling back is up to you: python invoice_html.py "
                f"{image_path}",
                file=sys.stderr,
            )
        sys.exit(1)

    output = invoice.model_dump_json(indent=2)
    saved = image_path.with_suffix(".json")
    saved.write_text(output + "\n", encoding="utf-8")

    # stdout: pure JSON (pipeable); bookkeeping on stderr.
    print(output)
    print(f"\nSaved to: {saved}", file=sys.stderr)
