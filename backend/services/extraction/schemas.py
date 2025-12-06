"""
Extraction schema registry.

Defines the "shape" of metadata to extract from each document category.
Each schema serves two purposes:
1. Prompt Generation: Telling the LLM what to extract
2. Table Definition: Defining columns for the dedicated SQL table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single extraction field."""
    name: str
    sql_type: str  # TEXT, REAL, INTEGER, DATE (stored as TEXT in SQLite)
    description: str
    required: bool = False
    examples: Optional[List[str]] = None


@dataclass(frozen=True)
class ExtractionSchema:
    """
    Schema definition for extracting metadata from a document category.
    
    Attributes:
        schema_id: Unique identifier for the schema
        target_l1_category: L1 category ID from taxonomy (e.g., "financial_accounting")
        target_l2_category: L2 category ID from taxonomy (e.g., "invoices")
        table_name: SQL table name (e.g., "meta_invoices")
        display_name: Human-readable name
        description: Description of what this schema extracts
        fields: List of field specifications
    """
    schema_id: str
    target_l1_category: str
    target_l2_category: Optional[str]
    table_name: str
    display_name: str
    description: str
    fields: tuple[FieldSpec, ...]

    def get_csv_headers(self) -> List[str]:
        """Return CSV column headers for LLM output."""
        return [f.name for f in self.fields]

    def get_field_descriptions(self) -> str:
        """Generate field descriptions for LLM prompt."""
        lines = []
        for f in self.fields:
            req = " (required)" if f.required else ""
            examples = ""
            if f.examples:
                examples = f" Examples: {', '.join(f.examples)}"
            lines.append(f"- {f.name}: {f.description}{req}{examples}")
        return "\n".join(lines)

    def to_create_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement."""
        columns = ["doc_hash TEXT PRIMARY KEY"]
        for f in self.fields:
            col_def = f"{f.name} {f.sql_type}"
            columns.append(col_def)
        columns.append("extracted_at TEXT NOT NULL")
        columns.append("extraction_model TEXT")
        columns.append("FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE")
        
        return f"""
CREATE TABLE IF NOT EXISTS {self.table_name} (
    {','.join(f'{chr(10)}    {c}' for c in columns)}
)
"""


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

INVOICES_SCHEMA = ExtractionSchema(
    schema_id="invoices",
    target_l1_category="financial_accounting",
    target_l2_category="invoices",
    table_name="meta_invoices",
    display_name="Invoices",
    description="Extract structured metadata from invoice documents",
    fields=(
        FieldSpec(
            name="invoice_number",
            sql_type="TEXT",
            description="Unique invoice identifier or number",
            required=True,
            examples=["INV-2024-001", "12345", "A-2023-0042"],
        ),
        FieldSpec(
            name="invoice_date",
            sql_type="TEXT",
            description="Date when the invoice was issued (ISO format YYYY-MM-DD)",
            required=True,
            examples=["2024-01-15", "2023-12-01"],
        ),
        FieldSpec(
            name="due_date",
            sql_type="TEXT",
            description="Payment due date (ISO format YYYY-MM-DD)",
            required=False,
            examples=["2024-02-15", "2024-01-31"],
        ),
        FieldSpec(
            name="vendor_name",
            sql_type="TEXT",
            description="Name of the company or person who issued the invoice",
            required=True,
            examples=["Acme Corp", "John's Consulting LLC"],
        ),
        FieldSpec(
            name="vendor_address",
            sql_type="TEXT",
            description="Address of the vendor/issuer",
            required=False,
        ),
        FieldSpec(
            name="customer_name",
            sql_type="TEXT",
            description="Name of the customer/recipient of the invoice",
            required=False,
            examples=["My Company Inc", "Jane Doe"],
        ),
        FieldSpec(
            name="subtotal",
            sql_type="REAL",
            description="Subtotal amount before tax (numeric value only)",
            required=False,
            examples=["1000.00", "500.50"],
        ),
        FieldSpec(
            name="tax_amount",
            sql_type="REAL",
            description="Tax amount (numeric value only)",
            required=False,
            examples=["190.00", "50.05"],
        ),
        FieldSpec(
            name="total_amount",
            sql_type="REAL",
            description="Total invoice amount including tax (numeric value only)",
            required=True,
            examples=["1190.00", "550.55"],
        ),
        FieldSpec(
            name="currency",
            sql_type="TEXT",
            description="Currency code",
            required=False,
            examples=["USD", "EUR", "GBP"],
        ),
        FieldSpec(
            name="payment_terms",
            sql_type="TEXT",
            description="Payment terms or conditions",
            required=False,
            examples=["Net 30", "Due on receipt", "2/10 Net 30"],
        ),
        FieldSpec(
            name="po_number",
            sql_type="TEXT",
            description="Purchase order number if referenced",
            required=False,
            examples=["PO-2024-123", "4567890"],
        ),
    ),
)


# Registry of all extraction schemas
EXTRACTION_SCHEMAS: Dict[str, ExtractionSchema] = {
    "invoices": INVOICES_SCHEMA,
}


def get_schema(schema_id: str) -> Optional[ExtractionSchema]:
    """Get extraction schema by ID."""
    return EXTRACTION_SCHEMAS.get(schema_id)


def get_schema_for_category(l1_category: str, l2_category: Optional[str] = None) -> Optional[ExtractionSchema]:
    """Find extraction schema matching a document's classification."""
    for schema in EXTRACTION_SCHEMAS.values():
        if schema.target_l1_category == l1_category:
            if l2_category is None or schema.target_l2_category is None:
                return schema
            if schema.target_l2_category == l2_category:
                return schema
    return None


def list_schemas() -> List[Dict[str, Any]]:
    """List all available extraction schemas."""
    return [
        {
            "schema_id": s.schema_id,
            "display_name": s.display_name,
            "description": s.description,
            "target_l1_category": s.target_l1_category,
            "target_l2_category": s.target_l2_category,
            "table_name": s.table_name,
            "field_count": len(s.fields),
        }
        for s in EXTRACTION_SCHEMAS.values()
    ]
