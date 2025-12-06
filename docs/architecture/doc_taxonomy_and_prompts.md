
# Document Taxonomy & Classification Prompts

This file summarizes the design we discussed for:
- Using **one powerful LLM (e.g., Qwen3 30B A3B)** in multiple “roles” via different prompts.
- Defining a **two-layer document taxonomy** (L1 + L2).
- Classifying documents into that taxonomy using **short, two-step prompts**.

---

## 1. High-Level Architecture

You’re using **a single LLM** and switching behavior via different prompts (“modes”).  
Conceptually:

- **Same model**, different prompts:
  - *Classifier (L1)* – choose top-level category.
  - *Classifier (L2)* – choose subcategory under a known L1.
  - (Optionally later: Planner, Executor, Summarizer, etc.)

Multi-agent system here simply means:

> **Different prompts → same model → different roles.**

No need for multiple models or real “agents”.

---

## 2. Document Taxonomy (Two Layers)

You maintain the taxonomy **in your code** (JSON / dict).  
The LLM only needs the *IDs* at classification time.

### 2.1. Full Taxonomy JSON (L1 + L2)

```json
{
  "version": "1.0",
  "taxonomy": [
    {
      "id": "financial_accounting",
      "name": "Financial & Accounting",
      "description": "Documents related to money, payments, budgets, and financial records.",
      "subcategories": [
        { "id": "invoices", "name": "Invoices" },
        { "id": "receipts", "name": "Receipts" },
        { "id": "budgets", "name": "Budgets & Financial Plans" },
        { "id": "payroll", "name": "Payroll & Salary Records" },
        { "id": "tax_paperwork", "name": "Tax Documents & Declarations" },
        { "id": "purchase_orders", "name": "Purchase Orders (POs)" },
        { "id": "expense_reports", "name": "Expense Reports" },
        { "id": "other_financial", "name": "Other Financial Documents" }
      ]
    },
    {
      "id": "legal_compliance",
      "name": "Legal & Compliance",
      "description": "Documents related to laws, regulations, contracts, and compliance.",
      "subcategories": [
        { "id": "contracts", "name": "Contracts & Agreements" },
        { "id": "licenses_certificates", "name": "Licenses & Certificates" },
        { "id": "regulations_policies", "name": "Regulations & Policies" },
        { "id": "court_documents", "name": "Court & Legal Case Documents" },
        { "id": "compliance_reports", "name": "Compliance Reports & Audits" },
        { "id": "privacy_gdpr", "name": "Privacy / GDPR Documents" },
        { "id": "permits", "name": "Permits & Authorizations" },
        { "id": "other_legal", "name": "Other Legal / Compliance Documents" }
      ]
    },
    {
      "id": "administrative_hr",
      "name": "Administrative & HR",
      "description": "Internal administration and human resources documentation.",
      "subcategories": [
        { "id": "employee_records", "name": "Employee Records & Files" },
        { "id": "recruitment_docs", "name": "Recruitment & Hiring Documents" },
        { "id": "hr_policies", "name": "HR Policies & Procedures" },
        { "id": "attendance_timesheets", "name": "Attendance & Timesheets" },
        { "id": "internal_memos", "name": "Internal Memos & Notices" },
        { "id": "meeting_minutes", "name": "Meeting Minutes" },
        { "id": "training_materials", "name": "Training & Onboarding Materials" },
        { "id": "other_admin_hr", "name": "Other Administrative / HR Documents" }
      ]
    },
    {
      "id": "technical_engineering",
      "name": "Technical & Engineering",
      "description": "Technical documentation, engineering specs, and manuals.",
      "subcategories": [
        { "id": "machine_manuals", "name": "Machine / Equipment Manuals" },
        { "id": "software_manuals", "name": "Software Manuals & Guides" },
        { "id": "sops", "name": "Standard Operating Procedures (SOPs)" },
        { "id": "technical_specs", "name": "Technical Specifications" },
        { "id": "maintenance_logs", "name": "Maintenance & Service Logs" },
        { "id": "installation_guides", "name": "Installation Guides" },
        { "id": "engineering_drawings", "name": "Engineering Drawings & Schematics" },
        { "id": "other_technical", "name": "Other Technical / Engineering Documents" }
      ]
    },
    {
      "id": "scientific_academic",
      "name": "Scientific & Academic",
      "description": "Research, scientific, and academic documents.",
      "subcategories": [
        { "id": "research_papers", "name": "Research Papers & Articles" },
        { "id": "theses_dissertations", "name": "Theses & Dissertations" },
        { "id": "lab_reports", "name": "Lab Reports & Experiment Logs" },
        { "id": "clinical_studies", "name": "Clinical Studies & Trials" },
        { "id": "conference_papers", "name": "Conference Papers & Posters" },
        { "id": "scientific_summaries", "name": "Scientific Summaries & Reviews" },
        { "id": "other_scientific", "name": "Other Scientific / Academic Documents" }
      ]
    },
    {
      "id": "forms_templates",
      "name": "Forms & Templates",
      "description": "Blank and filled forms, especially administrative or governmental.",
      "subcategories": [
        { "id": "blank_forms", "name": "Blank Forms & Templates" },
        { "id": "filled_forms", "name": "Filled Forms" },
        { "id": "government_forms", "name": "Government / Public Administration Forms" },
        { "id": "application_forms", "name": "Application Forms" },
        { "id": "questionnaires", "name": "Questionnaires & Surveys" },
        { "id": "other_forms", "name": "Other Forms" }
      ]
    },
    {
      "id": "business_operations",
      "name": "Business Operations",
      "description": "Documents about projects, plans, operations, and performance.",
      "subcategories": [
        { "id": "project_plans", "name": "Project Plans & Charters" },
        { "id": "status_reports", "name": "Status & Progress Reports" },
        { "id": "strategy_documents", "name": "Strategy & Planning Documents" },
        { "id": "rfp_rfq", "name": "RFP / RFQ / Tender Documents" },
        { "id": "performance_reports", "name": "KPIs & Performance Reports" },
        { "id": "audit_reports", "name": "Audit & Evaluation Reports" },
        { "id": "presentations", "name": "Presentations & Slide Decks" },
        { "id": "other_business_ops", "name": "Other Business / Operational Documents" }
      ]
    },
    {
      "id": "case_customer_files",
      "name": "Case & Customer Files",
      "description": "Cases, incidents, and customer-related records.",
      "subcategories": [
        { "id": "case_files", "name": "Case Files & Dossiers" },
        { "id": "incident_reports", "name": "Incident & Accident Reports" },
        { "id": "customer_requests", "name": "Customer / Citizen Requests" },
        { "id": "complaints_claims", "name": "Complaints & Claims" },
        { "id": "support_tickets", "name": "Support / Helpdesk Tickets" },
        { "id": "service_logs", "name": "Service & Intervention Logs" },
        { "id": "other_case_customer", "name": "Other Case / Customer Documents" }
      ]
    },
    {
      "id": "marketing_communications",
      "name": "Marketing & Communications",
      "description": "Public-facing and internal communication material.",
      "subcategories": [
        { "id": "announcements", "name": "Announcements & Notices" },
        { "id": "newsletters", "name": "Newsletters & Bulletins" },
        { "id": "press_releases", "name": "Press Releases" },
        { "id": "brochures_flyers", "name": "Brochures & Flyers" },
        { "id": "web_content", "name": "Website & Social Media Content" },
        { "id": "branding_materials", "name": "Branding & Communication Guidelines" },
        { "id": "other_marketing", "name": "Other Marketing / Communication Documents" }
      ]
    },
    {
      "id": "media_images",
      "name": "Media & Images",
      "description": "Non-text or image-heavy documents.",
      "subcategories": [
        { "id": "scanned_documents", "name": "Scanned Documents (all types)" },
        { "id": "invoice_images", "name": "Images / Scans of Invoices" },
        { "id": "form_images", "name": "Images / Scans of Forms" },
        { "id": "photos", "name": "Photos & Photographic Evidence" },
        { "id": "diagrams_charts", "name": "Diagrams, Charts & Drawings" },
        { "id": "screenshots", "name": "Screenshots" },
        { "id": "handwritten_notes", "name": "Handwritten Notes (images)" },
        { "id": "other_media", "name": "Other Media / Image Documents" }
      ]
    },
    {
      "id": "books_long_docs",
      "name": "Books & Long Documents",
      "description": "Large, book-like or very long documents.",
      "subcategories": [
        { "id": "books", "name": "Books & E-books" },
        { "id": "training_manuals", "name": "Training Manuals & Handbooks" },
        { "id": "reference_guides", "name": "Reference Guides" },
        { "id": "policy_handbooks", "name": "Policy & Procedure Handbooks" },
        { "id": "other_long_docs", "name": "Other Long / Book-Like Documents" }
      ]
    },
    {
      "id": "data_files",
      "name": "Data Files & Datasets",
      "description": "Structured data files and datasets.",
      "subcategories": [
        { "id": "csv_files", "name": "CSV Files" },
        { "id": "excel_files", "name": "Excel / Spreadsheet Files" },
        { "id": "json_xml", "name": "JSON / XML / Data Exports" },
        { "id": "databases_extracts", "name": "Database Extracts & Dumps" },
        { "id": "statistical_reports", "name": "Statistical Tables & Reports" },
        { "id": "other_data_files", "name": "Other Data / Dataset Files" }
      ]
    },
    {
      "id": "emails_correspondence",
      "name": "Emails & Correspondence",
      "description": "Communication via email or formal letters.",
      "subcategories": [
        { "id": "emails", "name": "Emails & Email Threads" },
        { "id": "letters", "name": "Official Letters & Correspondence" },
        { "id": "memos_correspondence", "name": "Memos & Internal Correspondence" },
        { "id": "email_with_attachments", "name": "Emails with Attachments" },
        { "id": "other_correspondence", "name": "Other Correspondence" }
      ]
    },
    {
      "id": "miscellaneous",
      "name": "Miscellaneous / Uncategorized",
      "description": "Documents that do not clearly fit in other categories or need manual review.",
      "subcategories": [
        { "id": "mixed_content", "name": "Mixed / Multi-type Content" },
        { "id": "unclear_type", "name": "Unclear Type / To Review" },
        { "id": "temporary_docs", "name": "Temporary / Draft Documents" },
        { "id": "other_misc", "name": "Other Miscellaneous Documents" }
      ]
    }
  ]
}
```

---

## 3. Classification Workflow (Two-Step: L1 then L2)

You classify in **two short, cheap steps**:

1. **L1** – pick top-level category ID.  
2. **L2** – given L1, pick subcategory ID from its subcategory list.

### 3.1. High-Level Flow

1. Ingest document (possibly run OCR).
2. Extract:
   - filename
   - metadata (MIME type, source system, etc.)
   - text (or best-effort OCR result).
3. **Call L1 classifier prompt** → get `top_level_category_id`.
4. From your taxonomy JSON, get the subcategories for that L1.
5. **Call L2 classifier prompt** (with just the subcategory IDs for that L1) → get `subcategory_id`.
6. Store:
   - `top_level_category_id`
   - `subcategory_id`
   - confidence scores
   - reasoning (optional, for audit/debugging).

### 3.2. Pseudocode

```python
def classify_document(doc_text: str, metadata: dict, filename: str, llm_client) -> dict:
    # 1) L1 classification
    l1_user = build_l1_user_prompt(doc_text, metadata, filename)
    l1_response = llm_client.chat(
        system=L1_SYSTEM_PROMPT,
        user=l1_user,
        temperature=0.0
    )
    l1_json = parse_json(l1_response)
    top_level = l1_json["top_level_category_id"]

    # 2) Get allowed subcategories for this top-level category
    subcats = get_subcategories_for_l1(top_level)  # from taxonomy JSON

    # 3) L2 classification
    l2_user = build_l2_user_prompt(doc_text, metadata, filename, top_level, subcats)
    l2_response = llm_client.chat(
        system=L2_SYSTEM_PROMPT,
        user=l2_user,
        temperature=0.0
    )
    l2_json = parse_json(l2_response)
    subcategory = l2_json["subcategory_id"]

    return {
        "top_level_category_id": top_level,
        "subcategory_id": subcategory,
        "l1_confidence": l1_json.get("confidence"),
        "l2_confidence": l2_json.get("confidence"),
        "l1_reasoning": l1_json.get("reasoning"),
        "l2_reasoning": l2_json.get("reasoning"),
    }
```

---

## 4. L1 Classification Prompt (Top-Level Category)

You wanted **short prompts**, so L1 only knows about the list of top-level IDs.

### 4.1. L1 System Prompt

```text
You are a document classifier.

Goal:
- Read the document.
- Choose EXACTLY ONE top-level category ID from this list:
  - financial_accounting
  - legal_compliance
  - administrative_hr
  - technical_engineering
  - scientific_academic
  - forms_templates
  - business_operations
  - case_customer_files
  - marketing_communications
  - media_images
  - books_long_docs
  - data_files
  - emails_correspondence
  - miscellaneous

Rules:
- Pick the best fit.
- If unclear, use "miscellaneous".
- Output ONLY a JSON object:

{
  "top_level_category_id": "<one_id_from_list>",
  "confidence": "high|medium|low",
  "reasoning": "<max 1–2 short sentences>"
}
```

### 4.2. L1 User Prompt Template

```text
Classify this document into ONE top-level category.

Filename: {{filename}}
Metadata (JSON): {{metadata_json}}

Text:
{{document_text}}
```

Expected model output (example):

```json
{
  "top_level_category_id": "financial_accounting",
  "confidence": "high",
  "reasoning": "The text includes invoice numbers, payment terms, and amounts due."
}
```

---

## 5. L2 Classification Prompt (Subcategory for a Known L1)

After L1, you know the **top-level category**.  
You then pass only the subcategories for that L1.

### 5.1. L2 System Prompt

```text
You are a document classifier.

You already know the document's top-level category (L1).
Now choose EXACTLY ONE subcategory ID from the allowed list.

Rules:
- Use ONLY IDs from the provided list.
- If nothing fits well, use the generic "other_..." subcategory from the list.
- Output ONLY a JSON object:

{
  "subcategory_id": "<one_id_from_list>",
  "confidence": "high|medium|low",
  "reasoning": "<max 1–2 short sentences>"
}
```

### 5.2. L2 User Prompt Template

```text
The document has top-level category (L1):
{{top_level_category_id}}

Allowed subcategory IDs for this L1:
{{list_of_subcategory_ids_and_names_for_this_L1}}

Now pick EXACTLY ONE subcategory.

Filename: {{filename}}
Metadata (JSON): {{metadata_json}}

Text:
{{document_text}}
```

Example for `financial_accounting`:

```text
The document has top-level category (L1):
financial_accounting

Allowed subcategory IDs for this L1:
- invoices
- receipts
- budgets
- payroll
- tax_paperwork
- purchase_orders
- expense_reports
- other_financial

Now pick EXACTLY ONE subcategory.

Filename: invoice_2024_001.pdf
Metadata (JSON): {"source": "upload", "mime_type": "application/pdf"}

Text:
[...document text here...]
```

Expected model output:

```json
{
  "subcategory_id": "invoices",
  "confidence": "high",
  "reasoning": "The document is clearly a sales invoice with a list of items, totals, and payment instructions."
}
```

---

## 6. Notes & Extensions

- Use **low temperature (0.0–0.1)** to keep IDs deterministic.
- Log `confidence` and `reasoning` for:
  - debugging
  - manual review UI for low-confidence docs.
- You can later add:
  - a **query router** that uses the same taxonomy to route user queries to specific indexes (e.g., only invoices, only manuals).
  - separate **vector indexes** per L1 (and maybe per important L2).

This markdown file should be enough to:
- Implement the taxonomy.
- Wire up L1/L2 classification with a single powerful LLM.
- Keep prompts short and maintainable.
