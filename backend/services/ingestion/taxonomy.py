from __future__ import annotations

from typing import Any, Dict, List, Optional

DOCUMENT_TAXONOMY: List[Dict[str, Any]] = [
    {
        "id": "financial_accounting",
        "name": "Financial & Accounting",
        "description": "Documents related to money, payments, budgets, and financial records.",
        "subcategories": [
            {"id": "invoices", "name": "Invoices"},
            {"id": "receipts", "name": "Receipts"},
            {"id": "budgets", "name": "Budgets & Financial Plans"},
            {"id": "payroll", "name": "Payroll & Salary Records"},
            {"id": "tax_paperwork", "name": "Tax Documents & Declarations"},
            {"id": "purchase_orders", "name": "Purchase Orders (POs)"},
            {"id": "expense_reports", "name": "Expense Reports"},
            {"id": "other_financial", "name": "Other Financial Documents"},
        ],
    },
    {
        "id": "legal_compliance",
        "name": "Legal & Compliance",
        "description": "Documents related to laws, regulations, contracts, and compliance.",
        "subcategories": [
            {"id": "contracts", "name": "Contracts & Agreements"},
            {"id": "licenses_certificates", "name": "Licenses & Certificates"},
            {"id": "regulations_policies", "name": "Regulations & Policies"},
            {"id": "court_documents", "name": "Court & Legal Case Documents"},
            {"id": "compliance_reports", "name": "Compliance Reports & Audits"},
            {"id": "privacy_gdpr", "name": "Privacy / GDPR Documents"},
            {"id": "permits", "name": "Permits & Authorizations"},
            {"id": "other_legal", "name": "Other Legal / Compliance Documents"},
        ],
    },
    {
        "id": "administrative_hr",
        "name": "Administrative & HR",
        "description": "Internal administration and human resources documentation.",
        "subcategories": [
            {"id": "employee_records", "name": "Employee Records & Files"},
            {"id": "recruitment_docs", "name": "Recruitment & Hiring Documents"},
            {"id": "hr_policies", "name": "HR Policies & Procedures"},
            {"id": "attendance_timesheets", "name": "Attendance & Timesheets"},
            {"id": "internal_memos", "name": "Internal Memos & Notices"},
            {"id": "meeting_minutes", "name": "Meeting Minutes"},
            {"id": "training_materials", "name": "Training & Onboarding Materials"},
            {"id": "other_admin_hr", "name": "Other Administrative / HR Documents"},
        ],
    },
    {
        "id": "technical_engineering",
        "name": "Technical & Engineering",
        "description": "Technical documentation, engineering specs, and manuals.",
        "subcategories": [
            {"id": "machine_manuals", "name": "Machine / Equipment Manuals"},
            {"id": "software_manuals", "name": "Software Manuals & Guides"},
            {"id": "sops", "name": "Standard Operating Procedures (SOPs)"},
            {"id": "technical_specs", "name": "Technical Specifications"},
            {"id": "maintenance_logs", "name": "Maintenance & Service Logs"},
            {"id": "installation_guides", "name": "Installation Guides"},
            {"id": "engineering_drawings", "name": "Engineering Drawings & Schematics"},
            {"id": "other_technical", "name": "Other Technical / Engineering Documents"},
        ],
    },
    {
        "id": "scientific_academic",
        "name": "Scientific & Academic",
        "description": "Research, scientific, and academic documents.",
        "subcategories": [
            {"id": "research_papers", "name": "Research Papers & Articles"},
            {"id": "theses_dissertations", "name": "Theses & Dissertations"},
            {"id": "lab_reports", "name": "Lab Reports & Experiment Logs"},
            {"id": "clinical_studies", "name": "Clinical Studies & Trials"},
            {"id": "conference_papers", "name": "Conference Papers & Posters"},
            {"id": "scientific_summaries", "name": "Scientific Summaries & Reviews"},
            {"id": "other_scientific", "name": "Other Scientific / Academic Documents"},
        ],
    },
    {
        "id": "forms_templates",
        "name": "Forms & Templates",
        "description": "Blank and filled forms, especially administrative or governmental.",
        "subcategories": [
            {"id": "blank_forms", "name": "Blank Forms & Templates"},
            {"id": "filled_forms", "name": "Filled Forms"},
            {"id": "government_forms", "name": "Government / Public Administration Forms"},
            {"id": "application_forms", "name": "Application Forms"},
            {"id": "questionnaires", "name": "Questionnaires & Surveys"},
            {"id": "other_forms", "name": "Other Forms"},
        ],
    },
    {
        "id": "business_operations",
        "name": "Business Operations",
        "description": "Documents about projects, plans, operations, and performance.",
        "subcategories": [
            {"id": "project_plans", "name": "Project Plans & Charters"},
            {"id": "status_reports", "name": "Status & Progress Reports"},
            {"id": "strategy_documents", "name": "Strategy & Planning Documents"},
            {"id": "rfp_rfq", "name": "RFP / RFQ / Tender Documents"},
            {"id": "performance_reports", "name": "KPIs & Performance Reports"},
            {"id": "audit_reports", "name": "Audit & Evaluation Reports"},
            {"id": "presentations", "name": "Presentations & Slide Decks"},
            {"id": "other_business_ops", "name": "Other Business / Operational Documents"},
        ],
    },
    {
        "id": "case_customer_files",
        "name": "Case & Customer Files",
        "description": "Cases, incidents, and customer-related records.",
        "subcategories": [
            {"id": "case_files", "name": "Case Files & Dossiers"},
            {"id": "incident_reports", "name": "Incident & Accident Reports"},
            {"id": "customer_requests", "name": "Customer / Citizen Requests"},
            {"id": "complaints_claims", "name": "Complaints & Claims"},
            {"id": "support_tickets", "name": "Support / Helpdesk Tickets"},
            {"id": "service_logs", "name": "Service & Intervention Logs"},
            {"id": "other_case_customer", "name": "Other Case / Customer Documents"},
        ],
    },
    {
        "id": "marketing_communications",
        "name": "Marketing & Communications",
        "description": "Public-facing and internal communication material.",
        "subcategories": [
            {"id": "announcements", "name": "Announcements & Notices"},
            {"id": "newsletters", "name": "Newsletters & Bulletins"},
            {"id": "press_releases", "name": "Press Releases"},
            {"id": "brochures_flyers", "name": "Brochures & Flyers"},
            {"id": "web_content", "name": "Website & Social Media Content"},
            {"id": "branding_materials", "name": "Branding & Communication Guidelines"},
            {"id": "other_marketing", "name": "Other Marketing / Communication Documents"},
        ],
    },
    {
        "id": "media_images",
        "name": "Media & Images",
        "description": "Non-text or image-heavy documents.",
        "subcategories": [
            {"id": "scanned_documents", "name": "Scanned Documents (all types)"},
            {"id": "invoice_images", "name": "Images / Scans of Invoices"},
            {"id": "form_images", "name": "Images / Scans of Forms"},
            {"id": "photos", "name": "Photos & Photographic Evidence"},
            {"id": "diagrams_charts", "name": "Diagrams, Charts & Drawings"},
            {"id": "screenshots", "name": "Screenshots"},
            {"id": "handwritten_notes", "name": "Handwritten Notes (images)"},
            {"id": "other_media", "name": "Other Media / Image Documents"},
        ],
    },
    {
        "id": "books_long_docs",
        "name": "Books & Long Documents",
        "description": "Large, book-like or very long documents.",
        "subcategories": [
            {"id": "books", "name": "Books & E-books"},
            {"id": "training_manuals", "name": "Training Manuals & Handbooks"},
            {"id": "reference_guides", "name": "Reference Guides"},
            {"id": "policy_handbooks", "name": "Policy & Procedure Handbooks"},
            {"id": "other_long_docs", "name": "Other Long / Book-Like Documents"},
        ],
    },
    {
        "id": "data_files",
        "name": "Data Files & Datasets",
        "description": "Structured data files and datasets.",
        "subcategories": [
            {"id": "csv_files", "name": "CSV Files"},
            {"id": "excel_files", "name": "Excel / Spreadsheet Files"},
            {"id": "json_xml", "name": "JSON / XML / Data Exports"},
            {"id": "databases_extracts", "name": "Database Extracts & Dumps"},
            {"id": "statistical_reports", "name": "Statistical Tables & Reports"},
            {"id": "other_data_files", "name": "Other Data / Dataset Files"},
        ],
    },
    {
        "id": "emails_correspondence",
        "name": "Emails & Correspondence",
        "description": "Communication via email or formal letters.",
        "subcategories": [
            {"id": "emails", "name": "Emails & Email Threads"},
            {"id": "letters", "name": "Official Letters & Correspondence"},
            {"id": "memos_correspondence", "name": "Memos & Internal Correspondence"},
            {"id": "email_with_attachments", "name": "Emails with Attachments"},
            {"id": "other_correspondence", "name": "Other Correspondence"},
        ],
    },
    {
        "id": "miscellaneous",
        "name": "Miscellaneous / Uncategorized",
        "description": "Documents that do not clearly fit in other categories or need manual review.",
        "subcategories": [
            {"id": "mixed_content", "name": "Mixed / Multi-type Content"},
            {"id": "unclear_type", "name": "Unclear Type / To Review"},
            {"id": "temporary_docs", "name": "Temporary / Draft Documents"},
            {"id": "other_misc", "name": "Other Miscellaneous Documents"},
        ],
    },
]

_L1_MAP: Dict[str, Dict[str, Any]] = {entry["id"]: entry for entry in DOCUMENT_TAXONOMY}
_SUBCATEGORY_MAP: Dict[str, Dict[str, Dict[str, str]]] = {}
for category in DOCUMENT_TAXONOMY:
    sub_map: Dict[str, Dict[str, str]] = {}
    for subcat in category.get("subcategories", []) or []:
        sub_map[subcat["id"]] = subcat
    _SUBCATEGORY_MAP[category["id"]] = sub_map


def get_top_level_categories() -> List[Dict[str, Any]]:
    return list(DOCUMENT_TAXONOMY)


def get_top_level_category(category_id: str) -> Optional[Dict[str, Any]]:
    return _L1_MAP.get(str(category_id).strip())


def get_subcategories(category_id: str) -> List[Dict[str, str]]:
    return list((_SUBCATEGORY_MAP.get(str(category_id).strip()) or {}).values())


def get_subcategory(category_id: str, subcategory_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not subcategory_id:
        return None
    return (_SUBCATEGORY_MAP.get(str(category_id).strip()) or {}).get(str(subcategory_id).strip())


def describe_top_level_categories() -> str:
    lines = []
    for entry in DOCUMENT_TAXONOMY:
        lines.append(f"- {entry['id']}: {entry['name']} — {entry['description']}")
    return "\n".join(lines)


def describe_subcategories(category_id: str) -> str:
    subcategories = get_subcategories(category_id)
    if not subcategories:
        return ""
    lines = [f"- {sub['id']}: {sub['name']}" for sub in subcategories]
    return "\n".join(lines)
