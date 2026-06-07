"""Document card rendering helpers."""

from datetime import datetime


def format_file_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_timestamp(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%b %d, %Y · %I:%M %p")
    except (ValueError, TypeError):
        return ts


def render_doc_card_html(doc: dict) -> str:
    """Render a document card as HTML."""
    name = doc.get("doc_id", "Unknown")
    pages = doc.get("total_pages", "?")
    chunks = doc.get("chunks", "?")
    size = format_file_size(doc.get("file_size"))
    indexed_at = format_timestamp(doc.get("indexed_at"))
    status = doc.get("status", "indexed")

    badge_class = {
        "indexed": "dm-badge-indexed",
        "complete": "dm-badge-indexed",
        "processing": "dm-badge-processing",
        "uploading": "dm-badge-uploading",
        "indexing": "dm-badge-processing",
        "error": "dm-badge-error",
        "failed": "dm-badge-error",
    }.get(status, "dm-badge-indexed")

    badge_label = {
        "indexed": "Indexed",
        "complete": "Complete",
        "processing": "Processing",
        "uploading": "Uploading",
        "indexing": "Indexing",
        "error": "Error",
        "failed": "Failed",
    }.get(status, "Indexed")

    return f"""
    <div class="dm-doc-card">
        <div class="dm-doc-card-header">
            <div>
                <div class="dm-doc-name">{name}</div>
                <div class="dm-doc-meta">
                    {pages} pages · {chunks} chunks · {size}
                </div>
                <div class="dm-doc-meta">{indexed_at}</div>
            </div>
            <span class="dm-badge {badge_class}">{badge_label}</span>
        </div>
    </div>
    """
