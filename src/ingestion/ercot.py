"""
ERCOT data ingestion.

Fetches publicly available grid condition data from ERCOT via gridstatus library:
- Transmission outage reports (TOLAR)
- Planned outage reports
- System outages

gridstatus library: https://docs.gridstatus.io/
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import gridstatus
import pandas as pd

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class GridDocument:
    """A single ingested document from ERCOT, ready for chunking + embedding."""
    source: str                  # URL or feed identifier
    doc_type: str                # "grid_notice" | "outage_report" | "system_status"
    title: str
    content: str
    published_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "title": self.title,
            "content": self.content,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            **self.metadata,
        }


# ── Fetchers ─────────────────────────────────────────────────────────────────

def fetch_grid_notices(limit: int = 50) -> list[GridDocument]:
    """
    Fetch recent ERCOT outage reports using gridstatus library.
    Returns transmission and generation outage data.
    """
    docs = []
    try:
        # Initialize ERCOT client
        ercot = gridstatus.Ercot()

        # Fetch hourly resource outage capacity data for today
        # This returns a DataFrame with outage information
        today = datetime.now().date()
        outages_df = ercot.get_hourly_resource_outage_capacity(date=today)

        if outages_df is None or outages_df.empty:
            print("[ingestion] No outages data returned from gridstatus")
            return docs

        # Limit the number of outages to process
        outages_df = outages_df.head(limit)

        # Convert each outage row to a GridDocument
        for idx, row in outages_df.iterrows():
            # Build a descriptive title from available fields
            outage_type = row.get('Outage Type', 'Unknown')
            equipment = row.get('Equipment Name', row.get('Unit', 'Unknown Equipment'))
            title = f"{outage_type} - {equipment}"

            # Build content from all available fields
            content_parts = []
            for col in outages_df.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    content_parts.append(f"{col}: {value}")
            content = "\n".join(content_parts)

            # Extract metadata
            metadata = {
                "outage_type": str(row.get('Outage Type', '')),
                "equipment": str(equipment),
            }

            # Try to parse publish date if available
            published_at = None
            for date_col in ['Publish Time', 'Start Time', 'Post Time']:
                if date_col in row and pd.notna(row[date_col]):
                    try:
                        published_at = pd.to_datetime(row[date_col])
                        break
                    except:
                        pass

            docs.append(GridDocument(
                source=f"ercot://outages/{idx}",
                doc_type="outage_report",
                title=title,
                content=content,
                published_at=published_at,
                metadata=metadata,
            ))

        print(f"[ingestion] Fetched {len(docs)} outage reports from gridstatus")

    except Exception as e:
        print(f"[ingestion] fetch_grid_notices failed: {e}")
        import traceback
        traceback.print_exc()

    return docs


def fetch_mock_documents() -> list[GridDocument]:
    """
    Returns mock ERCOT documents for local testing without hitting ERCOT servers.
    Replace with real fetchers once you've confirmed the pipeline works end-to-end.
    """
    return [
        GridDocument(
            source="mock://ercot/notice/001",
            doc_type="grid_notice",
            title="ERCOT Issues Weather Watch - Extreme Heat Event",
            content=(
                "ERCOT has issued a Weather Watch for the period June 14-16, 2026 due to "
                "forecast high temperatures exceeding 105°F across the ERCOT footprint. "
                "Peak demand is projected to reach 77,500 MW on June 15. "
                "All available generation resources have been notified. "
                "Transmission constraints on the South Texas Import Path (STIP) may limit "
                "available capacity from the Rio Grande Valley wind generation fleet. "
                "ERCOT is monitoring the situation and may issue a Conservation Appeal."
            ),
            published_at=datetime(2026, 6, 13, 8, 0),
            metadata={"severity": "watch", "region": "statewide"},
        ),
        GridDocument(
            source="mock://ercot/notice/002",
            doc_type="grid_notice",
            title="Transmission Outage - Lamar 345kV Line",
            content=(
                "Planned transmission outage on the Lamar 345kV line (Lamar to Paris) "
                "scheduled for June 14, 2026 06:00 - 18:00 CPT. "
                "This outage will affect the Northeast Texas load pocket. "
                "Estimated impact: 850 MW of import capability reduced. "
                "Switching orders have been coordinated with Oncor. "
                "Local generation dispatch may be required to maintain voltage support."
            ),
            published_at=datetime(2026, 6, 12, 16, 30),
            metadata={"severity": "planned", "region": "northeast_texas"},
        ),
        GridDocument(
            source="mock://ercot/notice/003",
            doc_type="system_status",
            title="ERCOT System Status - Current Operating Conditions",
            content=(
                "Current ERCOT System Status as of June 13, 2026 09:00 CPT:\n"
                "- System frequency: 59.98 Hz (normal operating range)\n"
                "- Total load: 58,240 MW\n"
                "- Wind generation: 14,200 MW (24.4% of load)\n"
                "- Solar generation: 8,900 MW (15.3% of load)\n"
                "- Available reserves: 4,100 MW above minimum\n"
                "- Active constraints: South Texas Import Path (STIP) congested\n"
                "- Houston Hub real-time price: $45.23/MWh\n"
                "- West Hub real-time price: $38.10/MWh\n"
                "No emergency conditions. Normal grid operations."
            ),
            published_at=datetime(2026, 6, 13, 9, 0),
            metadata={"severity": "normal", "region": "statewide"},
        ),
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> Optional[datetime]:
    formats = ["%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None
