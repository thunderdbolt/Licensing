
"""
flexlog_parser.py

Parser utilities for FLEXlm / FlexNet-style vendor daemon debug logs.
Designed for Streamlit ingestion (file uploader -> parse -> dataframe).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Iterable, Optional, Dict, Any, List, Tuple

# -----------------------------
# Regex patterns (tuned to your sample)
# -----------------------------

# Typical line prefix: "21:10:19 (slbsls) MESSAGE..."
LINE_PREFIX_RE = re.compile(
    r'^\s*(?P<hms>\d{1,2}:\d{2}:\d{2})\s+\((?P<daemon>[^)]+)\)\s+(?P<msg>.*)\s*$'
)

# Date anchor lines that appear occasionally (we use these to anchor the day)
# Example: "Reread time: Sun Jun 01 2025 00:09:31 Central Daylight Time"
REREAD_TIME_RE = re.compile(
    r'Reread time:\s+(?P<dow>\w+)\s+(?P<mon>\w+)\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})\s+(?P<hms>\d{1,2}:\d{2}:\d{2})'
)

# Example: "Time: Sun Jun 01 2025 03:06:51 Central Daylight Time"
GEN_TIME_RE = re.compile(
    r'\bTime:\s+(?P<dow>\w+)\s+(?P<mon>\w+)\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})\s+(?P<hms>\d{1,2}:\d{2}:\d{2})'
)

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

# Reserving features:
# "RESERVING 5 compmodule/server2  licenses for USER_GROUP WAN"
RESERVING_RE = re.compile(
    r'^RESERVING\s+(?P<count>\d+)\s+(?P<feature>[^/]+)/(?P<server>.+?)\s+licenses?\s+for\s+USER_GROUP\s+(?P<user_group>\S+)\s*$'
)

# Reservation ignored:
# "RESERVATION of water for USER_GROUP WAN exceeds license count - ignored."
RESERVATION_IGNORED_RE = re.compile(
    r'^RESERVATION\s+of\s+(?P<feature>\S+)\s+for\s+USER_GROUP\s+(?P<user_group>\S+)\s+exceeds\s+license\s+count\s+-\s+ignored\.\s*$'
)

# MAX lines:
# "MAX 1 USER kyuu for gui"
MAX_RE = re.compile(
    r'^MAX\s+(?P<max>\d+)\s+USER\s+(?P<user>\S+)\s+for\s+(?P<feature>\S+)\s*$'
)

# Include/exclude groups:
INCLUDE_EXCLUDE_RE = re.compile(
    r'^(?P<action>INCLUDE|EXCLUDE)\s+USER_GROUP\s+(?P<user_group>\S+)\s+in\s+(?P<scope>.*)$'
)

# Usage events (your file uses USER-1 / USER-2, but many logs include user@host; we keep it flexible)
IN_OUT_RE = re.compile(
    r'^(?P<event>IN|OUT):\s+"(?P<feature>[^"]+)"\s+(?P<client>.+?)\s*$'
)

# DENIED events can have an optional quoted version then a reason in parentheses
# Examples:
#   DENIED: "gui" USER-2  "v2022.06"(License server system does not support this version of this feature. (-25,334))
#   DENIED: "gui" USER-2  (Checkout exceeds MAX specified in options file. (-87,353))
DENIED_RE = re.compile(
    r'^DENIED:\s+"(?P<feature>[^"]+)"\s+(?P<client>\S+)\s*(?P<tail>.*)$'
)
DENIED_VERSION_RE = re.compile(r'"(?P<version>v[^"]+)"\s*\((?P<reason>.*)\)\s*$')
DENIED_REASON_ONLY_RE = re.compile(r'^\((?P<reason>.*)\)\s*$')

# UNSUPPORTED events
# Example:
#   UNSUPPORTED: "pe_prosper_olgas_3p" (PORT_AT_HOST_PLUS   ) USER  (No such feature exists. (-5,346))
UNSUPPORTED_RE = re.compile(
    r'^UNSUPPORTED:\s+"(?P<feature>[^"]+)"\s+\((?P<request>[^)]*)\)\s+(?P<client>\S+)\s+\((?P<reason>.*)\)\s*$'
)


def _parse_explicit_date(line: str) -> Optional[date]:
    """Return date if the line contains a strong date anchor (Reread time / Time:)."""
    m = REREAD_TIME_RE.search(line) or GEN_TIME_RE.search(line)
    if not m:
        return None
    mon = MONTHS.get(m.group("mon")[:3], None)
    if mon is None:
        return None
    return date(int(m.group("year")), mon, int(m.group("day")))


def _hms_to_time(hms: str) -> Tuple[int, int, int]:
    parts = hms.split(":")
    return int(parts[0]), int(parts[1]), int(parts[2])


def parse_flex_debug_log(lines: Iterable[str]) -> Dict[str, Any]:
    """
    Parse a vendor daemon debug log into structured tables.

    Returns a dict with:
      - events: list[dict]   (IN/OUT/DENIED/UNSUPPORTED + anything else that matches)
      - reservations: list[dict]
      - max_rules: list[dict]
      - group_rules: list[dict]  (INCLUDE/EXCLUDE rules seen in log)
      - meta: dict  (observations, warnings)
    """
    events: List[Dict[str, Any]] = []
    reservations: List[Dict[str, Any]] = []
    max_rules: List[Dict[str, Any]] = []
    group_rules: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"date_anchors": [], "warnings": []}

    current_date: Optional[date] = None
    last_dt: Optional[datetime] = None

    for raw in lines:
        line = raw.rstrip("\n")

        # Update date anchor if present (doesn't require a prefix)
        d = _parse_explicit_date(line)
        if d is not None:
            current_date = d
            meta["date_anchors"].append({"date": str(d), "line": line[:200]})
            # Don't 'continue' because this line might also include other info; but in your sample it doesn't matter.

        m = LINE_PREFIX_RE.match(line)
        if not m:
            continue

        hms = m.group("hms")
        daemon = m.group("daemon")
        msg = m.group("msg")

        if current_date is None:
            # If a log chunk starts without anchors, we still capture "relative" timestamps.
            # Streamlit can show these (hour/min/sec) while user chooses a base date.
            base_dt = None
            dt = None
        else:
            hh, mm, ss = _hms_to_time(hms)
            dt = datetime(current_date.year, current_date.month, current_date.day, hh, mm, ss)

            # Handle midnight rollover if time goes backwards significantly
            if last_dt is not None and dt < last_dt and (last_dt - dt) > timedelta(hours=6):
                dt = dt + timedelta(days=1)
                current_date = dt.date()

            last_dt = dt

        # --- Reservations
        rm = RESERVING_RE.match(msg)
        if rm:
            reservations.append({
                "timestamp": dt,
                "daemon": daemon,
                "feature": rm.group("feature"),
                "server": rm.group("server").strip(),
                "reserved_count": int(rm.group("count")),
                "user_group": rm.group("user_group"),
                "raw": msg
            })
            continue

        rim = RESERVATION_IGNORED_RE.match(msg)
        if rim:
            reservations.append({
                "timestamp": dt,
                "daemon": daemon,
                "feature": rim.group("feature"),
                "server": None,
                "reserved_count": None,
                "user_group": rim.group("user_group"),
                "status": "ignored_exceeds_license_count",
                "raw": msg
            })
            continue

        # --- MAX rules
        mmx = MAX_RE.match(msg)
        if mmx:
            max_rules.append({
                "timestamp": dt,
                "daemon": daemon,
                "user": mmx.group("user"),
                "feature": mmx.group("feature"),
                "max": int(mmx.group("max")),
                "raw": msg
            })
            continue

        # --- Group include/exclude rules
        g = INCLUDE_EXCLUDE_RE.match(msg)
        if g:
            group_rules.append({
                "timestamp": dt,
                "daemon": daemon,
                "action": g.group("action"),
                "user_group": g.group("user_group"),
                "scope": g.group("scope").strip(),
                "raw": msg
            })
            continue

        # --- IN/OUT
        io = IN_OUT_RE.match(msg)
        if io:
            events.append({
                "timestamp": dt,
                "daemon": daemon,
                "event": io.group("event"),
                "feature": io.group("feature"),
                "client": io.group("client").strip(),
                "reason": None,
                "version": None,
                "raw": msg
            })
            continue

        # --- DENIED
        de = DENIED_RE.match(msg)
        if de:
            tail = de.group("tail").strip()
            version = None
            reason = None

            mv = DENIED_VERSION_RE.match(tail)
            if mv:
                version = mv.group("version")
                reason = mv.group("reason").strip()
            else:
                mr = DENIED_REASON_ONLY_RE.match(tail)
                if mr:
                    reason = mr.group("reason").strip()
                else:
                    # Sometimes DENIED formats vary; keep tail as reason-ish.
                    reason = tail or None

            events.append({
                "timestamp": dt,
                "daemon": daemon,
                "event": "DENIED",
                "feature": de.group("feature"),
                "client": de.group("client"),
                "reason": reason,
                "version": version,
                "raw": msg
            })
            continue

        # --- UNSUPPORTED
        un = UNSUPPORTED_RE.match(msg)
        if un:
            events.append({
                "timestamp": dt,
                "daemon": daemon,
                "event": "UNSUPPORTED",
                "feature": un.group("feature"),
                "client": un.group("client"),
                "request_type": un.group("request").strip(),
                "reason": un.group("reason").strip(),
                "version": None,
                "raw": msg
            })
            continue

    return {
        "events": events,
        "reservations": reservations,
        "max_rules": max_rules,
        "group_rules": group_rules,
        "meta": meta,
    }
