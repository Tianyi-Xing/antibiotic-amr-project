#!/usr/bin/env python3
"""
Download public England data for the project:
"Which areas experienced declining antimicrobial resistance after reductions in
broad-spectrum antibiotic prescribing?"

The script supports:
1) Modern public-data panel (recommended):
   - UKHSA E. coli bacteraemia antimicrobial-resistance API
   - OpenPrescribing broad-spectrum prescribing and list sizes
   - Optional GP population, QOF and deprivation resources
2) Historical prescribing backfill from NHSBSA EPD.
3) Generic historical indicator downloads from Fingertips by indicator ID.

Important methodological note:
UKHSA's public resistance metric is a three-month rolling percentage. It is not
independent month-by-month data. Downstream analysis should use quarterly
snapshots or explicitly model serial correlation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import quote, urlencode, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UKHSA_API = "https://api.ukhsa-dashboard.data.gov.uk"
UKHSA_BULK_PAGE = "https://ukhsa-dashboard.data.gov.uk/bulk-downloads"
OPENPRESCRIBING_API = "https://openprescribing.net/api/1.0"
OPENPRESCRIBING_MEASURE = (
    "https://openprescribing.net/measure/broad_spectrum_list/definition/"
)
NHSBSA_EPD_DATASET = (
    "https://opendata.nhsbsa.net/dataset/"
    "english-prescribing-dataset-epd-with-snomed-code"
)
NHSBSA_CKAN = "https://opendata.nhsbsa.net/api/3/action"
FINGERTIPS_API = "https://fingertips.phe.org.uk/api"
FINGERTIPS_PROFILE = "https://fingertips.phe.org.uk/profile/amr-local-indicators"
GP_POP_DEFAULT_PAGE = (
    "https://digital.nhs.uk/data-and-information/publications/statistical/"
    "patients-registered-at-a-gp-practice/january-2026"
)
QOF_DEFAULT_PAGE = (
    "https://digital.nhs.uk/data-and-information/publications/statistical/"
    "quality-and-outcomes-framework-achievement-prevalence-and-exceptions-data/2024-25"
)
IMD_2019_PAGE = (
    "https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019"
)

# OpenPrescribing's current numerator definition for the measure
# "co-amoxiclav, cephalosporins & quinolones prescribing volume".
# Source: https://openprescribing.net/measure/broad_spectrum_list/definition/
BROAD_SPECTRUM_BNF_CODES: tuple[str, ...] = (
    "0501013K0AAAAAA", "0501013K0AAABAB", "0501013K0AAADAD",
    "0501013K0AAAGAG", "0501013K0AAAHAH", "0501013K0AAAIAI",
    "0501013K0AAAJAJ", "0501013K0AAAKAK", "0501013K0AAANAN",
    "0501013K0AAAPAP", "0501013K0BBAAAA", "0501013K0BBALAJ",
    "0501013K0BBANAI", "0501013K0BBAPAH", "0501013K0BBAQAP",
    "0501021A0AAAAAA", "0501021A0AAABAB", "0501021A0AAACAC",
    "0501021A0AAAEAE", "0501021A0AAAGAG", "0501021A0AAAJAJ",
    "0501021A0AAAKAK", "0501021A0BBABAB", "0501021A0BBACAC",
    "0501021A0BBADAE", "0501021A0BBAEAG", "0501021B0AAAAAA",
    "0501021B0AAACAC", "0501021C0AAAAAA", "0501021C0AAACAC",
    "0501021C0BBAAAA", "0501021K0AAAAAA", "0501021K0AAABAB",
    "0501021K0AAACAC", "0501021K0BBAAAA", "0501021K0BBABAB",
    "0501021K0BBACAC", "0501021L0AAAAAA", "0501021L0AAABAB",
    "0501021L0AAACAC", "0501021L0AAADAD", "0501021L0AAAEAE",
    "0501021L0AAAGAG", "0501021L0AAAHAH", "0501021L0AAANAN",
    "0501021L0AAAPAP", "0501021L0BBACAG", "0501021L0BCAAAA",
    "0501021L0BCABAB", "0501021L0BCACAG", "0501021L0BCADAH",
    "0501021L0BCAEAC", "0501021L0BCAFAD", "0501021M0AAAAAA",
    "0501021M0AAABAB", "0501120L0AAAAAA", "0501120L0AAAFAF",
    "0501120L0AAAGAG", "0501120L0AAAJAJ", "0501120L0AABGBG",
    "0501120L0BBAAAA", "0501120L0BBABAF", "0501120L0BBADAJ",
    "0501120L0BBAJBG", "0501120P0AAAAAA", "0501120P0AAABAB",
    "0501120Q0AAAAAA", "0501120Q0BBAAAA", "0501120X0AAAAAA",
    "0501120X0AAABAB", "0501120X0BCAAAB", "0501120Y0AAAAAA",
    "0501120Y0BBAAAA",
)

UKHSA_ECOLI_METRICS: tuple[str, ...] = (
    "e-coli_testing_bacteraemiaPercentResistantRollingMonth",
    "e-coli_testing_bacteraemiaNumberTestedRollingMonth",
    "e-coli_testing_bacteraemiaPercentTestedRollingMonth",
)

VALIDATION_URLS = {
    "UKHSA API root": f"{UKHSA_API}/themes?format=json",
    "UKHSA bulk-download page": UKHSA_BULK_PAGE,
    "OpenPrescribing API documentation": "https://openprescribing.net/api/",
    "OpenPrescribing broad-spectrum definition": OPENPRESCRIBING_MEASURE,
    "NHSBSA EPD dataset page": NHSBSA_EPD_DATASET,
    "NHSBSA CKAN package metadata": (
        f"{NHSBSA_CKAN}/package_show?id="
        "english-prescribing-dataset-epd-with-snomed-code"
    ),
    "Fingertips API documentation": FINGERTIPS_API,
    "Fingertips AMR profile": FINGERTIPS_PROFILE,
    "GP registered population": GP_POP_DEFAULT_PAGE,
    "QOF 2024-25": QOF_DEFAULT_PAGE,
    "English IMD 2019": IMD_2019_PAGE,
}


class DownloadError(RuntimeError):
    """Raised for a recoverable download or schema problem."""


@dataclass(frozen=True)
class UKHSAPath:
    theme: str
    sub_theme: str
    topic: str
    geography_type: str


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "England-AMR-public-data-downloader/1.0 "
                "(academic reproducibility project)"
            )
        }
    )
    return session


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.csv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(out_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 90,
) -> Any:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        preview = response.text[:300]
        raise DownloadError(f"Expected JSON from {response.url}; got {preview!r}") from exc


def flatten_entity_values(payload: Any) -> list[str]:
    """Extract hierarchy entity strings from several plausible DRF response shapes."""
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("results")
        if values is None:
            values = payload.get("data")
        if values is None:
            values = payload.get("items")
        if values is None:
            # Some hierarchy endpoints return one named array.
            list_values = [v for v in payload.values() if isinstance(v, list)]
            values = list_values[0] if len(list_values) == 1 else []
    else:
        values = []

    output: list[str] = []
    for item in values or []:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, dict):
            for key in (
                "name", "value", "slug", "theme", "sub_theme", "topic",
                "geography_type", "geography", "metric",
            ):
                value = item.get(key)
                if isinstance(value, str) and value:
                    output.append(value)
                    break
    return list(dict.fromkeys(output))


def next_url(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("next")
    if isinstance(value, str) and value:
        return value
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        value = pagination.get("next")
        if isinstance(value, str) and value:
            return value
    return None


def get_hierarchy_entities(session: requests.Session, url: str) -> list[str]:
    values: list[str] = []
    current: str | None = url
    seen: set[str] = set()
    while current:
        if current in seen:
            raise DownloadError(f"Pagination loop detected at {current}")
        seen.add(current)
        payload = request_json(session, current, params={"format": "json"})
        values.extend(flatten_entity_values(payload))
        current = next_url(payload)
    return list(dict.fromkeys(values))


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def choose_entity(
    values: Sequence[str],
    preferred: str,
    *,
    contains: Sequence[str] = (),
) -> str:
    if not values:
        raise DownloadError(f"No API entities were returned while looking for {preferred!r}.")
    preferred_norm = normalize_token(preferred)
    for value in values:
        if normalize_token(value) == preferred_norm:
            return value
    for value in values:
        value_norm = normalize_token(value)
        if all(normalize_token(token) in value_norm for token in contains):
            return value
    raise DownloadError(
        f"Could not identify {preferred!r}. Available values include: {list(values)[:30]}"
    )


def api_path(*parts: str) -> str:
    encoded = "/".join(quote(part, safe="-_.~") for part in parts)
    return f"{UKHSA_API}/{encoded}"


def discover_ukhsa_path(
    session: requests.Session,
    requested_geography_type: str,
    *,
    theme_override: str | None = None,
    sub_theme_override: str | None = None,
    topic_override: str | None = None,
) -> UKHSAPath:
    themes = get_hierarchy_entities(session, api_path("themes"))
    theme = theme_override or choose_entity(
        themes, "infectious_disease", contains=("infectious", "disease")
    )

    subthemes_url = api_path("themes", theme, "sub_themes")
    subthemes = get_hierarchy_entities(session, subthemes_url)

    candidates = [sub_theme_override] if sub_theme_override else subthemes
    chosen_subtheme: str | None = None
    chosen_topic: str | None = None
    errors: list[str] = []
    for subtheme in candidates:
        if subtheme is None:
            continue
        try:
            topics = get_hierarchy_entities(
                session,
                api_path("themes", theme, "sub_themes", subtheme, "topics"),
            )
        except Exception as exc:  # continue discovery across other subthemes
            errors.append(f"{subtheme}: {exc}")
            continue
        try:
            topic = topic_override or choose_entity(
                topics, "E-coli", contains=("ecoli",)
            )
        except DownloadError:
            continue
        chosen_subtheme = subtheme
        chosen_topic = topic
        break

    if chosen_subtheme is None or chosen_topic is None:
        detail = "; ".join(errors[:5])
        raise DownloadError(
            "Could not locate the E. coli topic in the UKHSA hierarchy. "
            f"Discovery notes: {detail}"
        )

    geography_types = get_hierarchy_entities(
        session,
        api_path(
            "themes", theme, "sub_themes", chosen_subtheme,
            "topics", chosen_topic, "geography_types",
        ),
    )
    geography_type = choose_entity(
        geography_types,
        requested_geography_type,
        contains=("sub", "integrated", "care", "board")
        if "sub" in requested_geography_type.lower()
        else ("integrated", "care", "board"),
    )
    return UKHSAPath(theme, chosen_subtheme, chosen_topic, geography_type)


def extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("results")
        if records is None:
            records = payload.get("data")
        if records is None:
            records = payload.get("items")
        if records is None:
            records = []
    else:
        records = []
    return [record for record in records if isinstance(record, dict)]


def fetch_ukhsa_metric(
    session: requests.Session,
    path: UKHSAPath,
    geography: str,
    metric: str,
) -> list[dict[str, Any]]:
    url = api_path(
        "themes", path.theme,
        "sub_themes", path.sub_theme,
        "topics", path.topic,
        "geography_types", path.geography_type,
        "geographies", geography,
        "metrics", metric,
    )
    records: list[dict[str, Any]] = []
    current: str | None = url
    first = True
    seen: set[str] = set()
    while current:
        if current in seen:
            raise DownloadError(f"Pagination loop detected at {current}")
        seen.add(current)
        params = {"format": "json", "page_size": 365} if first else None
        payload = request_json(session, current, params=params)
        records.extend(extract_records(payload))
        current = next_url(payload)
        first = False
    for record in records:
        record.setdefault("requested_geography", geography)
        record.setdefault("requested_geography_type", path.geography_type)
        record.setdefault("requested_metric", metric)
    return records


def parse_date_like(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


def find_date_column(df: pd.DataFrame) -> str | None:
    candidates = (
        "date", "reporting_date", "period_end", "period_start", "timestamp",
        "x_axis", "time_period", "timePeriod", "month",
    )
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        low = column.lower()
        if "date" in low or "period" in low or "month" in low:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() > 0.5:
                return column
    return None


def filter_date_range(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    column = find_date_column(df)
    if column is None:
        logging.warning("No date-like column found; date filtering was skipped.")
        return df
    parsed = pd.to_datetime(df[column], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= parsed >= pd.Timestamp(start)
    if end:
        mask &= parsed <= pd.Timestamp(end)
    result = df.loc[mask].copy()
    result["parsed_date"] = parsed.loc[mask].dt.strftime("%Y-%m-%d")
    return result


def download_ukhsa_ecoli(
    session: requests.Session,
    out_dir: Path,
    *,
    start: str,
    end: str,
    geography_type: str,
    metrics: Sequence[str],
    area_limit: int | None,
    theme_override: str | None,
    sub_theme_override: str | None,
    topic_override: str | None,
) -> pd.DataFrame:
    logging.info("Discovering UKHSA E. coli API hierarchy...")
    path = discover_ukhsa_path(
        session,
        geography_type,
        theme_override=theme_override,
        sub_theme_override=sub_theme_override,
        topic_override=topic_override,
    )
    (out_dir / "ukhsa_discovered_path.json").write_text(
        json.dumps(path.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    geographies = get_hierarchy_entities(
        session,
        api_path(
            "themes", path.theme,
            "sub_themes", path.sub_theme,
            "topics", path.topic,
            "geography_types", path.geography_type,
            "geographies",
        ),
    )
    if area_limit is not None:
        geographies = geographies[:area_limit]
    logging.info("Downloading %d areas x %d metrics", len(geographies), len(metrics))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for area_no, geography in enumerate(geographies, start=1):
        for metric in metrics:
            try:
                rows.extend(fetch_ukhsa_metric(session, path, geography, metric))
            except Exception as exc:
                logging.error("UKHSA failed for %s / %s: %s", geography, metric, exc)
                failures.append(
                    {"geography": geography, "metric": metric, "error": str(exc)}
                )
        if area_no % 10 == 0 or area_no == len(geographies):
            logging.info("UKHSA progress: %d/%d areas", area_no, len(geographies))
        time.sleep(0.05)

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "ukhsa_failures.csv", index=False)
    df = filter_date_range(pd.DataFrame(rows), start, end)
    df.to_csv(out_dir / "ukhsa_ecoli_bacteraemia_long.csv", index=False)
    return df


def chunks(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def normalize_openprescribing_rows(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise DownloadError(f"Unexpected OpenPrescribing payload: {type(payload).__name__}")
    return pd.DataFrame(payload)


def download_openprescribing(
    session: requests.Session,
    out_dir: Path,
    *,
    start: str,
    end: str,
    chunk_size: int = 18,
) -> pd.DataFrame:
    raw_frames: list[pd.DataFrame] = []
    for chunk_no, code_chunk in enumerate(chunks(BROAD_SPECTRUM_BNF_CODES, chunk_size), 1):
        logging.info(
            "OpenPrescribing broad-spectrum chunk %d (%d codes)",
            chunk_no,
            len(code_chunk),
        )
        payload = request_json(
            session,
            f"{OPENPRESCRIBING_API}/spending_by_org/",
            params={
                "org_type": "sicbl",
                "code": ",".join(code_chunk),
                "format": "json",
            },
            timeout=180,
        )
        frame = normalize_openprescribing_rows(payload)
        frame["source_chunk"] = chunk_no
        raw_frames.append(frame)
        time.sleep(0.2)

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    raw = filter_date_range(raw, start, end)
    raw.to_csv(out_dir / "openprescribing_broad_spectrum_raw.csv", index=False)

    required = {"date", "row_id", "row_name", "items"}
    missing = required.difference(raw.columns)
    if missing:
        raise DownloadError(
            f"OpenPrescribing broad-spectrum data lack required columns: {sorted(missing)}"
        )

    numeric_columns = [column for column in ("items", "quantity", "actual_cost") if column in raw]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0)
    grouped = (
        raw.groupby(["date", "row_id", "row_name"], as_index=False)[numeric_columns]
        .sum()
        .rename(columns={"items": "broad_spectrum_items"})
    )

    logging.info("Downloading OpenPrescribing Sub-ICB list sizes...")
    list_payload = request_json(
        session,
        f"{OPENPRESCRIBING_API}/org_details/",
        params={"org_type": "ccg", "keys": "total_list_size", "format": "json"},
        timeout=180,
    )
    list_size = normalize_openprescribing_rows(list_payload)
    list_size = filter_date_range(list_size, start, end)
    list_size.to_csv(out_dir / "openprescribing_list_size_raw.csv", index=False)

    # Normalize schema variants observed across API releases.
    rename_map: dict[str, str] = {}
    for candidate in ("org_id", "row_id", "code"):
        if candidate in list_size.columns:
            rename_map[candidate] = "row_id"
            break
    for candidate in ("org_name", "row_name", "name"):
        if candidate in list_size.columns:
            rename_map[candidate] = "list_size_row_name"
            break
    for candidate in ("total_list_size", "list_size", "value"):
        if candidate in list_size.columns:
            rename_map[candidate] = "total_list_size"
            break
    list_size = list_size.rename(columns=rename_map)
    needed = {"date", "row_id", "total_list_size"}
    if not needed.issubset(list_size.columns):
        raise DownloadError(
            "Could not identify date, organisation code and total list size in "
            f"OpenPrescribing org_details columns: {list(list_size.columns)}"
        )
    list_size["total_list_size"] = pd.to_numeric(
        list_size["total_list_size"], errors="coerce"
    )
    list_size = list_size.drop_duplicates(["date", "row_id"])

    merged = grouped.merge(
        list_size[["date", "row_id", "total_list_size"]],
        on=["date", "row_id"],
        how="left",
        validate="many_to_one",
    )
    merged["broad_spectrum_items_per_1000"] = (
        merged["broad_spectrum_items"] / merged["total_list_size"] * 1000
    )
    merged.to_csv(out_dir / "openprescribing_broad_spectrum_sicbl.csv", index=False)

    logging.info("Downloading OpenPrescribing Sub-ICB boundaries...")
    boundary_response = session.get(
        f"{OPENPRESCRIBING_API}/org_location/",
        params={"org_type": "ccg", "format": "json"},
        timeout=180,
    )
    boundary_response.raise_for_status()
    (out_dir / "openprescribing_sicbl_boundaries.geojson").write_bytes(
        boundary_response.content
    )
    return merged


def validate_url(
    session: requests.Session,
    name: str,
    url: str,
    timeout: int = 45,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "url": url, "checked_at": datetime.utcnow().isoformat() + "Z"}
    try:
        response = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
        result.update(
            {
                "ok": response.ok,
                "status_code": response.status_code,
                "final_url": response.url,
                "content_type": response.headers.get("content-type"),
                "content_length": response.headers.get("content-length"),
            }
        )
        next(response.iter_content(chunk_size=1), b"")
        response.close()
    except Exception as exc:
        result.update({"ok": False, "error": str(exc)})
    return result


def command_validate(session: requests.Session, out_dir: Path) -> None:
    rows = [validate_url(session, name, url) for name, url in VALIDATION_URLS.items()]
    (out_dir / "url_validation.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(out_dir / "url_validation.csv", index=False)
    failed = [row for row in rows if not row.get("ok")]
    logging.info("Validated %d URLs; %d failed", len(rows), len(failed))


def safe_filename_from_url(url: str, fallback: str) -> str:
    path_name = Path(url.split("?", 1)[0]).name
    return path_name or fallback


def download_stream(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout: int = 300,
) -> Path:
    partial = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        with partial.open("wb") as fh:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    fh.write(block)
    partial.replace(destination)
    return destination


def discover_resource_link(
    session: requests.Session,
    page_url: str,
    *,
    required_terms: Sequence[str],
    preferred_extensions: Sequence[str] = (".zip", ".csv", ".xlsx", ".ods"),
) -> tuple[str, str]:
    response = session.get(page_url, timeout=90)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    candidates: list[tuple[int, str, str]] = []
    normalized_terms = [term.lower() for term in required_terms]
    for anchor in soup.find_all("a", href=True):
        href = urljoin(response.url, anchor["href"])
        text = " ".join(anchor.get_text(" ", strip=True).split())
        combined = f"{text} {href}".lower()
        combined_normalized = normalize_token(combined)
        if not all(normalize_token(term) in combined_normalized for term in normalized_terms):
            continue
        score = 0
        for rank, extension in enumerate(preferred_extensions[::-1], start=1):
            if href.lower().split("?", 1)[0].endswith(extension):
                score += rank * 10
        if "download" in combined:
            score += 3
        if "files.digital.nhs.uk" in href or "assets.publishing.service.gov.uk" in href:
            score += 5
        candidates.append((score, href, text))
    if not candidates:
        raise DownloadError(
            f"No resource link on {page_url} matched all terms {list(required_terms)}"
        )
    candidates.sort(reverse=True)
    _, href, text = candidates[0]
    return href, text


def command_covariates(
    session: requests.Session,
    out_dir: Path,
    *,
    gp_page: str,
    qof_page: str,
    skip_gp: bool,
    skip_qof: bool,
    skip_imd: bool,
) -> None:
    resources: list[dict[str, str]] = []
    if not skip_gp:
        for label, terms in (
            ("gp_population_5year_age_groups", ("5-year age groups",)),
            ("gp_population_organisation_mapping", ("mapping", "sicbl")),
        ):
            url, link_text = discover_resource_link(session, gp_page, required_terms=terms)
            destination = out_dir / safe_filename_from_url(url, label + ".zip")
            download_stream(session, url, destination)
            resources.append({"label": label, "url": url, "file": destination.name, "link_text": link_text})

    if not skip_qof:
        url, link_text = discover_resource_link(
            session, qof_page, required_terms=("sub icb", "xlsx")
        )
        destination = out_dir / safe_filename_from_url(url, "qof_sub_icb.xlsx")
        download_stream(session, url, destination)
        resources.append({"label": "qof_sub_icb", "url": url, "file": destination.name, "link_text": link_text})

    if not skip_imd:
        url, link_text = discover_resource_link(
            session, IMD_2019_PAGE, required_terms=("file 7",), preferred_extensions=(".csv",)
        )
        destination = out_dir / safe_filename_from_url(url, "imd2019_file7.csv")
        download_stream(session, url, destination)
        resources.append({"label": "imd2019_file7", "url": url, "file": destination.name, "link_text": link_text})

    (out_dir / "covariate_resources.json").write_text(
        json.dumps(resources, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_ckan_packages(session: requests.Session) -> list[dict[str, Any]]:
    """Return current and retired EPD packages when they are available."""
    candidates = (
        "english-prescribing-dataset-epd-with-snomed-code",
        "english-prescribing-data-epd",
    )
    packages: list[dict[str, Any]] = []
    errors: list[str] = []
    for dataset_id in candidates:
        try:
            payload = request_json(
                session,
                f"{NHSBSA_CKAN}/package_show",
                params={"id": dataset_id},
                timeout=90,
            )
            if payload.get("success") and isinstance(payload.get("result"), dict):
                package = dict(payload["result"])
                package["_queried_dataset_id"] = dataset_id
                packages.append(package)
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")
    if not packages:
        raise DownloadError("NHSBSA package_show failed. " + "; ".join(errors))
    if errors:
        logging.warning("Some NHSBSA packages were unavailable: %s", "; ".join(errors))
    return packages

def month_tokens(month: str) -> tuple[str, ...]:
    stamp = pd.Timestamp(month + "-01")
    return (
        stamp.strftime("%Y%m").lower(),
        stamp.strftime("%Y-%m").lower(),
        stamp.strftime("%b %Y").lower(),
        stamp.strftime("%B %Y").lower(),
    )


def select_epd_resource(resources: Sequence[dict[str, Any]], month: str) -> dict[str, Any]:
    tokens = month_tokens(month)
    matches: list[tuple[int, dict[str, Any]]] = []
    for resource in resources:
        combined = " ".join(
            str(resource.get(key, ""))
            for key in ("name", "description", "url", "format")
        ).lower()
        if not any(token in combined for token in tokens):
            continue
        score = 0
        fmt = str(resource.get("format", "")).lower()
        url = str(resource.get("url", "")).lower()
        if fmt == "zip" or url.endswith(".zip"):
            score += 20
        elif fmt == "csv" or url.endswith(".csv"):
            score += 10
        if resource.get("url"):
            score += 2
        matches.append((score, resource))
    if not matches:
        raise DownloadError(f"No NHSBSA EPD resource matched month {month}")
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def unwrap_ckan_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap standard CKAN and NHSBSA's historically nested response shapes."""
    result: Any = payload.get("result") or {}
    for _ in range(3):
        if isinstance(result, dict) and isinstance(result.get("result"), dict):
            result = result["result"]
        else:
            break
    return result if isinstance(result, dict) else {}


def get_epd_fields(
    session: requests.Session,
    resource_id: str,
) -> list[str]:
    payload = request_json(
        session,
        f"{NHSBSA_CKAN}/datastore_search",
        params={"resource_id": resource_id, "limit": 0},
        timeout=120,
    )
    result = unwrap_ckan_result(payload)
    return [
        str(field.get("id"))
        for field in result.get("fields") or []
        if isinstance(field, dict) and field.get("id")
    ]


def choose_column(fields: Sequence[str], candidates: Sequence[str]) -> str:
    by_norm = {normalize_token(field): field for field in fields}
    for candidate in candidates:
        match = by_norm.get(normalize_token(candidate))
        if match:
            return match
    raise DownloadError(
        f"None of the expected columns {list(candidates)} were present. Fields: {list(fields)}"
    )


def query_epd_broad_spectrum(
    session: requests.Session,
    resource_id: str,
    month: str,
    *,
    chunk_size: int = 18,
) -> pd.DataFrame:
    """Query only broad-spectrum rows from one monthly EPD datastore."""
    fields = get_epd_fields(session, resource_id)
    drug_column = choose_column(
        fields,
        ("bnf_code", "BNF_CODE", "bnf_chemical_substance", "BNF_CHEMICAL_SUBSTANCE"),
    )
    item_column = choose_column(fields, ("items", "ITEMS", "total_items", "TOTAL_ITEMS"))
    pco_code_column = choose_column(fields, ("pco_code", "PCO_CODE", "icb_code", "ICB_CODE"))
    pco_name_column = choose_column(fields, ("pco_name", "PCO_NAME", "icb_name", "ICB_NAME"))

    use_chemical_codes = "chemicalsubstance" in normalize_token(drug_column)
    query_codes = (
        tuple(dict.fromkeys(code[:9] for code in BROAD_SPECTRUM_BNF_CODES))
        if use_chemical_codes
        else BROAD_SPECTRUM_BNF_CODES
    )

    def q(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    frames: list[pd.DataFrame] = []
    for code_chunk in chunks(query_codes, chunk_size):
        quoted_codes = ", ".join(
            "'" + code.replace("'", "''") + "'" for code in code_chunk
        )
        sql = (
            f"SELECT {q(pco_code_column)} AS pco_code, "
            f"{q(pco_name_column)} AS pco_name, "
            f"SUM({q(item_column)}) AS broad_spectrum_items "
            f"FROM `{resource_id}` WHERE {q(drug_column)} IN ({quoted_codes}) "
            f"GROUP BY {q(pco_code_column)}, {q(pco_name_column)}"
        )
        payload = request_json(
            session,
            f"{NHSBSA_CKAN}/datastore_search_sql",
            params={"resource_id": resource_id, "sql": sql},
            timeout=300,
        )
        if not payload.get("success"):
            raise DownloadError(f"NHSBSA datastore query failed for {resource_id}")
        records = unwrap_ckan_result(payload).get("records") or []
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["month"] = month + "-01"
            frame["drug_code_field"] = drug_column
            frames.append(frame)
        time.sleep(0.1)
    if not frames:
        return pd.DataFrame(
            columns=[
                "pco_code", "pco_name", "broad_spectrum_items", "month",
                "drug_code_field",
            ]
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["broad_spectrum_items"] = pd.to_numeric(
        combined["broad_spectrum_items"], errors="coerce"
    ).fillna(0)
    return combined.groupby(
        ["month", "pco_code", "pco_name", "drug_code_field"], as_index=False
    )["broad_spectrum_items"].sum()

def command_epd(
    session: requests.Session,
    out_dir: Path,
    months: Sequence[str],
    metadata_only: bool,
    full_download: bool,
) -> None:
    packages = get_ckan_packages(session)
    resources: list[dict[str, Any]] = []
    for package in packages:
        for raw_resource in package.get("resources") or []:
            resource = dict(raw_resource)
            resource["_dataset_id"] = package.get("_queried_dataset_id")
            resources.append(resource)

    selected: list[dict[str, Any]] = []
    filtered_frames: list[pd.DataFrame] = []
    for month in months:
        resource = select_epd_resource(resources, month)
        resource_name = str(resource.get("id") or resource.get("name") or "")
        selected.append(
            {
                "month": month,
                "dataset_id": resource.get("_dataset_id"),
                "name": resource.get("name"),
                "format": resource.get("format"),
                "url": resource.get("url"),
                "resource_id": resource.get("id"),
            }
        )
        if metadata_only:
            continue
        if full_download:
            url = str(resource.get("url") or "")
            if not url:
                raise DownloadError(f"Resource for {month} does not contain a URL")
            extension = Path(url.split("?", 1)[0]).suffix or ".dat"
            destination = out_dir / f"EPD_{month.replace('-', '')}{extension}"
            logging.info("Downloading full NHSBSA EPD %s to %s", month, destination)
            download_stream(session, url, destination, timeout=900)
        else:
            if not resource_name:
                raise DownloadError(f"Resource for {month} lacks a datastore name/id")
            logging.info("Querying filtered broad-spectrum EPD rows for %s", month)
            try:
                filtered_frames.append(
                    query_epd_broad_spectrum(session, resource_name, month)
                )
            except Exception as exc:
                resource_id = str(resource.get("id") or "")
                if resource_id and resource_id != resource_name:
                    logging.warning(
                        "Query by resource name failed for %s; retrying UUID: %s",
                        month,
                        exc,
                    )
                    filtered_frames.append(
                        query_epd_broad_spectrum(session, resource_id, month)
                    )
                else:
                    raise

    (out_dir / "nhsbsa_epd_selected_resources.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if filtered_frames:
        pd.concat(filtered_frames, ignore_index=True).to_csv(
            out_dir / "nhsbsa_epd_broad_spectrum_by_pco.csv", index=False
        )

def command_fingertips(
    session: requests.Session,
    out_dir: Path,
    indicator_ids: Sequence[int],
    child_area_type_id: int | None,
    parent_area_type_id: int | None,
) -> None:
    for indicator_id in indicator_ids:
        params: dict[str, Any] = {"indicator_ids": indicator_id}
        if child_area_type_id is not None:
            params["child_area_type_id"] = child_area_type_id
        if parent_area_type_id is not None:
            params["parent_area_type_id"] = parent_area_type_id
        response = session.get(
            f"{FINGERTIPS_API}/all_data/csv/by_indicator_id",
            params=params,
            timeout=300,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "csv" not in content_type.lower() and not response.text.startswith("Indicator ID"):
            logging.warning(
                "Fingertips indicator %s returned unexpected content-type %s",
                indicator_id,
                content_type,
            )
        destination = out_dir / f"fingertips_indicator_{indicator_id}.csv"
        destination.write_bytes(response.content)


def command_modern(
    session: requests.Session,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    command_validate(session, out_dir)
    download_ukhsa_ecoli(
        session,
        out_dir,
        start=args.start,
        end=args.end,
        geography_type=args.geography_type,
        metrics=args.metric,
        area_limit=args.area_limit,
        theme_override=args.ukhsa_theme,
        sub_theme_override=args.ukhsa_subtheme,
        topic_override=args.ukhsa_topic,
    )
    download_openprescribing(
        session,
        out_dir,
        start=args.start,
        end=args.end,
        chunk_size=args.openprescribing_chunk_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download public England prescribing, AMR and covariate data."
    )
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Check source URLs and save a validation report")

    modern = subparsers.add_parser(
        "modern",
        help="Download the recommended 2021-2026 Sub-ICB prescribing/AMR panel",
    )
    modern.add_argument("--start", default="2021-04-01")
    modern.add_argument("--end", default="2026-04-30")
    modern.add_argument(
        "--geography-type",
        default="Sub-Integrated Care Board",
        help="UKHSA geography type; ICB is an alternative",
    )
    modern.add_argument(
        "--metric",
        action="append",
        default=None,
        help="UKHSA metric API name; may be repeated",
    )
    modern.add_argument("--area-limit", type=int, default=None, help="For smoke tests only")
    modern.add_argument("--ukhsa-theme", default=None)
    modern.add_argument("--ukhsa-subtheme", default=None)
    modern.add_argument("--ukhsa-topic", default=None)
    modern.add_argument("--openprescribing-chunk-size", type=int, default=18)

    covariates = subparsers.add_parser("covariates", help="Download population, QOF and IMD files")
    covariates.add_argument("--gp-page", default=GP_POP_DEFAULT_PAGE)
    covariates.add_argument("--qof-page", default=QOF_DEFAULT_PAGE)
    covariates.add_argument("--skip-gp", action="store_true")
    covariates.add_argument("--skip-qof", action="store_true")
    covariates.add_argument("--skip-imd", action="store_true")

    epd = subparsers.add_parser("epd", help="Download selected historical NHSBSA EPD months")
    epd.add_argument(
        "--month",
        action="append",
        required=True,
        help="Month in YYYY-MM format; repeat for multiple months",
    )
    epd.add_argument(
        "--metadata-only",
        action="store_true",
        help="Resolve and record resource URLs without querying/downloading data",
    )
    epd.add_argument(
        "--full-download",
        action="store_true",
        help="Download full monthly files; default is a filtered broad-spectrum API query",
    )

    fingertips = subparsers.add_parser(
        "fingertips",
        help="Download generic historical Fingertips indicators by numeric ID",
    )
    fingertips.add_argument("--indicator-id", action="append", type=int, required=True)
    fingertips.add_argument("--child-area-type-id", type=int, default=None)
    fingertips.add_argument("--parent-area-type-id", type=int, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out_dir = ensure_dir(Path(args.out))
    session = make_session()

    try:
        if args.command == "validate":
            command_validate(session, out_dir)
        elif args.command == "modern":
            if args.metric is None:
                args.metric = list(UKHSA_ECOLI_METRICS)
            command_modern(session, out_dir, args)
        elif args.command == "covariates":
            command_covariates(
                session,
                out_dir,
                gp_page=args.gp_page,
                qof_page=args.qof_page,
                skip_gp=args.skip_gp,
                skip_qof=args.skip_qof,
                skip_imd=args.skip_imd,
            )
        elif args.command == "epd":
            command_epd(
                session, out_dir, args.month, args.metadata_only, args.full_download
            )
        elif args.command == "fingertips":
            command_fingertips(
                session,
                out_dir,
                args.indicator_id,
                args.child_area_type_id,
                args.parent_area_type_id,
            )
        else:
            parser.error(f"Unknown command {args.command}")
        write_manifest(out_dir)
        logging.info("Completed. Output directory: %s", out_dir.resolve())
        return 0
    except (requests.RequestException, DownloadError, ValueError, KeyError) as exc:
        logging.error("Download failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
