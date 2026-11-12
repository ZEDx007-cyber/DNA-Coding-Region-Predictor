"""Search and download public nucleotide sequences from NCBI and ENA."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENA_PORTAL = "https://www.ebi.ac.uk/ena/portal/api"
ENA_BROWSER = "https://www.ebi.ac.uk/ena/browser/api"
USER_AGENT = "DNA-Coding-Region-Predictor/1.0"
REQUEST_TIMEOUT = 60
MAX_RECORD_DOWNLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class SequenceHit:
    source: str
    accession: str
    title: str


def _read_url(
    url: str, params: dict[str, str], max_bytes: int | None = None
) -> str:
    request_url = f"{url}?{urlencode(params)}" if params else url
    request = Request(
        request_url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            data = response.read() if max_bytes is None else response.read(max_bytes + 1)
            if max_bytes is not None and len(data) > max_bytes:
                raise ValueError(
                    f"Archive response exceeds the {max_bytes // (1024 * 1024)} MiB "
                    "per-record download limit"
                )
            return data.decode("utf-8")
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        if len(details) > 400:
            details = details[:400] + "..."
        message = f"Archive request failed with HTTP {error.code}"
        if details:
            message += f": {details}"
        raise ValueError(message) from error


def _json_request(url: str, params: dict[str, str]) -> dict[str, Any]:
    try:
        result = json.loads(_read_url(url, params))
    except json.JSONDecodeError as error:
        raise ValueError("The sequence archive returned invalid JSON") from error
    if not isinstance(result, dict):
        raise ValueError("The sequence archive returned an unexpected response")
    return result


def search_sequences(
    source: str, query: str, limit: int = 20, email: str = ""
) -> list[SequenceHit]:
    """Search NCBI Nucleotide or ENA sequence descriptions."""
    query = query.strip()
    if not query:
        raise ValueError("Enter a search term")
    if source not in {"ncbi", "ena"}:
        raise ValueError(f"Unsupported sequence archive: {source}")
    if not 1 <= limit <= 50:
        raise ValueError("Search limit must be between 1 and 50")

    if source == "ncbi":
        return _search_ncbi(query, limit, email.strip())
    return _search_ena(query, limit)


def _search_ncbi(query: str, limit: int, email: str) -> list[SequenceHit]:
    params = {
        "db": "nuccore",
        "term": query,
        "retmax": str(limit),
        "retmode": "json",
        "tool": "DNA_Coding_Region_Predictor",
    }
    if email:
        params["email"] = email
    response = _json_request(f"{NCBI_EUTILS}/esearch.fcgi", params)
    search_result = response.get("esearchresult")
    identifiers = search_result.get("idlist") if isinstance(search_result, dict) else None
    if not isinstance(identifiers, list):
        raise ValueError("NCBI returned an unexpected search response")
    if not identifiers:
        return []

    summary_params = {
        "db": "nuccore",
        "id": ",".join(str(identifier) for identifier in identifiers),
        "retmode": "json",
        "tool": "DNA_Coding_Region_Predictor",
    }
    if email:
        summary_params["email"] = email
    summaries = _json_request(f"{NCBI_EUTILS}/esummary.fcgi", summary_params)
    result = summaries.get("result")
    uids = result.get("uids") if isinstance(result, dict) else None
    if not isinstance(result, dict) or not isinstance(uids, list):
        raise ValueError("NCBI returned unexpected sequence summaries")

    hits = []
    for uid in uids:
        summary = result.get(str(uid))
        if not isinstance(summary, dict):
            continue
        accession = summary.get("accessionversion") or summary.get("caption") or str(uid)
        title = summary.get("title") or accession
        hits.append(SequenceHit("ncbi", str(accession), str(title)))
    return hits


def _search_ena(query: str, limit: int) -> list[SequenceHit]:
    escaped_query = query.replace("\\", "\\\\").replace('"', '\\"')
    response = _read_url(
        f"{ENA_PORTAL}/search",
        {
            "result": "sequence",
            "query": f'description="*{escaped_query}*"',
            "fields": "accession,description",
            "format": "tsv",
            "limit": str(limit),
        },
    )
    rows = csv.DictReader(io.StringIO(response), delimiter="\t")
    if rows.fieldnames is None or not {"accession", "description"}.issubset(rows.fieldnames):
        raise ValueError("ENA returned an unexpected search response")
    return [
        SequenceHit("ena", row["accession"], row.get("description") or row["accession"])
        for row in rows
        if row.get("accession")
    ]


def download_fasta(hit: SequenceHit, email: str = "") -> str:
    """Download a sequence record and return its FASTA text."""
    if not hit.accession.strip():
        raise ValueError("Sequence accession cannot be empty")
    if hit.source == "ncbi":
        params = {
            "db": "nuccore",
            "id": hit.accession,
            "rettype": "fasta",
            "retmode": "text",
            "tool": "DNA_Coding_Region_Predictor",
        }
        if email.strip():
            params["email"] = email.strip()
        fasta = _read_url(
            f"{NCBI_EUTILS}/efetch.fcgi", params, max_bytes=MAX_RECORD_DOWNLOAD_BYTES
        )
    elif hit.source == "ena":
        fasta = _read_url(
            f"{ENA_BROWSER}/fasta/{hit.accession}", {}, max_bytes=MAX_RECORD_DOWNLOAD_BYTES
        )
    else:
        raise ValueError(f"Unsupported sequence archive: {hit.source}")

    if not fasta.lstrip().startswith(">"):
        raise ValueError(f"{hit.accession}: archive response did not contain FASTA data")
    return fasta.rstrip() + "\n"
