"""Candidate discovery with dated, auditable SEC evidence; no paid API needed."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BENCHMARKS = {
    "SPY": "broad_market", "QQQ": "growth_market", "SOXX": "semiconductors",
    "XLK": "technology", "XLE": "energy", "XLU": "utilities",
    "XLI": "industrials", "XLF": "financials", "XLV": "healthcare",
    "XLY": "consumer_discretionary", "XLP": "consumer_staples",
    "XLC": "communication_services", "XLRE": "real_estate", "XLB": "materials",
}


class SecClient:
    def __init__(self, user_agent: str, cache: Path):
        if "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must identify your app and a real contact email.")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.last_request = 0.0

    def get(self, url: str) -> str:
        key = self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".txt")
        # Refresh metadata daily; archived filings are immutable enough to cache too.
        if key.exists() and time.time() - key.stat().st_mtime < 86400:
            return key.read_text(encoding="utf-8")
        for attempt in range(3):
            time.sleep(max(0.0, 0.25 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            response = self.session.get(url, timeout=30)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            key.write_text(response.text, encoding="utf-8")
            return response.text
        raise RuntimeError(f"SEC request failed: {url}")

    def json(self, url: str):
        return json.loads(self.get(url))


def company_alias(title: str) -> str:
    value = re.sub(r"\s+(?:incorporated|inc\.?|corporation|corp\.?|ltd\.?|plc|co\.?)\b.*$", "", title, flags=re.I)
    return value.strip(" ,./")


def extract_mentions(html: str, companies: list[dict], subject: str,
                     filing_date: str, source_url: str) -> list[dict]:
    """Conservative name matching. Context labels do NOT imply direction or verification."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    plain = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    sentences = re.split(r"(?<=[.!?])\s+", plain)
    cues = {
        "competitor_context": r"competit\w*",
        "supplier_context": r"suppl\w*|manufactur\w*|foundr\w*",
        "customer_context": r"customer\w*|purchas\w*",
        "partner_context": r"partner\w*|collaborat\w*",
    }
    result = []
    for company in companies:
        ticker = company["ticker"].upper().replace(".", "-")
        alias = company_alias(company["title"])
        if ticker == subject or len(alias) < 5:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I)
        for sentence in sentences:
            match = pattern.search(sentence)
            if not match:
                continue
            # Limit cue matching to nearby text, not an entire filing table.
            evidence = sentence[max(0, match.start() - 300):match.end() + 300]
            labels = [name for name, regex in cues.items() if re.search(regex, evidence, re.I)]
            if not labels:
                continue
            result.append({"ticker": ticker, "company": company["title"],
                           "subject": subject, "source": "sec_mention",
                           "relationship": ",".join(labels), "status": "unverified_candidate",
                           "filing_date": filing_date, "known_since": filing_date,
                           "source_url": source_url, "evidence": evidence})
            break
    return result


def sec_candidates(ticker: str, cutoff: str, client: SecClient,
                   max_filings: int = 4) -> list[dict]:
    companies = list(client.json("https://www.sec.gov/files/company_tickers.json").values())
    company = next((c for c in companies if c["ticker"].replace(".", "-") == ticker), None)
    if company is None:
        raise ValueError(f"No SEC CIK mapping for {ticker}")
    cik = int(company["cik_str"])
    submissions = client.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    frames = [pd.DataFrame(submissions["filings"]["recent"])]
    # Follow historical pages so an older training cutoff is respected.
    for item in submissions["filings"].get("files", []):
        if item.get("filingFrom", "") <= cutoff:
            frames.append(pd.DataFrame(client.json("https://data.sec.gov/submissions/" + item["name"])))
    filings = pd.concat(frames, ignore_index=True)
    filings = filings[(filings["filingDate"] < cutoff) & filings["form"].isin(["10-K", "10-Q", "20-F"])]
    filings = filings.sort_values("filingDate", ascending=False).head(max_filings)
    if filings.empty:
        raise ValueError(f"No eligible SEC filings before {cutoff}")
    rows = []
    for filing in filings.to_dict("records"):
        accession = filing["accessionNumber"].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filing['primaryDocument']}"
        rows.extend(extract_mentions(client.get(url), companies, ticker, filing["filingDate"], url))
    return rows


def discover(ticker: str, cutoff: str, sec_user_agent: str | None,
             cache: Path, extra: list[str], use_yahoo: bool = True) -> tuple[list[dict], list[str]]:
    rows = [{"ticker": t, "source": "benchmark", "relationship": role} for t, role in BENCHMARKS.items()]
    rows.extend({"ticker": t.upper(), "source": "manual", "relationship": "user_candidate"} for t in extra)
    warnings = []
    if use_yahoo:
        try:
            import yfinance as yf
            key = yf.Ticker(ticker).info.get("industryKey")
            if not key:
                raise ValueError("Yahoo returned no industryKey")
            peers = yf.Industry(key).top_companies
            if peers is None or peers.empty:
                raise ValueError("Yahoo returned no industry peers")
            rows.extend({"ticker": str(t).upper(), "source": "yahoo_industry",
                         "relationship": key} for t in peers.index)
            warnings.append("Yahoo industry membership is a current snapshot, not historical membership; survivorship bias remains.")
        except Exception as exc:
            warnings.append(f"Yahoo industry discovery unavailable: {exc}")
    if sec_user_agent:
        try:
            evidence = sec_candidates(ticker, cutoff, SecClient(sec_user_agent, cache))
            rows.extend(evidence)
            if not evidence:
                warnings.append("No named SEC relationship candidates matched; this does not mean no relationships exist.")
        except Exception as exc:
            warnings.append(f"SEC discovery unavailable: {exc}")
    else:
        warnings.append("SEC discovery skipped: set SEC_USER_AGENT to an app name and real contact email.")
    now = datetime.now(timezone.utc).isoformat()
    rows = [dict(row, retrieved_at=now) for row in rows if row["ticker"] != ticker]
    return rows, warnings
