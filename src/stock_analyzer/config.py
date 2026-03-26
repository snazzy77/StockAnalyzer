from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    raw_data: Path = root / "data" / "raw"
    processed_data: Path = root / "data" / "processed"
    models: Path = root / "models"
    reports: Path = root / "reports"
    curves: Path = reports / "curves"
    markdown_reports: Path = reports / "markdown"
    metrics: Path = reports / "metrics"


@dataclass(frozen=True)
class Defaults:
    ticker: str = "NVDA"
    start_date: str = "2019-01-01"
    end_date: str = "2026-01-01"
    horizon_days: int = 5
    random_state: int = 42
