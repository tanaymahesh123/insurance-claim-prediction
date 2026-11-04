"""Load the project's raw CSV without changing it."""

from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_PATH


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a CSV into memory; the source file is opened read-only."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV not found: {source}")
    try:
        return pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse CSV {source}: {exc}") from exc
