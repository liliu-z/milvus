"""One explicit workspace shared by the benchmark, collector and analyzers."""
import os
from pathlib import Path

workspace = Path(os.environ.get('COLD_SEARCH_WORK', Path(__file__).resolve().parents[2] / '.cold-search-results')).resolve()
results = workspace / 'results'
results.mkdir(parents=True, exist_ok=True)
