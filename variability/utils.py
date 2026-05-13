import re
from pathlib import Path
from typing import List, Tuple


REAL_RE = re.compile(r"_r\d+i\d+p\d+f\d+")
REAL_CAPTURE_RE = re.compile(r"r\d+i\d+p\d+f\d+")


def find_nc_files(input_dir: str, glob_pattern: str = "*.nc") -> List[Path]:
    p = Path(input_dir)
    return sorted(p.rglob(glob_pattern))


def extract_realization(fname: str) -> str:
    """Extract the realization code (e.g. r35i1p1f1) from a filename."""
    m = REAL_CAPTURE_RE.search(fname)
    return m.group(0) if m else ""


def base_name_without_realization(path: Path) -> str:
    """Return a base filename with the realization code removed.

    This helps grouping files that are the same pattern but different
    realizations (e.g. PatternScalingCoefficients_tas_historical_r01...)
    by removing the `_r000i...` part.
    """
    name = path.name
    base = REAL_RE.sub("", name)
    # normalize repeated underscores
    base = base.replace("__", "_")
    return base
