# conftest.py
# Override tmp_path to avoid Windows PermissionError on %TEMP%/pytest-of-<user>
import itertools
import os
import shutil
from pathlib import Path

import pytest

_COUNTER = itertools.count()
_TMP_ROOT = Path(__file__).parent / ".pytest_tmp"


@pytest.fixture()
def tmp_path() -> Path:
    """Project-local tmp_path that avoids Windows AppData/Temp permission issues."""
    p = _TMP_ROOT / f"test_{next(_COUNTER)}"
    p.mkdir(parents=True, exist_ok=True)
    yield p
    shutil.rmtree(p, ignore_errors=True)
