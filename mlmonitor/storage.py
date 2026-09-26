"""Small atomic records shared by the monitor, dashboard, and optional worker."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile


def finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def read(path, limit=1_000_000):
    if path is None:
        return {}
    try:
        with Path(path).open('rb') as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            return {}
        value = json.loads(data)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, allow_nan=False) + '\n'
    with path.open('a') as handle:
        handle.write(payload)


def field(row, dotted):
    if dotted is None:
        return None
    for part in dotted.split('.'):
        if not isinstance(row, dict):
            return None
        row = row.get(part)
    return row
