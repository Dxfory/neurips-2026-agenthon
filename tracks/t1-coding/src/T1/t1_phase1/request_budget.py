"""Local resource overrides may lower the controller's request allowance."""
import re


def parse_request_cap(value: str | None, *, default: int = 25) -> int:
    """Accept a canonical ASCII integer within the existing request ceiling.

    Absence preserves the default. An explicit malformed or larger value is
    rejected; it never silently restores a larger allowance.
    """
    if type(default) is not int or not 1 <= default <= 25:
        raise ValueError('Invalid default request cap')
    if value is None:
        return default
    if not isinstance(value, str) or not re.fullmatch(r'[1-9][0-9]?', value):
        raise ValueError('T1_REQUEST_CAP must be an ASCII integer within the request ceiling')
    cap = int(value)
    if cap > default:
        raise ValueError('T1_REQUEST_CAP exceeds the request ceiling')
    return cap
