"""Source locators and explicit point-in-time selection, using only stdlib.

These helpers operate on sources and records supplied by the current caller.
They neither extract facts nor establish that a quotation supports a value.
An event date describes the fact; availability describes when it became known;
a revision date describes a changed version. Only the latter two gate use.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math

__all__ = ["EvidenceLedger", "availability_filter", "asof_vintages"]

_MAX_QUOTE_CHARS = 500


def _text(value, name):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _json_value(value, active=None):
    """Validate strict finite JSON without coercing keys or custom objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if not isinstance(value, (list, dict)):
        raise TypeError("values must contain only JSON types")
    active = set() if active is None else active
    marker = id(value)
    if marker in active:
        raise ValueError("JSON values must not contain cycles")
    active.add(marker)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        children = value.values()
    else:
        children = value
    for child in children:
        _json_value(child, active)
    active.remove(marker)


def _canonical(value):
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class EvidenceLedger:
    """Retain finite JSON facts, short exact source spans, and visible conflicts.

    ``key``, ``unit``, ``period`` and ``source_id`` are caller-defined nonempty
    strings. A span uses Python Unicode character offsets, not byte offsets;
    it must select 1--500 characters with some non-whitespace content. The
    source SHA256 covers the full source's UTF-8 bytes; the source itself is
    not retained. ``quote`` always equals ``source_text[start:end]``.

    A locator records where the caller looked. It does not verify the value,
    source identity, or semantic support. Conflicts are grouped by key/period;
    different units and different JSON values are explicit, never converted
    or overwritten. Values use structural JSON equality, so 12 and 12.0 are
    conservatively distinct. Returned records and reports are defensive copies.
    """

    def __init__(self):
        self._facts = []

    def add(self, key, value, *, unit, period, source_text, source_id, start, end):
        """Add one caller-provided fact and return its record with a local ID.

        Parameters
        ----------
        key, unit, period, source_id : str
            Nonempty fact identity, declared unit/period, and source identity.
        value : JSON value
            Finite scalar, list, or dict with string keys; copied at insertion.
        source_text : str
            Current source text, hashed in full but never stored in full.
        start, end : int
            Half-open Unicode character span for a short exact quotation.

        Returns
        -------
        dict
            id, key, value, unit, period, and source metadata with exact quote.

        Example
        -------
        >>> ledger = EvidenceLedger()
        >>> ledger.add('cash', 12, unit='USD', period='2025',
        ...            source_text='Cash: 12 USD.', source_id='note', start=0, end=13)['value']
        12
        """
        for name, item in [("key", key), ("unit", unit), ("period", period), ("source_id", source_id)]:
            _text(item, name)
        if not isinstance(source_text, str):
            raise TypeError("source_text must be a string")
        if type(start) is not int or type(end) is not int:
            raise TypeError("start and end must be integers, not booleans")
        if not 0 <= start < end <= len(source_text):
            raise ValueError("source span must be nonempty and within source_text")
        if end - start > _MAX_QUOTE_CHARS:
            raise ValueError(f"source quote exceeds {_MAX_QUOTE_CHARS} characters")
        quote = source_text[start:end]
        if not quote.strip():
            raise ValueError("source quote must have non-whitespace content")
        _canonical(value)
        try:
            digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        except UnicodeEncodeError as exc:
            raise ValueError("source_text must be valid UTF-8 text") from exc
        record = {
            "id": len(self._facts) + 1,
            "key": key, "value": deepcopy(value), "unit": unit, "period": period,
            "source": {"id": source_id, "sha256": digest, "start": start, "end": end, "quote": quote},
        }
        self._facts.append(record)
        return deepcopy(record)

    def report(self):
        """Return JSON-safe ``facts`` and grouped ``conflicts`` without mutation."""
        groups = {}
        for record in self._facts:
            groups.setdefault((record["key"], record["period"]), []).append(record)
        conflicts = []
        for (key, period), records in groups.items():
            reasons = []
            if len({_canonical(record["value"]) for record in records}) > 1:
                reasons.append("value")
            if len({record["unit"] for record in records}) > 1:
                reasons.append("unit")
            if reasons:
                conflicts.append({"key": key, "period": period,
                                  "record_ids": [record["id"] for record in records], "reasons": reasons})
        return {"facts": deepcopy(self._facts), "conflicts": conflicts}


def _timestamp(value, name):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO datetime with an explicit timezone") from exc
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be an aware datetime or ISO datetime string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include an explicit timezone; naive dates are not inferred")
    return value.astimezone(timezone.utc)


def _field_time(record, field):
    if field not in record:
        raise ValueError(f"missing required timestamp field: {field}")
    return _timestamp(record[field], field)


def _records(records):
    if isinstance(records, (str, bytes, Mapping)):
        raise TypeError("records must be an iterable of mappings")
    result = []
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("each record must be a mapping")
        result.append(deepcopy(dict(row)))
    return result


def _field_names(available_field, revision_field, event_field=None):
    _text(available_field, "available_field")
    if revision_field is not None:
        _text(revision_field, "revision_field")
    if event_field is not None:
        _text(event_field, "event_field")


def availability_filter(records, *, as_of, available_field="available_at",
                        event_field="event_at", revision_field="revision_at"):
    """Keep records whose availability and enabled revision are <= as_of.

    Parameters
    ----------
    records : iterable of mappings
        Records supplied by the caller. Availability is required in every row.
        Revision is required unless ``revision_field=None``; the optional
        event field, when present, must be an aware timestamp but may be future.
    as_of : aware datetime or str
        Inclusive cutoff, compared in UTC. ISO strings must include a timezone.
    available_field, event_field, revision_field : str or None
        Field names. Event and revision may be disabled with None; availability
        cannot be disabled. Missing required or malformed timestamps raise.

    Returns
    -------
    list of dict
        Defensive copies in input order. This filter cannot reconstruct a
        past value from a later revision; supply all versions to asof_vintages.

    Example
    -------
    >>> availability_filter([{'published': '2025-01-01T00:00:00Z'}],
    ...     as_of='2025-02-01T00:00:00Z', available_field='published', revision_field=None)
    [{'published': '2025-01-01T00:00:00Z'}]
    """
    _field_names(available_field, revision_field, event_field)
    cutoff = _timestamp(as_of, "as_of")
    selected = []
    for row in _records(records):
        available = _field_time(row, available_field)
        revision = available if revision_field is None else _field_time(row, revision_field)
        if event_field is not None and event_field in row:
            _field_time(row, event_field)
        if available <= cutoff and revision <= cutoff:
            selected.append(row)
    return selected


def asof_vintages(records, *, as_of, key_fields, available_field="available_at",
                  revision_field="revision_at"):
    """Select the latest known version per caller-defined key at an instant.

    Parameters
    ----------
    records : iterable of mappings
        Historical versions, including previous values rather than just the
        latest overwritten value. Payload and keys must be finite JSON values;
        enabled timestamp fields may alternatively hold aware datetimes.
    as_of : aware datetime or str
        Inclusive cutoff for both availability and revision, compared in UTC.
    key_fields : sequence of str
        Nonempty, unique names defining a series and observation, e.g. a series
        name plus reference period. Missing keys always raise.
    available_field, revision_field : str or None
        Availability is required; revision is required unless set to None.
        Latest means greatest (revision, availability) UTC tuple. Without a
        revision field, availability alone orders versions. Event dates never
        decide availability or version priority.

    Returns
    -------
    list of dict
        One defensive copy per known key, sorted by canonical JSON key. Exact
        duplicates collapse; equivalent timezone representations also collapse.
        Conflicting records at the same eligible version tuple raise, including
        older tied versions. No row-order tie break silently picks a value.

    Example
    -------
    >>> asof_vintages([{'series': 'A', 'value': 1, 'date': '2025-01-01T00:00:00Z'}],
    ...     as_of='2025-02-01T00:00:00Z', key_fields=['series'],
    ...     available_field='date', revision_field=None)[0]['value']
    1
    """
    _field_names(available_field, revision_field)
    cutoff = _timestamp(as_of, "as_of")
    if isinstance(key_fields, (str, bytes)) or not isinstance(key_fields, Sequence):
        raise TypeError("key_fields must be a sequence of field names")
    if not key_fields:
        raise ValueError("key_fields must not be empty")
    for key in key_fields:
        _text(key, "key field")
    if len(set(key_fields)) != len(key_fields):
        raise ValueError("key_fields must be unique")
    versions = {}
    selected = {}
    for row in _records(records):
        missing = [key for key in key_fields if key not in row]
        if missing:
            raise ValueError(f"missing key fields: {missing}")
        key = _canonical([row[field] for field in key_fields])
        available = _field_time(row, available_field)
        revision = available if revision_field is None else _field_time(row, revision_field)
        normalized = dict(row)
        normalized[available_field] = available.isoformat()
        if revision_field is not None:
            normalized[revision_field] = revision.isoformat()
        payload = _canonical(normalized)
        if available > cutoff or revision > cutoff:
            continue
        version = (revision, available)
        version_key = (key, version)
        if version_key in versions and versions[version_key] != payload:
            raise ValueError(f"conflicting records for key {key} at the same version timestamps")
        versions[version_key] = payload
        if key not in selected or version > selected[key][0]:
            selected[key] = (version, row)
    return [selected[key][1] for key in sorted(selected)]
