#!/usr/bin/env python3
"""Fetch pinned public units for host evaluation; never print task contents.

Git blob SHA-1 is checked against the pinned source index. SHA-256 is checked
after writing and against the previous local manifest on repeat runs. Checker
and reference files stay here; only stage_unit's participant view enters agents.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


REPOSITORY = 'Agenthon-2026/track1-coding-public'
MAX_FILE_BYTES = 64 * 1024 * 1024


def _download(url: str, expected_size: int, attempts: int = 4) -> bytes:
    """Retry transient transport errors and short bodies; the blob hash is still checked later."""
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={'User-Agent': 'track1-phase1-pinned-fetch/1'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(expected_size + 1)
            if len(payload) == expected_size or attempt == attempts - 1:
                return payload
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError('unreachable')


def _entry_path(entry: dict) -> PurePosixPath:
    value = entry.get('path', '')
    if (not isinstance(value, str) or '\\' in value or any(ord(c) < 32 for c in value)
            or any(part in {'', '.', '..'} for part in value.split('/'))):
        raise ValueError('Invalid source index path')
    path = PurePosixPath(value)
    if (len(path.parts) < 3 or path.parts[0] != 'units'
            or not re.fullmatch(r't1-[A-Za-z0-9_-]+', path.parts[1])):
        raise ValueError('Only units/t1-*/... source paths are allowed')
    if entry.get('type') != 'blob' or entry.get('mode') not in {'100644', '100755'}:
        raise ValueError('Only ordinary Git blob files are allowed; symlinks are rejected')
    if not re.fullmatch(r'[a-f0-9]{40}', str(entry.get('sha', ''))):
        raise ValueError('Invalid Git blob hash')
    size = entry.get('size')
    if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
        raise ValueError('Missing or excessive source file size')
    return path


def _safe_target(root: Path, relative: PurePosixPath) -> Path:
    target = root.joinpath(*relative.parts)
    for count in range(len(relative.parts) + 1):
        current = root.joinpath(*relative.parts[:count])
        if current.is_symlink():
            raise ValueError('Symlink in dataset destination')
        if current.exists() and count < len(relative.parts) and not current.is_dir():
            raise ValueError('Dataset parent is not a directory')
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Dataset destination escapes its root')
    return target


def _digests(payload: bytes, entry: dict) -> tuple[str, str]:
    blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
    if len(payload) != entry['size'] or blob != entry['sha']:
        raise ValueError('Source content does not match pinned Git blob and size')
    return blob, hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.download-', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def fetch_sources(index_path: Path, dataset_dir: Path, manifest_path: Path, *, workers: int = 6) -> dict:
    """Download or revalidate every indexed file, then atomically publish provenance."""
    if not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    index_path, dataset_dir, manifest_path = map(Path, (index_path, dataset_dir, manifest_path))
    if any(path.is_symlink() for path in (index_path, dataset_dir, manifest_path)):
        raise ValueError('Index, datasets and manifest must not be symlinks')
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    commit = index.get('commit', '')
    if not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('A pinned 40-character commit is required')
    entries = index.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Source index must contain files')
    paths = [_entry_path(entry) for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths in source index')
    dataset_dir = dataset_dir.resolve()
    for path in paths:
        _safe_target(dataset_dir, PurePosixPath(*path.parts[1:]))
    indexed = {str(PurePosixPath(*path.parts[1:])) for path in paths}
    for existing in dataset_dir.rglob('*'):
        if existing.is_symlink():
            raise ValueError('Symlink in existing dataset tree')
        if not existing.is_dir() and (not existing.is_file()
                or existing.relative_to(dataset_dir).as_posix() not in indexed):
            raise ValueError('Unindexed file in existing dataset tree')
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    prior = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('source_commit', manifest.get('commit')) != commit
                or manifest.get('source_index_sha256') != index_sha256):
            raise ValueError('Existing dataset manifest belongs to a different source index')
        prior = {record['source_path']: record for record in manifest['files']}
        if set(prior) != {str(path) for path in paths}:
            raise ValueError('Existing dataset manifest file set does not match source index')
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def fetch_one(entry: dict) -> tuple[dict, bool]:
        path = _entry_path(entry)
        relative = PurePosixPath(*path.parts[1:])
        target = _safe_target(dataset_dir, relative)
        url = f'https://raw.githubusercontent.com/{REPOSITORY}/{commit}/' + urllib.parse.quote(str(path))
        reused = target.exists()
        if reused:
            if not stat.S_ISREG(target.lstat().st_mode) or target.stat().st_size != entry['size']:
                raise ValueError('Existing dataset file has wrong type or size')
            payload = target.read_bytes()
        else:
            payload = _download(url, entry['size'])
        blob, sha256 = _digests(payload, entry)
        if str(path) in prior and prior[str(path)]['sha256'] != sha256:
            raise ValueError('Dataset SHA-256 differs from previous manifest')
        if not reused:
            _safe_target(dataset_dir, relative)
            _atomic_write(target, payload, int(entry['mode'][-3:], 8))
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
                raise ValueError('Dataset readback SHA-256 mismatch')
        return {'source_path': str(path), 'local_path': str(relative),
                'git_blob_sha': blob, 'sha256': sha256, 'size': entry['size'],
                'mode': entry['mode'], 'source_url': url}, reused

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_one, entries))
    records = sorted((record for record, _ in results), key=lambda record: record['source_path'])
    manifest = {'version': 1, 'repository': REPOSITORY, 'source_commit': commit,
                'source_index_sha256': index_sha256, 'file_count': len(records),
                'unit_count': len({PurePosixPath(record['local_path']).parts[0] for record in records}),
                'total_bytes': sum(record['size'] for record in records), 'files': records,
                'visibility': 'host-evaluator-source; stage participant inputs separately'}
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    reused = sum(was_reused for _, was_reused in results)
    return {'file_count': len(records), 'unit_count': manifest['unit_count'],
            'total_bytes': manifest['total_bytes'], 'downloaded': len(records) - reused, 'reused': reused}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=root / 'evidence/source_files.json')
    parser.add_argument('--datasets', type=Path, default=root / 'datasets')
    parser.add_argument('--manifest', type=Path, default=root / 'evidence/dataset_manifest.json')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(fetch_sources(args.index, args.datasets, args.manifest, workers=args.workers), sort_keys=True))


if __name__ == '__main__':
    main()
