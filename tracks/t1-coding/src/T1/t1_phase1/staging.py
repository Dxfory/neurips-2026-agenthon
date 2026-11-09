"""Create a local participant view; never modify the trusted source unit.

The manifest here is local provenance, not the official g0 integrity manifest.
Static Docker COPY mappings are compatibility metadata, not a Docker build.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import tempfile
import tomllib


PROFILE = 'local-participant-v1'
_BLOCKED = re.compile(
    r'^(?:(?:reference|expected|oracle|answer_key|outcome)|'
    r'(?:checks?|checker|tests?|ref|solutions?|dev|reward|pytest_report)(?:$|[._-]))', re.I)
_PROTECTED_APP = {'output', 'agent', 'agent.py', 'main.py', 'solve.py', '__main__.py',
                  'entrypoint.sh', 'code', 'src', 't1_phase1', 'bin', 'lib'}


def _relative(value: str) -> PurePosixPath:
    if (not value or '\\' in value or any(ord(c) < 32 for c in value)
            or any(part in {'.', '..'} for part in value.split('/'))):
        raise ValueError('Unsafe relative input path')
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError('Absolute input path is not allowed')
    return path


def _safe_tree(path: Path, root: Path) -> list[Path]:
    """Validate the entire tree, including files that will later be excluded."""
    relative = path.relative_to(root)
    if any((root / Path(*relative.parts[:index])).is_symlink()
           for index in range(1, len(relative.parts) + 1)):
        raise ValueError('Symlinks are not allowed in participant inputs')
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Input escapes its source directory')
    _relative(path.relative_to(root).as_posix())
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        return [path]
    if not stat.S_ISDIR(mode):
        raise ValueError('Only regular files and directories may be staged')
    result = []
    for child in sorted(path.iterdir()):
        result.extend(_safe_tree(child, root))
    return result


def _allowed(relative: PurePosixPath) -> bool:
    return not any(_BLOCKED.match(part) or part.lower() in {
        'dockerfile', 'manifest.json', 'staging_manifest.json', '__pycache__', '.git'
    } for part in relative.parts)


def participant_file(relative: PurePosixPath) -> bool:
    """The same participant allowlist is rechecked before every agent launch."""
    if relative.as_posix() in {'instruction.md', 'card.toml'}:
        return True
    roots = (('data',), ('environment', 'data'), ('templates',), ('starter',))
    return _allowed(relative) and any(relative.parts[:len(root)] == root for root in roots)


def _destination(value: str, source: Path) -> str:
    if any(char in value for char in ('\\', '$', '*', '?', '[', ']', ',', '\n', '\r')):
        raise ValueError('Docker data destination must be a literal /app path')
    if '..' in value.split('/') or '.' in value.split('/'):
        raise ValueError('Docker data destination contains traversal')
    target = PurePosixPath(value)
    if source.is_file() and value.endswith('/'):
        target /= source.name
    if (not target.is_absolute() or len(target.parts) < 3 or target.parts[1] != 'app'
            or target.parts[2] in _PROTECTED_APP):
        raise ValueError('Docker data mapping would replace a protected path')
    return target.as_posix()


def data_mounts(source: Path) -> list[dict]:
    """Read only literal COPY data/... mappings; RUN/ADD are never executed."""
    source = Path(source).resolve()
    mounts = []
    for relative in ('Dockerfile', 'environment/Dockerfile'):
        dockerfile = source / relative
        if not dockerfile.exists() and not dockerfile.is_symlink():
            continue
        _safe_tree(dockerfile, source)
        for line in dockerfile.read_text().splitlines():
            match = re.match(r'^\s*COPY\s+(.+)$', line, re.I)
            if not match:
                continue
            text = match.group(1).strip()
            try:
                operands = json.loads(text) if text.startswith('[') else shlex.split(text)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError('Unsupported Docker COPY syntax') from exc
            if not any(isinstance(x, str) and x.startswith('data') for x in operands[:-1]):
                continue
            if (len(operands) != 2 or not all(isinstance(x, str) for x in operands)
                    or any(c in operands[0] for c in '$*?[]') or operands[0].startswith('--')):
                raise ValueError('Only a single literal Docker data COPY is supported')
            src_relative = _relative(operands[0].rstrip('/'))
            if src_relative.parts[0] != 'data' or not _allowed(src_relative):
                raise ValueError('Docker COPY points outside participant data')
            src = dockerfile.parent / Path(src_relative)
            safe_files = _safe_tree(src, source)
            if src.is_dir() and operands[1] in {'/app', '/app/'}:
                # Legacy COPY-to-root semantics: bind each allowed file, never /app.
                candidates = [(path, '/app/' + path.relative_to(src).as_posix())
                              for path in safe_files
                              if _allowed(PurePosixPath(path.relative_to(src).as_posix()))]
            else:
                candidates = [(src, operands[1])]
            for path, destination in candidates:
                mapping = {'source_relative': path.relative_to(source).as_posix(),
                           'destination': _destination(destination, path)}
                if mapping not in mounts:
                    mounts.append(mapping)
    destinations = [m['destination'] for m in mounts]
    if len(destinations) != len(set(destinations)):
        raise ValueError('Conflicting Docker data destinations')
    return mounts


# The benchmark's standard training-data banner is canary metadata, like the canary line it stands beside. Left in
# place as a heading once that line goes, it would become the parent section of the task text after it.
_CANARY_BANNER = re.compile(r'^\s*(?:<!--)?[\s#>*_]*BENCHMARK\s+DATA\s+SHOULD\s+NEVER\s+APPEAR\s+IN\s+TRAINING\s+'
                            r'CORPORA[\s.!*_]*(?:-->)?\s*$', re.I)


def canary_banner(line: str) -> bool:
    """Whether a whole line is the standard banner (as text, a markdown heading, emphasis or an HTML comment)."""
    return _CANARY_BANNER.match(line) is not None


def _canary_values(value) -> list[str]:
    if not isinstance(value, dict):
        return []
    result = []
    for key, item in value.items():
        if 'canary' in key.lower() and isinstance(item, str) and item:
            result.append(item)
        result.extend(_canary_values(item))
    return result


def stage_unit(source: Path, target: Path) -> dict:
    """Stage a fresh, allowlisted participant directory and return its manifest.

    An existing empty target is accepted. Any nonempty target, symlink, invalid
    TOML after sanitization, or unsupported data mapping fails closed.
    """
    source, target = Path(source), Path(target)
    if source.is_symlink() or target.is_symlink():
        raise ValueError('Source and target must not be symlinks')
    source, target = source.resolve(), target.resolve()
    if not source.is_dir():
        raise ValueError('Source unit must be a directory')
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Source and target must not contain one another')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('Refusing to overwrite a nonempty staging target')
    files = []
    for name in ('instruction.md', 'card.toml'):
        path = source / name
        _safe_tree(path, source)
        if not path.is_file():
            raise ValueError('Required task metadata must be regular files')
        files.append(path)
    for name in ('environment/data', 'data', 'templates', 'starter'):
        path = source / name
        if path.exists() or path.is_symlink():
            files.extend(p for p in _safe_tree(path, source)
                         if _allowed(PurePosixPath(p.relative_to(source).as_posix())))
    mounts = data_mounts(source)
    card = (source / 'card.toml').read_text()
    canaries = _canary_values(tomllib.loads(card))
    cleaned = {}
    removed = 0
    for name in ('instruction.md', 'card.toml'):
        lines = (source / name).read_text().splitlines(keepends=True)
        kept = [line for line in lines if 'canary' not in line.lower()
                and not any(value in line for value in canaries) and not canary_banner(line)]
        removed += len(lines) - len(kept)
        cleaned[name] = ''.join(kept).encode()
    # Invalid multiline fields fail instead of leaking the remaining canary text.
    tomllib.loads(cleaned['card.toml'].decode())
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.participant-', dir=target.parent) as temporary:
        staging = Path(temporary) / 'view'
        staging.mkdir()
        hashes = {}
        for path in sorted(set(files)):
            relative = path.relative_to(source).as_posix()
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative in cleaned:
                destination.write_bytes(cleaned[relative])
            else:
                shutil.copyfile(path, destination)
            destination.chmod(0o644)
            with destination.open('rb') as stream:
                hashes[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
        for mapping in mounts:
            if not (staging / mapping['source_relative']).exists():
                raise ValueError('Docker mapping has no participant-safe input')
        manifest = {'version': 1, 'profile': PROFILE, 'files': hashes,
                    'data_mounts': mounts, 'removed_canary_lines': removed}
        (staging / 'staging_manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.replace(staging, target)
    return manifest
