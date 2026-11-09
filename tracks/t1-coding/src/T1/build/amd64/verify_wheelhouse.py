"""Verify downloaded AMD64 wheel artifacts without importing or installing them."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pins(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        if not match:
            raise ValueError(f'Unexpected lock entry in {path.name}')
        name = re.sub(r'[-_.]+', '-', match[1]).lower()
        if name in result:
            raise ValueError('Duplicate locked package')
        result[name] = match[2]
    return result


def compatible_tag(tag):
    python, abi, platform = tag.split('-')
    if platform == 'any':
        return abi == 'none' and bool(set(python.split('.')) & {'py3', 'py313'})
    platforms = platform.split('.')
    for entry in platforms:
        if entry == 'manylinux2014_x86_64':
            continue
        match = re.fullmatch(r'manylinux_2_(\d+)_x86_64', entry)
        if not match or not 5 <= int(match[1]) <= 36:
            return False
    if python == 'cp313' and abi == 'cp313':
        return True
    match = re.fullmatch(r'cp3(\d+)', python)
    return abi == 'abi3' and bool(match) and 2 <= int(match[1]) <= 13


def expand_tags(tags):
    """Expand PEP 425 compressed tag sets such as cp310-abi3-a.b into single tags."""
    result = set()
    for tag in tags:
        python, abi, platform = tag.split('-')
        result.update(f'{p}-{a}-{s}' for p in python.split('.')
                      for a in abi.split('.') for s in platform.split('.'))
    return result


def internal_tags_match(filename, headers):
    """The WHEEL Tag lines must name exactly the filename's tags (maturin compresses them)."""
    tags = [line[5:] for line in headers.splitlines() if line.startswith('Tag: ')]
    try:
        internal = expand_tags(tags)
        declared = expand_tags(['-'.join(filename[:-4].rsplit('-', 3)[1:])])
    except ValueError:
        return False
    return bool(internal) and internal == declared and any(compatible_tag(tag) for tag in internal)


def main():
    metadata_path = ROOT / 'wheel-metadata.json'
    manifest_path = ROOT / 'wheelhouse-manifest.json'
    lock_path = ROOT / 'requirements-linux-amd64.lock'
    source_lock = ROOT / 'requirements-reference.lock'
    direct_path = ROOT.parent.parent / 'requirements-container.txt'
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert metadata['python'] == manifest['python'] == '3.13'
    assert metadata['platform'] == manifest['platform'] == 'linux/x86_64'
    assert manifest['status'] == 'verified'
    assert manifest['metadata_sha256'] == sha256(metadata_path)
    assert manifest['lock_sha256'] == sha256(lock_path)
    versions = pins(lock_path)
    assert versions == pins(source_lock)
    direct = pins(direct_path)
    assert direct and all(versions[k] == v for k, v in direct.items())
    files = metadata['files']
    assert len(files) == len(versions) == manifest['file_count']
    assert manifest['wheels'] == [{**item, 'local_path': 'wheels/' + item['filename']} for item in files]
    assert {re.sub(r'[-_.]+', '-', x['name']).lower(): x['version'] for x in files} == versions
    wheel_dir = ROOT / 'wheels'
    assert not wheel_dir.is_symlink()
    assert {p.name for p in wheel_dir.iterdir()} == {x['filename'] for x in files}
    checked = []
    for item in files:
        filename = item['filename']
        assert re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', filename)
        url = urlsplit(item['url'])
        assert url.scheme == 'https' and url.hostname == 'files.pythonhosted.org'
        assert not (url.username or url.password or url.query or url.fragment)
        assert unquote(url.path.rsplit('/', 1)[-1]) == filename
        wheel = wheel_dir / filename
        assert wheel.is_file() and not wheel.is_symlink()
        assert wheel.stat().st_size == item['size'] and sha256(wheel) == item['sha256']
        filename_tag = '-'.join(filename[:-4].rsplit('-', 3)[1:])
        assert compatible_tag(filename_tag), filename
        with zipfile.ZipFile(wheel) as archive:
            records = [x for x in archive.infolist() if x.filename.endswith('.dist-info/WHEEL')]
            assert len(records) == 1 and records[0].file_size < 65536
            headers = archive.read(records[0]).decode('utf-8')
            assert internal_tags_match(filename, headers), filename
            tags = sorted(expand_tags(line[5:] for line in headers.splitlines() if line.startswith('Tag: ')))
        checked.append({'filename': filename, 'size': item['size'], 'sha256': item['sha256'],
                        'wheel_tags': tags})
    total = sum(x['size'] for x in checked)
    assert total == manifest['total_bytes']
    report = {'status': 'verified', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': '3.13', 'platform': 'linux/x86_64', 'file_count': len(checked),
              'total_bytes': total, 'all_sizes_sha256_match': True,
              'exact_wheel_tree': True, 'all_versions_match_reference_lock': True,
              'direct_pins_preserved': len(direct), 'compatible_filename_and_internal_tags': True,
              'wheel_execution_or_installation': False, 'manifest_sha256': sha256(manifest_path),
              'metadata_sha256': sha256(metadata_path), 'lock_sha256': sha256(lock_path),
              'source_lock_sha256': sha256(source_lock), 'files': checked}
    (ROOT / 'wheelhouse-verification.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, sort_keys=True))


if __name__ == '__main__':
    main()
