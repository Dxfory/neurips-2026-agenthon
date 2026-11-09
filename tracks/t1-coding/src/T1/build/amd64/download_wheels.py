"""Fetch official wheels as bounded parallel ranges; never install or execute them."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import unquote, urlsplit

CHUNK = 1048576
ROOT = Path(__file__).resolve().parent


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_file(item):
    name = item['filename']
    if not re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', name):
        raise ValueError('Unsafe wheel filename')
    parsed = urlsplit(item['url'])
    if (parsed.scheme != 'https' or parsed.hostname != 'files.pythonhosted.org'
            or parsed.username or parsed.password or parsed.query
            or unquote(parsed.path.rsplit('/', 1)[-1]) != name):
        raise ValueError('Wheel URL must be the exact public official artifact')
    if type(item['size']) is not int or not 0 < item['size'] < 200 * 1024 * 1024:
        raise ValueError('Unexpected wheel size')
    if not re.fullmatch(r'[a-f0-9]{64}', item['sha256']):
        raise ValueError('Invalid official SHA256')
    return item


def part_path(item, begin, end):
    directory = ROOT / 'parts' / item['filename']
    if directory.is_symlink():
        raise ValueError('Part directory must not be a symlink')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{begin}-{end}.part'
    if path.is_symlink():
        raise ValueError('Part must not be a symlink')
    return path


def fetch_part(job):
    item, begin, end = job
    size = end - begin + 1
    target = part_path(item, begin, end)
    if target.exists() and target.stat().st_size == size:
        return size, True
    temporary = target.with_suffix('.partial')
    if temporary.is_symlink():
        raise ValueError('Temporary part must not be a symlink')
    for attempt in range(2):
        try:
            command = ['curl', '-q', '--noproxy', '*', '--silent', '--show-error',
                       '--fail', '--proto', '=https', '--connect-timeout', '15',
                       '--max-time', '90', '--max-filesize', str(size),
                       '--range', f'{begin}-{end}', '--dump-header', '-',
                       '--output', str(temporary), item['url']]
            process = subprocess.run(command, capture_output=True, text=True, timeout=95)
            expected = f'bytes {begin}-{end}/{item["size"]}'
            ranges = re.findall(r'^content-range:\s*(.+?)\s*$', process.stdout, re.I | re.M)
            statuses = re.findall(r'^HTTP/\S+\s+(\d+)', process.stdout, re.M)
            if (process.returncode != 0 or not statuses or statuses[-1] != '206'
                    or ranges != [expected] or not temporary.exists()
                    or temporary.stat().st_size != size):
                raise ValueError('Bounded range transfer did not satisfy status/range/length')
            os.replace(temporary, target)
            return size, False
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if temporary.exists():
                temporary.unlink()
            if attempt == 1:
                raise
            time.sleep(1)


def assemble(item):
    output = ROOT / 'wheels' / item['filename']
    if output.is_symlink():
        raise ValueError('Wheel output must not be a symlink')
    if output.exists():
        if output.stat().st_size != item['size'] or file_hash(output) != item['sha256']:
            raise ValueError('Existing wheel differs from the official hash')
        return
    temporary = output.with_suffix('.assembling')
    if temporary.is_symlink():
        raise ValueError('Temporary wheel must not be a symlink')
    with temporary.open('wb') as stream:
        for begin in range(0, item['size'], CHUNK):
            end = min(item['size'] - 1, begin + CHUNK - 1)
            with part_path(item, begin, end).open('rb') as fragment:
                while chunk := fragment.read(CHUNK):
                    stream.write(chunk)
    if temporary.stat().st_size != item['size'] or file_hash(temporary) != item['sha256']:
        raise ValueError('Assembled wheel does not match official SHA256 and size')
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--budget-sec', type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 60 <= args.budget_sec <= 3600:
        raise ValueError('Expected bounded workers and duration')
    metadata = json.loads((ROOT / 'wheel-metadata.json').read_text())
    files = [validate_file(item) for item in metadata['files']]
    if len({item['filename'] for item in files}) != len(files):
        raise ValueError('Duplicate wheel filenames')
    for path in (ROOT / 'parts', ROOT / 'wheels'):
        if path.is_symlink():
            raise ValueError('Wheelhouse directories must not be symlinks')
        path.mkdir(exist_ok=True)
    jobs = []
    reused_bytes = 0
    for item in files:
        output = ROOT / 'wheels' / item['filename']
        if output.exists():
            assemble(item)
            reused_bytes += item['size']
            continue
        for begin in range(0, item['size'], CHUNK):
            end = min(item['size'] - 1, begin + CHUNK - 1)
            part = part_path(item, begin, end)
            if part.exists() and part.stat().st_size == end - begin + 1:
                reused_bytes += end - begin + 1
            else:
                jobs.append((item, begin, end))
    start = time.monotonic()
    downloaded = 0
    print(json.dumps({'phase': 'range_download', 'wheels': len(files), 'parts_needed': len(jobs),
                      'reused_bytes': reused_bytes, 'total_bytes': sum(x['size'] for x in files)}), flush=True)
    for offset in range(0, len(jobs), args.workers):
        if time.monotonic() - start > args.budget_sec:
            raise TimeoutError('Download budget reached; completed parts retained')
        batch = jobs[offset:offset + args.workers]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(fetch_part, job) for job in batch]):
                count, was_reused = future.result()
                if not was_reused:
                    downloaded += count
        elapsed = time.monotonic() - start
        print(json.dumps({'parts_completed': min(offset + len(batch), len(jobs)),
                          'parts_needed': len(jobs), 'downloaded_bytes': downloaded,
                          'elapsed_sec': round(elapsed, 3),
                          'average_bytes_per_sec': round(downloaded / elapsed)}), flush=True)
    for item in files:
        assemble(item)
    manifest = {'status': 'verified', 'python': metadata.get('python'),
                'platform': metadata.get('platform'), 'file_count': len(files),
                'total_bytes': sum(item['size'] for item in files),
                'elapsed_sec': round(time.monotonic() - start, 3),
                'metadata_sha256': file_hash(ROOT / 'wheel-metadata.json'),
                'lock_sha256': file_hash(ROOT / 'requirements-linux-amd64.lock'),
                'wheels': [{**item, 'local_path': 'wheels/' + item['filename']} for item in files]}
    (ROOT / 'wheelhouse-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'phase': 'complete_sha256_verified', 'wheels': len(files),
                      'bytes': manifest['total_bytes'], 'elapsed_sec': manifest['elapsed_sec']}), flush=True)


if __name__ == '__main__':
    main()
