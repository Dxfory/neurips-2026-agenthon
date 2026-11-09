#!/usr/bin/env python3
"""Test real Docker outer-timeout cleanup with a fixed, offline sleep program.

Uses an already prepared synthetic participant directory. No House client,
generated task program, image build, or official task evaluation is performed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t1_phase1.docker_runner import build_agent_command, resource_profile, run_container


OUTER_TIMEOUT_SEC = 10
SLEEP_SEC = 120
START_MARKER = 'T1_TIMEOUT_FIXTURE_SLEEP_STARTED'
SLEEP_PROGRAM = (
    f'import time; print({START_MARKER!r}, flush=True); time.sleep({SLEEP_SEC})'
)


def execute(args, run_dir: Path, report: dict) -> None:
    participant = args.participant.resolve(strict=True)
    card = tomllib.loads((participant / 'card.toml').read_text())
    if card.get('id') != 'fixture-only-addition':
        raise ValueError('Use participant inputs prepared by integration_fixture.py')

    inspect = subprocess.run(
        ['docker', '--context', args.context, 'image', 'inspect', args.image],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if inspect.returncode != 0:
        raise RuntimeError('Local image inspection failed; no container launched')
    image = json.loads(inspect.stdout)[0]
    if image.get('Os') != 'linux' or image.get('Architecture') != args.platform.split('/')[-1]:
        raise ValueError('Local image does not match the requested platform')
    if image.get('Config', {}).get('Entrypoint'):
        raise ValueError('The fixed sleep fixture requires an image without ENTRYPOINT')
    image_id = image['Id']
    report['image_id'] = image_id

    output = run_dir / 'output'
    output.mkdir(mode=0o777)
    output.chmod(0o777)
    command = build_agent_command(
        image_id, participant, output, context=args.context, platform=args.platform,
        timeout_sec=SLEEP_SEC, model_env=False, cpus=args.cpus, memory=args.memory,
    )
    image_index = command.index(image_id)
    command[image_index:image_index] = ['--pull=never']
    image_index = command.index(image_id)
    command[image_index + 1:] = ['python', '-c', SLEEP_PROGRAM]
    if '--network=none' not in command:
        raise RuntimeError('The sleep fixture must be offline')
    container_name = command[command.index('--name') + 1]
    report['container_name'] = container_name
    report['process'] = process = run_container(command, timeout_sec=OUTER_TIMEOUT_SEC)

    # A successful empty listing proves both daemon responsiveness and absence.
    # A failed `docker inspect` alone could instead mean the daemon is unavailable.
    listing = subprocess.run(
        ['docker', '--context', args.context, 'container', 'ls', '--all',
         '--format', '{{.Names}}'],
        capture_output=True, text=True, timeout=15, check=False,
    )
    names = set(listing.stdout.splitlines()) if listing.returncode == 0 else set()
    report['post_cleanup'] = {
        'listing_returncode': listing.returncode,
        'container_present': container_name in names if listing.returncode == 0 else None,
    }
    cleanup = process.get('cleanup', {})
    checks = {
        'fixed_sleep_started': START_MARKER in process.get('stdout', '').splitlines(),
        'outer_timeout_observed': process.get('timed_out') is True and process.get('status') == 'timeout',
        'cleanup_attempted': cleanup.get('attempted') is True,
        'cleanup_succeeded': cleanup.get('ok') is True and cleanup.get('returncode') == 0,
        'daemon_listing_succeeded': listing.returncode == 0,
        'container_absent': listing.returncode == 0 and container_name not in names,
    }
    report['acceptance_checks'] = checks
    report['status'] = 'passed' if all(checks.values()) else 'failed'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='t1-phase1:b0')
    parser.add_argument('--context', default='colima')
    parser.add_argument('--platform', choices=['linux/arm64', 'linux/amd64'], default='linux/arm64')
    parser.add_argument('--cpus', type=float, default=4)
    parser.add_argument('--memory', default='8g')
    parser.add_argument('--participant', required=True, type=Path)
    parser.add_argument('--out-dir', type=Path, help='Fresh evidence directory; never overwritten')
    args = parser.parse_args()
    try:
        profile = resource_profile(args.cpus, args.memory)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = (args.out_dir.resolve() if args.out_dir else ROOT / 'evidence' / 'timeout' /
               (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]))
    if run_dir.exists():
        parser.error('--out-dir must not exist; previous evidence is never overwritten')
    run_dir.mkdir(parents=True)
    report = {
        'status': 'not_started', 'fixture_only': True, 'official_house_calls': 0,
        'official_tasks_used': False, 'official_score': None,
        'context': args.context, 'platform': args.platform, 'image': args.image,
        'profile': profile, 'platform_equivalent': False,
        'network': 'none', 'sleep_sec': SLEEP_SEC, 'outer_timeout_sec': OUTER_TIMEOUT_SEC,
        'scope': 'local Docker outer-timeout cleanup only; not platform-equivalent',
        'run_directory': str(run_dir),
    }
    try:
        execute(args, run_dir, report)
    except Exception as exc:
        report.update(status='fixture_error', reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        (run_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        acceptance = {
            'status': report['status'], 'fixture_only': True,
            'image_id': report.get('image_id'), 'platform': args.platform,
            'profile': profile, 'platform_equivalent': False,
            'checks': report.get('acceptance_checks', {'fixture_completed': False}),
            'official_house_calls': 0, 'official_score': None,
            'report': str(run_dir / 'report.json'),
        }
        (run_dir / 'result.json').write_text(json.dumps(acceptance, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'fixture_only': True,
                      'report': str(run_dir / 'report.json')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
