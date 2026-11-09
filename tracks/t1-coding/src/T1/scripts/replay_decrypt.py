#!/usr/bin/env python3
"""Open encrypted replay shards on the developer machine (local use only).

public_eval.py --replay-pubkey seals each shard's raw replay (prompt, replies, programs, run logs and the
checker's reports) as replay-shard-K.tar.gz.enc plus replay-shard-K.key.enc: the archive is AES-256-CBC
(openssl enc, PBKDF2-SHA256) under a random 256-bit key, and that key is RSA-OAEP-SHA256 wrapped for the
public key in evidence/replay_pubkey.pem. The private key stays outside every git work tree (this checkout
or any other) and is named only on this command line; the plaintext is extracted outside every git work
tree and never written as a tar.

The archive is confidential but not authenticated: anyone holding the committed public key can seal a shard
this script will open. Open only shards downloaded with `gh run download <run_id>` from a known workflow run
of this repository, and treat the extracted programs and replies as untrusted data (run them, if at all,
only in the sandboxed image).
"""
import argparse
import io
import os
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[2]
CIPHER = ('-aes-256-cbc', '-pbkdf2', '-iter', '100000', '-md', 'sha256')
OAEP = ('-pkeyopt', 'rsa_padding_mode:oaep', '-pkeyopt', 'rsa_oaep_md:sha256', '-pkeyopt', 'rsa_mgf1_md:sha256')


def _openssl(args, data=b''):
    # The key travels on stdin only: never in argv, the environment or a file.
    return subprocess.run(['openssl', *args], input=data, capture_output=True, check=True, timeout=600).stdout


def seal(plain: Path, pubkey: Path, data_out: Path, key_out: Path) -> None:
    secret = os.urandom(32).hex().encode()
    _openssl(['pkeyutl', '-encrypt', '-pubin', '-inkey', str(pubkey), *OAEP, '-out', str(key_out)], secret)
    _openssl(['enc', '-e', *CIPHER, '-salt', '-pass', 'stdin', '-in', str(plain), '-out', str(data_out)],
             secret + b'\n')


def unseal(data_in: Path, key_in: Path, private_key: Path) -> bytes:
    secret = _openssl(['pkeyutl', '-decrypt', '-inkey', str(private_key), *OAEP, '-in', str(key_in)])
    return _openssl(['enc', '-d', *CIPHER, '-pass', 'stdin', '-in', str(data_in)], secret + b'\n')


def in_work_tree(path: Path) -> bool:
    """True when path or any ancestor holds a .git entry (a checkout's directory or a worktree's file)."""
    return any(os.path.lexists(directory / '.git') for directory in (path, *path.parents))


def open_shards(shards, private_key, target) -> list[Path]:
    """Each replay-shard-K.tar.gz.enc (its key.enc beside it) is extracted to <target>/replay-shard-K."""
    target, private_key = Path(target).resolve(), Path(private_key).resolve()
    if target.is_relative_to(REPO) or REPO.is_relative_to(target) or in_work_tree(target):
        raise SystemExit('Refusing to write replay plaintext inside a git work tree; choose a target outside it')
    if private_key.is_relative_to(REPO) or in_work_tree(private_key.parent):
        raise SystemExit('The replay private key must stay outside every git work tree')
    opened = []
    for shard in map(Path, shards):
        stem = shard.name.removesuffix('.tar.gz.enc')
        if stem == shard.name:
            raise SystemExit(f'Not a sealed replay shard: {shard.name}')
        plain = unseal(shard, shard.with_name(stem + '.key.enc'), private_key)
        with tarfile.open(fileobj=io.BytesIO(plain), mode='r:gz') as archive:
            archive.extractall(target / stem, filter='data')
        opened.append(target / stem)
    return opened


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--key', type=Path, required=True, help='private key matching evidence/replay_pubkey.pem')
    parser.add_argument('--out', type=Path, required=True, help='target directory outside any git work tree')
    parser.add_argument('shards', type=Path, nargs='+', help='replay-shard-K.tar.gz.enc files')
    args = parser.parse_args()
    for path in open_shards(args.shards, args.key, args.out):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
