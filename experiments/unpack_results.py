#!/usr/bin/env python3
"""Restore shipped raw records after verifying archive and per-file checksums.

Existing records are retained by default, so regenerated experiments are not silently
replaced. Use --overwrite to restore the exact released records. Only flat regular
files listed in the manifest are accepted; archive paths and links are never trusted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'results'
    archive, manifest_path = out / 'raw_records.tar.xz', out / 'RAW_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Raw-record archive checksum mismatch')
    restored = kept = 0
    with tarfile.open(archive, 'r:xz') as tar:
        seen: set[str] = set()
        for item in tar.getmembers():
            name = item.name
            if (not item.isfile() or Path(name).name != name or name in seen
                    or name not in manifest['files']):
                raise ValueError(f'Unexpected or unsafe archive member: {name!r}')
            seen.add(name)
            stream = tar.extractfile(item)
            if stream is None:
                raise ValueError(f'Missing archive content: {name}')
            data = stream.read()
            if hashlib.sha256(data).hexdigest() != manifest['files'][name]:
                raise ValueError(f'Raw-record checksum mismatch: {name}')
            destination = out / name
            if destination.is_symlink():
                raise ValueError(f'Refusing to follow output symlink: {name}')
            if destination.exists() and not args.overwrite:
                kept += 1
                continue
            destination.write_bytes(data)
            restored += 1
        if seen != set(manifest['files']):
            raise ValueError('Archive does not contain the complete manifest')
    print(f'Raw records: {restored} restored, {kept} existing files retained.')


if __name__ == '__main__':
    main()
