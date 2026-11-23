#!/usr/bin/env python3
"""Read-only, model-free measurements; no diagnosis or card changes.

Directories contribute only their immediate .mp3 files (case insensitive).
Explicit files are measured first, with duplicate resolved paths removed.
Uses the ears' mono decoder, so these are not source-channel clipping counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raven_ears_plus import SR_SED, load_audio
from doctored_audio import measure


def input_files(paths):
    seen = set()
    for path in paths:
        files = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.mp3'),
                       key=lambda p: p.name.casefold()) if path.is_dir() else [path]
        for file in files:
            resolved = file.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield file


def table(rows):
    lines = ['| File | s | Peak | ≥1 (%) | ≥.99 (%) | Runs ≥3 | Runs/s | RMS dBFS | Crest dB | Bass share | Tilt dB/oct |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fields = [('duration_s', '.2f'), ('peak', '.4f'), ('full_scale_pct', '.5f'),
              ('near_scale_pct', '.5f'), ('near_runs', 'd'), ('near_runs_per_s', '.3f'),
              ('rms_dbfs', '.2f'), ('crest_db', '.2f'), ('bass_share', '.5f'),
              ('tilt_db_per_oct', '.3f')]
    for row in rows:
        name = row['file'].replace('|', '&#124;').replace('\n', ' ')
        if 'error' in row:
            lines.append('| ' + name + ' | ERROR |' + ' — |' * 10)
        else:
            cells = ['—' if row[k] is None else format(row[k], fmt) for k, fmt in fields]
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', help='audio files or nonrecursive MP3 directories')
    parser.add_argument('--sample-rate', type=int, default=SR_SED)
    parser.add_argument('--json', action='store_true', help='full precision and input SHA-256 hashes')
    args = parser.parse_args(argv)
    from network_policy import local_audio_path
    for path in args.paths:
        local_audio_path(path, parser)
    args.paths = [Path(path) for path in args.paths]
    rows = []
    failed = False
    for path in input_files(args.paths):
        row = dict(file=path.name)
        try:
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            row.update(measure(load_audio(str(path), args.sample_rate), args.sample_rate))
        except Exception as exc:
            # Do not put private source paths/decoder diagnostics in shareable output.
            row['error'] = type(exc).__name__
            failed = True
        rows.append(row)
        print(f"{len(rows)}: {path.name}: {'ERROR' if 'error' in row else 'ok'}", file=sys.stderr)
    if not rows:
        parser.error('no input audio files')
    if args.json:
        print(json.dumps(dict(sample_rate=args.sample_rate, rows=rows), indent=2, allow_nan=False))
    else:
        print(table(rows))
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
