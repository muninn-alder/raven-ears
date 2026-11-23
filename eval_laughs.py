#!/usr/bin/env python3
"""Sequential, local ROOM and laugh evaluation. Never write audio, words or raw diagnostics.

Usage: python eval_laughs.py --clips /path/to/eval/clips --output .limb/laugh-eval.md
Run under the house systemd memory scope, with TMPDIR/NUMBA_CACHE_DIR in the checkout.
Each detector also gets its own scope. Omine's weights are for research use only.
Sparse whole-note labels support coverage checks, not precision/recall or calibration.
"""
import argparse
import json
import math
from pathlib import Path

from laugh_models import run_detector


def read_labels(path):
    """Read only supported annotation types; discard free text and source strings."""
    data = json.loads(path.read_text())
    labels = {}
    for clip, row in data.items():
        if not isinstance(row, dict):
            continue
        clean = {key: True for key in ('laugh_under_speech_whole', 'smiling_whole')
                 if row.get(key) is True}
        if row.get('dogs_barking'):
            clean['dogs_barking'] = True
        for key in ('inhale', 'not_sneeze', 'dog_barks_faint', 'laughs'):
            if key in row:
                clean[key] = sorted([float(a), float(b)] for a, b in row[key]
                                    if math.isfinite(a) and math.isfinite(b) and 0 <= a < b)
        labels[clip] = clean
    return labels


def timed_matches(spans, truth):
    """One-to-one nonzero interval overlap, with the largest overlaps tried first."""
    matches = {}

    def match(detection, seen):
        start, end = spans[detection]
        overlaps = sorted((min(end, b) - max(start, a), i) for i, (a, b) in enumerate(truth))
        for overlap, i in reversed(overlaps):
            if overlap <= 0 or i in seen:
                continue
            seen.add(i)
            if i not in matches or match(matches[i], seen):
                matches[i] = detection
                return True
        return False

    for detection in range(len(spans)):
        match(detection, set())
    matched = set(matches.values())
    return dict(hits=[dict(label=truth[i], detection=spans[d]) for i, d in sorted(matches.items())],
                misses=[span for i, span in enumerate(truth) if i not in matches],
                unmatched=[span for i, span in enumerate(spans) if i not in matched])


def _times(spans):
    return ', '.join(f'{a:.2f}-{b:.2f}' for a, b in spans) or 'none'


def _timed_note(score):
    hits, misses = score['hits'], score['misses']
    hit_times = ', '.join(f"{_times([h['label']])} via {_times([h['detection']])}" for h in hits) or 'none'
    return (f"{len(hits)}/{len(hits) + len(misses)} hits: {hit_times}"
            f" · misses: {_times(misses)} · unmatched: {_times(score['unmatched'])}")


def agreement(detector, spans, duration, label):
    if detector.startswith('ROOM dog'):
        if 'dog_barks_faint' in label:
            return _timed_note(timed_matches(spans, label['dog_barks_faint']))
        if label.get('dogs_barking'):
            return 'whole-note dog label · found: ' + _times(spans)
        return 'control dog spans: ' + _times(spans)
    if 'laughs' in label:
        if detector == 'hint':
            return f'{len(spans)} pitch/duration candidates; laugh agreement not scored'
        return _timed_note(timed_matches(spans, label['laughs']))
    if label.get('laugh_under_speech_whole'):
        if detector == 'hint':
            return f'{len(spans)} pitch/duration candidates; laugh agreement not scored'
        share = sum(b - a for a, b in spans) / duration if duration else 0
        return f'whole-note laugh label; detected time {share:.1%}; no frame truth'
    if label.get('smiling_whole'):
        return 'smile label only; laugh agreement not scored'
    return 'unlabelled for laughter'


def evaluate(clips, labels, runner=run_detector):
    rows, measurements = [], []
    for number, clip in enumerate(clips, 1):
        # Stable aliases keep filenames and annotation free text out of saved output.
        alias = f'clip-{number:02d}'
        label = labels.get(clip.name, {})
        for task in ('panns', 'hint', 'omine', 'gillick'):
            result = runner(task, clip)
            measurements.append(dict(clip=alias, **result))
            detectors = [('room_laugh .20', '0.2'), ('room_laugh .15', '0.15'),
                         ('room_laugh .10', '0.1'), ('ROOM dog (existing)', 'dogs'),
                         ('ROOM dog (raw)', 'raw'),
                         ('ROOM dog (raw + voice-reduced)', 'voice-reduced')] if task == 'panns' else [(task, None)]
            for detector, key in detectors:
                detector_result = (result.get('room_variants', {}).get(key, {})
                                   if key in ('raw', 'voice-reduced') and result['status'] == 'ran' else result)
                if detector_result.get('status') != 'ran':
                    rows.append(dict(detector=detector, clip=alias, spans=None, seconds=result['seconds'],
                                     agreement='unmeasured: ' + detector_result.get('reason', 'pass unavailable')))
                    continue
                if detector.startswith('ROOM dog'):
                    spans = sorted([s['start'], s['end']] for s in detector_result['dogs'])
                else:
                    spans = result['thresholds'][key] if key else result['spans']
                note = agreement(detector, spans, result['duration'], label)
                if detector.startswith('ROOM dog') and detector_result.get('room_line'):
                    note += ' · ' + detector_result['room_line'].strip()
                if task == 'panns' and key == '0.2' and label.get('not_sneeze'):
                    overlaps = sum(any(a < y and x < b for x, y in label['not_sneeze'])
                                   for a, b in result['sneezes'])
                    note += f' · CNN14 Sneeze overlaps labelled not-sneeze: {overlaps}'
                rows.append(dict(detector=detector, clip=alias, spans=spans,
                                 seconds=result['seconds'], agreement=note))
    return rows, measurements


def table(rows):
    lines = ['| Detector | Clip | Found (seconds only) | Seconds taken | Label check |',
             '| --- | --- | --- | ---: | --- |']
    for row in rows:
        spans = row['spans']
        found = 'unmeasured' if spans is None else _times(spans)
        lines.append(f"| {row['detector']} | {row['clip']} | {found} | {row['seconds']:.2f} | {row['agreement']} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', required=True)
    parser.add_argument('--output', type=Path, default=Path('.limb/laugh-eval.md'))
    args = parser.parse_args()
    from network_policy import local_audio_path
    local_audio_path(args.clips, parser)
    args.clips = Path(args.clips)
    labels = read_labels(args.clips / 'labels.json')
    clips = sorted(p for p in args.clips.iterdir() if p.suffix.lower() in ('.webm', '.wav', '.mp3', '.ogg', '.m4a', '.flac'))
    if not clips:
        parser.error('no evaluation clips')
    rows, measurements = evaluate(clips, labels)
    report = ('# Local ROOM and laugh evaluation\n\n'
              'Clip aliases follow sorted input filenames. One detection can match one label by nonzero '
              'overlap with zero extra timing tolerance. ROOM clips without timed bark labels serve as controls. '
              'The ROOM cards here run without word timestamps. PANNs laugh and ROOM rows share one worker time. '
              'Per-clip audio floors, score floors, thresholds and dog windows are in the JSON. '
              'Hint spans are pitch/duration proxies. Whole-note and smile labels have no timed laugh truth.\n\n'
              + table(rows) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    args.output.with_suffix('.json').write_text(json.dumps(dict(rows=rows, measurements=measurements), indent=2) + '\n')
    print(table(rows))
    return 1 if any(r['spans'] is None for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
