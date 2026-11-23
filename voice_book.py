"""Optional, local speaker resemblance. Voiceprints never authenticate a person.

Idea and method from Kael & Élyahna: Resemblyzer enrolment, and the warning that
mic hiss can match itself.
"""
from pathlib import Path
import json
import os
import re
import stat
import tempfile

import numpy as np

SR = 16000
MIN_SECONDS = 1.6
MIN_ENROL_SECONDS = 10
DIMENSIONS = 256
# 15 Nov 2026, real held-out speech: strongest wrong reference .770 (human
# .707); .80 leaves four weak human clips unknown, accepts the other 19/23.
# Require separation too. This small, closed household is NOT open-set safety
# evidence: even very high guesses must NEVER append to anyone's baseline.
SCORE_MIN = .80
GAP_MIN = .10


def home():
    return Path(os.environ.get('RAVEN_EARS_HOME', str(Path.home() / '.local/share/raven-ears'))) / 'voices'


def case_variants(directory, stem, suffix):
    """Files in `directory` named `stem`+`suffix` in any letter case: [(exact_case_name, ...)]."""
    want = (stem + suffix).casefold()
    try:
        return sorted(p.name for p in Path(directory).iterdir() if p.name.casefold() == want and not p.name.startswith('.'))
    except OSError:
        return []


def case_path(directory, stem, suffix):
    """Speaker names are case-insensitive; the canonical file is lowercase.

    An exact-as-typed legacy file is still used when it is the only one, so no history is lost or
    duplicated. When several cases exist the canonical one wins (or the as-typed one, if there is
    no canonical); the caller says so with case_variants() and merges nothing.
    """
    directory = Path(directory)
    canon = stem.casefold() + suffix
    found = case_variants(directory, stem, suffix)
    if canon in found or not found:
        return directory / canon
    exact = stem + suffix
    return directory / (exact if exact in found else found[0])


def voice_path(name):
    # Case-insensitive: --enrol/--forget fox finds Fox.json instead of making a second print. A new print
    # keeps the case it was enrolled with (it is the display name); no traversal or aliases.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
        raise ValueError('voice name: use 1–64 ASCII letters, numbers, dots, underscores or hyphens; start with a letter or number')
    found = case_variants(home(), name, '.json')
    if found and name + '.json' not in found:
        return home() / found[0]
    return home() / (name + '.json')


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    if v.shape != (DIMENSIONS,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-8:
        raise ValueError('invalid voiceprint')
    return v / np.linalg.norm(v)


def voices():
    result = {}
    for path in sorted(home().glob('*.json')):
        try:
            if path.is_symlink() or voice_path(path.stem).name.casefold() != path.name.casefold():
                continue
            row = json.loads(path.read_text(encoding='utf-8'))
            row['embedding'] = _unit(row['embedding'])
            if (type(row['count']) is not int or row['count'] < 1
                    or not np.isfinite(row['seconds']) or row['seconds'] < MIN_ENROL_SECONDS):
                continue
            result[path.stem] = row
        except (OSError, ValueError, KeyError, TypeError):
            continue  # One damaged print must not break ordinary listening.
    return result


def _backend():
    from resemblyzer import VoiceEncoder, preprocess_wav
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return VoiceEncoder(device='cpu', verbose=False), preprocess_wav


def _prepare(wav, preprocess):
    wav = np.asarray(wav, dtype=np.float32)
    if not len(wav) or not np.all(np.isfinite(wav)) or np.max(np.abs(wav)) < 1e-6:
        return np.empty(0, dtype=np.float32)
    return preprocess(wav, source_sr=SR)


def _embed(wav, encoder):
    # Bound model batches, also on long files sent by friends. Same policy as the
    # real-clip measurement: duration-weighted mean of normalized 20 s blocks.
    blocks = [wav[i:i+20*SR] for i in range(0, len(wav), 20*SR)]
    blocks = [b for b in blocks if len(b) >= int(MIN_SECONDS*SR)]
    if not blocks:
        raise ValueError('not enough retained voice (need 1.6 seconds)')
    return _unit(np.average([encoder.embed_utterance(b) for b in blocks],
                            axis=0, weights=[len(b) for b in blocks]))


def enrol(name, clips, load_audio, *, warn=None):
    """Replace this name's print; retain no audio, paths, words or per-clip vectors."""
    path = voice_path(name)
    encoder, preprocess = _backend()
    embeddings, seconds = [], 0.
    for clip in clips:
        wav = _prepare(load_audio(clip, SR), preprocess)
        if len(wav) < MIN_SECONDS*SR:
            raise ValueError('an enrolment clip has less than 1.6 seconds of retained voice')
        embeddings.append(_embed(wav, encoder))
        seconds += len(wav)/SR
    if seconds < MIN_ENROL_SECONDS:
        raise ValueError('enrol at least 10 seconds of retained voice; 20–40 seconds across clips is preferable')
    row = dict(embedding=_unit(np.mean(embeddings, axis=0)).tolist(),
               count=len(embeddings), seconds=round(seconds, 3))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.voice-', delete=False) as fh:
            temporary = Path(fh.name)
            os.fchmod(fh.fileno(), 0o600)
            private = stat.S_IMODE(os.fstat(fh.fileno()).st_mode) == 0o600
            json.dump(row, fh, allow_nan=False)
            fh.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if not private and warn is not None:
        warn('Warning: the voiceprint could not be made private (expected file mode 0600).')
    return row


def forget(name):
    voice_path(name).unlink(missing_ok=True)


def _rank(embedding, book):
    return sorted(((name, float(np.clip(np.dot(embedding, row['embedding']), -1, 1)))
                   for name, row in book.items()), key=lambda item: (-item[1], item[0]))


def _confident(ranked):
    return ranked[0][1] >= SCORE_MIN and (len(ranked) == 1 or ranked[0][1]-ranked[1][1] >= GAP_MIN)


def identify(wav):
    """None preserves the old card when the book or optional encoder is absent."""
    book = voices()
    if not book:
        return None
    try:
        encoder, preprocess = _backend()
    except Exception:
        return None  # Optional package or bundled weights unavailable: old behaviour.
    clean = _prepare(wav, preprocess)
    if len(clean) < MIN_SECONDS*SR:
        return dict(status='unknown', reason='too little voice after VAD', ranked=[])
    ranked = _rank(_embed(clean, encoder), book)
    confident = _confident(ranked)
    # A whole-file average can hide a second voice. Disagreeing confident 5 s
    # segments veto the call; this is a mixed-voice warning, not diarization.
    segment_names = set()
    for i in range(0, len(wav), 5*SR):
        part = _prepare(wav[i:i+5*SR], preprocess)
        if len(part) >= MIN_SECONDS*SR:
            scores = _rank(_embed(part, encoder), book)
            if _confident(scores):
                segment_names.add(scores[0][0])
    mixed = bool(segment_names - {ranked[0][0]})
    return dict(status='matched' if confident and not mixed else 'unknown',
                reason='segments disagree; possibly more than one voice' if mixed else '',
                ranked=ranked, seconds=round(len(clean)/SR, 3))


def voice_line(result):
    ranked = result['ranked']
    if not ranked:
        return '  VOICE: not sure who this is (too little voice after VAD)'
    name, score = ranked[0]
    if result['status'] == 'matched':
        next_one = f'; next {ranked[1][0]} {ranked[1][1]:.2f}' if len(ranked) > 1 else '; only enrolled voice'
        return f'  VOICE: sounds like {name} ({score:.2f}{next_one}) · a resemblance, not a lock'
    why = '; ' + result['reason'] if result.get('reason') else ''
    return f'  VOICE: not sure who this is (closest {name} {score:.2f}{why})'
