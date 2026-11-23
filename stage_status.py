"""Run-local stage outcomes. No models, lookups, audio or cache access here.

`stage_status` is a list of {stage, status, kind, reason?, source?}. A completed
scan with no detections is `ran`; a missing database match is `not-found`.
Unknown old cards never acquire a claim that their backends all ran.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import re
import textwrap

_run = ContextVar('ears_stage_status', default=None)
_progress = ContextVar('ears_song_progress', default=None)
STATES = {'ran', 'skipped', 'failed', 'degraded', 'not-found'}

_NAMES_NOTE = "lyrics: names list fed to Whisper"
_SAVED_NAMES_NOTE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?P<label>(?:\[[^\]\r\n]+\][ \t]+)?(?:NOTE:[ \t]*)?" +
    re.escape(_NAMES_NOTE) + r") "
    r"\((?P<payload>[^\r\n]*(?:\r?\n(?P=indent) {10}\S[^\r\n]*)*?)\)"
    r"(?P<cache>(?:[ \t]+|\r?\n(?P=indent) {10})\[shared cache\])?[ \t]*(?=\r?$)")


def whisper_names_note(count):
    return f"{_NAMES_NOTE} ({count} {'name' if count == 1 else 'names'})"


def names_note_text(text):
    """Remove prompt previews from old cards and cache notes, including wrapped notes."""
    def replace(match):
        if re.fullmatch(r"\d+ names?|count unavailable", match["payload"]):
            return match[0]
        # Old notes stored up to 90 characters of a prompt, without a count.
        cache = " [shared cache]" if match["cache"] else ""
        return f"{match['indent']}{match['label']} (count unavailable){cache}"
    return _SAVED_NAMES_NOTE.sub(replace, text) if _NAMES_NOTE in text else text


def names_note_bytes(raw):
    return names_note_text(raw.decode("utf-8", errors="surrogateescape")).encode("utf-8", errors="surrogateescape")


@contextmanager
def progress_to(report):
    token = _progress.set(report)
    try:
        yield
    finally:
        _progress.reset(token)


def progress(label):
    report = _progress.get()
    if report is not None:
        report(label)


def cnn_key_text(text, analysis=None):
    """Keep a missing CNN vote beside the key claim, including on saved cards."""
    issues = [row for row in card_notes(text, analysis)
              if re.match(r'CNN key voter: (?:skipped|failed)\b', row)]
    if not issues:
        return text
    if any(row.startswith('CNN key voter: failed') for row in issues):
        note = 'CNN key voter not run (execution error)'
    elif any('venv-madmom not installed' in row for row in issues):
        legacy = 'CNN key voter not run (madmom not installed)'
        note = legacy if legacy in text else 'CNN key voter: optional, not installed'
        if note != legacy and isinstance(analysis, dict):
            claim = analysis.get('key_claim') or analysis.get('key_verdict')
            voters = claim.get('voters') if isinstance(claim, dict) else None
            if isinstance(voters, dict) and voters:
                note += f' ({len(voters)} voter{"s" if len(voters) != 1 else ""} used)'
    else:
        note = 'CNN key voter not run (skipped)'
    def annotate(match):
        line = match[0]
        if 'CNN key voter not run' in line or 'CNN key voter: optional' in line:
            return line
        return line + ' · ' + note
    return re.sub(r'(?m)^[ \t]*KEY:[^\n]*', annotate, text)


def cpu_requested():
    """True when the caller chose CPU (--cpu / RAVEN_EARS_DEVICE=cpu): running there is no fallback."""
    return os.environ.get('RAVEN_EARS_DEVICE', '').strip().lower() == 'cpu'


def reason_of(exc, limit=120):
    """Reason text for an exception. Remedy-first exceptions (remedy_first=True) read as written,
    instruction before cause, with no type-name prefix to push the instruction off the line."""
    if getattr(exc, 'remedy_first', False):
        return clean(exc)[:limit]
    return f'{type(exc).__name__}: {clean(exc)[:limit]}'


def clean(value):
    return ' '.join(str(value).split())


def outcome(stage, status='ran', reason='', *, kind='audio', source=None):
    if status not in STATES:
        raise ValueError('unknown stage status: ' + status)
    row = dict(stage=stage, status=status, kind=kind)
    if reason:
        row['reason'] = clean(reason)
    if source:
        row['source'] = source
    return row


def collect(rows):
    target = _run.get()
    if target is not None:
        for row in rows:
            key = (row['stage'], row.get('source'))
            same = [r for r in target if (r['stage'], r.get('source')) == key]
            if row['status'] == 'ran' and any(r['status'] != 'ran' for r in same):
                continue
            if row['status'] != 'ran':
                target[:] = [r for r in target if r not in same or r['status'] != 'ran']
            if row not in target:
                target.append(row)
    return rows


def mark(stage, status='ran', reason='', **kwargs):
    return collect([outcome(stage, status, reason, **kwargs)])


def snapshot():
    return list(_run.get() or [])


def engine(fn):
    """Keep nested engines isolated, then pass their outcomes to the caller."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        token = _run.set([])
        rows = []
        try:
            result = fn(*args, **kwargs)
            rows = snapshot()
            if isinstance(result, dict):
                rows = merge(rows, result.get('stage_status', []))
                result['stage_status'] = rows
                if 'data' in result and isinstance(result['data'], dict):
                    result['data']['stage_status'] = rows
                if 'lines' in result and 'data' in result:  # speech's rendered card, not harmony's lyric rows.
                    result['lines'] = add_notes(result['lines'], rows)
            return result
        finally:
            rows = merge(rows, snapshot())
            _run.reset(token)
            collect(rows)
    return wrapped


def stage(name, *, kind='audio', classify=None):
    """Record a stage at its existing failure boundary; exceptions still propagate."""
    def decorate(fn):
        @wraps(fn)
        @engine
        def wrapped(*args, **kwargs):
            try:
                value = fn(*args, **kwargs)
            except Exception as exc:
                mark(name, 'skipped' if getattr(exc, 'remedy_first', False) else 'failed', reason_of(exc), kind=kind)
                raise
            try:
                if classify:
                    rows = classify(value)
                elif value is None and any(r['stage'] == name and r['status'] != 'ran' for r in snapshot()):
                    rows = []
                else:
                    rows = [outcome(name, 'skipped' if value is None else 'ran',
                                    'no usable result' if value is None else '', kind=kind)]
            except Exception as exc:
                mark(name, 'failed', f'invalid result ({type(exc).__name__})', kind=kind)
                raise
            if isinstance(value, dict):
                rows = merge(value.get('stage_status', []), rows)
                value = dict(value, stage_status=rows)
            collect(rows)
            return value
        return wrapped
    return decorate


def merge(*groups):
    rows = []
    for group in groups:
        for row in group:
            if row not in rows:
                rows.append(row)
    bad = {(r['stage'], r.get('source')) for r in rows if r['status'] != 'ran'}
    return [r for r in rows if r['status'] != 'ran' or (r['stage'], r.get('source')) not in bad]


def lookup_rows(name, value):
    state = value.get('status')
    reason = value.get('note') or '; '.join(value.get('issues', []))
    if name == 'LRCLIB':
        status = {'off': 'skipped', 'skipped': 'skipped', 'failed': 'failed',
                  'nomatch': 'not-found', 'empty': 'not-found'}.get(state, 'ran')
        reason = reason or {'nomatch': 'no match', 'empty': 'no lyrics'}.get(state, '')
        if value.get('issues') and status == 'ran':
            status = 'degraded'
        return [outcome('LRCLIB', status, reason, kind='input', source='LRCLIB')]
    if value.get('stage_status'):
        return value['stage_status']
    status = ('skipped' if value.get('off') else 'ran' if value.get('entries') else
              'failed' if re.search(r'unreadable|could not|failed|unreachable|timed out', reason, re.I) else
              'skipped' if 'no artist/title' in reason else 'not-found')
    return [outcome('credits', status, reason or ('disabled' if value.get('off') else 'no usable credits'), kind='input')]


def credit_sources(value):
    rows = []
    for key, name in (('mb', 'MusicBrainz'), ('dc', 'Discogs')):
        source = value.get(key) or {}
        state = source.get('status', 'notried')
        status = {'ok': 'ran', 'uncertain': 'degraded', 'nomatch': 'not-found', 'noperf': 'not-found',
                  'failed': 'failed', 'skipped': 'skipped', 'notried': 'skipped'}[state]
        reason = source.get('detail') or {'nomatch': 'no match', 'noperf': 'no performer credits',
                                        'notried': 'not needed'}.get(state, '')
        rows.append(outcome('credits', status, reason, kind='input', source=name))
    return rows


def from_notes(notes, stage_name='audio'):
    """Preserve existing diagnostics while their producers gain explicit hooks."""
    rows = []
    for note in notes:
        text = clean(note)
        if re.search(r'fell back|retried|approximat|fusing without|using the chords-only', text, re.I):
            status = 'degraded'
        elif re.search(r'fail|not importable|unavailable|unreachable|\b\w*(?:Error|Exception)\b', text, re.I):
            status = 'failed'
        elif re.search(r'skipp|disabled|without word times', text, re.I):
            status = 'skipped'
        else:
            continue
        match = re.match(r'([\w /-]+?)(?: skipped)?\s*:\s*(.*)', text)
        if match:
            label, reason = match.groups()
            label = label.lower()
            if status != 'degraded' and re.search(r'\b\w*(?:Error|Exception)\b', reason):
                status = 'failed'
            rows.append(outcome(label, status, reason))
        else:
            rows.append(outcome(stage_name, status, text))
    return rows


def word_rows(words, notes, *, skipped=False):
    """A successful retry is degraded; [] is a scan with no detections, not failure."""
    rows = from_notes(notes, 'Whisper')
    for row in rows:
        if words is not None and row['status'] == 'failed':
            row['status'] = 'degraded'
        row['stage'] = 'Whisper'
    for note in notes:
        device = re.search(r'\bon (cpu|cuda) \(([^)]+)\)', str(note), re.I)
        chosen = device and device[1].lower() == 'cpu' and cpu_requested()
        if device and not chosen and (device[1].lower() == 'cpu' or device[2] != 'float16'):
            rows.append(outcome('Whisper', 'degraded',
                                ('cached transcription used ' if '[shared cache]' in note else 'fell back to ')
                                + device[1].upper() + ' ' + device[2]))
    rows.append(outcome('Whisper', 'skipped' if skipped else 'failed' if words is None else 'ran',
                        'LRCLIB instrumental designation' if skipped else 'words unavailable' if words is None else ''))
    rows.append(outcome('words', 'skipped' if skipped else 'failed' if words is None else 'ran',
                        'LRCLIB instrumental designation' if skipped else 'Whisper words unavailable' if words is None else
                        "Whisper's guesses", kind='input'))
    return merge(rows)


def engine_notes(notes, stage_name):
    # LRCLIB's resolver already classifies transcription attempts using their
    # final result. Re-reading an earlier retry here would invent a failure.
    return from_notes([n for n in notes if not str(n).lower().startswith('lyrics')], stage_name)


def speech_rows(data, notes, *, transcribed=True):
    """Use the completed transcript to distinguish retries from final failure."""
    word_notes = [n for n in notes if str(n).lower().startswith('words:')]
    rows = from_notes([n for n in notes if n not in word_notes], 'speech')
    if transcribed and word_notes and (data.get('words_status') or '').startswith('failed'):
        rows += word_rows(None, word_notes)
    elif transcribed and word_notes and data.get('words') is not None:
        rows += word_rows(data['words'], word_notes)
    # These card lines have their own eligibility checks after the backend ran.
    events = data.get('events')
    breath = (events or {}).get('breath')
    if not breath or 'per_min' not in breath:
        rows.append(outcome('breath', 'skipped', (breath or {}).get('skipped') or
                            ('no breath-rate evidence' if events else 'events unavailable or disabled')))
    affect = data.get('affect')
    if affect is not None and not affect.get('ok'):
        rows.append(outcome('affect', 'skipped', 'no usable voiced windows'))
    emphasis = data.get('emphasis')
    if emphasis is not None and not emphasis.get('ok'):
        rows.append(outcome('emphasis', 'skipped', emphasis.get('reason', 'no usable word evidence')))
    return merge(rows)


def cli(fn):
    """A fatal backend still produces a short failure card and a nonzero exit."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            import json
            import sys
            from pathlib import Path
            name = fn.__module__ if fn.__module__ != '__main__' else Path(sys.argv[0]).stem
            message = reason_of(exc)
            rows = merge(snapshot(), [outcome(name, 'failed', message)])
            argv = args[0] if args and args[0] is not None else sys.argv[1:]
            if '--json' in argv:
                from ears_handoff import finish_json
                print(json.dumps(finish_json({'stage_status': rows, 'notes': [message]})))
            else:
                print('\n'.join(add_notes([name.upper() + ': analysis unavailable'], rows)))
            return 1
    return wrapped


def describe(row):
    label = row['stage']
    if row.get('source') and row['source'] != label:
        label += ' (' + row['source'] + ')'
    state, reason = row['status'], row.get('reason', '')
    if state == 'not-found':
        return label + ': ' + (reason or 'none found')
    return label + ': ' + (reason if state == 'ran' else state + (': ' + reason if reason else ''))


_TYPE_PREFIX = re.compile(r'^[A-Za-z_]*(?:Error|Exception): ')
_SEVERITY = {'failed': 0, 'degraded': 1}


def _same_problem(a, b):
    """Two non-ran rows for one stage that say the same thing (one may be a truncated copy)."""
    if (a['stage'], a.get('source')) != (b['stage'], b.get('source')):
        return False
    ra, rb = (_TYPE_PREFIX.sub('', r.get('reason', '')).lower() for r in (a, b))
    return ra == rb or bool(ra and rb and (ra.startswith(rb) or rb.startswith(ra)))


def _dedupe(rows):
    out = []
    for row in rows:
        twin = next((i for i, r in enumerate(out) if r['status'] != 'ran' and row['status'] != 'ran'
                     and _same_problem(r, row)), None)
        if twin is None:
            out.append(row)
        elif _SEVERITY.get(row['status'], 2) < _SEVERITY.get(out[twin]['status'], 2):
            out[twin] = row
    return out


def summary(rows, *, unknown='stage status not recorded (older card)'):
    return ' · '.join(summary_items(rows, unknown=unknown))


def summary_items(rows, *, unknown='stage status not recorded (older card)'):
    rows = _dedupe(merge(rows))
    if not rows:
        return [unknown]
    grouped = set()
    issues = []
    guesses = next((r for r in rows if r['stage'] == 'words' and r['status'] == 'ran'
                    and r.get('reason') == "Whisper's guesses"), None)
    if guesses:
        inputs = [r for r in rows if r['stage'] in ('LRCLIB', 'lyrics file') and r['status'] != 'ran']
        details = [r.get('reason', r['status']) if r['stage'] == 'lyrics file' else
                   describe(r) for r in inputs]
        issues.append("words: Whisper's guesses" + (' (' + '; '.join(details) + ')' if details else ''))
        grouped.update(id(r) for r in [guesses] + inputs)
    credit_misses = [r for r in rows if r['stage'] == 'credits' and r['status'] == 'not-found' and r.get('source')]
    if credit_misses:
        issues.append('credits: none found (' + ', '.join(dict.fromkeys(r['source'] for r in credit_misses)) + ')')
        grouped.update(id(r) for r in credit_misses)
    issues.extend(describe(r) for r in rows if id(r) not in grouped and
                  (r['status'] != 'ran' or ((r['kind'] == 'input' or r['stage'] == 'quality') and r.get('reason'))))
    audio = [r for r in rows if r['kind'] == 'audio']
    if audio and all(r['status'] == 'ran' for r in audio):
        issues.append('all audio stages ran' if any(r['kind'] == 'input' for r in rows) else 'all stages ran')
    return list(dict.fromkeys(issues)) or ['all stages ran']


def add_notes(lines, rows):
    from ears_handoff import finish_lines
    body = list(lines)
    while body and (not body[-1].strip() or body[-1] == 'and now, you'):
        body.pop()
    # One NOTES line. The engine's own raw "NOTES  ..." digest is dropped: this line says
    # which stages failed, were skipped or degraded, i.e. what the reader can trust.
    body = [line for line in body if not line.startswith('NOTES  ')]
    import lyrics_shared
    context = '\n'.join(body)
    source, lyric_rows = lyrics_shared.supplied_lyrics(text=context)
    body = lyrics_shared.supplied_gap_text(lyrics_shared.alignment_note(context, source), lyric_rows).splitlines()
    items = display_items('\n'.join(body), summary_items(rows))
    body = [part for line in body for part in
            (wrap_note(line) if line.lstrip().startswith('NOTE:') else [line])]
    digest = notes_line(items)
    if digest != 'NOTES: ' + ' · '.join(items):
        # Keep the bounded digest, but give text-only readers every full entry.
        for item in items:
            body += textwrap.wrap(item, width=NOTES_WIDTH, initial_indent='  STATUS: ',
                                  subsequent_indent='          ', break_on_hyphens=False)
    return finish_lines(body + [digest])


def wrap_note(line):
    indent = line[:len(line) - len(line.lstrip())][:NOTES_WIDTH // 4]  # extreme indents must not defeat the width
    return textwrap.wrap(line.strip(), width=NOTES_WIDTH, initial_indent=indent,
                         subsequent_indent=indent + '          ', break_on_hyphens=False)


def instrumental_text(text, evidence=""):
    """Describe word absence using the singer's explicit verdict, not a new decision.

    Only generic absence messages change; actual backend errors stay visible.
    Saved stage rows remain untouched.
    """
    if not re.search(r'^SINGER: instrumental\b', text + "\n" + evidence, re.M):
        return text
    text = re.sub(r'(?m)(^\s*(?:(?:STATUS|NOTE|NOTES):\s*)?| · )'
                  r'(Whisper|words): (?:failed: (?:Whisper )?words unavailable|'
                  r'skipped: too little vocal-stem energy for transcription)'
                  r'(?=$| · )',
                  lambda m: m[1] + m[2] + ': no words to hear (instrumental)', text)
    text = re.sub(r'(?m)^(\s*(?:NOTE: )?LYRICS:) LRCLIB had no matching lyrics; '
                  r'no Whisper words available; not evidence of silence$',
                  r'\1 no words to hear (instrumental)', text)
    text = re.sub(r'(?m)^(\s*)unavailable \(no usable lyric lines, see NOTES\)$',
                  r'\1no words to hear (instrumental)', text)
    text = re.sub(r'(?m)^(LYRICS:) no saved lyric rows supplied$',
                  r'\1 no words to hear (instrumental)', text)
    return text


def display_items(text, items, analysis=None):
    """The printed lyric label describes displayed words; Whisper remains evidence."""
    import lyrics_shared
    source, rows = lyrics_shared.supplied_lyrics(analysis, text)
    items = [lyrics_shared.supplied_gap_text(lyrics_shared.alignment_note(item, source), rows) for item in items]
    items = list(dict.fromkeys(instrumental_text(item, text) for item in items))
    origin = lyrics_shared.displayed_lyric_source(analysis, text)
    if origin == 'file':
        items = [item.replace("words: Whisper's guesses",
                              "words: writer's text (Whisper guesses are alignment evidence)", 1)
                 if item.startswith("words: Whisper's guesses") else item for item in items]
    elif origin in ('plain', 'synced'):
        items = list(dict.fromkeys(bit for item in items
                     if item != 'lyrics file: no lyrics file'
                     for bit in re.sub(r"^words: Whisper's guesses(?: \(no lyrics file\))?$",
                                       'lyrics: LRCLIB ' + origin + ' (displayed) · Whisper: alignment check',
                                       item).split(' · ')))
    return items


def display_text(text, analysis=None):
    """Refresh saved status prose using its displayed lyric label; keep raw rows."""
    # NOTE rows may be wrapped in saved cards. Read their continuation before
    # refreshing, then use the usual width for the changed note.
    for note in source_notes(text):
        refreshed = next(iter(display_items(text, [note], analysis)), note)
        if refreshed == note:
            continue
        pattern = r"(?m)^([ \t]*)" + re.escape(note).replace(r"\ ", r"\s+")
        text = re.sub(pattern, lambda m: "\n".join(wrap_note(m[1] + refreshed)), text, count=1)
    return re.sub(r'(?m)^(\s*(?:STATUS|NOTES):\s*)(.+)$',
                  lambda m: m[1] + ' · '.join(display_items(text, m[2].split(' · '), analysis)), text)


def source_notes(text):
    """Read NOTE rows including the continuations emitted by wrap_note."""
    notes, prefix = [], None
    for line in text.splitlines():
        if line.lstrip().startswith('NOTE:'):
            notes.append(line.strip())
            prefix = ' ' * (len(line) - len(line.lstrip()) + 10)
        elif prefix is not None and line.startswith(prefix):
            notes[-1] += ' ' + line.strip()
        else:
            prefix = None
    return notes


NOTES_WIDTH = 118  # same budget as raven_ears_plus.MAX_LINE


def notes_line(items, width=NOTES_WIDTH, pre='NOTES: '):
    """One line of unique entries. Failures and degradations come first when the line must be cut;
    what does not fit is counted as '+N more' instead of being silently lost."""
    items = list(dict.fromkeys(items))
    line = pre + ' · '.join(items)
    if len(line) <= width:
        return line
    cap = 70  # long causes (paths, tracebacks) are shortened before whole entries are dropped
    items = [t if len(t) <= cap else t[:cap - 1].rstrip() + '…' for t in items]
    line = pre + ' · '.join(items)
    if len(line) <= width:
        return line
    def rank(t):  # what to trust first, then what the reader can fix
        return (0 if re.search(r'\bfailed\b', t) else 1 if re.search(r'\bdegraded\b', t) else
                2 if re.search(r'\brun --', t) else 3)
    items.sort(key=rank)
    for count in range(len(items) - 1, 0, -1):
        line = pre + ' · '.join(items[:count]) + f' · +{len(items) - count} more'
        if len(line) <= width:
            return line
    tail = f' · +{len(items) - 1} more' if len(items) > 1 else ''
    room = width - len(pre) - len(tail) - 1
    return pre + items[0][:room].rstrip() + '…' + tail


def card_notes(text, analysis=None):
    """Read explicit NOTES/STATUS rows, never quoted lyrics or arbitrary JSON text."""
    notes, details, current = [], [], None
    for line in text.splitlines():
        if line.startswith('  STATUS: '):
            details.append(line[10:])
            current = len(details) - 1
        elif current is not None and line.startswith('          '):
            details[current] += ' ' + line.strip()
        else:
            current = None
        if line.strip().startswith('NOTES: '):
            notes.append(' · '.join(details) if details else line.strip()[7:])
            details, current = [], None
    if details:
        notes.append(' · '.join(details))
    rows = []
    def visit(value):
        if isinstance(value, dict):
            found = value.get('stage_status')
            if isinstance(found, list):  # malformed saved rows are absent, not fatal
                rows.extend(r for r in found if isinstance(r, dict) and isinstance(r.get('stage'), str)
                            and isinstance(r.get('status'), str) and isinstance(r.get('kind'), str))
            for key, child in value.items():
                if key != 'stage_status' and isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(analysis)
    if rows:
        # Only a matching digest establishes that this JSON recovers its omissions.
        # Unrelated or partial stage rows must not erase an older card's +N warning.
        recovered = summary_items(rows)
        digest = notes_line(display_items(text, recovered, analysis))[7:]
        notes = [' · '.join(recovered) if note == digest else note for note in notes]
        notes.append(summary(rows))
    if not notes:
        # Old cards can still state a lookup miss explicitly. Read those labels,
        # never infer success from the lack of warnings or quote lyric payloads.
        import lyrics_shared
        legacy = []
        if not lyrics_shared.lyric_lookup(analysis).get('status'):
            for label in lyrics_shared.lyric_labels(text):
                if label.startswith('LYRICS: LRCLIB had no matching lyrics'):
                    legacy.append(outcome('LRCLIB', 'not-found', 'no match', kind='input'))
                elif label.startswith('LYRICS: LRCLIB had no lyrics'):
                    legacy.append(outcome('LRCLIB', 'not-found', 'no lyrics', kind='input'))
                else:
                    continue
                if lyrics_shared.displayed_lyric_source(text=label) == 'whisper':
                    legacy.append(outcome('words', reason="Whisper's guesses", kind='input'))
        for line in text.splitlines():
            line = line.strip().removeprefix('NOTE:').strip()
            if line.startswith('CREDITS:'):
                for source in ('MusicBrainz', 'Discogs'):
                    if re.search(source + r' (?:has no match|.*?no performer credits)', line):
                        legacy.append(outcome('credits', 'not-found', 'no match or performer credits', kind='input', source=source))
        if legacy:
            notes += [summary(legacy), 'audio stage status not recorded (older card)']
    bits = list(dict.fromkeys(bit for note in notes for bit in note.split(' · ')))
    bits = [re.sub(r'\+(\d+) more', r'\1 more status entries omitted by source; supply original JSON for details', bit)
            if re.fullmatch(r'\+\d+ more', bit) else bit for bit in bits]
    # Success in one engine cannot erase a different engine's missing backend.
    if any(bit not in ('all stages ran', 'all audio stages ran') for bit in bits):
        bits = [bit for bit in bits if bit != 'all stages ran']
    if len(notes) > 1 and any(re.search(r'\bfailed\b|\bdegraded\b|\bskipped\b', bit) for bit in bits):
        bits = [bit for bit in bits if bit not in ('all stages ran', 'all audio stages ran')]
    return display_items(text, bits, analysis)
