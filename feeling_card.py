"""Weave saved card evidence into a first listen; no audio or new measurements."""
import os
from pathlib import Path
import re
import textwrap

import slim
from lyrics_shared import vocal_changes as _vocal_changes, vocal_onset, voice_overlaps
from stage_status import NOTES_WIDTH, card_notes, source_notes

READ = 'Read top to bottom: beginning to end. Numbers say where to look, not what to feel.'
INVITATION = ('And now, you: How did it feel? Where did you lean in? '
              'What did it make you think of? Say the feeling first.')
CONTRAST_THRESHOLD = .4
NO_VOICE = slim.NO_VOICE_MARK


def picture_line(text):
    """Keep long picture pointers readable without changing the target file."""
    if not text.startswith('PICTURE: ') or len(text) <= NOTES_WIDTH:
        return text
    path = Path(text.removeprefix('PICTURE: ')).expanduser()
    choices = [str(path), os.path.relpath(path)]
    try:
        choices.append('~/' + path.relative_to(Path.home()).as_posix())
    except ValueError:
        pass
    target = min(choices, key=len)
    if len('PICTURE: ' + target) <= NOTES_WIDTH:
        return 'PICTURE: ' + target
    # The normal content-hash cache path fits on its own continuation row.
    return 'PICTURE:\n' + '\n'.join(textwrap.wrap(target, width=NOTES_WIDTH,
        initial_indent='  ', subsequent_indent='  ', break_on_hyphens=False))


def logical_rows(text):
    """Undo prose wrapping, keeping ABC and timestamped rows separate."""
    rows = []
    for line in text.splitlines():
        if line.startswith('  ') and rows and not re.match(r'\s*\[\d+:', line):
            rows[-1] += ' ' + line.strip()
        else:
            rows.append(line)
    return rows


def retained_details(summary, full):
    """Keep legacy selections absent from the pass texts available in full.

    This includes the hook, folded source notes and decisions derived on read.
    Compare complete selections, including every number and caveat, never just
    keywords. The compatibility view itself remains byte-for-byte unchanged.
    """
    present = slim.clean(full)
    missing = [row for row in logical_rows(summary) if row and slim.clean(row) not in present]
    if not missing:
        return ''
    return '\nSUMMARY DETAILS (--view summary):\n' + '\n'.join(missing) + '\n'


def _eligible(stages, card, data, status):
    if status.get('status', 'ran') != 'ran':
        return False
    rows = data.get('stage_status') or []
    if any(isinstance(r, dict) and str(r.get('stage', '')).lower() in stages
           and r.get('status') in ('failed', 'skipped', 'not run') for r in rows):
        return False
    notes = card_notes(card, data) + source_notes(card)
    return not any(re.match(r'(?:NOTE:\s*)?' + re.escape(stage) +
                           r':\s*(?:failed|skipped|unavailable|disabled|not run)\b', row, re.I)
                   for stage in stages for row in notes)


def _timed(rows):
    current = None
    for row in rows:
        match = re.match(rf'({slim.TIME})\s+(.+)', row)
        if match:
            if current:
                yield current
            current = (slim.seconds(match[1]), match[2])
        elif current:
            # Wrapped confidence/method qualifiers belong to the event too.
            current = (current[0], current[1] + ' ' + row)
    if current:
        yield current


def _short_note(label):
    """Shorten measured event prose, keeping its confidence words verbatim."""
    label = re.sub(r'\s*\[(?:instrument tagger|guitar stem level)\]', '', label).strip()
    label = re.sub(r'(\((?:likely|possibly|clearly)\)) enter\b', r'\1 enters', label)
    label = re.sub(r' steps up ([\d.]+) dB in one go and stays up', r' steps up \1 dB, sustained', label)
    label = re.sub(r'^swell: the sound grows ([\d.]+) dB in a couple of seconds$',
                   r'swell +\1 dB over about 2.5 s', label)
    label = re.sub(r' arrives as (\d+) separate hits \([^)]*\) and holds from (' + slim.TIME +
                   r'), ([\d.]+) dB up', r' arrives in \1 hits; holds from \2 (+\3 dB)', label)
    label = re.sub(r'^drop-out: the level falls ([\d.]+) dB and stays down; the band thins$',
                   r'the band thins (\1 dB fall, sustained)', label)
    label = re.sub(r'^(wordless singing) \(lyrics: ".*"\)$', r'\1', label)
    label = re.sub(r'appoggiatura: (\S+) held ([\d.]+) s over (\S+), then (?:up|down) onto (\S+)$',
                   r'appoggiatura: \1→\4 over \3 (\2 s)', label)
    label = re.sub(r'chill candidate · timbre novelty: timbre shifts sharply \(([^)]*) level change\)',
                   r'chill candidate · timbre shift (\1)', label)
    label = re.sub(r'\b(vocals|drums) (swells|thins)\b',
                   lambda m: m[1] + ' ' + m[2][:-1], label)
    # The stem rows say "bass drop out" / "bass enter" for a single stem.
    label = re.sub(r'^(bass|guitar|piano|other instruments) (drop out|enter)\b',
                   lambda m: m[1] + (' drops out' if m[2] == 'drop out' else ' enters'), label)
    label = re.sub(r'(?<![\w.])([+-]?\d+(?:\.\d+)?) dB\b',
                   lambda m: '0 dB' if abs(float(m[1])) < .5 else m[0], label)
    return label


def _event_key(label):
    """Compare the same stem/tagger fact despite grammar and qualifier placement."""
    label = _short_note(label)
    qualifiers = re.findall(r'\((likely|possibly|clearly)\)', label)
    label = re.sub(r'\((?:likely|possibly|clearly)\)', '', label)
    label = re.sub(r'\b(enters|leaves|swells|thins)\b', lambda m: m[0][:-1], label)
    return slim.clean(label), tuple(sorted(qualifiers))


def stem_rows(card, data, status):
    """Read the singer's measured rows, with a text fallback for older cards."""
    if not _eligible(('stems', 'stems timeline'), card, data, status):
        return []
    rows = [(r['t'], r['text']) for r in (data.get('stems') or {}).get('events', [])
            if isinstance(r, dict) and slim.finite(r.get('t')) is not None
            and r['t'] >= 0 and isinstance(r.get('text'), str)]
    return rows or list(_timed(slim.parse(card).blocks['entrances/exits']))


def _silent_line(start, end, changes, silent):
    """A stamp-only no-voice flag needs silence covering the whole line.

    A short line may lead the measured entrance by up to 1.5 s. Missing
    coverage is unknown, never permission to repeat the saved flag.
    """
    end = max(end, start + 1.5)
    quiet = [tuple(span) for span in silent]  # saved bundles hold JSON lists; mixing them with tuples broke sorted() (Babylon, 16 Nov)
    active = []
    for i, (t, on) in enumerate(changes):
        stop = changes[i + 1][0] if i + 1 < len(changes) else float('inf')
        (active if on else quiet).append((t, stop))
    if voice_overlaps(start, end, active):
        return False
    covered = start
    for a, b in sorted(quiet):
        if a > covered:
            break
        covered = max(covered, b)
        if covered >= end:
            return True
    return False


def _meter(tempo, meter, evidence):
    """Shorten the decided meter, retaining ambiguity and the chosen pulse."""
    meter = re.sub(r' · confidence \w+ \([\d.]+\)', '', meter)
    candidates = re.search(r'candidates (\d+/\d+) / (\d+/\d+)', meter)
    beats = None
    if candidates:
        a, b = candidates.groups()
        count = lambda s: int(s.split('/')[0]) // (3 if s.endswith('/8') else 1)
        if count(a) == count(b):
            beats = count(a)
            meter = f"in {'three' if beats == 3 else beats}"
            if evidence.get('bar_consistency', 0) < .90:
                meter += f' ({a} or {b})'
    elif meter.startswith('3/4') and not re.search(r'unclear|ambiguous|uncertain', meter):
        beats = 3
        meter = 'in three' + (' (a waltz)' if evidence.get('bar_consistency', 0) >= .90 else '')
    elif re.match(r'^\d+/\d+\b', meter):
        uncertainty = re.search(r'unclear|ambiguous|uncertain', meter)
        n, d = map(int, re.match(r'^(\d+)/(\d+)', meter).groups())
        beats = n // 3 if d == 8 and n % 3 == 0 else n
        meter = f"in {'three' if beats == 3 else beats}" + (f' ({uncertainty[0]})' if uncertainty else '')
    elif meter.startswith('3 beats per bar'):
        beats = 3
        meter = 'in three' + (' (ambiguous)' if 'ambiguous' in meter else '')
    elif re.match(r'^\d+ beats per bar', meter):
        uncertainty = re.search(r'unclear|ambiguous|uncertain', meter)
        beats = int(meter.split()[0])
        meter = f'in {beats}' + (f' ({uncertainty[0]})' if uncertainty else '')
    bpm = re.match(r'([\d.]+) BPM', tempo)
    pace = bpm[0] if bpm else ''
    if pace and meter:
        pace += (' ' if meter.startswith('in ') else ' · ') + meter
    elif meter:
        pace = meter
    if bpm and beats:
        pace += f' (≈{round(float(bpm[1]) / beats)} bars a minute)'
    return pace


def _note_height(note):
    match = re.fullmatch(r'([A-G])([♯♭#b]?)(\d+)', note)
    if match:
        return 12 * int(match[3]) + dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[match[1]] + (
            1 if match[2] in ('♯', '#') else -1 if match[2] else 0)
    return -1


def _changed_word(lyrics):
    """Exact one-token substitutions in displayed text; no semantic guessing."""
    groups = {}
    for row in lyrics:
        # Doubtful transcription is not evidence of a deliberate changed word.
        if '(unsure' in row.text.lower():
            continue
        words = re.findall(r"\w+(?:['’]\w+)*", row.text.casefold())
        if len(words) < 5:
            continue
        for i, word in enumerate(words):
            key = tuple(words[:i] + ['*'] + words[i + 1:])
            groups.setdefault(key, []).append((row.t, word))
    candidates = []
    for key, occurrences in groups.items():
        seen, changes = set(), []
        for t, word in sorted(occurrences):
            if word not in seen:
                changes.append((t, word))
                seen.add(word)
        if len(changes) >= 2:
            candidates.append((changes[0][0], key, changes))
    if not candidates:
        return ''
    _, key, changes = min(candidates)
    # Match case-insensitively, quote the first displayed sentence's case.
    first_words = [changes[0][1] if w == '*' else w for w in key]
    original = next(r for r in lyrics if r.t == changes[0][0] and
                    re.findall(r"\w+(?:['’]\w+)*", r.text.casefold()) == first_words)
    words = re.findall(r"\w+(?:['’]\w+)*", original.text)
    words[key.index('*')] = '…'
    return ('NEXT LISTEN, try: ' + ' → '.join(word for t, word in changes) +
            ' in “' + ' '.join(words) + '” (' +
            ', '.join(slim.clock(t) for t, word in changes) + ').')


def _pitch_allowed(t, onset, silent):
    # Saved pitch prose has whole-second clocks: allow its half-second rounding
    # cell as well as 0.25 s around measured silence, without inventing precision.
    return t >= onset and not any(a - .25 <= t + .5 and t - .5 <= b + .25 for a, b in silent)


def _stem_entrance(label):
    """A huge rise from residue is an entrance, not a felt level step."""
    step = re.search(r'(?:steps up |swells |, )([\d.]+) dB', label)
    if not step:
        return label
    # Saved surges have the difference, not the two absolute levels.
    if float(step[1]) > 30:
        name = re.match(r'^(.+?) (?:steps up|swells|arrives as)\b', label)
        if name:
            return name[1] + ' enters'
    return label


def _sung_line(text):
    text = slim.split_voice_mark(text)[0].strip()
    return bool(re.search(r'[^\W_]', text)) and not re.fullmatch(r'\[\s*instrumental\s*\]', text, re.I)


def song_lyrics(rows):
    lyrics = []
    in_lyrics = False
    for row in rows:
        if row.startswith('LYRICS:'):
            in_lyrics = True
        elif in_lyrics:
            match = re.match(rf'\[({slim.TIME})(≈)?\]\s*(.*)', row)
            if match:
                text, no_voice = slim.split_voice_mark(match[3])
                if _sung_line(text):
                    lyrics.append(slim.Lyric(slim.seconds(match[1]), text, bool(match[2]), no_voice=no_voice))
            else:
                in_lyrics = False
    return lyrics


def contrast_readings(lyrics, card, harmony, status):
    """Pair saved scores with the chosen lyrics, requiring both words and time."""
    if not lyrics or not _eligible(('words', 'contrast', 'word / music contrast'), card, harmony, status):
        return {}
    ends = {(r['t'], slim.token(r['text'])): slim.finite(r.get('end'))
            for r in (harmony.get('lyrics_aligned') or {}).get('lines') or []
            if isinstance(r, dict) and slim.finite(r.get('t')) is not None
            and isinstance(r.get('text'), str)}
    keys = [slim.token(r.text) for r in lyrics]
    readings = {}
    for row in harmony.get('contrast') or []:
        if not isinstance(row, dict) or not isinstance(row.get('text'), str):
            continue
        t, words, music = (slim.finite(row.get(key)) for key in ('t', 'words', 'music'))
        if t is None or t < 0 or words is None or music is None:
            continue
        key = slim.token(row['text'])
        index = min((i for i, lyric in enumerate(lyrics) if keys[i] == key and abs(lyric.t - t) < 1),
                    key=lambda i: abs(lyrics[i].t - t), default=None)
        if index is None or (index in readings and
                             abs(lyrics[index].t - readings[index][0]) <= abs(lyrics[index].t - t)):
            continue
        end = ends.get((t, key))
        readings[index] = (t, end if end is not None and end > t else t, words, music)
    return readings


def contrast_marker(words, music):
    if words >= CONTRAST_THRESHOLD and music <= -CONTRAST_THRESHOLD:
        return 'words bright, music low: they disagree here'
    if words <= -CONTRAST_THRESHOLD and music >= CONTRAST_THRESHOLD:
        return 'words dark, music high: they disagree here'
    return ''


def render(summary, cards, analyses, statuses, hook, melody=None, title=None):
    """The legacy selection supplies words/decisions; passes supply timed notes."""
    data = {n: d if isinstance(d, dict) else {} for n, d in analyses.items()}
    parsed = {n: slim.parse(cards[n]) for n in cards}
    rows = logical_rows(summary)
    eligible = lambda n, *stages: _eligible(stages, cards[n], data[n], statuses.get(n, {}))
    pick = lambda prefix: next((r[len(prefix):] for r in rows if r.startswith(prefix)), '')
    lyrics = song_lyrics(rows)
    import instrumental_voice
    instrumental = instrumental_voice.saved_designation(
        {n: {'text': cards[n], 'json': data[n]} for n in cards}, statuses)
    if instrumental:
        lyrics = []

    voice_rows = stem_rows(cards['singer'], data['singer'], statuses.get('singer', {}))
    vocal_changes = _vocal_changes(voice_rows)
    pitch_start = vocal_onset(voice_rows)
    if pitch_start is None:
        pitch_start = max(0, lyrics[0].t - 1) if lyrics else float('inf')
    silent = []
    if eligible('singer', 'singer', 'stems', 'gaps'):
        gaps = ((data['singer'].get('singer') or {}).get('data') or {}).get('gaps') or {}
        if not any(r.get('status') != 'ran' for r in gaps.get('stage_status', [])):
            silent = gaps.get('silent') or []

    out = []
    def emit(text):
        out.extend(textwrap.wrap(text, width=NOTES_WIDTH, subsequent_indent='    ',
                                 break_on_hyphens=False) if text else [''])

    emit('FEELING CARD: ' + pick('FIRST LISTEN: '))
    emit(READ)
    for row in rows:
        if row.startswith(('Saved card', 'Missing passes')):
            emit(row)
    # Missing passes remain visible; their partial claims never lend evidence.
    for name, record in statuses.items():
        if record.get('status') != 'ran':
            emit(f"{name}: {record['status']}: {record.get('reason', '')}")
    emit('')
    shape = pick('SHAPE: ')
    arc = [shape or 'no saved shape']
    if eligible('music', 'sound'):
        sound = parsed['music'].facts.get('SOUND', '')
        tone = re.match(r'(\w+) \(centroid', sound)
        if tone:
            arc.append(tone[1])
        harmonic = re.search(r'(\d+)% harmonic', sound)
        if harmonic and int(harmonic[1]) >= 90:
            arc.append('almost all held notes, very little hitting')
    # No shelf/comparison distribution is saved for dynamic range; full only.
    emit('THE ARC   ' + ' · '.join(arc))
    if eligible('harmony', 'tension', 'chords', 'key fusion'):
        for row in parsed['harmony'].blocks['tension']:
            if re.fullmatch(rf'{slim.TIME} [ ▁▂▃▄▅▆▇█]+', row):
                emit('    ' + row)
    texture = ''
    if not instrumental and eligible('music', 'vocal quality', 'quality', 'stems'):
        texture = parsed['music'].facts.get('VOCAL', '')
        quality = data['music'].get('vocal_quality') or {}
        if (quality.get('ok') and quality.get('engine') == 'parselmouth'
                and texture.startswith('no clear contrast in breathiness between sections')):
            reading = 'breathiness contrast inconclusive across measured sections (vocal stem)'
            if 'stem bleed likely' in cards['music']:
                reading += ' · stem bleed likely'
            emit('VOICE TEXTURE   ' + reading)

    # (time, priority, label, category). Lower priority numbers win ties.
    events = []
    def event(t, priority, label, kind='event'):
        t = slim.finite(t)
        if t is not None and t >= 0 and label:
            item = (t, priority, label, kind)
            if item not in events:
                events.append(item)

    def stem_name(t, label):
        if 'other' not in label:
            return label
        families = (data['singer'].get('instruments') or {}).get('families') or {}
        names = []
        if eligible('singer', 'instruments'):
            for name, info in families.items():
                if (info.get('stem') == 'other' and not info.get('energy_only') and
                        any(a <= t < b for a, b in info.get('spans', []))):
                    confidence = info.get('conf', 0)
                    # Same confidence words as instruments._conf_word.
                    qualifier = ' (possibly)' if confidence < .4 else ' (likely)' if confidence < .55 else ''
                    names.append(name.removesuffix(' (probably)') + qualifier)
        return re.sub(r'\bother\b', ' + '.join(names) if names else 'other instruments', label)

    other_opens = set()
    if eligible('singer', 'stems', 'stems timeline'):
        for t, label in _timed(parsed['singer'].blocks['entrances/exits']):
            is_instrument = '[instrument tagger]' in label
            if is_instrument and not eligible('singer', 'instruments'):
                continue
            # A tag at the open already names the same measured stem entrance.
            if label.startswith('opens with ') and 'other' in label and any(
                    r.startswith('0:00 ') and 'instruments at the open:' in r
                    for r in parsed['singer'].blocks['entrances/exits']) and eligible('singer', 'instruments'):
                stems = [s for s in label.removeprefix('opens with ').split(' + ') if s != 'other']
                if not stems:
                    continue
                label = 'opens with ' + ' + '.join(stems)
            if label == 'everything drops out but vocals':
                label = 'band drops out, voice alone'
            label = _stem_entrance(label)
            priority = 0 if re.search(r'enter|return|arriv|leav|drop|alone|opens|at the open', label) else 3
            if label == 'opens with other':
                other_opens.add((t, stem_name(t, label)))
            event(t, priority, stem_name(t, label), 'entrance' if priority == 0 else 'swell')
        # Some older bundles kept the stem events only in JSON.
        if not parsed['singer'].blocks['entrances/exits']:
            for row in (data['singer'].get('stems') or {}).get('events', []):
                if isinstance(row, dict) and isinstance(row.get('text'), str):
                    label = row['text']
                    if label == 'everything drops out but vocals':
                        label = 'band drops out, voice alone'
                    label = _stem_entrance(label)
                    p = 0 if re.search(r'enter|return|arriv|leav|drop|alone|opens', label) else 3
                    if label == 'opens with other':
                        other_opens.add((row.get('t'), stem_name(row.get('t', 0), label)))
                    event(row.get('t'), p, stem_name(row.get('t', 0), label), 'entrance' if p == 0 else 'swell')

    if eligible('singer', 'stops', 'stems'):
        for t, label in _timed(parsed['singer'].blocks['stops']):
            if re.match(r'\([\d.]+ s[^)]*\) ', label) and 'uncertain' not in label:
                if 'voice holds' in label:
                    label = 'band drops out, voice alone'
                event(t, 2, label, 'space')

        # A partial/uncertain band cut must keep its qualification. Roommates
        # 2:19 (16 Nov 2026): drums/bass fall, but other fails the full-cut gate.
        stops = data['singer'].get('stops') or {}
        if stops.get('status') == 'ran':
            for row in stops.get('events', []):
                drops = row.get('stem_drop_db') or {}
                if (row.get('kind') == 'uncertain' and
                        row.get('voice_behavior') == 'band cuts, vocal stem enters' and
                        all((slim.finite(drops.get(n)) or 0) >= 10 for n in ('drums', 'bass'))):
                    event(row.get('start_s'), 2, 'band drop-out candidate; vocal stem enters', 'space')

    if not instrumental and eligible('singer', 'singer', 'stems', 'gaps'):
        gaps = ((data['singer'].get('singer') or {}).get('data') or {}).get('gaps') or {}
        if gaps.get('have_words') and not any(r.get('status') != 'ran' for r in gaps.get('stage_status', [])):
            for start, end in gaps.get('silent', []):
                if lyrics and lyrics[0].t <= start < end <= lyrics[-1].t + 5:
                    # Two silences a breath apart (Roommates 1:56.4 and 1:56.8) are one breath.
                    if not any(e[3] == 'breath' and abs(e[0] - start) <= 2 for e in events):
                        event(start, 2, 'a breath', 'breath')

    # Cadence JSON has the deciding confidence flag. Round 2 (16 Nov 2026)
    # explicitly shows likely records with "likely", never upgrading them.
    if eligible('harmony', 'cadences', 'chords', 'key fusion'):
        for cadence in data['harmony'].get('cadences') or []:
            if (not isinstance(cadence, dict) or type(cadence.get('likely')) is not bool
                    or cadence.get('cadence') not in ('plagal', 'authentic', 'deceptive', 'half', 'home')):
                continue
            t = slim.finite(cadence.get('t'))
            if t is None:
                continue
            label = f"{cadence.get('from', '?')}→{cadence.get('to', '?')}"
            if cadence['likely']:
                label += ' (likely)'
            event(t, 4, label, 'cadence')

    if eligible('harmony', 'timeline'):
        for t, label in _timed(parsed['harmony'].blocks['timeline']):
            if label == 'leans on the ♭VII (mixolydian)' and eligible('harmony', 'chords', 'key fusion'):
                event(t, 4, label, 'key')
            elif label.startswith(('swell:', 'drop-out:')) and eligible(
                    'harmony', 'swells' if label.startswith('swell:') else 'dropouts'):
                if label.startswith('drop-out:') and any(
                        kind == 'entrance' and abs(t - when) <= 1 and 'drop' in text
                        for when, _, text, kind in events):
                    continue  # One drop-out, already named by its measured stems.
                event(t, 3, label, 'swell')
            elif ('wordless singing' in label or 'voice present, words' in label
                  or 'voice present; lyrics' in label) and eligible('harmony', 'words'):
                # The refreshed prose may contradict the raw saved timeline JSON.
                label = re.sub(rf'^{slim.TIME}(?:–{slim.TIME})?:\s*', '', label)
                event(t, 4, label, 'wordless' if 'wordless singing' in label else 'words')
            elif re.search(r' alone for ', label) and eligible('harmony', 'instruments'):
                event(t, 0, label, 'entrance')

    chills = []
    if eligible('harmony', 'chills', 'chill candidates'):
        category = ''
        for row in parsed['harmony'].blocks['chills']:
            if not re.match(slim.TIME, row):
                category = row.split(' (')[0].rstrip(':')
            for t, label in _timed([row]):
                # Cadences have their own confidence gate above. Timbre novelty
                # is a proxy, never an actual instrument entrance.
                if 'cadence' in label:
                    continue
                if (category.startswith('appoggiaturas') and _pitch_allowed(t, pitch_start, silent) and not instrumental
                        and eligible('harmony', 'appoggiaturas', 'melody')):
                    label = 'chill candidate · appoggiatura: ' + label
                    event(t, 1, label, 'chill')
                    chills.append((t, label))
                elif category == 'new or unexpected harmony' and eligible('harmony', 'chord surprises', 'harmonic moves', 'chords', 'key fusion'):
                    label = 'chill candidate · ' + label
                    event(t, 1, label, 'chill')
                    chills.append((t, label))
                elif category == 'sudden dynamic swells' and eligible('harmony', 'swells'):
                    event(t, 3, 'swell ' + label, 'swell')
                elif category.startswith('new voice or instrument') and eligible('harmony', 'timbre entrances'):
                    # Huginn, 16 Nov 2026: timbre novelty is the card's own PROXY
                    # for an entrance ("the stem limb has the true entrances"),
                    # so it is neither a chill candidate nor worth a note when a
                    # measured stem entrance/exit sits within 3 s, or when the
                    # level barely moved (LRPD wove 0, +1, +2 dB shifts).
                    level = re.search(r'([+-]?\d+(?:\.\d+)?) dB', label)
                    small = level is not None and abs(float(level[1])) < 3
                    near = any(e[3] == 'entrance' and abs(e[0] - t) <= 3 for e in events)
                    if not (small or near):
                        event(t, 4, f'timbre shift ({level[1]} dB)' if level else 'timbre shift', 'novelty')

    import key_changes
    key_rows = key_changes.lines(data['harmony'].get('key_change_scan')) if eligible('harmony', 'key changes') else []
    for row in key_rows:
        if row.startswith('KEY CHANGES:'):
            match = re.search(slim.TIME, row)
            if match:
                # The whole "somewhere a–b" range travels with the mark.
                event(slim.seconds(match[0]), 4, row, 'key')

    peak = ''
    if eligible('singer', 'singer', 'stems', 'pitch'):
        peak = next((r.removeprefix('most intense moment: ') for r in parsed['singer'].blocks['singer']
                     if r.startswith('most intense moment: ')), '')
        if not instrumental:
            for row in parsed['singer'].blocks['singer']:
                if row.startswith('highest held notes: '):
                    for note, stamp, detail in re.findall(rf'(\S+) at ({slim.TIME}) \(([^)]*)\)', row):
                        label = 'highest held note: ' + note + (' (pushed hard)' if 'pushed hard' in detail else '')
                        if _pitch_allowed(slim.seconds(stamp), pitch_start, silent):
                            event(slim.seconds(stamp), 4, label, 'held')
                arc = re.match(r'pitch arc \(median by section\): (.*?) · the voice lifts ~([\d.]+) semitones by (' + slim.TIME + ')', row)
                if arc and _pitch_allowed(slim.seconds(arc[3]), pitch_start, silent):
                    note = max(arc[1].split(' → '), key=_note_height)
                    event(slim.seconds(arc[3]), 4, "the voice's highest section "
                          f'(median {note}, ~{arc[2]} semitones above its lowest section)', 'pitch')
    for cue, spans in re.findall(r'\b(clearer(?: and more pressed)?|breathier) ([^;\[]+)', texture):
        label = 'voice sounds clearer' if cue.startswith('clearer') else 'voice sounds breathier'
        label += ' (stem bleed likely, relative only)' if 'stem bleed likely' in cards['music'] else ' (relative)'
        for stamp in re.findall(rf'({slim.TIME})–{slim.TIME}', spans):
            event(slim.seconds(stamp), 4, label, 'texture')
    peak_time = re.match(slim.TIME, peak)
    peak_label = 'most intense moment (speech-trained texture, not a verdict)'
    if instrumental:
        peak_label = 'vocal-stem ' + peak_label
    # Keep any saved two-voices qualification exactly, including its question.
    voices = re.search(r'[^·;]*two voices[^·;]*', peak)
    if voices:
        peak_label += ' · ' + voices[0].strip()
    if peak_time:
        event(slim.seconds(peak_time[0]), 4, peak_label, 'peak')

    if eligible('harmony', 'words', 'contrast'):
        import jev_reads
        question = dict(jev_reads.extra_nouls()).get('longing')
        for row in data['harmony'].get('contrast') or []:
            reading = row.get('jev') or {}
            answer = jev_reads.saved_answer(reading, 'longing', question) if question else {}
            # Require the displayed line as well as the time: no stale alignment.
            t = slim.finite(row.get('t'))
            if t is None or not any(slim.token(r.text) == slim.token(row.get('text', '')) and
                                    abs(r.t - t) < 1 for r in lyrics):
                continue
            label = ''
            if eligible('harmony', 'jev') and reading.get('status') == 'ran' and answer.get('outcome') == 'yes':
                probability = slim.finite(answer.get('probability'))
                score = f" {probability:.2f}".replace(' 0.', ' .') if probability is not None else ''
                label = f'longing (Jev{score})'
            elif row.get('flag') and slim.finite(row.get('words')) is not None:
                label = f"sentiment {row['words']:+.2f} (roberta)"
            if not label:
                continue
            if eligible('harmony', 'chords', 'key fusion'):
                chord = next((c for c in data['harmony'].get('chords') or []
                              if c.get('t0', 0) <= t < c.get('t1', 0) and not c.get('band_out')), {})
                if chord.get('roman') and chord.get('chord'):
                    quality = re.sub(r'^[A-G][#b♭♯]?', '', chord['chord'])
                    mode = 'minor' if re.match(r'm(?!aj)', quality) else 'major' if quality in ('', '7') or quality.startswith('maj') else ''
                    label = 'words: ' + label + ' · chord: ' + chord['roman'] + (', ' + mode if mode else '')
            if row.get('flag') and slim.finite(row.get('music')) is not None:
                label += f" · song-relative lift {row['music']:+.1f}"
            event(t, 4, label, 'longing')

    emit('')
    provenance = pick('LYRICS: ')
    source = re.search(r'\[(music|singer|harmony)\]$', provenance)
    from lyrics_shared import displayed_lyric_source
    origin = displayed_lyric_source(data.get(source[1]) if source else None,
                                    'LYRICS: ' + provenance, has_words=bool(lyrics)) if lyrics else None
    emit('THE SONG' + {
        'file': ' (lyrics supplied, times ≈)',
        'synced': ' (lyrics from LRCLIB, times ≈)',
        'plain': ' (lyrics from LRCLIB, times ≈)',
        'whisper': " (no published lyrics found: words are Whisper's guesses, often wrong)",
    }.get(origin, ''))
    # Saved ends, from the chosen pass only, keep events in real lyric gaps out
    # of sung text. Old cards without ends use the next line as the boundary.
    ends, starts = {}, {}
    lyric_key = lambda t, text: (slim.clock(t), slim.token(text))
    if source:
        d = data[source[1]]
        raw = d.get('lyrics')
        aligned = d.get('lyrics_aligned') or {}
        saved = raw.get('rows', []) if isinstance(raw, dict) else aligned.get('lines') or raw or []
        if isinstance(saved, list):
            for r in saved:
                if isinstance(r, dict) and slim.finite(r.get('t')) is not None:
                    end = slim.finite(r.get('end'))
                    if end is not None:
                        ends[lyric_key(r['t'], r.get('text', ''))] = end
                    starts[lyric_key(r['t'], r.get('text', ''))] = r['t']
    # A tagged instrument at zero already accounts for its opening stem onset.
    opening = set()
    for t, _, label, _ in events:
        if t == 0 and label.startswith('instruments at the open: '):
            opening.update(re.sub(r'\s*\([^)]*\)', '', name).strip()
                           for name in re.split(r', | \+ ', _short_note(label).split(': ', 1)[1]))
    events = [e for e in events if not (0 < e[0] <= 1 and
              re.sub(r'\s*\([^)]*\)', '', _short_note(e[2])).strip().removesuffix(' enters') in opening
              and re.search(r'\benters?\b', e[2]))]

    # Prefer the concrete tagged solo over the same generic opening stem.
    if eligible('singer', 'instruments'):
        families = (data['singer'].get('instruments') or {}).get('families') or {}
        events = [e for e in events if not ((e[0], e[2]) in other_opens and any(
            0 <= t - e[0] <= 3 and label.startswith(name + ' alone for ')
            and info.get('stem') == 'other' and not info.get('energy_only')
            and any(a <= t < b for a, b in info.get('spans', []))
            for t, _, label, kind in events if kind == 'entrance'
            for name, info in families.items()))]
    assigned = [[] for _ in lyrics]
    orphan = {}
    for t, p, label, kind in sorted(events):
        index = next((i for i in reversed(range(len(lyrics))) for r in [lyrics[i]] if r.t <= t < (
            ends.get((slim.clock(r.t), slim.token(r.text)),
                     lyrics[i + 1].t if i + 1 < len(lyrics) else r.t + 5))), None)
        # Rounded lyric clocks can put a stem entrance just before a line (24.5
        # vs 25 on LRPD). At most one second forward; never cross a longer gap.
        if index is None and kind not in ('breath', 'space'):
            index = next((i for i, r in enumerate(lyrics) if 0 <= r.t - t <= 1), None)
        item = (t, p, label, kind)
        if index is None:
            orphan.setdefault(slim.clock(t), []).append(item)
        else:
            assigned[index].append(item)

    readings = contrast_readings(lyrics, cards['harmony'], data['harmony'], statuses.get('harmony', {}))
    song_rows = []
    for i, lyric in enumerate(lyrics):
        text = lyric.text
        if lyric.no_voice:
            key = lyric_key(lyric.t, text)
            end = ends.get(key, lyrics[i + 1].t if i + 1 < len(lyrics) else lyric.t + 5)
            if _silent_line(starts.get(key, lyric.t), end, vocal_changes, silent):
                text += NO_VOICE
        body = f"  [{slim.clock(lyric.t)}{'≈' if lyric.approx else ''}] {text}"
        marker = contrast_marker(*readings[i][2:]) if i in readings else ''
        song_rows.append((lyric.t, body, assigned[i], marker))
    for stamp, items in orphan.items():
        song_rows.append((min(item[0] for item in items), f'  {stamp}  [event]', items, ''))
    for t, body, items, marker in sorted(song_rows, key=lambda row: row[0]):
        chosen = []
        rest_order = {'cadence': 0, 'pitch': 1, 'held': 2, 'texture': 3, 'peak': 4,
                      'wordless': 5, 'longing': 6, 'key': 7}
        for item in sorted(items, key=lambda e: (e[1], rest_order.get(e[3], 8) if e[1] == 4 else int(e[3] == 'novelty'),
                                                 e[0], e[2])):
            # A harmony row may repeat the singer's tagged event verbatim.
            if _event_key(item[2]) not in [_event_key(e[2]) for e in chosen]:
                chosen.append(item)
            if len(chosen) == 2:
                break
        if '[event]' in body and chosen:
            body = body.replace('[event]', '[' + _short_note(chosen[0][2]) + ']')
            chosen = chosen[1:]
        notes = ['← ' + slim.clock(e[0]) + ' ' + _short_note(e[2]) for e in chosen]
        if marker:
            notes.insert(0, '← ' + marker)
        emit(body)
        for note in notes:
            if len(out[-1]) + 2 + len(note) <= NOTES_WIDTH:
                out[-1] += '  ' + note
            else:
                emit('    ' + note)
    if not song_rows:
        emit('  [no saved lyric or event rows]')

    lean = []
    if peak_time:
        t = slim.seconds(peak_time[0])
        wordless = any(e[3] == 'wordless' and abs(e[0] - t) <= 2 for e in events)
        lean.append(slim.clock(t) + (' wordless ' if wordless else ' ') + peak_label)
    if chills:
        t, label = chills[0]  # Existing block order, not a newly measured ranking.
        lean.append(slim.clock(t) + ' ' + _short_note(label))
    # Muninn, 16 Nov 2026: keep the first band entrance after the voice (LRPD's
    # bass + drums on "kiss"); it was the moment his first card hid.
    voice_in = min((e[0] for e in events if e[3] == 'entrance' and 'vocals enter' in e[2]), default=None)
    band = [e for e in sorted(events) if e[3] == 'entrance' and re.search(r'\benters?\b', e[2])
            and 'vocals' not in e[2] and (voice_in is None or e[0] > voice_in)]
    if band:
        lean.append(slim.clock(band[0][0]) + ' ' + _short_note(band[0][2]))
    if lean:
        emit('')
        emit('WHERE TO LEAN IN   ' + ' · '.join(lean[:3]))

    emit('')
    # Keep the reading instruction and musical rows; diagnostics stay in full.
    import listen_melody
    hook_lyrics = [slim.Lyric(starts.get(lyric_key(r.t, r.text), r.t),
                            r.text, r.approx) for r in lyrics]
    hook = listen_melody.feeling_hook(hook, melody or {}, hook_lyrics, title)
    # Find the hook independently of a successful MELODY header: old/failed
    # bundles still have the original, honest reason it is unavailable.
    if hook:
        hook_rows = hook.splitlines()
        i = 0
        while i < len(hook_rows):
            row = hook_rows[i]
            if row.startswith(('MELODY:', 'rhythm:', '% pitches and words only', '% words without a rendered note:')):
                i += 1
                continue
            if i + 1 < len(hook_rows) and hook_rows[i + 1].startswith('w: '):
                # ABC must split pitches and syllables together, never wrap a
                # w: row as prose. Old saved hooks can exceed today's width.
                out.extend(listen_melody.split_rows(row.split(), hook_rows[i + 1][3:].split(), NOTES_WIDTH))
                i += 2
                continue
            emit(row)
            i += 1
    else:
        emit('  no saved hook')

    emit('')
    key = pick('KEY: ')
    meter = pick('METER: ')
    if not meter and eligible('singer', 'meter', 'beat tracker'):
        import meter_trust
        meter = meter_trust.assess(data['singer'].get('meter'))[0] or parsed['singer'].facts.get('METER', '')
    pace = _meter(pick('TEMPO: '), meter, data['singer'].get('meter') or {})
    home = [x for x in (key, pace) if x]
    # The old summary already gated key/chords/changes and intro corrections.
    ending = next((r.split(': ', 1)[1] for r in rows if r.startswith('HARMONY (') and ': ends on ' in r), '')
    if ending:
        reference = next((r.split('):', 1)[0] for r in rows if r.startswith('HARMONY (') and ': ends on ' in r), '')
        uncertain = re.compile(r'unclear|between|working|uncertain|ambiguous|\blean\b', re.I)
        if reference and (not key or uncertain.search(reference)):
            home.append(reference.removeprefix('HARMONY ('))
        match = re.fullmatch(r'ends on (\S+)(?: \(([^)]*)\))?; opens on (\S+)', ending)
        if match and key and not uncertain.search(key + ' ' + reference):
            last, chord, first = match.groups()
            home.append(f'opens on {first}, ends on {last}' + (' (home)' if last in ('I', 'i') else ''))
        else:
            home.append(ending)
    if eligible('harmony', 'cadences', 'chords', 'key fusion'):
        # Reuse RESOLVES' count and uncertainty, not unique transition times:
        # several phrase ends can refer to the same detected IV→I transition.
        resolves = re.search(r'^RESOLVES: (\d+) of (\d+) phrase ends (likely )?land on I(?: \(([^)]*)\))?',
                             cards['harmony'], re.M)
        if resolves:
            n, total, likely, kinds = resolves.groups()
            label = f'{n} of {total} judged phrase ends {likely or ""}land on I'
            if kinds == f'{n} plagal':
                label += ', all by IV→I'
            elif kinds:
                label += ' (' + kinds.replace('plagal', 'IV→I').replace('authentic', 'V→I') + ')'
            home.append(label)
    emit('HOME   ' + (' · '.join(home) if home else 'no saved home or meter decision'))
    emit(INVITATION)
    next_listen = _changed_word(lyrics)
    if next_listen:
        emit(next_listen)
    return '\n'.join(out) + '\n'
