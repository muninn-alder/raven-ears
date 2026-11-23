"""ROOM and WHISPER from the speech engine's existing windows and frame masks.

ROOM thresholds are first guesses, not calibrated probabilities of a finding.
Exact labels checked against AudioSet's class_labels_indices.csv (527 classes);
there is no Kettle whistle class. Related labels use their maximum, never a sum.
Two overlapping model windows are supporting scores, not independent evidence.
Times quote the event engine's center/hop convention, with ~2 s model context.

WHISPER uses timed words (>=80% active, <20% voiced, >=0.6 s per run), or
>=2 s of active, unvoiced stretch (pauses under 0.4 s bridged, >=1.5 s unvoiced) with AudioSet support.
Gaps >= the engine's GAP_MIN break word runs. Whispering cannot identify a speaker
or exclude breathy speech, masked pitch, bad word times or background sounds.
Frame-only and narrowband candidates remain ≈likely. No models run here.
"""
import os

import numpy as np

import stage_status as ST


# (display name, exact AudioSet labels, initial per-window threshold)
ROOM_CLASSES = (
    ('dog', ('Dog', 'Bark', 'Howl', 'Yip', 'Growling', 'Whimper (dog)'), .35),
    ('cat', ('Cat', 'Meow'), .35),
    ('bird song', ('Bird vocalization, bird call, bird song',), .35),
    ('rain', ('Rain',), .30),
    ('wind', ('Wind',), .35),
    ('tap/sink', ('Water tap, faucet', 'Sink (filling or washing)'), .35),
    ('boiling', ('Boiling',), .40),
    ('dishes/cutlery', ('Dishes, pots, and pans', 'Cutlery, silverware'), .35),
    ('door', ('Door',), .35),
    ('footsteps', ('Walk, footsteps',), .35),
    ('television', ('Television',), .30),
    ('music', ('Music',), .40),
    ('typing', ('Typing', 'Computer keyboard'), .35),
    ('vehicle/car', ('Vehicle', 'Car'), .35),
    ('telephone ringing', ('Telephone bell ringing',), .35),
    ('microwave', ('Microwave oven',), .35),
    ('frying', ('Frying (food)',), .35),
    ('clock tick', ('Tick', 'Tick-tock'), .35),
)

DOG_LABELS = ROOM_CLASSES[0][1]
DOG_WIN = 1.0
DOG_MIN_SCORE = .08


def voice_reduced(y32, fe, sr):
    """Attenuate voiced frames by 15 dB with ramps between frame centers."""
    gain = np.where(fe['voiced'], .18, 1.)
    envelope = np.interp(np.arange(len(y32)) / sr, fe['times'], gain).astype(np.float32)
    return y32 * envelope


def faint_dogs(centers, probs, labels, y32, fe, dur, sr, hop, *, reduced=False):
    """A per-clip score floor and an acoustic gate for the short dog windows."""
    from raven_ears_plus import _runs

    ix = {label: i for i, label in enumerate(labels)}
    family = [label for label in DOG_LABELS if label in ix]
    if not family or len(centers) < 3:
        return None
    scores = probs[:, [ix[label] for label in family]].max(axis=1)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('invalid dog scores')
    floor = float(np.median(scores))
    mad = float(np.median(np.abs(scores - floor)))
    threshold = max(DOG_MIN_SCORE, floor + .04, 2.5 * floor, floor + 4 * 1.4826 * mad)

    # Measure the acoustic floor on the waveform we actually scored. Use short
    # bins so a transient can qualify even below the original speech-active gate.
    indices = (fe['times'] * sr).round().astype(int)
    indices = indices[indices < len(y32)]
    energy = np.add.reduceat(np.square(y32), indices) / np.diff(np.r_[indices, len(y32)])
    db = 10 * np.log10(np.maximum(energy, 1e-10))
    floor_db = float(np.percentile(db, 20))
    salient = (db >= max(-75., floor_db + 6.)) & ~fe['voiced'][:len(db)]
    edges = np.searchsorted(fe['times'][:len(db)], [centers - DOG_WIN / 2, centers + DOG_WIN / 2])
    # A voiced-only window, digital silence, or steady room tone cannot open the lower gate.
    support = np.array([salient[a:b].sum() >= 4 for a, b in zip(*edges)])
    spans = []
    for a, b in _runs((scores >= threshold) & support):
        peak = a + int(scores[a:b].argmax())
        # One isolated hit needs extra margin. Two overlapping windows can localize
        # the same short bark, but are not independent evidence.
        if b - a == 1 and scores[peak] < threshold + .06:
            continue
        spans.append(dict(start=round(float(max(0., centers[a] - hop / 2)), 2),
                          end=round(float(min(dur, centers[b - 1] + hop / 2)), 2),
                          time=round(float(min(dur, centers[peak])), 2),
                          score=round(float(scores[peak]), 3)))
    return dict(method='voice-reduced' if reduced else 'raw', labels=family, spans=spans,
                score_floor=round(floor, 4), score_mad=round(mad, 4), threshold=round(threshold, 4),
                audio_floor_db=round(floor_db, 1), windows=[
                    dict(center=float(c), score=round(float(p), 4), audible=bool(ok))
                    for c, p, ok in zip(centers, scores, support)])


def event_context(centers, probs, labels, dur, hop, whisper_threshold):
    """Consume the SAME gated probabilities used by EVENTS; keep JSON numbers only."""
    from raven_ears_plus import _runs
    centers, probs = np.asarray(centers, dtype=float), np.asarray(probs, dtype=float)
    if (centers.ndim != 1 or probs.shape != (len(centers), len(labels))
            or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0)):
        raise ValueError('invalid event window times or probability shape')
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('unusable event probabilities; expected finite values in [0,1]')
    ix = {label: i for i, label in enumerate(labels)}
    hits, missing = [], []
    for name, family, threshold in ROOM_CLASSES:
        columns = [ix[label] for label in family if label in ix]
        missing.extend(label for label in family if label not in ix)
        if not columns:
            continue
        p = probs[:, columns].max(axis=1)
        hit = p >= threshold
        count = int(hit.sum())
        if count < 2:
            continue
        spans = [dict(start=round(float(max(0, centers[a] - hop / 2)), 2),
                      end=round(float(min(dur, centers[b - 1] + hop / 2)), 2),
                      score=round(float(p[a:b].max()), 3)) for a, b in _runs(hit)]
        hits.append(dict(name=name, labels=[label for label in family if label in ix],
                         score=round(float(p[hit].max()), 3), threshold=threshold,
                         windows=count, window_share=count / len(centers),
                         throughout=count / len(centers) > .60, spans=spans))
    hits.sort(key=lambda h: (-h['score'], h['name']))
    whisper = (probs[:, ix['Whispering']] if 'Whispering' in ix else None)
    return dict(room=dict(hits=hits, total_windows=len(centers), missing_labels=missing,
                          reason='' if hits else 'no ambient class cleared its threshold in at least 2 windows'),
                whisper_windows=(None if whisper is None else
                                 [dict(center=float(c), score=float(p)) for c, p in zip(centers, whisper)]),
                whisper_threshold=whisper_threshold)


def room_result(events, gaps, dog_passes=()):
    """WORDLESS's existing decision controls the lone-Music exception."""
    source = events['room']
    hits = source['hits']
    if len(hits) == 1 and hits[0]['name'] == 'music' and gaps.get('wordless'):
        source = dict(source, hits=[], reason='Music alone with WORDLESS voice; humming/singing can trigger Music')
    hits = list(source['hits'])
    dogs = [span for hit in hits if hit['name'] == 'dog' for span in hit['spans']]
    faint = []
    for result in dog_passes:
        for span in result['spans']:
            if any(span['start'] <= old['end'] and old['start'] <= span['end'] for old in dogs + faint):
                continue
            faint.append(dict(span, method=result['method']))
    if faint:
        faint.sort(key=lambda s: s['start'])
        hits.append(dict(name='dog', faint=True, spans=faint, score=max(s['score'] for s in faint)))
    return dict(source, hits=hits, reason='' if hits else source['reason'])


def whisper_result(words, fe, events, channel, gap_min, tag_win, duration=None, masked=None):
    """All candidates remain in data; rendering chooses the longest three.

    masked: timed ROOM masker hits, a supplied simultaneous masker name, or None.
    Huginn's review on real audio, 9 Nov 2026: with a song playing quietly behind a normal
    speaking voice, the pitch tracker lost the voice under the music and three ordinary
    words came back "likely whispered". Under a background sound, frame evidence alone is
    not evidence of a whisper, so a candidate without model support is withheld and counted.
    ROOM cannot place a sound behind the speaker: only overlapping hits can mask it.
    """
    from raven_ears_plus import FPS, _runs
    words = words or []
    times = np.asarray(fe['times'], dtype=float)
    active, voiced = np.asarray(fe['active']), np.asarray(fe['voiced'])
    if (times.ndim != 1 or active.shape != times.shape or voiced.shape != times.shape
            or not np.isfinite(times).all() or np.any(times < 0)
            or not np.isin(active, [False, True]).all() or not np.isin(voiced, [False, True]).all()):
        raise ValueError('invalid frame times or active/voiced masks')
    if len(times) < 2 or not np.allclose(np.diff(times), 1 / FPS, rtol=.01, atol=1e-6):
        raise ValueError('incomplete frame coverage; expected front-end cadence')
    limit = times[-1] if duration is None else duration
    active, voiced = active.astype(bool), voiced.astype(bool)
    windows = (events or {}).get('whisper_windows')
    threshold = (events or {}).get('whisper_threshold')
    if windows is not None:
        scores = np.asarray([w['score'] for w in windows], dtype=float)
        centers = np.asarray([w['center'] for w in windows], dtype=float)
        if (not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
                or not np.isfinite(centers).all()
                or np.any(centers < 0) or np.any(np.diff(centers) <= 0)
                or threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('unusable Whispering probabilities or window times')
    narrow = (channel or {}).get('classification') == 'narrowband'
    spans, run, withheld = [], [], []

    def support_at(start, end):
        # A supporting 2 s window must overlap at least half the shorter span.
        local = [w for w in (windows or [])
                 if min(end, w['center'] + tag_win / 2) - max(start, w['center'] - tag_win / 2)
                 >= .5 * min(end - start, tag_win)]
        support = [w for w in local if w['score'] >= threshold]
        # Count windows AND their union in time: overlapping scores are not independent,
        # and two isolated windows must not confirm a minute of unsupported words.
        covered, right = 0., start
        for w in sorted(support, key=lambda w: w['center']):
            a, b = max(start, w['center'] - tag_win / 2), min(end, w['center'] + tag_win / 2)
            covered += max(0., b - max(a, right))
            right = max(right, b)
        coverage = covered / (end - start)
        supported = len(support) >= 2 and len(support) / len(local) >= .6 and coverage >= .6
        return supported, dict(whisper_score=max((w['score'] for w in local), default=None),
                               support_windows=len(support), total_windows=len(local),
                               support_time_share=coverage, model_available=windows is not None)

    def finish():
        if not run:
            return
        start, end = words[run[0]][1], max(words[i][2] for i in run)
        if end - start < .6:
            return
        supported, evidence = support_at(start, end)
        confirmed = supported and not narrow
        maskers = ([masked] if isinstance(masked, str) else
                   [h['name'] for h in (masked or []) if any(
                       s['start'] < end and s['end'] > start for s in h['spans'])])
        if maskers and not supported:
            withheld.append(dict(start=float(start), end=float(end), maskers=maskers))
            return
        reason = ('frame evidence and AudioSet Whispering' if confirmed else
                  'narrowband CHANNEL; voicing unreliable' if narrow else
                  'event probabilities unavailable' if windows is None else
                  'insufficient overlapping Whispering support')
        selected = np.zeros(len(times), bool)
        for i in run:
            selected |= (times >= words[i][1]) & (times < words[i][2])
        spans.append(dict(start=float(start), end=float(end), words=len(run), indices=list(run),
                          confirmed=confirmed, label='whispered' if confirmed else 'likely whispered',
                          reason=reason, voiced_share=float(voiced[selected].mean()), **evidence))

    for i, (_word, start, end) in enumerate(words):
        sel = (times >= start) & (times < end)
        if (0 <= start < end <= limit and
                (start < times[0] - 1 / FPS or end > times[-1] + 1 / FPS)):
            raise ValueError('incomplete frame coverage at supplied word times')
        ok = (0 <= start < end <= limit and sel.any()
              and active[sel].mean() >= .8 and voiced[sel].mean() < .20)
        if run and (not ok or start - max(words[j][2] for j in run) >= gap_min):
            finish()
            run = []
        if ok:
            run.append(i)
    finish()

    # Two seconds matches one model context and excludes brief unvoiced bursts.
    # A real whisper pauses between words and catches its voice now and then, so
    # unvoiced runs separated by under BRIDGE_S of other frames join into one
    # stretch, which must still hold UNVOICED_MIN_S of active, unvoiced sound
    # (Fox's quiet whisper, 10 Nov 2026: Whispering x5, 11.4 s unvoiced, and no
    # single unbroken 2 s run). Never extend beyond the supplied masks.
    BRIDGE_S, UNVOICED_MIN_S = 0.4, 1.5
    unvoiced = active & ~voiced & (times < limit)
    joined = []
    for a, b in _runs(unvoiced):
        start, end = float(times[a]), float(min(limit, times[b - 1] + 1 / FPS))
        if joined and start - joined[-1][1] < BRIDGE_S:
            joined[-1][1] = end
        else:
            joined.append([start, end])
    frame_len = np.maximum(0., np.minimum(times + 1 / FPS, limit) - times)
    for start, end in joined:
        if end - start < 2.0:
            continue
        inside = (times >= start) & (times < end)
        if float(frame_len[inside & unvoiced].sum()) < UNVOICED_MIN_S:
            continue
        supported, evidence = support_at(start, end)
        if not supported:
            continue  # Breath, fans and rustle also produce active, unvoiced frames.
        overlapping = [s for s in spans if s['start'] < end and s['end'] > start]
        start = min([start] + [s['start'] for s in overlapping])
        end = max([end] + [s['end'] for s in overlapping])
        supported, evidence = support_at(start, end)
        if not supported:
            continue  # An enlarged span must pass the same coverage safeguards.
        indices = [i for i, (_word, s, e) in enumerate(words) if s < end and e > start]
        selected = (times >= start) & (times < end)
        lengths = np.maximum(0., np.minimum(times + 1 / FPS, end) - np.maximum(times, start))
        spans = [s for s in spans if s not in overlapping]
        voiced_here = float(voiced[selected].mean())
        # Whisper is a spectrum (Fox, 10 Nov 2026): a stretch that is partly
        # voiced is a soft, half-voiced whisper, not a plain one.
        label = 'likely whispered' if voiced_here < .20 else 'likely half-whispered (partly voiced)'
        spans.append(dict(start=start, end=end, words=len(indices), indices=indices,
                          confirmed=False, label=label,
                          reason='active unvoiced frames and AudioSet Whispering; transcription may miss words',
                          unvoiced_s=float(lengths[active & ~voiced].sum()),
                          voiced_share=float(voiced[selected].mean()), **evidence))
    spans.sort(key=lambda s: s['start'])
    reason = '' if spans else 'no qualifying word run or model-supported active, unvoiced run of at least 2 s'
    if withheld:
        names = ', '.join(dict.fromkeys(name for item in withheld for name in item['maskers']))
        reason = (f'{len(withheld)} unsupported candidate(s) withheld: overlapping {names} '
                  'can mask the pitch of a normal voice')
    return dict(spans=spans, reason=reason, withheld=len(withheld), withheld_spans=withheld)


def exclude_unvoiced(gaps, whisper):
    """Subtract every accepted candidate, including spans omitted from the compact card."""
    remaining = list(gaps.get('unvoiced') or [])
    for span in whisper['spans']:
        start, end = span['start'], span['end']
        cut = []
        for a, b in remaining:
            if b <= start or a >= end:
                cut.append((a, b))
            else:
                if a < start:
                    cut.append((a, start))
                if end < b:
                    cut.append((end, b))
        remaining = cut
    gaps.update(unvoiced=remaining, unvoiced_s=round(sum(b - a for a, b in remaining), 2))


def enrich(data, fe, words, notes, gap_min, tag_win, duration, dog_passes=(), *, room=None):
    """Small engine hook; failures reach NOTES, never look like zero detections."""
    data.update(room=None, whisper=None)
    if room is None:
        room = (data.get('events') or {}).get('room')
    for stage in ('room', 'whisper'):
        if stage == 'room' and room is None:
            continue  # The scan already owns the unavailable/disabled reason.
        if os.environ.get('RAVEN_EARS_' + stage.upper(), '1') == '0':
            ST.mark(stage, 'skipped', 'RAVEN_EARS_' + stage.upper() + '=0')
            continue
        try:
            if stage == 'room':
                data[stage] = room_result({'room': room}, data['gaps'], dog_passes)
            else:
                heard = (room or {}).get('hits') or []
                # Continuous sounds that can bury a voice's pitch. A dog or a door cannot.
                maskers = ('music', 'television', 'rain', 'wind', 'tap/sink', 'frying', 'vehicle/car')
                loud = [h for h in heard if h['name'] in maskers
                        and (h['throughout'] or h['name'] in ('music', 'television'))]
                data[stage] = whisper_result(words, fe, data.get('events'), data.get('channel'),
                                             gap_min, tag_win, duration, masked=loud)
                exclude_unvoiced(data['gaps'], data[stage])
            ST.mark(stage)
        except Exception as exc:
            data[stage] = None
            notes.append(f'{stage.upper()} skipped: {type(exc).__name__}: {str(exc)[:100]}')
            ST.mark(stage, 'failed', f'{type(exc).__name__}: {str(exc)[:100]}')


def room_line(result):
    from raven_ears_plus import _fit, _t
    if result is None:
        return None
    hits = result.get('hits', [])
    faint = next((hit for hit in hits if hit.get('faint')), None)
    pieces = []
    if faint:
        times = [_t(span['time']) for span in faint['spans']]
        count = min(8, len(times))
        omitted = f' +{len(times) - count} spans' if count < len(times) else ''
        pieces.append(f"dog, faint ({', '.join(times[:count])}){omitted}")
    for hit in [h for h in hits if not h.get('faint')][:4 - len(pieces)]:
        span = max(hit['spans'], key=lambda s: (s['score'], s['end'] - s['start'], -s['start']))
        a, b = _t(span['start']), _t(span['end'])
        when = a if a == b else f'{a}-{b}'
        if len(hit['spans']) > 1:
            when += f" +{len(hit['spans']) - 1} spans"
        pieces.append(f"{hit['name']} {when} ({hit['score']:.2f})")
    return _fit('ROOM', pieces) if pieces else '  ROOM:    nothing stands out'


def whisper_line(result):
    from raven_ears_plus import _fit, _t, MAX_LINE
    spans = sorted((result or {}).get('spans', []), key=lambda s: (s['start'] - s['end'], s['start']))[:3]
    # Fit by duration priority, then show the retained spans in time order.
    while spans:
        pieces = []
        for span in sorted(spans, key=lambda s: s['start']):
            a, b = _t(span['start']), _t(span['end'])
            when = a if a == b else f'{a}–{b}'
            label = 'whispered' if span['confirmed'] else '≈likely whispered'
            if 'unvoiced_s' in span:
                # The frame route's own label carries "half-whispered" (Fox: a spectrum).
                label = '≈' + span.get('label', 'likely whispered')
                count = span['words']
                piece = (f"{when} ({label}; {span['unvoiced_s']:.1f}s unvoiced; "
                         f"Whispering {span['support_windows']} windows, peak {span['whisper_score']:.2f}; "
                         f"{count} {'word' if count == 1 else 'words'} transcribed)")
                # Shorten in steps until it fits; never let _fit drop it to a bare label.
                for long, short in ((' windows, peak ', 'w peak '), (' (partly voiced)', ''),
                                    (' transcribed)', ')')):
                    if len('  WHISPER:  ' + piece) <= MAX_LINE:
                        break
                    piece = piece.replace(long, short)
                pieces.append(piece)
                continue
            voicing = f"{span['voiced_share']:.0%} voiced"
            support = (f"AS support {span['support_windows']}/{span['total_windows']}, "
                       f"{span['support_time_share']:.0%} time" if span['model_available'] else 'AS unavailable')
            pieces.append(f"{when} ({span['words']} words, {label}; {voicing}; {support})")
        line = _fit('WHISPER', pieces)
        if len(pieces) == 1 or all(piece in line for piece in pieces):
            return line
        spans.pop()
    return None


def whisper_tokens(result, words):
    """Use PEAK's open/close ordering, retaining emphasis and event markers."""
    tokens = []
    for span in (result or {}).get('spans', []):
        if not span['indices']:
            continue  # A frame-only span has no WORDS to bracket.
        label = 'WHISPER' if span['confirmed'] else '≈LIKELY WHISPER'
        tokens.extend([(span['start'], -1, f'⟪{label}⟫'),
                       (words[span['indices'][-1]][1], 3, '⟪/WHISPER⟫')])
    return tokens
