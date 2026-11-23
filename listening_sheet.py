"""Optional local PNGs of existing evidence; no model or network imports.

Song facts come from listen's decided first-listen card. Older bundles retain
section means rather than frame loudness: display those steps and say so.
The background is computed from audio, once per content-hash bundle. Recurrence
uses that spectrum; optional stem curves only read existing cached WAVs.
"""
import os
import math
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap

WIDTH, HEIGHT = 2400, 1200
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED = "#202C38", "#576574"
REPEATED_LYRIC = "#747D86"
DISAGREEMENT = "#F6E3D8"
CONTRAST_MIN_BAND_S = 1.0
AD_LIB = re.compile(r"(?:[ao]+h+|ha+|e+h+|h?m+|wo+(?:ah)?|who+a+|hey+|yea+h*|uh+|um+|la+|na+)", re.I)
STAMP = r"\d+:\d{2}(?:\.\d+)?"
UNCLEAR = re.compile(r"\b(unclear|uncertain|ambiguous|working|between|lean)\b", re.I)
LIKELY = re.compile(r"\b(likely|possible|candidate|somewhere)\b", re.I)


def add_picture_args(parser, *, help):
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--picture", action="store_true", default=None,
                       help=help + " (default on; also RAVEN_EARS_PICTURE=1)")
    flags.add_argument("--no-picture", dest="picture", action="store_false",
                       help="omit the listening sheet (also RAVEN_EARS_PICTURE=0; flags win)")


def prepare_picture(args, *, text_card, error):
    """Resolve CLI/env defaults before collecting frames or attempting a draw."""
    implied = bool(args.by_ear or getattr(args, "picture_out", None))
    explicit = args.picture is True or (args.picture is None and implied)
    if not text_card:
        if explicit:
            raise ValueError(error)
        return False, None
    enabled = (args.picture if args.picture is not None else
               implied or os.environ.get("RAVEN_EARS_PICTURE") != "0")
    if not enabled:
        return False, None
    if not explicit:
        try:
            import matplotlib
        except Exception:
            # An absent/broken optional install must not change the text listen.
            return False, "pip install -r requirements-picture.txt to draw the listening sheet"
    return True, None


def attempt(draw, *args, **kwargs):
    """A picture failure must never suppress the text card."""
    try:
        path = draw(*args, **kwargs)
        return f"PICTURE: {path}"
    except ImportError as error:
        if (error.name or "").startswith("matplotlib"):
            return "pip install matplotlib to draw the listening sheet"
        return f"PICTURE unavailable: {error}"
    except Exception as error:
        return f"PICTURE unavailable: {error}"


def _seconds(stamp):
    minutes, seconds = stamp.split(":")
    return int(minutes) * 60 + float(seconds)


def _clock(value, _position=None):
    seconds = max(0, round(value))
    return f"{seconds // 60}:{seconds % 60:02d}"


def by_ear_rows(path, duration):
    """Human annotations are picture input only; never attach them to a card."""
    if path is None:
        return []
    rows = []
    for number, row in enumerate(Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines(), 1):
        if not row.strip():
            continue
        match = re.fullmatch(r"\s*(\d+:[0-5]\d(?:\.\d+)?)\s+(.+?)\s*", row)
        if not match or _seconds(match[1]) > duration:
            raise ValueError(f"by ear line {number}: expected m:ss label within the recording")
        rows.append((_seconds(match[1]), _seconds(match[1]), match[2], False))
    return rows


def _song_gaps(song, full):
    """Use refreshed STOPS and explicit card gaps, never infer silence from RMS."""
    gaps = []
    for name in ("singer", "music", "harmony"):
        if song["manifest"]["passes"].get(name, {}).get("status") != "ran":
            continue
        data = song["cards"][name].get("json") or {}
        result = data.get("stops")
        if isinstance(result, dict):
            import stops
            refreshed = stops.refresh(result)
            for event in refreshed.get("events", []) if refreshed.get("status") == "ran" else []:
                if event["kind"] == "uncertain":
                    continue  # The full card does not call these stops either.
                start = event["start_s"]
                end = event.get("return_s", start + event["length_s"])
                label = event["kind"] + (" (partial cut)" if event.get("arrangement") == "partial cut" else "")
                gaps.append((start, end, label, False))
                voice = event.get("voice_break")
                if voice:
                    gaps.append((voice["start_s"], voice["return_s"], "voice break", False))
        singer = data.get("singer")
        vocal_data = (singer.get("data") or {}) if isinstance(singer, dict) else {}
        for start, end in (vocal_data.get("gaps") or {}).get("silent") or []:
            gaps.append((start, end, "vocal-stem silence", False))
    # Old harmony bundles save only the printed timeline. Its leading clock is
    # the onset; drop-out prose has no measured end, so draw an onset only.
    current, timeline = None, False
    for row in full.splitlines():
        clean = row.strip()
        if clean.startswith("HARMONY:"):
            current = "harmony"
        elif row and not row[0].isspace():
            timeline = clean == "TIMELINE"
        if current != "harmony" or not timeline:
            continue
        if song["manifest"]["passes"].get("harmony", {}).get("status") != "ran":
            continue
        match = re.match(rf"({STAMP})\s+(near-silence for (\d+(?:\.\d+)?) s|drop-out:)", clean)
        if match:
            start = _seconds(match[1])
            gaps.append((start, start + float(match[3]), "near-silence", False) if match[3] else
                        (start, start, "drop-out (end not measured)", False))
    return list(dict.fromkeys(gaps))


def _repeat_groups(power, duration):
    """Conservative picture-only spectral returns, not a section/key vote.

    Compare ~8 s means after removing level and the song's persistent spectral
    shape. Complete-link expansion prevents chains of merely similar pairs.
    A group needs a full intervening segment outside it to count as a return.
    Thresholds are display heuristics, not calibrated musical probabilities.
    """
    import numpy as np
    count = max(1, min(480, power.shape[1], round(duration / 8)))
    parts = np.array_split(np.maximum(power, -65), count, axis=1)
    pooled = np.stack([part.mean(axis=1) for part in parts], axis=1)
    edges = np.r_[0, np.cumsum([part.shape[1] for part in parts])] * (duration / power.shape[1])
    valid = np.isfinite(pooled).all(axis=0) & (pooled.max(axis=0) > -60)
    shaped = pooled - pooled.mean(axis=0)
    if valid.any():
        shaped -= shaped[:, valid].mean(axis=1, keepdims=True)
    norm = np.linalg.norm(shaped, axis=0)
    valid &= norm / np.sqrt(power.shape[0]) >= 1  # At least 1 dB residual contrast.
    shaped /= np.maximum(norm, 1e-8)
    matrix = np.clip(shaped.T @ shaped, -1, 1)
    returns = []
    available = valid.copy()
    separated = abs(np.arange(count)[:, None] - np.arange(count)) > 1
    while True:
        # Seed the strongest nonadjacent pair before adding nearby continuations;
        # adjacent sustained timbre must not consume a later returning passage.
        pairs = np.where(separated & available[:, None] & available[None, :], matrix, -np.inf)
        if pairs.max() < .80:
            break
        a, b = np.unravel_index(np.argmax(pairs), pairs.shape)
        members = [int(a), int(b)]
        available[members] = False
        minimum = np.minimum(matrix[a], matrix[b])
        while True:
            minimum[~available] = -np.inf
            if minimum.max() < .80:
                break
            i = int(np.argmax(minimum))
            members.append(i)
            available[i] = False
            minimum = np.minimum(minimum, matrix[i])
        members.sort()
        if not any(b - a > 1 for a, b in zip(members, members[1:])):
            continue  # Sustained adjacent material alone is not a return.
        score = float(matrix[np.ix_(members, members)].min())
        returns.append((members, score))
    # Six distinct colour-blind-safe hues; never reuse a hue for another group.
    # If unusually many groups qualify, display the six strongest, rest grey.
    returns = sorted(sorted(returns, key=lambda r: (-r[1], r[0][0]))[:6], key=lambda r: r[0][0])
    groups, strength = np.full(count, -1, dtype=int), np.zeros(count)
    for group, (members, score) in enumerate(returns):
        groups[members], strength[members] = group, score
    return edges, groups, strength


def _cached_stem_energy(audio_hash, duration):
    """Read only hash-matched cached WAVs. Never import singer or separate.

    Four-stem and six-stem bass/drums/other are different model estimates; prefer
    the four-stem set, falling back to six, keeping its source visible.
    """
    import numpy as np
    root = Path(os.environ.get("RAVEN_STEMS_CACHE", "~/.cache/raven-stems")).expanduser()
    if not re.fullmatch(r"[a-fA-F0-9]{64}", audio_hash):
        return []
    curves, seen = [], set()
    for suffix, names in (("", ("drums", "bass", "other", "vocals")),
                          ("-6s", ("drums", "bass", "other", "vocals", "guitar", "piano"))):
        for name in names:
            if name in seen:
                continue
            path = root / (audio_hash[:16] + "-a0" + suffix) / (name + ".wav")
            if not path.is_file():
                continue
            try:
                import soundfile as sf
                with sf.SoundFile(path) as audio:
                    if abs(audio.frames / audio.samplerate - duration) > .1:
                        continue  # No stretching, guessing offsets or mismatched endings.
                    hop = max(1, round(audio.samplerate * max(.1, duration / 2400)))
                    energy = np.array([np.mean(block.astype(np.float64) ** 2)
                                       for block in audio.blocks(blocksize=hop, dtype="float32", always_2d=True)])
                    if not len(energy) or not np.isfinite(energy).all():
                        continue
                    db = 10 * np.log10(np.maximum(energy, 1e-12))
                    curves.append((name + (" (6s)" if suffix else ""),
                                   (np.arange(len(db)) + .5) * hop / audio.samplerate, db))
                    seen.add(name)
            except (ImportError, OSError, RuntimeError, ValueError):
                continue  # Optional cached evidence must not prevent the sheet.
    return curves


def _facts(text):
    """Keep complete wrapped decisions, including their uncertainty qualifiers."""
    result, current = {}, None
    for row in text.splitlines():
        match = re.match(r"^(KEY|TEMPO|METER):\s*(.*)", row)
        if match:
            current = match[1]
            result[current] = match[2]
        elif current and row.startswith("  "):
            result[current] += " " + row.strip()
        else:
            current = None
    return result


def _lyric_phrases(lyrics, words, *, line_starts=False):
    """Split card words only when their saved word times match, never rewrite.

    Writer/LRCLIB lines can differ from Whisper. Unmatched lines retain the
    card's timing and text rather than borrowing times from different words.
    For line labels, only a capitalised restart after a saved pause splits a
    card line; punctuation and shorter phrases still contribute ticks.
    """
    import slim
    timed = []
    for word in words or []:
        if (not isinstance(word, (list, tuple)) or len(word) != 3
                or not isinstance(word[0], str)):
            return lyrics
        text, start, end = word
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end < start or (timed and start < timed[-1][1])):
            return lyrics
        timed.append((slim.token(text), start, end))
    result = []
    for row in lyrics:
        body = re.sub(r"\s*\(unsure[^)]*\)", "", row.text, flags=re.I).strip()
        tokens = body.split()
        keys = [slim.token(t) for t in tokens]
        matches = [i for i, (_, start, _) in enumerate(timed)
                   if abs(start - row.t) <= 1 and keys and all(keys)
                   and [w[0] for w in timed[i:i + len(keys)]] == keys]
        if len(matches) != 1:
            result.append(row)
            continue
        words_here = timed[matches[0]:matches[0] + len(keys)]
        boundaries = [0]
        for i in range(1, len(tokens)):
            # Capital I inside a clause ("that I could") is not a restart.
            # A fresh I after several words is useful when punctuation is lost.
            restart = (i + 1 < len(tokens) and tokens[i + 1].lower()
                       in ("only", "don't", "want", "wanna", "just"))
            capital = tokens[i][0].isupper() and (tokens[i] != "I" or restart or i - boundaries[-1] >= 4)
            pause = words_here[i][1] - words_here[i - 1][2] >= .3 - 1e-8
            boundary = (tokens[i][0].isupper() and pause if line_starts else
                        re.search(r"[,;:.!?][\"'’”]*$", tokens[i - 1]) or capital or pause)
            if boundary:
                boundaries.append(i)
        boundaries.append(len(tokens))
        unsure = re.search(r"\(unsure[^)]*\)", row.text, flags=re.I)
        for a, b in zip(boundaries, boundaries[1:]):
            text = " ".join(tokens[a:b]) + (" " + unsure[0] if unsure else "")
            result.append(slim.Lyric(words_here[a][1], text, row.approx, words_here[b - 1][2]))
    return result


def song_evidence(song, decided, full):
    """Projection of saved cards. No alternative key, meter, event or lyric vote."""
    import feeling_card
    import instrumental_voice
    import slim
    if re.search(r'(?m)^FEELING CARD:', decided):
        # The picture keeps the same decided evidence through the presentation change.
        import listen
        decided = listen.render(song, 'summary')[0]
    facts = _facts(decided)
    manifest = song["manifest"]
    options = manifest.get("options") or {}
    title = " — ".join(x for x in (options.get("artist"), options.get("title")) if x)
    if not title:
        header = next((r for r in decided.splitlines() if r.startswith("FIRST LISTEN:")), "")
        title = header.removeprefix("FIRST LISTEN:").split(" · ")[0].strip() or "Saved song"
    music = song["cards"].get("music", {})
    singer = song["cards"].get("singer", {})
    ran = lambda name: manifest["passes"].get(name, {}).get("status") == "ran"
    parsed = slim.parse(music.get("text", ""))
    sections = []
    # SHAPE must have survived the card's stage-status policy.
    if ran("music") and re.search(r"(?m)^SHAPE:", decided):
        for row in parsed.blocks["shape"]:
            match = re.match(rf"({STAMP})[–-]({STAMP})\s+(.+?)\s+(-?\d+(?:\.\d+)?) dB", row)
            if match:
                sections.append({"start": _seconds(match[1]), "end": _seconds(match[2]),
                                 "label": match[3], "db": float(match[4]), "peak": "PEAK" in row})
        labels = parsed.facts.get("SHAPE", "").split(" → ")
        if len(labels) == len(sections):
            for section, label in zip(sections, labels):
                section["label"] = re.sub(STAMP, "", label).strip()
    # Only the decided KEY CHANGES rows supply ranges. Never use raw voters or
    # harmony's older raw_changes, which may disagree with the displayed card.
    keys = []
    for row in decided.splitlines():
        if row.startswith("KEY CHANGES:") and not UNCLEAR.search(row):
            times = re.findall(STAMP, row)
            if len(times) == 2:
                keys.append((_seconds(times[0]), _seconds(times[1]), row.removeprefix("KEY CHANGES: ")))
    events = [(s["start"], s["end"], "section PEAK", False) for s in sections if s["peak"]]
    # Card event prose stays attached to the time it actually names. In
    # particular, a range's beginning is not presented as its peak's instant.
    block, current_pass = "", None
    for row in full.splitlines():
        clean = row.strip()
        for name, prefix in (("music", "MUSIC:"), ("singer", "SINGER.PY"), ("harmony", "HARMONY:")):
            if clean.startswith((prefix, name + ":")):
                current_pass = name
        if current_pass and not ran(current_pass):
            continue
        if clean.startswith(("CHILL CANDIDATES", "ACHE")):
            block = "candidate"
        elif clean in ("TENSION", "TIMELINE"):
            block = clean
        elif row and not row[0].isspace():
            block = ""
        event_row = block in ("TENSION", "TIMELINE", "candidate") or clean.startswith("most intense moment:")
        wanted = event_row and re.search(r"peak at|most intense moment:|\bdrop\b|likely to land", clean, re.I)
        if not wanted and not (block == "candidate" and re.match(STAMP, clean)):
            continue
        if UNCLEAR.search(clean):
            continue
        if re.search(r"\bmodulates?\b|key (?:change|area|moves)", clean, re.I):
            continue  # Older harmony candidates are not the deciding key-change scan.
        times = re.findall(STAMP, clean)
        if times:
            # A tension range line also names a peak; only its 'peak at' time
            # is the event, not the preceding range's bounds.
            peak = re.search(rf"peak at ({STAMP})", clean)
            a = _seconds(peak[1] if peak else times[0])
            b = a if peak or len(times) < 2 else _seconds(times[1])
            label = ("tension peak" if peak else
                     re.sub(rf"{STAMP}(?:[–-]{STAMP})?", "", clean).strip(" :·,[]"))
            likely = block == "candidate" or bool(LIKELY.search(clean))
            if likely and not LIKELY.search(label):
                label = "candidate: " + label
            events.append((a, b, label, likely))
    lyrics = slim.parse(decided).lyrics
    contrast = {}
    if not instrumental_voice.saved_designation(song["cards"], manifest["passes"]):
        harmony = song["cards"].get("harmony", {})
        contrast = feeling_card.contrast_readings(
            feeling_card.song_lyrics(feeling_card.logical_rows(decided)),
            harmony.get("text", ""), harmony.get("json") or {}, manifest["passes"].get("harmony", {}))
    pitch = ((singer.get("json") or {}).get("melody_input") or {}) if ran("singer") else {}
    stages = (singer.get("json") or {}).get("stage_status") or []
    words_ok = all(r.get("status") == "ran" for r in stages if r.get("stage") == "words")
    phrases = _lyric_phrases(lyrics, pitch.get("words")) if words_ok else lyrics
    lines = _lyric_phrases(lyrics, pitch.get("words"), line_starts=True) if words_ok else lyrics
    curve = ((music.get("json") or {}).get("listening_sheet") or {}) if ran("music") else {}
    return {"title": title, "facts": facts, "sections": sections, "keys": keys,
            "events": list(dict.fromkeys(events)), "lyrics": lyrics, "phrases": phrases, "lyric_lines": lines,
            "pitch": pitch, "curve": curve, "contrast": sorted(contrast.values()),
            "gaps": _song_gaps(song, full)}


def _spectrum(audio, cache, audio_hash):
    """Stream a cheap full-mix STFT; retain at most 2400 display columns.

    Cache contains numeric arrays/strings only (never pickle). FFT batches keep
    working memory bounded; no full-song complex STFT is held in memory.
    """
    from network_policy import local_audio_path
    local_audio_path(audio)
    import numpy as np
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                if saved["version"].item() == 2 and saved["sha256"].item() == audio_hash:
                    return saved["power"], saved["frequencies"], float(saved["duration"])
        except (OSError, ValueError, KeyError, EOFError):
            pass
    sr, frame, hop = 22050, 2048, 512
    edges = np.geomspace(30, sr / 2, 129)
    frequencies = np.fft.rfftfreq(frame, 1 / sr)
    bins = [np.flatnonzero((frequencies >= a) & (frequencies < b)) for a, b in zip(edges[:-1], edges[1:])]
    bins = [indices if len(indices) else np.array([np.argmin(abs(frequencies - a))])
            for indices, a in zip(bins, edges[:-1])]
    window = np.hanning(frame)
    pieces, tail, count = [], np.empty(0, dtype=np.float32), 0
    # stderr goes to a temporary file so a decoder error cannot fill a pipe.
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-protocol_whitelist", "file,pipe", "-i", str(audio), "-map", "0:a:0",
                               "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
                              stdout=subprocess.PIPE, stderr=errors) as process:
            while True:
                raw = process.stdout.read(hop * 256 * 4)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype="<f4")
                count += len(samples)
                tail = np.concatenate((tail, samples))
                if len(tail) < frame:
                    continue
                frames = np.lib.stride_tricks.sliding_window_view(tail, frame)[::hop]
                power = abs(np.fft.rfft(frames * window, axis=1)) ** 2
                pieces.append(np.stack([power[:, indices].mean(axis=1) for indices in bins]))
                tail = tail[len(frames) * hop:].copy()
            if process.wait():
                raise ValueError("could not decode full mix for the picture")
    if not count:
        raise ValueError("no audio samples for the picture")
    # Keep the final partial frame, including recordings shorter than one FFT.
    if len(tail):
        power = abs(np.fft.rfft(np.pad(tail, (0, frame - len(tail))) * window)) ** 2
        pieces.append(np.array([[power[indices].mean()] for indices in bins]))
    power = np.concatenate(pieces, axis=1)
    if power.shape[1] > 2400:
        power = np.stack([part.mean(axis=1) for part in np.array_split(power, 2400, axis=1)], axis=1)
    power = (10 * np.log10(np.maximum(power, 1e-20) / max(float(power.max()), 1e-20))).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, version=2, sha256=audio_hash, power=power,
                            frequencies=edges, duration=count / sr)
        os.replace(temporary, cache)
    finally:
        temporary.unlink(missing_ok=True)
    return power, edges, count / sr


def speech_spectrum(samples, sr):
    """A bounded display STFT from speech's already-decoded mono samples."""
    import numpy as np
    frame = 512
    columns = min(2400, math.ceil(len(samples) / (frame // 2)))
    edges = np.linspace(0, len(samples), columns + 1)
    centres = ((edges[:-1] + edges[1:]) / 2).astype(int)
    padded = np.pad(samples, frame // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[centres]
    window = np.hanning(frame)
    amplitude = abs(np.fft.rfft(frames * window, axis=1)[:, 1:]) / (window.sum() / 2)
    frequencies = (np.arange(frame // 2 + 1) + .5) * sr / frame
    return edges / sr, frequencies, (20 * np.log10(np.maximum(amplitude, 1e-5))).T.astype(np.float32)


def _plotting():
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    # rc_context also isolates caller preferences (including TeX and fonts).
    settings = dict(matplotlib.rcParamsDefault)
    settings.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "text.usetex": False, "text.parse_math": False,
                     "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "axes.edgecolor": MUTED,
                     "text.color": INK, "axes.labelcolor": INK})
    context = matplotlib.rc_context(settings)
    return context, Figure, FigureCanvasAgg


def _axes(figure, heights, left=.065):
    axes = figure.subplots(len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights, "hspace": .15})
    figure.subplots_adjust(left=left, right=.94, bottom=.065, top=.79)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.13)
    return axes


def _wrap_measured(artist, text, width, renderer):
    """Wrap in display units, including arbitrarily long unbroken tokens."""
    font = artist.get_fontproperties()
    measure = lambda value: renderer.get_text_width_height_descent(value, font, ismath=False)[0]
    lines = []
    for row in text.splitlines():
        line = ""
        for word in row.split():
            candidate = (line + " " + word).strip()
            if line and measure(candidate) > width:
                lines.append(line)
                line = ""
            while measure(word) > width:
                lo, hi = 1, len(word)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if measure(word[:mid]) <= width:
                        lo = mid
                    else:
                        hi = mid - 1
                lines.append(word[:lo])
                word = word[lo:]
            line = (line + " " + word).strip()
        lines.append(line)
    return "\n".join(lines)


def _header(figure, title, subtitle, footer, plot_height=5.8):
    """Stack measured blocks; grow very long sheets to preserve plot height."""
    renderer = figure.canvas.get_renderer()
    blocks = []
    width = figure.bbox.width * (.94 - .065)
    for value, size, colour, weight in ((title, 18, INK, "bold"),
                                       (subtitle, 12, INK, "normal"),
                                       (footer, 10, MUTED, "normal")):
        artist = figure.text(.065, 0, "", fontsize=size, color=colour, weight=weight, va="top")
        artist.set_text(_wrap_measured(artist, value, width, renderer))
        blocks.append((artist, artist.get_window_extent(renderer).height / figure.dpi))
    # Inches stay constant when the figure grows; font size and plot readability
    # do not depend on how much quoted card prose happens to precede the lanes.
    header_height = sum(height for _, height in blocks) + .12
    key_blocks, key_height = _bottom_key(figure)
    bottom = .52 + key_height
    height = max(HEIGHT / 150, .22 + header_height + .18 + plot_height + bottom)
    figure.set_size_inches(WIDTH / 150, height)
    top = height - .22
    for artist, block_height in blocks:
        artist.set_y(top / height)
        top -= block_height + .06
    figure.subplots_adjust(top=(height - .22 - header_height - .18) / height, bottom=bottom / height)
    for artist, offset in key_blocks:
        artist.set_y((key_height - offset) / height)


def _short_event(text):
    """Remove stock wording, retaining every measured number and qualifier."""
    text = re.sub(r"^candidate:\s*", "", text, flags=re.I)
    text = text.replace("the level falls", "level falls")
    text = text.replace(" and stays down; the band thins", "; stays down, band thins")
    return " ".join(text.split())


def _key_clock(value):
    """Keep subsecond saved edges distinguishable in the key."""
    minutes, seconds = divmod(round(max(0, value), 2), 60)
    if seconds % 1:
        return f"{int(minutes)}:{seconds:05.2f}".rstrip("0").rstrip(".")
    return f"{int(minutes)}:{int(seconds):02d}"


def _numbered_evidence(events, gaps, ear, duration):
    rows = [(a, b, label, likely, kind) for kind, source in
            (("card", events), ("gap", gaps), ("by ear", ear))
            for a, b, label, likely in source if 0 <= a <= duration]
    return [dict(number=i, start=a, end=b, label=_short_event(label) if kind == "card" else label,
                 likely=likely, kind=kind)
            for i, (a, b, label, likely, kind) in enumerate(sorted(rows, key=lambda r: r[0]), 1)]


def _bottom_key(figure):
    """Two measured columns below the time axis; long entries continue intact."""
    renderer = figure.canvas.get_renderer()
    entries = getattr(figure, "_sheet_key", [])
    wrapped = []
    for row in entries:
        at = _key_clock(row["start"])
        if row["end"] > row["start"]:
            at += "–" + _key_clock(row["end"])
        source = (row["kind"] + ": ") if row["kind"] != "card" else ""
        value = f'{row["number"]}  {at}  {source}{row["label"]}'
        lines = textwrap.wrap(value, width=60, subsequent_indent="    ")
        wrapped.append((row, "\n".join(lines)))
    total = sum(value.count("\n") + 1 for _, value in wrapped)
    blocks, offset, column = [], 0, 0
    line_height = .21
    split = max(1, math.ceil(total / 2))
    for row, value in wrapped:
        if column == 0 and offset / line_height >= split:
            column, offset = 1, 0
        artist = figure.text(.11 + column * .425, 0, value, fontsize=11, va="top",
                             color=BLUE if row["kind"] == "by ear" else INK)
        artist.set_text(_wrap_measured(artist, value, figure.bbox.width * .40, renderer))
        blocks.append((artist, offset))
        offset += max(line_height, artist.get_window_extent(renderer).height / figure.dpi + .045)
    key_height = max((offset + artist.get_window_extent(renderer).height / figure.dpi
                      for artist, offset in blocks), default=0)
    notes = getattr(figure, "_sheet_notes", "")
    if notes:
        artist = figure.text(.11, 0, "", fontsize=9, color=MUTED, va="top")
        artist.set_text(_wrap_measured(artist, notes, figure.bbox.width * .83, renderer))
        blocks.append((artist, key_height + .14))
        key_height += .14 + artist.get_window_extent(renderer).height / figure.dpi
    return blocks, key_height + .15 if blocks else 0


def _finish(figure, axes, duration, output, title, subtitle, footer):
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    axes[-1].set_xlim(0, max(duration, .1))
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 3, 5, 10]))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_clock))
    axes[-1].set_xlabel("Time (m:ss)")
    plot_height = getattr(figure, "_sheet_plot_height", 5.8 + max(0, figure.get_figheight() - HEIGHT / 150))
    _header(figure, title, subtitle, footer, plot_height)
    return _save(figure, output)


def _save(figure, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: a failed draw leaves any previous sheet intact.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=150, facecolor="white", metadata={"Software": "raven-ears listening sheet"})
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        figure.clear()
    return output


def _section_labels(ax, sections, duration):
    """Centre labels inside their own spans; shorten or omit sub-glyph spans."""
    renderer = ax.figure.canvas.get_renderer()
    for i, section in enumerate(sections):
        start, end = max(0, section["start"]), min(duration, section["end"])
        ax.axvline(start, color="white", alpha=.6, lw=.8)
        # Also respect the next boundary if a saved row has overlapping ends.
        if i + 1 < len(sections):
            end = min(end, sections[i + 1]["start"])
        if end <= start:
            continue
        artist = ax.text((start + end) / 2, .97 - (i % 2) * .17, section["label"],
                         ha="center", va="top", transform=ax.get_xaxis_transform(),
                         color="white", fontsize=11, clip_on=True, zorder=4,
                         bbox=dict(facecolor="black", edgecolor="none", alpha=.65, pad=1.5))
        available = (end - start) / max(duration, .1) * ax.bbox.width - 12
        limit = len(section["label"])
        while limit > 1 and artist.get_window_extent(renderer).width > available:
            limit -= 1
            artist.set_text(textwrap.shorten(section["label"], width=limit, placeholder="…"))
        if artist.get_window_extent(renderer).width > available:
            artist.set_text("")  # Even an ellipsis would cross this boundary.


def _labels(ax, rows, duration, lanes=5, color=INK, bottom=0):
    """Stagger abbreviated text with a visible anchor at its measured time."""
    ax.set_ylim(-.5, lanes + bottom - .2)
    ax.set_yticks([])
    rows = sorted((r for r in rows if 0 <= r[0] <= duration), key=lambda r: r[0])
    renderer = ax.figure.canvas.get_renderer()
    for i, (start, end, text, likely) in enumerate(rows):
        lane = i % lanes
        next_start = rows[i + lanes][0] if i + lanes < len(rows) else duration
        # Reserve each label's actual available width; crowded lyrics become
        # shorter excerpts rather than overlapping illegibly. The full words
        # remain on the accompanying card.
        limit = max(1, min(34, int((next_start - start) / max(duration, .1) / .006) - 1))
        label = textwrap.shorten(" ".join(text.split()), width=limit, placeholder="…")
        y = lanes + bottom - 1 - lane
        style = "--" if likely else "-"
        alpha = .55 if likely else 1
        ax.plot([start, max(start, end)], [y - .18, y - .18], color=color,
                lw=2, ls=style, alpha=alpha, marker="|", markersize=7)
        artist = ax.text(start, y + .04, label, fontsize=11, color=color, alpha=alpha, clip_on=True)
        # Use the fixed font's actual extents too: wide capitals and symbols
        # must fit the same slot as lowercase excerpts.
        available = max(0, (next_start - start) / max(duration, .1) * ax.bbox.width - 8)
        while limit > 1 and artist.get_window_extent(renderer).width > available:
            limit -= 1
            artist.set_text(textwrap.shorten(" ".join(text.split()), width=limit, placeholder="…"))


def _fit_text(artist, text, available, renderer, prefix="", suffix="", min_words=1):
    """Keep the beginning, with whole words where they fit; never a tail."""
    measure = lambda value: renderer.get_text_width_height_descent(
        value, artist.get_fontproperties(), ismath=False)[0]
    words = lambda value: sum(bool(re.search(r"\w", word)) for word in value.split())
    if words(text) < min_words:
        return ""
    if measure(prefix + text + suffix) <= available:
        return prefix + text + suffix
    for end in reversed([m.start() for m in re.finditer(r"\s+", text)]):
        value = prefix + text[:end] + "…" + suffix
        if words(text[:end]) >= min_words and measure(value) <= available:
            return value
    if min_words > 1:
        return ""
    for end in range(len(text), 0, -1):
        value = prefix + text[:end] + "…" + suffix
        if measure(value) <= available:
            return value
    return ""


def _phrase_key(text):
    """Match repeats across the song, ignoring punctuation and parenthesised ad-libs."""
    from slim import token

    def without_ad_lib(match):
        words = token(match[1]).split()
        return " " if words and all(AD_LIB.fullmatch(word) for word in words) else match[0]

    text = text.replace("'", "").replace("’", "")
    return token(re.sub(r"\(([^()]*)\)", without_ad_lib, text)) or token(text)


def _collapsed_phrases(phrases, duration):
    """Collapse adjacent repeats only, ignoring case, spacing and punctuation.

    Different words are not interchangeable: 'could' and 'would' stay separate.
    Every original onset survives as a tick, including collapsed repeats.
    """
    groups = []
    for row in sorted((r for r in phrases if 0 <= r.t <= duration), key=lambda r: r.t):
        body = re.sub(r"\s*\(unsure[^)]*\)", "", row.text, flags=re.I).strip()
        signature = tuple(re.findall(r"\w+", body.casefold()))
        if groups and signature and signature == groups[-1]["signature"]:
            groups[-1]["rows"].append(row)
        else:
            groups.append(dict(body=body, signature=signature, rows=[row]))
    return groups


def _merge_micro_phrases(phrases):
    """Combine close starts for labels only; saved phrase ticks stay intact."""
    from slim import Lyric
    result = []
    previous = None
    for row in sorted(phrases, key=lambda r: r.t):
        if previous is not None and row.t - previous < .6 - 1e-8:
            last = result[-1]
            unsure = re.search(r"\(unsure[^)]*\)", last.text + " " + row.text, re.I)
            body = re.sub(r"\s*\(unsure[^)]*\)", "", last.text + " " + row.text, flags=re.I).strip()
            result[-1] = Lyric(last.t, body + (" " + unsure[0] if unsure else ""),
                               last.approx or row.approx, row.end)
        else:
            result.append(row)
        previous = row.t
    return result


def _phrase_labels(ax, phrases, duration, *, lines=None):
    """Label lines with whole words; retain every saved phrase onset as a tick.

    Add rows when five opening words would collide. Three rows alone hid the
    chorus behind one-word fragments even after phrase text was removed.
    """
    ax.set_xlim(0, max(duration, .1))
    ax.set_yticks([])
    renderer = ax.figure.canvas.get_renderer()
    groups = _collapsed_phrases(_merge_micro_phrases(phrases if lines is None else lines), duration)
    all_approx = bool(phrases) and all(r.approx for r in phrases)
    ax.set_ylabel("Lyrics" + (" ≈" if all_approx else ""), fontsize=10,
                  rotation=0, ha="right", va="center")
    positions = [g["rows"][0].t for g in groups]
    ends, assignments, prefixes, suffixes = [0.] * 3, [], [], []
    probe = ax.text(0, 0, "", fontsize=11)
    for group, left in zip(groups, positions):
        rows = group["rows"]
        prefix = "≈ " if not all_approx and any(r.approx for r in rows) else ""
        prefix += "(unsure) " if any("(unsure" in r.text.lower() for r in rows) else ""
        suffix = f" ×{len(rows)}" if len(rows) > 1 else ""
        opening = " ".join(group["body"].split()[:5])
        probe.set_text(prefix + opening + "…" + suffix)
        width = probe.get_window_extent(renderer).width + 8
        start_px = left / max(duration, .1) * ax.bbox.width
        lane = next((i for i, end in enumerate(ends) if end <= start_px), len(ends))
        if lane == len(ends):
            ends.append(0.)
        ends[lane] = start_px + width
        assignments.append(lane)
        prefixes.append(prefix)
        suffixes.append(suffix)
    probe.remove()
    lanes = len(ends)
    ax.set_ylim(-.5, lanes - .2)
    seen = set()
    for i, group in enumerate(groups):
        rows = group["rows"]
        row, last = rows[0], rows[-1]
        lane = assignments[i]
        y = lanes - 1 - lane
        next_start = next((positions[j] for j in range(i + 1, len(groups))
                           if assignments[j] == lane), duration)
        left = positions[i]
        available = (next_start - left) / max(duration, .1) * ax.bbox.width - 8
        artist = ax.text(left, y + .04, "", fontsize=11, clip_on=True)
        label = _fit_text(artist, group["body"], available, renderer, prefixes[i], suffixes[i], min_words=2)
        artist.set_text(label)
        key = _phrase_key(group["body"])
        if key and key in seen:
            artist.set_color(REPEATED_LYRIC)
        seen.update(_phrase_key(member.text) for member in rows)
        # Interior phrase starts stay on their line's row, including repeats.
        stop = positions[i + 1] if i + 1 < len(groups) else duration + 1e-8
        for member in phrases:
            if not row.t <= member.t < stop:
                continue
            ax.plot([member.t, member.t], [y - .22, y - .08], color=INK, lw=.8,
                    marker="|", markersize=5)
        if len(rows) > 1:
            end = min(duration, max(last.t, getattr(last, "end", None) or last.t))
            ax.plot([row.t, end], [y - .22] * 2, color=MUTED, lw=.8, marker="|")
    return lanes


def _event_markers(ax, rows, duration):
    """Small numbered anchors, with extra row offsets for coincident times."""
    ax.set_xlim(0, max(duration, .1))
    renderer = ax.figure.canvas.get_renderer()
    ends = []
    for row in rows:
        colour = BLUE if row["kind"] == "by ear" else ORANGE if row["kind"] == "card" else INK
        artist = ax.text(row["start"], 0, str(row["number"]), fontsize=11,
                         color=colour, alpha=.55 if row["likely"] else 1, va="bottom",
                         bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.5))
        width = artist.get_window_extent(renderer).width
        left = min(row["start"] / max(duration, .1) * ax.bbox.width,
                   ax.bbox.width - width - 2)
        lane = next((i for i, end in enumerate(ends) if end + 9 <= left), len(ends))
        if lane == len(ends):
            ends.append(0)
        ends[lane] = left + width
        y = -lane
        artist.set_position((left / ax.bbox.width * duration, y))
        ax.plot([row["start"], row["end"]], [y - .16] * 2, color=colour, lw=1.3,
                ls="--" if row["likely"] else "-", alpha=.55 if row["likely"] else 1,
                marker="|", markersize=5)
        if row["kind"] == "by ear":
            ax.axvline(row["start"], color=BLUE, lw=.8, ls=":", alpha=.6)
            ax.plot(row["start"], .98, marker="v", markersize=5, color=BLUE,
                    transform=ax.get_xaxis_transform())
    lanes = max(1, len(ends))
    ax.set_ylim(-lanes + .5, .8)
    ax.set_yticks([])
    return lanes


def _ear_overlay(ax, rows, duration):
    if not rows:
        return
    _labels(ax, rows, duration, lanes=2, bottom=3, color=BLUE)
    for start, *_ in rows:
        ax.axvline(start, color=BLUE, lw=.8, ls=":", alpha=.6)
        ax.plot(start, .98, marker="v", markersize=6, color=BLUE,
                transform=ax.get_xaxis_transform())
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=ORANGE, marker="|", label="card"),
                       Line2D([], [], color=BLUE, marker="v", ls=":", label="by ear")],
              loc="upper right", fontsize=9, framealpha=.85, ncol=2)


def _gap_bands(ax, rows):
    for start, end, label, _ in rows:
        if end > start:
            ax.axvspan(start, end, facecolor="#DDE8EE", edgecolor=INK,
                       hatch="////", alpha=.55, lw=.8)
        else:
            ax.axvline(start, color=INK, ls=":", lw=1.2)


def _usual_pitch_limits(lo, hi, usual):
    if usual and "notes" in usual:
        if usual["low_hz"] > 0:
            lo = min(lo, usual["low_hz"] * .95)
        hi = max(hi, usual["high_hz"] * 1.05)
    return lo, hi


def _usual_band(ax, usual, colour, *, missing):
    if not usual or "reason" in usual:
        reason = usual["reason"] if usual else missing
        return f"Usual pitch band unavailable · {reason}"
    low = max(usual["low_hz"], ax.get_ylim()[0])
    label = f"pale band: usual median pitch (from {usual['notes']} notes)"
    ax.axhspan(low, usual["high_hz"], color=colour, alpha=.12, lw=0, zorder=0, label=label)
    return label


def _pitch(ax, times, values, max_ticks=4, *, usual=None):
    import numpy as np
    from matplotlib.ticker import NullLocator, ScalarFormatter
    f0 = np.array(values, dtype=float, copy=True)
    f0[~np.isfinite(f0) | (f0 <= 0)] = np.nan
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    if np.any(np.isfinite(f0)):
        ax.plot(times, f0, color=BLUE, lw=.9)
        lo, hi = np.nanmin(f0), np.nanmax(f0)
        lo, hi = (lo * .7, hi * 1.4) if lo == hi else (lo * .9, hi * 1.1)
    else:
        lo, hi = 60, 600
        ax.text(.01, .5, "No measured voiced frames available", transform=ax.transAxes, color=MUTED)
    lo, hi = _usual_pitch_limits(lo, hi, usual)
    ticks = np.array([30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400])
    ticks = ticks[(ticks >= lo) & (ticks <= hi)]
    if len(ticks) > max_ticks:
        ticks = ticks[np.linspace(0, len(ticks) - 1, max_ticks).astype(int)]
    ax.set_yticks(ticks if len(ticks) else np.geomspace(lo, hi, 3))
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Pitch (Hz)")


def _contrast_strip(ax, readings, duration):
    from feeling_card import contrast_marker
    from matplotlib.patches import Patch

    times, sentiment, lift, points = [], [], [], []
    for start, end, words, music in readings:
        end = min(end, duration)
        if contrast_marker(words, music):
            # At least CONTRAST_MIN_BAND_S wide, centred on the line, so a short
            # disagreement still shows on a phone (Muninn, 20 Nov: the 2:41 sliver).
            mid = (start + end) / 2 if end > start else start
            half = max((end - start) / 2, CONTRAST_MIN_BAND_S / 2)
            ax.axvspan(max(0, mid - half), min(duration, mid + half), color=DISAGREEMENT, lw=0)
        if end == start:
            points.append(len(times))
        times.extend((start, end, end))
        sentiment.extend((words, words, math.nan))
        lift.extend((music, music, math.nan))
    ax.axhline(0, color=MUTED, lw=.6, alpha=.5)
    words_line, = ax.plot(times, sentiment, color=BLUE, lw=1.2, marker=".", markevery=points, label="words sentiment")
    music_line, = ax.plot(times, lift, color=GREEN, lw=1.2, marker=".", markevery=points, label="song-relative music lift")
    limit = max(1., *(abs(v) for row in readings for v in row[2:]))
    ax.set_ylim(-limit - .08, limit + .08)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Contrast", fontsize=10, rotation=0, ha="right", va="center")
    ax.legend(handles=[words_line, music_line, Patch(facecolor=DISAGREEMENT, label="words and music disagree")],
              loc="lower right", bbox_to_anchor=(1, 1), borderaxespad=0, borderpad=0,
              fontsize=8, frameon=False, ncol=3)


def song_sheet(audio, song, decided, full, output, *, by_ear=None):
    import numpy as np
    context, Figure, Canvas = _plotting()  # Fail before decoding if matplotlib is absent.
    evidence = song_evidence(song, decided, full)
    output = Path(output)
    power, frequencies, duration = _spectrum(audio, output.with_name("listening-spectrum.npz"),
                                            song["manifest"]["file_sha256"])
    ear = by_ear_rows(by_ear, duration)
    stems = _cached_stem_energy(song["manifest"]["file_sha256"], duration)
    pitch = evidence["pitch"]
    have_pitch = bool(pitch.get("f0")) and pitch.get("fps", 0) > 0
    contrast = [row for row in evidence["contrast"] if row[0] <= duration]
    key_rows = [(a, b, label, True) for a, b, label in evidence["keys"]]
    numbered = _numbered_evidence(evidence["events"] + key_rows, evidence["gaps"], ear, duration)
    heights = [3.4, .6] + ([.8] if evidence["gaps"] else []) + ([1.8] if stems else [])
    heights += [1.] + ([1.1] if contrast else []) + ([1.2] if have_pitch else []) + [1.8]
    with context:
        figure = Figure(figsize=(WIDTH / 150, HEIGHT / 150), dpi=150)
        Canvas(figure)
        figure._sheet_key = numbered
        axes = _axes(figure, heights, left=.11)
        spec, repeats = axes[:2]
        index = 2
        gap_ax = axes[index] if evidence["gaps"] else None
        index += bool(evidence["gaps"])
        stem_ax = axes[index] if stems else None
        index += bool(stems)
        events = axes[index]
        times = np.linspace(0, duration, power.shape[1] + 1)
        spec.pcolormesh(times, frequencies, power, cmap="magma", vmin=-65, vmax=0, shading="flat", rasterized=True)
        spec.set_yscale("log")
        spec.set_yticks([50, 200, 1000, 5000], ["50", "200", "1k", "5k"])
        spec.set_ylabel("Full mix (Hz)")
        repeat_edges, groups, strength = _repeat_groups(power, duration)
        colours = (BLUE, ORANGE, GREEN, "#CC79A7", "#E69F00", "#56B4E9")
        for a, b, group, score in zip(repeat_edges[:-1], repeat_edges[1:], groups, strength):
            alpha = (.35 if score < .92 else .95) if group >= 0 else 1
            repeats.axvspan(a, b, facecolor=colours[group] if group >= 0 else "#E1E4E8",
                           alpha=alpha, edgecolor="none")
        repeats.set_yticks([])
        repeats.set_ylim(0, 1)
        repeats.set_ylabel("Sounds\nalike", fontsize=10, rotation=0, ha="right", va="center")
        if gap_ax is not None:
            _gap_bands(gap_ax, evidence["gaps"])
            gap_lanes = _event_markers(gap_ax, [r for r in numbered if r["kind"] == "gap"], duration)
            heights[2] = max(heights[2], gap_lanes * .6)
            gap_ax.set_ylabel("Gaps / cuts", fontsize=10, rotation=0, ha="right", va="center")
        if stem_ax is not None:
            for i, (name, at, db) in enumerate(stems):
                # Shared absolute dBFS scale, clipped to -80..0 for each row;
                # a weak/empty separated stem never establishes absence.
                stem_ax.plot(at, i + np.clip((db + 80) / 80, 0, 1) * .8, color=GREEN, lw=.8)
            stem_ax.set_yticks(np.arange(len(stems)) + .4, [name for name, *_ in stems], fontsize=9)
            stem_ax.set_ylim(-.1, len(stems))
            stem_ax.set_ylabel("Cached\nstems", fontsize=9, rotation=0, ha="right", va="center", labelpad=4)
        loud = spec.twinx()
        loud.set_ylabel("Loudness (dB)", color=GREEN)
        loud.tick_params(axis="y", colors=GREEN)
        curve = evidence["curve"]
        if curve.get("rms_db") and curve.get("step_s", 0) > 0:
            loud.plot(np.arange(len(curve["rms_db"])) * curve["step_s"], curve["rms_db"], color=GREEN, lw=1.3)
            loud_note = "green line: loudness"
        elif evidence["sections"]:
            for section in evidence["sections"]:
                loud.plot([section["start"], section["end"]], [section["db"]] * 2, color=GREEN, lw=2)
            loud_note = "green steps: average loudness by section"
        else:
            loud_note = "loudness not saved"
            loud.set_yticks([])
        # Use the twin axis so its curve is drawn below the section boxes.
        _section_labels(loud, evidence["sections"], duration)
        event_lanes = _event_markers(events, [r for r in numbered if r["kind"] != "gap"], duration)
        heights[index] = max(heights[index], event_lanes * .6)
        events.set_ylabel("Card events", fontsize=10, rotation=0, ha="right", va="center")
        if contrast:
            index += 1
            _contrast_strip(axes[index], contrast, duration)
        if have_pitch:
            _pitch(axes[index + 1], np.arange(len(pitch["f0"])) / pitch["fps"], pitch["f0"], max_ticks=2)
            axes[index + 1].set_ylabel("Vocal stem\n(Hz)", fontsize=10, rotation=0, ha="right", va="center")
        lyric_lanes = _phrase_labels(axes[-1], evidence["phrases"], duration, lines=evidence["lyric_lines"])
        heights[-1] = max(heights[-1], lyric_lanes * .6)
        axes[-1].get_gridspec().set_height_ratios(heights)
        figure._sheet_plot_height = max(5.8, sum(heights) * .52)
        if not evidence["lyrics"]:
            axes[-1].text(.01, .5, "No timed lyric lines on this card", transform=axes[-1].transAxes, color=MUTED)
        facts = evidence["facts"]
        # Quote complete decisions, including unclear prose, in separate rows.
        subtitle = "\n".join(label + ": " + facts[label] for label in ("KEY", "TEMPO", "METER") if label in facts)
        footer = ("Background: the whole mix (low notes at the bottom), brighter = louder · " + loud_note +
                  "\nSame colour in the strip = stretches sound alike (pale = a little alike; grey = no match)" +
                  "\nDashed or light = likely, not sure · ≈ = lyric times from the lyrics source · hatched = a measured gap")
        notes = ("Sound-alike strip: spectral match ≥0.80; pale <0.92; up to six groups. "
                 "Background: dB relative to mix maximum. Lyrics label line starts; ticks keep phrase starts; full words on the card.")
        if not np.any(groups >= 0):
            notes += " No supported sound-alike returns."
        if evidence["gaps"]:
            notes += " Cuts can retain voice; onset-only gaps have no measured end."
        if stems:
            notes += " Stems: −80 to 0 dBFS per row; level alone does not identify an instrument."
        if not have_pitch:
            notes += " Vocal pitch track not saved."
        figure._sheet_notes = notes.strip()
        return _finish(figure, axes, duration, output, evidence["title"], subtitle, footer)


def speech_sheet(data, measured, output, *, title="Speech", by_ear=None):
    context, Figure, Canvas = _plotting()
    duration = float(data.get("duration_s", 0))
    ear = by_ear_rows(by_ear, duration)
    with context:
        figure = Figure(figsize=(WIDTH / 150, HEIGHT / 150), dpi=150)
        Canvas(figure)
        axes = _axes(figure, [2, 1.6, 1.9 if ear else 1.2, 2.1])
        times, rms, f0 = (measured.get(k, []) for k in ("times", "rms_db", "f0"))
        axes[0].plot(times, rms, color=GREEN, lw=1.3)
        axes[0].set_ylabel("Energy (dB)")
        if not len(times):
            axes[0].text(.01, .5, "No frame energy retained (empty or too short clip)",
                         transform=axes[0].transAxes, color=MUTED)
        usual = measured.get("usual_pitch")
        _pitch(axes[1], times, f0, usual=usual)
        usual_note = _usual_band(axes[1], usual, BLUE,
                                 missing="no speaker identified" if len(times) else "speech measurements unavailable")
        silence = [(a, b, "silence", False) for a, b in (data.get("gaps") or {}).get("silent") or []]
        _gap_bands(axes[0], silence)
        _gap_bands(axes[1], silence)
        rows = []
        for event in (data.get("events") or {}).get("events", []):
            rows.append((event["start"], event["end"], event.get("name", event["label"]), False))
        for key, label in (("laughs", "Laughter"), ("snorts", "Snort")):
            for event in (data.get("laughs") or {}).get(key, []):
                if not any(a == event["start"] and b == event["end"] and name == label
                           for a, b, name, _ in rows):
                    rows.append((event["start"], event["end"], label, False))
        for a, b in (data.get("gaps") or {}).get("wordless") or []:
            rows.append((a, b, "WORDLESS", False))
        words = data.get("words") or []
        # This is the same one-second pause rule as inline WORDS, including its
        # overlap handling. Do not call all untranscribed stretches 'pauses'.
        from raven_ears_plus import PAUSE_SHOW
        end = None
        for _, start, stop in words:
            if end is not None and start - end >= PAUSE_SHOW:
                rows.append((end, start, "pause", False))
            end = max(end or 0, stop)
        _labels(axes[2], rows + silence, duration, lanes=3, color=ORANGE)
        _ear_overlay(axes[2], ear, duration)
        axes[2].set_ylabel("Card events")
        _labels(axes[3], [(a, b, word, False) for word, a, b in words], duration, lanes=6)
        axes[3].set_ylabel("Words")
        if not words:
            axes[3].text(.01, .5, "No timed words available", transform=axes[3].transAxes, color=MUTED)
        return _finish(figure, axes, duration, output, title, "Listening sheet · existing speech measurements",
                       "Energy: the card's RMS · pitch: measured Hz, log axis; gaps stay empty · hatched = measured silence · labels abbreviated\n"
                       + usual_note)
