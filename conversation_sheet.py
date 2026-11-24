#!/usr/bin/env python3
"""Draw one local PNG from ordered speech clips."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys

import listening_sheet as S

UNTIMED_GAP = .5
LONG_GAP = 10.
COMPRESSED_GAP = 5.
COLOURS = (S.BLUE, "#A34C88", "#007F65", "#7656A5", "#866421", "#3C7387")


@dataclass(frozen=True)
class Clip:
    speaker: str
    duration: float
    measured: dict
    laughs: list
    laugh_source: str
    words: list


def analyze_clips(paths, speakers, *, include_words=False):
    import raven_ears_plus as ears

    clips = []
    for path, speaker in zip(paths, speakers, strict=True):
        result = ears.analyze(str(path), speaker=speaker, fast=True, events=False, use_whisper=include_words,
                              update_baseline=False, identify_voice=not speaker,
                              capture_picture=True, capture_spectrum=True, laugh_model_default="omine")
        data = result["data"]
        measured = result.get("picture_data", {})
        voice = data.get("voice") or {}
        if not speaker and voice.get("status") == "matched":
            speaker = voice["ranked"][0][0]
        model = data.get("laugh_model") or {}
        room = data.get("laughs")
        if model.get("status") == "ran":
            laughs, source = ears._merge_spans(model.get("spans") or []), model["model"]
        elif room is not None:
            laughs, source = [(r["start"], r["end"]) for r in room["laughs"]], "CNN14"
        else:
            laughs, source = [], ""
        clips.append(Clip(speaker or "unknown", measured.get("duration_s", data["duration_s"]),
                          measured, laughs, source, (data.get("words") or []) if include_words else []))
    return clips


def _timestamps(values):
    if not values:
        return []
    if any("T" not in value and " " not in value for value in values):
        raise ValueError("timestamps must be ISO date-times")
    stamps = [datetime.fromisoformat(value) for value in values]
    aware = [stamp.utcoffset() is not None for stamp in stamps]
    if any(aware) and not all(aware):
        raise ValueError("timestamps must all include UTC offsets or all use the same local clock")
    if all(aware):
        stamps = [stamp.astimezone(timezone.utc) for stamp in stamps]
    if any(a > b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("timestamps must be in clip-start order")
    return stamps


def _timeline(clips, stamps):
    """Map clip starts to display seconds, removing time only from empty gaps."""
    starts, gaps = [], []
    end = removed = 0.
    for i, clip in enumerate(clips):
        start = (stamps[i] - stamps[0]).total_seconds() if stamps else end + (UNTIMED_GAP if i else 0.)
        gap = start - end
        if i and (not stamps or gap > LONG_GAP):
            width = COMPRESSED_GAP if stamps else gap
            gaps.append((end - removed, end - removed + width, gap))
            removed += gap - width
        starts.append(start - removed)
        end = max(end, start + clip.duration)
    return starts, gaps, end - removed


def _gap_label(seconds):
    minutes, seconds = divmod(round(seconds, 3), 60)
    parts = [f"{minutes:g} min"] if minutes else []
    if seconds or not parts:
        parts.append(f"{seconds:.3f}".rstrip("0").rstrip(".") + " s")
    return "… " + " ".join(parts) + " …"


def _timeline_labels(ax, labels, duration):
    """Reserve one row for starts and one for gap lengths, with room for box padding."""
    renderer = ax.figure.canvas.get_renderer()
    available = ax.bbox.width
    ends = [-float("inf"), -float("inf")]
    for start, text, colour, is_gap in sorted(labels, key=lambda label: label[0]):
        lane = int(is_gap)
        artist = ax.text(start, lane + .5, "", fontsize=10, va="center", color=colour,
                         bbox=dict(facecolor="white", alpha=.95, edgecolor="none", pad=1.))
        fitted = S._fit_text(artist, text, available - 8, renderer)
        artist.set_text(fitted)
        width = artist.get_window_extent(renderer).width
        x = start / duration * available - (width / 2 if is_gap else 0.)
        left = max(4., min(x, available - width - 4))
        if not fitted or left < ends[lane] + 8:
            artist.remove()
            continue
        ends[lane] = left + width
        artist.set_position((left / available * duration, lane + .5))
    lanes = 2 if math.isfinite(ends[1]) else 1
    ax.set_yticks([])
    ax.set_ylim(0, lanes)
    return lanes


WORD_LANES = 5


def _word_lanes(ax, words, duration, lanes=WORD_LANES):
    """Whole words only, packed by real width; a word that fits no lane keeps just its tick.

    The shared `_labels` slots each word by its neighbours' start times, which
    on a dense two-clip sheet leaves two characters and turns nearly every
    label into an ellipsis.
    """
    ax.set_ylim(-.9, lanes - .2)
    ax.set_yticks([])
    renderer = ax.figure.canvas.get_renderer()
    available = ax.bbox.width
    ends = [-float("inf")] * lanes
    for start, end, text, _likely in sorted((w for w in words if 0 <= w[0] <= duration), key=lambda w: w[0]):
        text = " ".join(str(text).split())
        x = start / max(duration, .1) * available
        ax.plot([start, start], [-.75, -.45], color=S.INK, lw=.8, alpha=.7)
        if not text:
            continue
        artist = ax.text(start, 0, text, fontsize=10, color=S.INK, va="center", clip_on=True)
        width = artist.get_window_extent(renderer).width
        lane = next((i for i, e in enumerate(ends) if x >= e + 5), None)
        if lane is None or x + width > available:
            artist.remove()
            continue
        ends[lane] = x + width
        artist.set_position((start, lanes - 1 - lane))


def _plot_clip(clip, start, spec, loud, pitch, colour):
    import numpy as np

    times = start + np.asarray(clip.measured.get("times", []))
    # Plot each clip separately so neither curve bridges a note boundary.
    S._plot_speech_spectrum(spec, loud, clip.measured.get("spectrum"), times,
                            clip.measured.get("rms_db", []), start=start)
    pitch.plot(times, clip.measured.get("f0", []), color=colour, lw=.95)
    for a, b in clip.laughs:
        a, b = max(0., a), min(clip.duration, b)
        if b > a:
            pitch.axvspan(start + a, start + b, color=S.ORANGE, alpha=.16, lw=0)
            pitch.plot([start + a, start + b], [.94, .94], color=S.ORANGE, lw=1.7,
                       ls="--" if clip.laugh_source == "CNN14" else "-",
                       transform=pitch.get_xaxis_transform())
    for ax in (spec, pitch):
        ax.axvline(start, color=colour, lw=.9, zorder=4)


def conversation_sheet(clips, output, *, stamps=(), include_words=False):
    """Render captured measurements only. Stamps are ordered clip-start date-times."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

    starts, gaps, duration = _timeline(clips, stamps)
    duration = max(duration, .1)
    speakers = list(dict.fromkeys(clip.speaker for clip in clips))
    usual_by_speaker = {}
    for clip in clips:
        usual = clip.measured.get("usual_pitch")
        if usual and (clip.speaker not in usual_by_speaker or "reason" in usual):
            usual_by_speaker[clip.speaker] = usual
    f0_by_speaker = {}
    for clip in clips:
        f0_by_speaker.setdefault(clip.speaker, []).append(clip.measured.get("f0"))
    heights = [1.] + [part for _ in speakers for part in (1.8, 1.1)]
    if include_words:
        heights.append(2.0)
    context, Figure, Canvas = S._plotting()
    with context:
        figure = Figure(figsize=(S.WIDTH / 150, S.HEIGHT / 150), dpi=150)
        Canvas(figure)
        axes = S._axes(figure, heights, left=.105)
        labels = []
        usual_notes = []
        panels = {}
        for i, speaker in enumerate(speakers):
            spec, pitch = axes[1 + 2 * i:3 + 2 * i]
            colour = COLOURS[i % len(COLOURS)]
            spec.set_ylabel(f"{speaker}\nSpectrum (Hz)", color=colour)
            loud = S._spectrum_rms_axis(spec)
            pitch.set_ylabel("Pitch (Hz)", color=colour)
            pitch.set_yscale("log")
            usual = usual_by_speaker.get(speaker)
            lo, hi = S._usual_pitch_limits(S._voice_pitch_floor(f0_by_speaker[speaker], usual), 600, usual)
            ticks = S._pitch_ticks(lo, hi)
            pitch.set_yticks(ticks, [f"{t:g}" for t in ticks])
            pitch.yaxis.set_minor_locator(NullLocator())
            pitch.set_ylim(lo, hi)
            missing = "no speaker identified" if speaker == "unknown" else "speech measurements unavailable"
            usual_notes.append(f"{speaker} · " + S._usual_band(pitch, usual, colour, missing=missing))
            panels[speaker] = (spec, loud, pitch, colour)
        multi_day = stamps and stamps[0].date() != stamps[-1].date()
        for i, (clip, start) in enumerate(zip(clips, starts)):
            spec, loud, pitch, colour = panels[clip.speaker]
            _plot_clip(clip, start, spec, loud, pitch, colour)
            when = (stamps[i].replace(tzinfo=None).isoformat(sep=" ") if multi_day else
                    stamps[i].time().isoformat() if stamps else S._key_clock(start))
            labels.append((start, when, colour, False))
            axes[0].axvline(start, color=colour, lw=.9, alpha=.4)

        for ax in axes:
            if stamps:
                S._gap_bands(ax, [(a, b, "", False) for a, b, _ in gaps])
            else:
                for a, b, _ in gaps:
                    ax.axvline((a + b) / 2, color=S.MUTED, ls=":", lw=.8)
        if stamps:
            labels.extend(((a + b) / 2, _gap_label(gap), S.INK, True) for a, b, gap in gaps)
        axes[-1].set_xlim(0, duration)
        heights[0] = _timeline_labels(axes[0], labels, duration) * .55
        if include_words:
            words = [(start + a, start + b, word, False)
                     for clip, start in zip(clips, starts) for word, a, b in clip.words]
            _word_lanes(axes[-1], words, duration)
            axes[-1].set_ylabel("Words")
            if not words:
                axes[-1].text(.01, .5, "No timed words available", transform=axes[-1].transAxes, color=S.MUTED)
        axes[-1].get_gridspec().set_height_ratios(heights)
        ticks = MaxNLocator(nbins=10).tick_values(0, duration)
        compressed = gaps if stamps else []
        axes[-1].xaxis.set_major_locator(FixedLocator(
            [tick for tick in ticks if tick <= duration and not any(a < tick < b for a, b, _ in compressed)]))

        def clock(shown, _):
            elapsed = shown + sum(gap - (b - a) for a, b, gap in compressed if shown >= b)
            return S._key_clock(elapsed) if duration < 5 else S._clock(elapsed)

        axes[-1].xaxis.set_major_formatter(FuncFormatter(clock))
        axes[-1].set_xlabel("Elapsed time (m:ss)" if stamps else "Assembled time (m:ss)")
        if stamps:
            zone = "UTC" if stamps[0].utcoffset() is not None else "local clock"
            timing = f"Clip starts · {zone} · hatched gaps are shortened · gap labels show true length"
        else:
            timing = f"Clip starts · dotted lines mark inserted {UNTIMED_GAP:g} s spacing"
        sources = sorted({clip.laugh_source for clip in clips} - {"", "CNN14"})
        note = "Green line · speech RMS"
        if sources:
            note += "    /    Solid orange spans · laughs from " + " / ".join(sources)
        if any(clip.laugh_source == "CNN14" for clip in clips):
            note += "    /    Dashed orange spans · laughs from CNN14 fallback"
        unavailable = sum(not clip.laugh_source for clip in clips)
        if unavailable:
            note += f"    /    Laugh analysis unavailable for {unavailable} clip(s)"
        S._header(figure, "Conversation", timing, "\n".join([note, *usual_notes]),
                  plot_height=max(5.8, sum(heights)))
        return S._save(figure, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", help="audio clips in conversation order")
    parser.add_argument("--speakers", nargs="+", metavar="NAME",
                        help="one label per clip, or a single label for every clip; - uses the voice book")
    parser.add_argument("--timestamps", nargs="+", metavar="ISO", help="one ISO start date-time per clip")
    parser.add_argument("--words", action="store_true", help="include local Whisper word labels in the PNG")
    parser.add_argument("--cpu", action="store_true", help="force CPU for speech analysis")
    parser.add_argument("--output", type=Path, default=Path("conversation.png"), help="PNG destination")
    args = parser.parse_args(argv)
    from network_policy import local_audio_path
    for clip in args.clips:
        local_audio_path(clip, parser)
    if args.speakers is not None and len(args.speakers) == 1 and len(args.clips) > 1:
        args.speakers = args.speakers * len(args.clips)
    for values, option in ((args.speakers, "--speakers"), (args.timestamps, "--timestamps")):
        if values is not None and len(values) != len(args.clips):
            parser.error(f"{option} needs one value per clip")
    try:
        stamps = _timestamps(args.timestamps)
    except ValueError as error:
        parser.error(str(error))
    paths = [Path(path).expanduser() for path in args.clips]
    output = args.output.expanduser()
    if output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if output.resolve() in (path.resolve() for path in paths):
        parser.error("--output must differ from the input clips")
    for index, path in enumerate(paths, 1):
        if not path.is_file():
            parser.error(f"clip {index} is not a readable audio file")
    speakers = [label.strip() if label != "-" else "" for label in args.speakers or ["-"] * len(paths)]
    os.environ["RAVEN_EARS_OFFLINE"] = "1"
    if args.cpu:
        os.environ.update(RAVEN_EARS_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
    try:
        S._plotting()  # Check the optional dependency before analyzing any clips.
        clips = analyze_clips(paths, speakers, include_words=args.words)
        print(conversation_sheet(clips, output, stamps=stamps, include_words=args.words))
    except ImportError as error:
        if (error.name or "").startswith("matplotlib"):
            parser.exit(1, "pip install matplotlib to draw the conversation sheet\n")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
