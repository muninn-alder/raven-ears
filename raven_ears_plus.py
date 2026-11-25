#!/usr/bin/env python3
"""raven_ears_plus: standalone local speech analysis with optional word timestamps.

Audio inference runs locally. Optional model downloads contact their hosting services.
``--setup [fast|speech|music|full]`` prepares selected weights; ``--check`` only checks local files.

    from raven_ears_plus import analyze
    r = analyze("note.mp3", words=[("hello", 0.1, 0.5), ...], speaker="speaker-1")
    print("\\n".join(r["lines"]))

Lines produced (compact fields usually <= ~118 chars):
    CHANNEL   observed narrow/wideband limits before interpreting vocal tone (warning may be longer)
    PITCH     real F0 (librosa.pyin): median, range, voiced share, phrase-end lift/drop
    EVENTS    non-speech sounds with times (PANNs CNN14 sound-event detection, AudioSet labels)
    ROOM      ambient-class candidates, with an additional faint-dog pass
    WHISPER   low-voicing word runs, or active unvoiced stretches with AudioSet Whispering support
    AFFECT    arousal / dominance / valence 0..1 (audeering wav2vec2 dim model), whole note + thirds
    SHAPE     thirds: loudness, pace (needs words), pitch
    QUALITY   voice quality on voiced frames: breathy / pressed / rough / clear (CPPS, HNR, H1-H2, tilt; Parselmouth)
    WORDLESS / SILENT   word gaps split by energy + voicing into "voice with no words" vs true silence
    CONTOUR   what the pitch did inside each wordless stretch of a second or more (rose, fell, held, arched)
    PEAK      the loudest, most aroused stretch of the note
    EMPHASIS  the words the speaker leaned on (pitch peak, loudness, duration; >=2 of 3), also *starred* in WORDS
    TUNE      strongest speech-pitch shapes, in semitones from the note's median F0
    WORDS vs VOICE   optional Jev text scores beside measurements from the same phrases
    BREATH    rough breaths per minute, only when the event model sees enough clean breathing
    VS USUAL  this note against the speaker's own rolling median (needs speaker + >=10 earlier notes)
    NOTES     compact raw diagnostics; full notes stay in the result and on stderr

Hard dependencies: numpy, scipy, librosa, ffmpeg on PATH.
Soft dependencies (skips appear on the card and in result["notes"]):
torch + panns_inference (EVENTS, BREATH), torch + transformers (AFFECT), faster-whisper (own words).
"""
from __future__ import annotations

import ears_profile as EP
from network_policy import offline, require_online, local_audio_path

import argparse
import ctypes
import gc
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  constants
# --------------------------------------------------------------------------- #
SR = 16000                 # analysis rate (pitch, energy, affect)
SR_SED = 32000             # PANNs wants 32 kHz
FRAME = 1024               # 64 ms
HOP = 256                  # 16 ms
FPS = SR / HOP
GAP_MIN = 0.5              # word gap that gets split into silent / wordless
MAX_LINE = 118             # card lines stay under ~120 chars including the prefix
AFFECT_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
AFFECT_WIN = 3.0
PANNS_DIR = Path(os.path.expanduser(os.environ.get("RAVEN_PANNS_CACHE", "~/.cache/raven-panns")))
PANNS_CKPT = PANNS_DIR / "Cnn14_mAP=0.431.pth"
PANNS_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_LABELS = PANNS_DIR / "class_labels_indices.csv"
PANNS_LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
BASELINE_MIN = 10          # notes needed before VS USUAL speaks
BASELINE_WINDOW = 60       # rolling median over the last N notes

# AudioSet labels we care about -> (display name, window-probability threshold)
EVENT_LABELS = {
    "Laughter": ("Laughter", 0.35),
    "Giggle": ("Giggle", 0.35),
    "Belly laugh": ("Belly laugh", 0.35),
    "Chuckle, chortle": ("Chuckle", 0.35),
    "Baby laughter": ("Baby laughter", 0.45),
    "Crying, sobbing": ("Crying", 0.35),
    "Whimper": ("Whimper", 0.30),
    "Wail, moan": ("Wail/moan", 0.25),
    "Sigh": ("Sigh", 0.35),
    "Groan": ("Groan", 0.30),
    "Grunt": ("Grunt", 0.35),
    "Gasp": ("Gasp", 0.35),
    "Pant": ("Pant", 0.30),
    "Breathing": ("Breathing", 0.30),
    "Sniff": ("Sniff", 0.35),
    "Snort": ("Snort", 0.40),
    "Cough": ("Cough", 0.40),
    # A labelled inhale reached 0.42 in one window; breath-related tags stayed weak.
    "Sneeze": ("Sneeze", 0.45),
    "Screaming": ("Screaming", 0.35),
    "Yell": ("Yell", 0.35),
    "Shout": ("Shout", 0.35),
    "Whispering": ("Whispering", 0.40),
    "Humming": ("Humming", 0.35),
    "Hum": ("Hum", 0.35),
    "Singing": ("Singing", 0.45),
    "Hiccup": ("Hiccup", 0.35),
    "Burping, eructation": ("Burp", 0.40),
    "Gurgling": ("Gurgling", 0.35),
}
LAUGH_LABELS = {"Laughter", "Giggle", "Belly laugh", "Chuckle, chortle", "Baby laughter"}


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
import stage_status as ST  # HONEST CARD: run-local outcomes.
import room_whisper as RW

def _t(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def _span(a: float, b: float) -> str:
    if _t(a) == _t(b):
        return f"{_t(a)} ({b - a:.1f}s)"
    return f"{_t(a)}–{_t(b)}"


def _spans(spans, k=3) -> str:
    """Longest k spans (shown in time order); '+n' for the rest."""
    top = sorted(sorted(spans, key=lambda s: s[0] - s[1])[:k])
    out = ", ".join(_span(a, b) for a, b in top)
    if len(spans) > k:
        out += f" +{len(spans) - k}"
    return out


def _runs(mask: np.ndarray):
    """Contiguous True runs of a bool array -> [(i0, i1_exclusive)]."""
    if mask.size == 0:
        return []
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _merge_runs(runs, max_gap):
    out = []
    for a, b in runs:
        if out and a - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _sem(f_hz, ref_hz):
    return 12.0 * np.log2(np.asarray(f_hz, dtype=float) / ref_hz)


def _fit(label: str, pieces, width=MAX_LINE) -> str:
    """label + pieces joined by ' · ', dropping the LAST pieces until it fits."""
    pieces = [p for p in pieces if p]
    pre = f"  {label + ':':<9}"
    while pieces:
        line = pre + " · ".join(pieces)
        if len(line) <= width:
            return line
        pieces = pieces[:-1]
    return pre.rstrip()


def notes_line(notes, width=MAX_LINE):
    """Keep whole diagnostics when possible, counting those omitted from the card."""
    pieces = [" ".join(str(n).split()) for n in notes]
    pieces = [p for p in pieces if p]
    if not pieces:
        return None
    pre = "NOTES  "
    for count in range(len(pieces), 0, -1):
        more = len(pieces) - count
        tail = f"; +{more} more" if more else ""
        line = pre + "; ".join(pieces[:count]) + tail
        if len(line) <= width:
            return line
    # Even a single diagnostic may exceed the budget. Show its beginning,
    # mark that truncation, and count any additional, wholly omitted notes.
    tail = f"; +{len(pieces) - 1} more" if len(pieces) > 1 else ""
    room = width - len(pre) - len(tail) - 1
    return pre + pieces[0][:room].rstrip() + "…" + tail


def _log(msg):
    print(msg, file=sys.stderr)


class _Clock:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        clock = self

        class _C:
            def __enter__(s):
                s.t0 = time.time()

            def __exit__(s, *a):
                clock.t[name] = round(clock.t.get(name, 0) + time.time() - s.t0, 2)

        return _C()


# --------------------------------------------------------------------------- #
#  audio loading
# --------------------------------------------------------------------------- #
def _ffmpeg_bin():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@ST.stage('decode')
def load_audio(path: str, sr: int) -> np.ndarray:
    """Decode anything ffmpeg can read to mono float32 at ``sr``."""
    path = local_audio_path(path)
    ff = _ffmpeg_bin()
    if ff:
        r = subprocess.run([ff, "-v", "error", "-protocol_whitelist", "file,pipe", "-i", path, "-map", "0:a:0", "-f", "f32le", "-ac", "1",
                            "-ar", str(sr), "-"], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return np.frombuffer(r.stdout, dtype=np.float32).copy()
        raise RuntimeError("ffmpeg could not decode audio: " + r.stderr.decode(errors="replace")[-300:])
    raise RuntimeError("FFmpeg is required to decode the first audio stream safely")


def _source_sample_rate(path):
    """Read the source rate, never mistake the analysis resample for the file rate."""
    path = local_audio_path(path)
    import wave
    try:
        with wave.open(str(path), "rb") as source:
            return source.getframerate()
    except (wave.Error, EOFError):
        pass  # compressed or floating-point audio: use the general metadata reader
    probe = shutil.which("ffprobe")
    if probe:
        result = subprocess.run(
            [probe, "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "json", str(path)], capture_output=True, check=True)
        rate = int(json.loads(result.stdout)["streams"][0]["sample_rate"])
    else:
        raise RuntimeError("ffprobe is required to inspect compressed audio")
    if rate <= 0:
        raise ValueError("source sample rate is not positive")
    return rate


def analyze_channel(y32, sample_rate, fe):
    """Observed bandwidth, not a device diagnosis. Use sounding (not just voiced) frames.

    The existing 32 kHz decode preserves both shelves of interest. Rolloff above
    its 16 kHz Nyquist is not measured; the original file rate is stored separately.
    A shelf means <0.1% of sounding-frame power above the specified frequency.
    """
    frame, hop = FRAME * 2, HOP * 2
    padded = np.pad(y32, (frame // 2, frame // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    active = np.asarray(fe["active"], dtype=bool)
    count = min(len(frames), len(active))
    idx = np.flatnonzero(active[:count])
    power = np.zeros(frame // 2 + 1)
    window = np.hanning(frame)
    # Bound FFT scratch space even for a long recording.
    for start in range(0, len(idx), 256):
        chunk = frames[idx[start:start + 256]].astype(float)
        chunk -= chunk.mean(axis=1, keepdims=True)
        power += np.sum(np.abs(np.fft.rfft(chunk * window, axis=1)) ** 2, axis=0)
    power[1:-1] *= 2  # one-sided power spectrum
    freq = np.fft.rfftfreq(frame, 1 / SR_SED)
    total = float(power.sum())
    out = dict(sample_rate_hz=int(sample_rate), analysis_rate_hz=SR_SED,
               sounding_frames=len(idx), rolloff_99_hz=None,
               energy_above_3600_fraction=None, energy_above_7500_fraction=None,
               shelf_above_3600=None, shelf_above_7500=None, classification=None)
    if total <= 0:
        return out
    rolloff = float(freq[min(np.searchsorted(np.cumsum(power), total * .99), len(freq) - 1)])
    high36 = float(power[freq > 3600].sum() / total)
    high75 = float(power[freq > 7500].sum() / total)
    out.update(rolloff_99_hz=rolloff, energy_above_3600_fraction=high36,
               energy_above_7500_fraction=high75,
               shelf_above_3600=high36 < .001, shelf_above_7500=high75 < .001)
    nyquist = sample_rate / 2
    if nyquist <= 4000 or (rolloff <= 4000 and out["shelf_above_3600"]):
        out["classification"] = "narrowband"
    elif nyquist <= 8000 or (rolloff <= 8000 and out["shelf_above_7500"]):
        out["classification"] = "wideband-limited"
    return out


def channel_line(channel):
    kind = (channel or {}).get("classification")
    if kind == "narrowband":
        return ("CHANNEL  narrowband (≈phone/Bluetooth, ~3.4 kHz): brightness, crest and AFFECT unreliable; "
                "trust words and timing")
    if kind == "wideband-limited":
        return ("CHANNEL  wideband-limited (~7 kHz, e.g. Bluetooth HFP/phone app): "
                "treat brightness and tone with care")
    return None


# --------------------------------------------------------------------------- #
#  device / model management  (lazy, freed after use, CPU fallback)
# --------------------------------------------------------------------------- #
_MODELS: dict = {}
_WHISPER_MODELS: dict = {}
_VRAM = {"peak_mb": 0.0}


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


class SetupNeeded(FileNotFoundError):
    """Something a stage needs is missing. str() leads with the remedy, then the cause.

    Card lines get cut at the right edge, so the instruction goes first:
    ``run --setup (PANNs checkpoint missing)``. The location is kept in ``detail``.
    """

    def __init__(self, remedy, cause, detail=""):
        super().__init__(f"{remedy} ({cause})")
        self.remedy_first = True
        self.remedy, self.cause, self.detail = remedy, cause, detail


def _skip_why(e) -> str:
    """Short reason for a 'STAGE skipped:' note; remedy-first exceptions are not prefixed."""
    return ST.reason_of(e, 100)


def _skip_line(label, why):
    line = f"  {label + ':':<9}skipped: {why}"
    return line if len(line) <= MAX_LINE else line[:MAX_LINE - 1].rstrip() + '…'


def _pick_device(need_mb=1500) -> str:
    if EP.cpu():
        return "cpu"
    want = os.environ.get("RAVEN_EARS_DEVICE", "auto").lower()
    if want == "cpu":
        return "cpu"
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    try:
        free, _tot = torch.cuda.mem_get_info()
        if free / 2**20 < need_mb:
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


def _note_vram():
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        try:
            _VRAM["peak_mb"] = max(_VRAM["peak_mb"], torch.cuda.max_memory_allocated() / 2**20)
        except Exception:
            pass


def release_models():
    """Drop every cached model and give the GPU its memory back."""
    _note_vram()
    _MODELS.clear()
    _WHISPER_MODELS.clear()
    gc.collect()
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _is_oom(e: Exception) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda" in s and "alloc" in s


def _with_fallback(name, fn, notes):
    """Run fn(device). On CUDA failure/OOM free memory and retry on CPU."""
    dev = _pick_device()
    try:
        return fn(dev), dev
    except Exception as e:
        if dev == "cuda" and not isinstance(e, (FileNotFoundError, ImportError)):
            ST.mark(name, "degraded", "GPU failed; retried on CPU")
            notes.append(f"{name}: GPU failed ({type(e).__name__}: {str(e)[:80]}); retried on CPU")
            _MODELS.pop(name, None)
            release_models()
            return fn("cpu"), "cpu"
        raise


# --------------------------------------------------------------------------- #
#  energy + pitch front end
# --------------------------------------------------------------------------- #
@ST.stage('pitch front end')
def _front_end(y16: np.ndarray, fmin=65.0, fmax=600.0):
    import librosa
    rms = librosa.feature.rms(y=y16, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = np.clip(20 * np.log10(rms + 1e-7), -100, 0)
    n = len(rms_db)
    floor = float(np.percentile(rms_db, 5))
    ref = float(np.percentile(rms_db, 95))
    if ref - floor >= 25.0:          # a real quiet floor exists: sit 10 dB above it
        thr = max(floor + 10.0, ref - 40.0, -70.0)
    else:                            # no quiet frames to learn from (hum/tone/dense speech)
        thr = max(ref - 35.0, -70.0)
    active = rms_db > thr
    f0, vflag, _vprob = librosa.pyin(y16, fmin=fmin, fmax=fmax, sr=SR, frame_length=FRAME,
                                     hop_length=HOP, fill_na=np.nan)
    f0 = np.asarray(f0, dtype=float)[:n]
    vflag = np.asarray(vflag, dtype=bool)[:n]
    vprob = np.asarray(_vprob, dtype=float)[:n] if _vprob is not None else np.full(n, np.nan)
    vprob = np.pad(vprob, (0, n - len(vprob)), constant_values=np.nan)
    if len(f0) < n:                       # defensive: lengths should match
        pad = n - len(f0)
        f0 = np.concatenate([f0, np.full(pad, np.nan)])
        vflag = np.concatenate([vflag, np.zeros(pad, bool)])
    # pyin pins breathy noise to the search-range edges; those are artefacts, not pitch
    edge = (f0 < fmin * 1.04) | (f0 > fmax * 0.96)
    voiced = vflag & active & np.isfinite(f0) & ~edge
    f0s = f0.copy()
    f0s[~voiced] = np.nan
    return {"rms_db": rms_db, "active": active, "thr": thr, "f0": f0s, "voiced": voiced,
            "voiced_prob": vprob,
            "times": np.arange(n) / FPS, "n": n}


# --------------------------------------------------------------------------- #
#  words
# --------------------------------------------------------------------------- #
def _parse_words(words):
    """Accept (word,start,end) tuples, dicts with start/end or startTimeMs/endTimeMs,
    or a whole hear_core result. Returns sorted [(word, start_s, end_s)] or None."""
    if words is None:
        return None
    if isinstance(words, dict):
        for path in (("raw_stt", "wordTimestamps"), ("transcription", "wordTimestamps"),
                     ("wordTimestamps",), ("words",)):
            cur = words
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
            if cur:
                words = cur
                break
        else:
            return None
    out = []
    for w in words:
        try:
            if isinstance(w, dict):
                txt = str(w.get("word", w.get("text", "")))
                if "startTimeMs" in w:
                    s, e = w["startTimeMs"] / 1000.0, w["endTimeMs"] / 1000.0
                else:
                    s, e = float(w["start"]), float(w["end"])
            else:
                txt, s, e = str(w[0]), float(w[1]), float(w[2])
            out.append((txt.strip(), s, max(e, s)))
        except Exception:
            continue
    out.sort(key=lambda x: x[1])
    return out or None


def _sanitize_words(words):
    """Whisper likes to stretch a word's end over a following hum/silence. Cap each word to a
    plausible length so the stretched tail is treated as a gap (then split by energy)."""
    out, capped = [], 0
    for txt, s, e in words:
        n = max(1, len(re.sub(r"\W", "", txt)))
        cap = max(1.0, 0.30 + 0.10 * n)
        if e - s > cap:
            e = s + cap
            capped += 1
        out.append((txt, s, e))
    return out, capped


def _preload_cuda12():
    """ctranslate2 wants CUDA-12 cuBLAS/cuDNN; pip's nvidia-*-cu12 wheels hold them."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
        for root in roots:
            for pat in ("cublas/lib/libcublas.so.12", "cublas/lib/libcublasLt.so.12",
                        "cudnn/lib/libcudnn.so.9", "cudnn/lib/libcudnn_ops.so.9",
                        "cudnn/lib/libcudnn_cnn.so.9"):
                for f in glob.glob(os.path.join(root, pat)):
                    try:
                        ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
    except Exception:
        pass


VERBATIM_STYLE = ("Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. "
                  "I, I, I like want to, uh, you know, say it.")
VERBATIM_KWARGS = dict(vad_filter=True,
                       vad_parameters=dict(threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=400),
                       hallucination_silence_threshold=2.0, condition_on_previous_text=False)


def verbatim_on():
    """RAVEN_EARS_VERBATIM=1 (or --verbatim): ask Whisper to keep fillers and stutters and to stay silent on silence."""
    return os.environ.get("RAVEN_EARS_VERBATIM", "").strip().lower() in ("1", "true", "yes", "on")


def verbatim_kwargs(nk=None, model=None, names=None, notes=None):
    """transcribe() kwargs for verbatim mode. The disfluent style text is the initial_prompt, joined after a names
    initial_prompt when names are delivered that way (names first: Whisper keeps only the LAST 223 prompt tokens,
    so the style must be the tail). With `model` and `names` the prompt is fitted to that token budget and a trim
    is said in `notes`; hotwords stay separate."""
    kw = dict(VERBATIM_KWARGS)
    names_prompt = (nk or {}).get("initial_prompt")
    style = verbatim_style()
    if model is not None and (names or not names_prompt):
        import lyrics_shared as LS
        kw["initial_prompt"], note = LS.fit_prompt(model, names if names_prompt else [], style)
        if note and notes is not None and note not in notes:
            notes.append(note)
    else:
        kw["initial_prompt"] = (names_prompt + " " if names_prompt else "") + style
    return kw


def verbatim_style():
    """The style text Whisper imitates. RAVEN_EARS_VERBATIM_STYLE may hold your own text, or a path to a file of
    it: a few lines in the way your person actually talks. Long text is cut, because Whisper's prompt is short."""
    custom = os.environ.get("RAVEN_EARS_VERBATIM_STYLE", "").strip()
    if not custom:
        return VERBATIM_STYLE
    path = Path(custom).expanduser()
    try:
        if len(custom) < 4096 and path.is_file():
            custom = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    text = " ".join(custom.split())
    return text[:600] or VERBATIM_STYLE


@ST.stage('Whisper')
def _whisper_words(y16: np.ndarray, notes, *, fast=False, keep_model=False):
    try:
        _preload_cuda12()
        from faster_whisper import WhisperModel
    except Exception as e:
        ST.mark("Whisper", "failed", f"faster-whisper not importable ({type(e).__name__})")
        notes.append(f"words: faster-whisper not importable ({type(e).__name__}); running without word times")
        return None
    size = EP.speech_model(fast)
    # Keep the same Whisper model while reducing precision to fit available memory.
    # large-v3 float16 wants ~4.5 GB; int8_float16 uses less memory;
    # CPU int8 is the last resort (slow, same model).
    if _pick_device(need_mb=4500) == "cuda":
        attempts = [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]
    elif _pick_device(need_mb=1800) == "cuda":
        attempts = [("cuda", "int8_float16"), ("cpu", "int8")]
    else:
        attempts = [("cpu", "int8")]
    for d, ct in attempts:
        model = None
        nk = {}
        key = (size, d, ct, EP.cpu_threads() if EP.cpu() or fast else None)
        try:
            kwargs = {"cpu_threads": EP.cpu_threads()} if EP.cpu() or fast else {}
            model = _WHISPER_MODELS.get(key) if keep_model else None
            if model is None:
                model = WhisperModel(size, device=d, compute_type=ct, local_files_only=offline(), **kwargs)
            if keep_model:
                _WHISPER_MODELS.clear()  # Retain only the current speech model.
                _WHISPER_MODELS[key] = model
            try:                                    # names list: hotwords (or a short initial_prompt)
                import lyrics_shared as _LS
                names = _LS.effective_names()
                nn = []
                nk = _LS.whisper_name_kwargs(model, names, notes=nn)
            except Exception:
                nk, nn = {}, []
            if nk:
                note = ST.whisper_names_note(len(names))
                if note not in notes:
                    notes.append(note)
            tk = dict(vad_filter=True, condition_on_previous_text=False)
            if verbatim_on():
                tk.update(verbatim_kwargs(nk, model, names, notes))
                nn = []                             # the verbatim fit speaks for the names prompt
                note = "words: verbatim mode (disfluent style prompt, tuned VAD, silence hallucination guard)"
                if note not in notes:
                    notes.append(note)
            notes.extend(n for n in nn if n not in notes)
            tk.update({k: v for k, v in nk.items() if k != "initial_prompt" or "initial_prompt" not in tk})
            segs, _info = model.transcribe(y16, language=os.environ.get("RAVEN_EARS_LANG", "en"),
                                           word_timestamps=True, **tk)
            words = [(w.word.strip(), float(w.start), float(w.end))
                     for s in segs for w in (s.words or [])]
            if d == "cpu" and ST.cpu_requested():
                ST.mark("Whisper", "ran", f"{d.upper()} {ct} (chosen)")   # chosen, not a fallback
            elif d == "cpu" or ct != "float16":
                ST.mark("Whisper", "degraded", f"fell back to {d.upper()} {ct}")
            notes.append(f"words: own faster-whisper {size} on {d} ({ct})")
            return words
        except Exception as e:
            _WHISPER_MODELS.pop(key, None)
            detail = "" if nk else f": {str(e)[:80]}"
            notes.append(f"words: whisper on {d} failed ({type(e).__name__}{detail})")
        finally:
            del model
            gc.collect()
            torch = _torch()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    ST.mark("Whisper", "failed", "all transcription attempts failed")
    return None


# --------------------------------------------------------------------------- #
#  1. PITCH
# --------------------------------------------------------------------------- #
def _word_phrases(words):
    """Yield each word run with its span, including overlapping word timestamps."""
    if not words:
        return
    left, start, end = 0, words[0][1], words[0][2]
    for right, (_, s, e) in enumerate(words[1:], 1):
        if s - end >= GAP_MIN:
            yield start, end, words[left:right]
            left, start, end = right, s, e
        else:
            end = max(end, e)
    yield start, end, words[left:]


def _phrases(fe, words):
    """Word runs split at gaps >= GAP_MIN, else active runs."""
    if words:
        return [(start, end) for start, end, _ in _word_phrases(words)]
    runs = _merge_runs(_runs(fe["active"]), int(0.4 * FPS))
    return [(a / FPS, b / FPS) for a, b in runs if (b - a) / FPS >= 0.3]


@ST.stage('pitch')
def analyze_pitch(fe, words):
    f0, voiced, times = fe["f0"], fe["voiced"], fe["times"]
    n_act = int(fe["active"].sum())
    n_voiced = int(voiced.sum())
    out = {"voiced_frac": round(n_voiced / n_act, 3) if n_act else 0.0}
    if n_voiced < 12:
        out["ok"] = False
        return out
    v = f0[voiced]
    med = float(np.median(v))
    lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 95))
    out.update(ok=True, median_hz=round(med, 1), lo_hz=round(lo), hi_hz=round(hi),
               range_st=round(float(_sem(hi, lo)), 2))
    ends = []
    for a, b in _phrases(fe, words):
        idx = np.where(voiced & (times >= a - 0.05) & (times <= b + 0.05))[0]
        if len(idx) < 8:
            continue
        t_last = times[idx[-1]]
        tail = idx[times[idx] >= t_last - 0.6]
        if len(tail) < 6:
            continue
        st = _sem(f0[tail], med)
        tt = times[tail]
        slope = float(np.polyfit(tt, st, 1)[0])
        delta = slope * 0.6
        ends.append("up" if delta >= 1.5 else "down" if delta <= -1.5 else "level")
    out["phrase_ends"] = ends
    out["phrases"] = len(_phrases(fe, words))
    return out


def pitch_line(p):
    if not p.get("ok"):
        return "  PITCH:   (too little voiced sound to measure a pitch)"
    pieces = [f"median {p['median_hz']:.0f} Hz",
              f"range {p['range_st']:.1f} st ({p['lo_hz']}–{p['hi_hz']} Hz)",
              f"voiced {p['voiced_frac'] * 100:.0f}%"]
    e = p.get("phrase_ends") or []
    if e:
        sym = {"up": "↑", "down": "↓", "level": "→"}
        cnt = " ".join(f"{sym[k]}{e.count(k)}" for k in ("up", "down", "level") if e.count(k))
        pieces.append(f"phrase ends {cnt} (last {sym[e[-1]]})")
    return _fit("PITCH", pieces)


def analyze_tune(fe, words, pitch, baseline):
    """Measure the voiced shape of each phrase without storing a second transcript."""
    if not pitch.get('ok'):
        return None
    f0 = fe['f0']
    good = fe['voiced'] & np.isfinite(f0) & (f0 > 0)
    times = fe['times'][good]
    st = _sem(f0[good], np.median(f0[good]))
    word_starts = np.array([s for _, s, _ in (words or [])], dtype=float)
    phrases = []
    for start, end in _phrases(fe, words):
        left, right = np.searchsorted(times, [start, end])
        if right - left < 8:
            continue
        tt, ss = times[left:right], st[left:right]
        # Take medians in 120 ms bins to keep short tracker spikes out of the shape.
        cuts = np.flatnonzero(np.diff(np.floor((tt - start) / .12))) + 1
        anchors = [(np.median(t), np.median(s)) for t, s in
                   zip(np.split(tt, cuts), np.split(ss, cuts)) if len(t) >= 3]
        if len(anchors) < 2:
            continue
        t, s = np.asarray(anchors).T
        first, last = s[0], s[-1]
        peak, trough = s.max(), s.min()
        arch, dip = peak - max(first, last), min(first, last) - trough
        points = [first]
        if max(arch, dip) >= 1.5:
            points.append(peak if arch >= dip else trough)
        points.append(last)

        # Only mark the ending if voiced evidence reaches it. Never bridge a long
        # dropout in the last part of the phrase to manufacture an end inflection.
        breaks = np.flatnonzero(np.diff(tt) > .18)
        tail = s[t >= max(tt[-1] - .45, tt[breaks[-1] + 1] if breaks.size else tt[0])]
        ending, delta = '?', None
        if end - tt[-1] <= .25 and len(tail) >= 2:
            edge = max(1, len(tail) // 2)
            delta = float(np.median(tail[-edge:]) - np.median(tail[:edge]))
            ending = '↑' if delta >= 1.5 else '↓' if delta <= -1.5 else '→'

        lo, hi = np.searchsorted(word_starts, [start, end])
        jump = None
        for j in range(1, len(s)):
            i = int(np.searchsorted(word_starts, t[j], side='right')) - 1
            if not (lo <= i < hi and t[j] < words[i][2]):
                continue
            # Look over at most 320 ms so a quick glide can count as well as a step.
            for k in range(j - 1, -1, -1):
                if t[j] - t[k] > .32 or t[k + 1] - t[k] > .18:
                    break
                change = float(s[j] - s[k])
                if abs(change) >= 2 and (jump is None or abs(change) > abs(jump['st'])):
                    jump = dict(word_index=i, st=change)
        if jump:
            jump['st'] = round(jump['st'], 1)
        a, b = np.percentile(s, [5, 95])
        phrases.append(dict(start=float(start), end=float(end), points_st=[round(float(x), 1) for x in points],
                            range_st=round(float(b - a), 2), ending=ending, end_st=delta, jump=jump,
                            tail_words=list(range(max(lo, hi - 2), hi))))

    history = [row.get('range_st') for row in baseline[-BASELINE_WINDOW:] if isinstance(row, dict)]
    history = [x for x in history if isinstance(x, (int, float)) and math.isfinite(x) and x >= 0]
    return dict(phrases=phrases, range_st=pitch['range_st'], baseline_n=len(history),
                usual_range_st=float(np.median(history)) if history else None)


def tune_lines(tune, words):
    if not tune or not tune['phrases']:
        return []
    head = '  TUNE:    semitones from the note median'
    if tune['baseline_n']:
        head += (f" · range {tune['range_st']:.1f} st, usual {tune['usual_range_st']:.1f} st"
                 f" (n={tune['baseline_n']})")
    phrases = sorted(tune['phrases'], key=lambda p: (-p['range_st'], -abs(p['end_st'] or 0)))[:6]
    if any(phrase['ending'] == '?' for phrase in phrases):
        head += ' · ? end pitch unclear'
    lines = [head]

    def quote_word(i):
        text = re.sub(r'["–—]', '', words[i][0].strip())
        text = ' '.join(text.split()[:1])
        return text if len(text) <= 23 else text[:22] + '…'

    for phrase in sorted(phrases, key=lambda p: p['start']):
        offsets = [int(round(s)) for s in phrase['points_st']]
        shape = f'{offsets[0]:+d}'
        for a, b in zip(offsets, offsets[1:]):
            if a != b:
                shape += f"{'↗' if b > a else '↘'}{b:+d}"
        tail = ' '.join(filter(None, map(quote_word, phrase['tail_words'])))
        excerpt = f' "…{tail}"' if tail else ''
        jump = phrase['jump']
        jump_word = quote_word(jump['word_index']) if jump else ''
        flag = f" ({'lift' if jump['st'] > 0 else 'drop'} on \"{jump_word}\")" if jump_word else ''
        lines.append(f"  {_t(phrase['start'])}{excerpt} {shape} st {phrase['ending']}{flag}")
    return lines


# --------------------------------------------------------------------------- #
#  2. EVENTS (+ 8. BREATH)   PANNs CNN14 audio tagging on sliding 2 s windows
# --------------------------------------------------------------------------- #
TAG_WIN, TAG_HOP = 2.0, 0.5          # seconds; event times are therefore good to ~0.5 s


def _load_tagger(device):
    key = ("tagger", device)
    if key in _MODELS:
        return _MODELS[key]
    if not PANNS_CKPT.is_file() or PANNS_CKPT.stat().st_size < 3e8:
        raise SetupNeeded("run --setup", "PANNs checkpoint missing", detail=str(PANNS_CKPT))
    # panns_inference 0.1.1 downloads this separate labels file on import.
    # PANNS_LABELS in our cache alone does not prevent that download.
    if (sys.modules.get("panns_inference.config") is None
            and not (Path.home() / "panns_data" / "class_labels_indices.csv").is_file()):
        require_online("PANNs cached weights not used offline: upstream labels missing")
    torch = _torch()
    from panns_inference.models import Cnn14
    from panns_inference.config import labels, classes_num
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
              classes_num=classes_num)
    ck = torch.load(str(PANNS_CKPT), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    _MODELS.clear()      # one big model at a time
    _MODELS[key] = (m, labels)
    return _MODELS[key]


def _tag_windows(y32: np.ndarray, device: str, *, win=TAG_WIN):
    """(centers_s [nW], probs [nW, 527], labels)."""
    torch = _torch()
    m, labels = _load_tagger(device)
    W, H = int(win * SR_SED), int(TAG_HOP * SR_SED)
    if len(y32) < W:
        y32 = np.pad(y32, (0, W - len(y32)))
    starts = list(range(0, len(y32) - W + 1, H))
    if (len(y32) - W) % H:
        starts.append(len(y32) - W)          # cover the tail
    out = []
    B = 16
    for i in range(0, len(starts), B):
        x = np.stack([y32[s:s + W] for s in starts[i:i + B]]).astype(np.float32)
        with torch.no_grad():
            p = m(torch.from_numpy(x).to(device))["clipwise_output"].float().cpu().numpy()
        out.append(p)
    _note_vram()
    centers = (np.array(starts) + W / 2) / SR_SED
    return centers, np.concatenate(out, axis=0), labels


def _breath_rate(centers, probs_b, fe, words, thr=0.2):
    """Breaths/min from the gaps between unvoiced breath-like bursts inside windows the tagger
    calls Breathing. Rough by design; returns {"per_min":...} or {"skipped": reason}."""
    n = fe["n"]
    flag = np.zeros(n, bool)
    for c, p in zip(centers, probs_b):
        if p >= thr:
            flag[int(max(0, (c - TAG_HOP) * FPS)):int(min(n, (c + TAG_HOP) * FPS))] = True
    if not flag.any():
        return {"skipped": "the event model does not hear breathing here"}
    mask = flag & fe["active"] & ~fe["voiced"]
    if words:                                    # speech consonants are unvoiced too: keep to the gaps
        for _w, s, e in words:
            mask[int(max(0, s * FPS)):int(min(n, e * FPS)) + 1] = False
    bursts = [(a, b) for a, b in _merge_runs(_runs(mask), int(0.15 * FPS)) if 0.25 <= (b - a) / FPS <= 4.0]
    breaths = _merge_runs(bursts, int(0.6 * FPS))        # inhale+exhale -> one breath
    if len(breaths) < 3:
        return {"skipped": f"only {len(breaths)} breath-like bursts in the breathing stretch"}
    onsets = np.array([a / FPS for a, _b in breaths])
    iv = np.diff(onsets)
    med = float(np.median(iv))
    mad = float(np.median(np.abs(iv - med)))
    if not (1.0 <= med <= 15.0) or mad > 0.5 * med:
        return {"skipped": f"breath spacing too irregular (median {med:.1f}s, spread {mad:.1f}s): noisy"}
    return {"per_min": round(60.0 / med, 1), "n": len(breaths), "span_s": round(float(onsets[-1] - onsets[0] + med), 1)}


def _event_windows(y32, notes, fe, stage):
    (centers, probs, labels), dev = _with_fallback(stage, lambda dev: _tag_windows(y32, dev), notes)
    if fe is not None:
        # the tagger happily "hears" sighs in near-silent room tone; ignore windows with no real sound
        keep = np.array([fe["active"][int(max(0, (c - TAG_WIN / 2) * FPS)):int((c + TAG_WIN / 2) * FPS) + 1].mean() >= 0.1
                         for c in centers])
        probs = probs * keep[:, None]
    return centers, probs, labels, dev


@ST.stage('events')
def analyze_events(y32: np.ndarray, dur: float, notes, fe=None, words=None):
    centers, probs, labels, dev = _event_windows(y32, notes, fe, "tagger")
    ix = {l: i for i, l in enumerate(labels)}
    events = []
    for lab, (disp, thr) in EVENT_LABELS.items():
        if lab not in ix:
            continue
        p = probs[:, ix[lab]]
        for a, b in _runs(p >= thr):
            t0 = max(0.0, centers[a] - 0.25)
            t1 = min(dur, centers[b - 1] + 0.25)
            events.append({"label": lab, "name": disp, "start": round(float(t0), 2), "end": round(float(t1), 2),
                           "score": round(float(p[a:b].max()), 2)})
    events.sort(key=lambda e: -e["score"])
    breath = None
    if "Breathing" in ix and fe is not None:
        breath = _breath_rate(centers, probs[:, ix["Breathing"]], fe, words)
    return {"events": events, "device": dev, "breath": breath,
            **RW.event_context(centers, probs, labels, dur, TAG_HOP, EVENT_LABELS['Whispering'][1]),
            "has_moan_label": "Moan" in ix, "wail_moan_label": "Wail, moan" in ix}


def analyze_faint_dogs(y32, dur, fe, notes, *, reduced=False):
    if reduced:
        if not fe['voiced'].any():
            return None
        y32 = RW.voice_reduced(y32, fe, SR_SED)
    (centers, probs, labels), _ = _with_fallback(
        'ROOM faint dogs', lambda dev: _tag_windows(y32, dev, win=RW.DOG_WIN), notes)
    return RW.faint_dogs(centers, probs, labels, y32, fe, dur, SR_SED, TAG_HOP, reduced=reduced)


def events_line(ev, k=5):
    events = ev["events"]
    if not events:
        return "  EVENTS:  none above threshold (no laughs, sighs, gasps, breaths, groans, cries heard)"
    by = {}
    for e in events:
        by.setdefault(e["label"], []).append(e)
    items = []
    for lab, es in by.items():
        top = max(es, key=lambda e: e["score"])
        if len(es) >= 3:
            txt = f"{top['name']} ×{len(es)} (first {_t(min(x['start'] for x in es))}, peak {top['score']:.2f})"
        else:
            spans = ", ".join(_t((x["start"] + x["end"]) / 2) if x["end"] - x["start"] < 1.0
                              else _span(x["start"], x["end"]) for x in sorted(es, key=lambda x: x["start"]))
            txt = f"{top['name']} {spans} ({top['score']:.2f})"
        items.append((top["score"], txt))
    items.sort(key=lambda x: -x[0])
    return _fit("EVENTS", [t for _s, t in items[:k]])


def breath_line(br):
    if not br or "per_min" not in br:
        return None
    return _fit("BREATH", [f"~{br['per_min']:.0f}/min", f"{br['n']} breaths in ~{br['span_s']:.0f}s",
                           "rough: speech masks breaths, so it undercounts"])


def pace_line(words):
    if not words or len(words) < 2:
        return "  PACE:    none (fewer than 2 words)"
    span = max(words[-1][2] - words[0][1], 1e-6)
    gaps = [(s - e, e) for (_w, _s, e), (_w2, s, _e2) in zip(words, words[1:])]
    g, at = max(gaps)
    pieces = [f"{len(words)} words / {span:.0f}s ≈ {len(words) / span * 60:.0f} wpm"]
    pieces.append(f"longest pause {g:.1f}s at {_t(at)}" if g >= 0.3 else "no pause over 0.3s")
    return _fit("PACE", pieces)


def _merge_spans(spans, gap=1.0):
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def laughs_line(r, hint=None, model=None):
    """LAUGHS. With a laugh model switched on (RAVEN_EARS_LAUGH_MODEL) the model leads:
    on Fox's timed note (17 Nov 2026) Omine heard 4 of her 4 laughs under speech and
    CNN14 heard 0, so the model's spans come first and the pitch hint is dropped.
    Without a model: room_laugh's 1 s / low-threshold pass, plus the hint."""
    def part(evs, k=4):
        top = sorted(evs, key=lambda e: -e["total"])[:k]
        out = ", ".join(f"{_t(e['start'])} ({e['total']:.2f})" for e in sorted(top, key=lambda e: e["start"]))
        return out + (f" +{len(evs) - k}" if len(evs) > k else "")
    model_ran = model and model.get('status') == 'ran'
    L, S = (r or {}).get("laughs", []), (r or {}).get("snorts", [])
    sp = f"{len(S)} ({part(S)})" if S else "none heard"
    pieces = []
    if model_ran:
        spans = _merge_spans(model.get('spans') or [])
        k = 5
        shown = ", ".join(_t(a) if b - a < 1.0 else f"{_t(a)}-{_t(b)}" for a, b in spans[:k])
        if len(spans) > k:
            shown += f" +{len(spans) - k}"
        pieces.append(f"{len(spans)} heard by {model['model']} ({shown})" if spans
                      else f"{model['model']}: none heard")
        pieces.append((f"CNN14 {len(L)}" if L else "CNN14 none heard") if r is not None else 'CNN14 unavailable')
        if not spans and not L and hint:   # the model heard nothing: the pitch hint may still be worth a look
            clocks = list(dict.fromkeys(_t(h['start']) for h in hint))
            pieces.append(f"laughing? big pitch jumps on short words at {', '.join(clocks[:2])}" + (f" +{len(clocks) - 2}" if len(clocks) > 2 else ""))
    else:
        pieces.append((f"{len(L)} ({part(L)})" if L else "none heard") if r is not None else 'CNN14 unavailable')
        if r is not None and not L and hint:
            clocks = list(dict.fromkeys(_t(h['start']) for h in hint))
            times = ", ".join(clocks[:3]) + (f" +{len(clocks) - 3}" if len(clocks) > 3 else "")
            pieces.append(f"laughing? big pitch jumps on short words at {times}")
    if r is not None:
        pieces.append(f"snorts {sp}")
    return _fit("LAUGHS", pieces)


# --------------------------------------------------------------------------- #
#  3. AFFECT  (audeering wav2vec2 dimensional emotion)
# --------------------------------------------------------------------------- #
def _load_affect(device):
    key = ("affect", device)
    if key in _MODELS:
        return _MODELS[key]
    from model_compat import check_transformers_torch
    check_transformers_torch()
    torch = _torch()
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kw):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            h = self.wav2vec2(input_values)[0]
            h = torch.mean(h, dim=1)
            return h, self.classifier(h)

    m = EmotionModel.from_pretrained(AFFECT_MODEL, local_files_only=offline()).to(device).eval()
    _MODELS.clear()
    _MODELS[key] = m
    return m


def _affect_windows(y16, fe, device, win_s, hop_s):
    torch = _torch()
    m = _load_affect(device)
    W, H = int(win_s * SR), int(hop_s * SR)
    starts = list(range(0, max(1, len(y16) - W + 1), H)) if len(y16) >= W else [0]
    rows, keep = [], []
    for s in starts:
        seg = y16[s:s + W]
        f0, f1 = int(s / HOP), int(min(len(y16), s + W) / HOP)
        act = fe["active"][f0:f1]
        if len(seg) < SR or act.size == 0 or act.mean() < 0.4:
            continue
        rows.append(seg)
        keep.append(s / SR)
    preds = []
    B = 8
    for i in range(0, len(rows), B):
        batch = rows[i:i + B]
        L = max(len(b) for b in batch)
        x = np.zeros((len(batch), L), np.float32)
        for j, b in enumerate(batch):
            bn = (b - b.mean()) / np.sqrt(b.var() + 1e-7)       # = Wav2Vec2FeatureExtractor(do_normalize)
            x[j, :len(b)] = bn
        with torch.no_grad():
            lg = m(torch.from_numpy(x).to(device))[1].float().cpu().numpy()
        preds.append(lg)
    _note_vram()
    if not preds:
        return np.zeros(0), np.zeros((0, 3))
    return np.array(keep) + win_s / 2.0, np.concatenate(preds, axis=0)


@ST.stage('affect')
def analyze_affect(y16, fe, dur, notes):
    if os.environ.get("RAVEN_EARS_NO_AFFECT") == "1":
        ST.mark("affect", "skipped", "RAVEN_EARS_NO_AFFECT=1")
        return {"ok": False}
    win_s = AFFECT_WIN
    hop_s = 1.5 if dur <= 75 else 3.0

    def run(dev):
        return _affect_windows(y16, fe, dev, win_s, hop_s)

    (centers, P), dev = _with_fallback("affect", run, notes)
    if len(centers) == 0:
        return {"ok": False, "device": dev}
    names = ("arousal", "dominance", "valence")
    out = {"ok": True, "device": dev, "n_windows": len(centers),
           "whole": {n: round(float(P[:, i].mean()), 3) for i, n in enumerate(names)},
           "centers": centers, "P": P}
    act = np.where(fe["active"])[0]
    if len(act) and (act[-1] - act[0]) / FPS >= 6.0:
        t0, t1 = act[0] / FPS, act[-1] / FPS
        edges = np.linspace(t0, t1, 4)
        th = []
        for k in range(3):
            sel = (centers >= edges[k]) & (centers < edges[k + 1] + (1e-6 if k == 2 else 0))
            th.append({n: (round(float(P[sel, i].mean()), 3) if sel.any() else None)
                       for i, n in enumerate(names)})
        out["thirds"] = th
    return out


def _phrase_affect(affect, start, end, dur):
    if not affect or not affect.get("ok"):
        return None
    centers, predictions = affect["centers"], affect["P"]
    left = centers - AFFECT_WIN / 2
    right = np.minimum(centers + AFFECT_WIN / 2, dur)
    overlap = np.minimum(end, right) - np.maximum(start, left)
    # Require at least a second and half of the actual window inside the phrase.
    use = ((overlap >= 1.0) & (overlap >= (right - left) / 2)
           & np.isfinite(predictions[:, [0, 2]]).all(axis=1))
    if not use.any():
        return None
    arousal, _, valence = np.average(predictions[use], axis=0, weights=overlap[use])
    return dict(arousal=round(float(arousal), 3), valence=round(float(valence), 3))


def _phrase_voice(data, fe, groups, baseline, dur):
    pitch, tune = data["pitch"], data["tune"] or {}
    ending = {(p["start"], p["end"]): p["ending"] for p in tune.get("phrases", [])}
    history = [r.get("pitch_hz") for r in baseline[-BASELINE_WINDOW:] if isinstance(r, dict)]
    history = [x for x in history if isinstance(x, (int, float)) and math.isfinite(x) and x > 0]
    usual_hz = np.median(history) if history else None
    active_db = fe["rms_db"][fe["active"]]
    note_db = np.percentile(active_db, 90) if active_db.size else None
    model = data["laugh_model"] or {}
    events = (data["events"] or {}).get("events", [])
    cnn = [e for e in events if e["label"] in LAUGH_LABELS] + (data["laughs"] or {}).get("laughs", [])
    laugh_spans = {"CNN14": [(e["start"], e["end"]) for e in cnn]}
    if model.get("status") == "ran":
        laugh_spans[model["model"]] = model["spans"]

    rows = []
    for start, end, words in groups:
        mask = (fe["times"] >= start) & (fe["times"] < end)
        f0 = fe["f0"][mask & fe["voiced"]]
        rms = fe["rms_db"][mask & fe["active"]]
        wpm = len(words) * 60.0 / (end - start) if len(words) >= 2 and end > start else None
        loud = float(np.percentile(rms, 90)) if rms.size else None
        row = dict(pitch_st=None, range_st=None, usual_pitch_st=None, wpm=wpm, loud_db=loud,
                   note_range_st=pitch.get("range_st"), usual_range_st=tune.get("usual_range_st"),
                   loud_vs_note_db=loud - float(note_db) if loud is not None else None,
                   affect=_phrase_affect(data["affect"], start, end, dur), ending=ending.get((start, end)),
                   laugh_sources=[name for name, spans in laugh_spans.items()
                                  if any(min(end, b) > max(start, a) for a, b in spans)])
        if pitch.get("ok") and len(f0) >= 8:
            lo, med, hi = np.percentile(f0, [5, 50, 95])
            row.update(pitch_st=float(_sem(med, pitch["median_hz"])), range_st=float(_sem(hi, lo)))
            if usual_hz is not None:
                row["usual_pitch_st"] = float(_sem(med, usual_hz))
        rows.append(row)
    return rows


def affect_line(a):
    if not a.get("ok"):
        return None
    w = a["whole"]
    pieces = [f"arousal {w['arousal']:.2f}", f"dominance {w['dominance']:.2f}", f"valence {w['valence']:.2f}"]
    th = a.get("thirds")
    if th:
        for n in ("arousal", "valence"):
            vals = [t[n] for t in th]
            if all(v is not None for v in vals):
                pieces.append(f"{n[:3]} thirds " + "→".join(f"{v:.2f}" for v in vals))
    return _fit("AFFECT", pieces)


# --------------------------------------------------------------------------- #
#  6. WORDLESS vs SILENT
# --------------------------------------------------------------------------- #
@ST.stage('gaps')
def analyze_gaps(fe, words, dur):
    active, voiced, n = fe["active"], fe["voiced"], fe["n"]
    f = lambda s: int(round(s * FPS))
    out = {"have_words": bool(words)}
    margin = 0.10
    if words:
        regions = []
        prev_end = 0.0
        first = True
        for _w, s, e in words:
            if s - prev_end >= GAP_MIN:
                regions.append((prev_end, s, "head" if first else "gap"))
            prev_end = max(prev_end, e)
            first = False
        if dur - prev_end >= GAP_MIN:
            regions.append((prev_end, dur, "tail"))
        wordless, silent, unvoiced = [], [], []
        for a, b, kind in regions:
            a2, b2 = a + (margin if kind != "head" else 0.0), b - (margin if kind != "tail" else 0.0)
            if b2 - a2 < 0.3:
                continue
            i0, i1 = f(a2), min(n, f(b2))
            seg_v = voiced[i0:i1]
            seg_a = active[i0:i1]
            # voice with no words: voiced frames, holes <= 0.2 s closed, runs >= 0.4 s kept
            vr = [(x + i0, y + i0) for x, y in _merge_runs(_runs(seg_v), f(0.2)) if (y - x) / FPS >= 0.4]
            wordless += vr
            # silence: not-active runs >= 0.3 s (head/tail silence is just recording padding)
            if kind == "gap":
                silent += [(x + i0, y + i0) for x, y in _runs(~seg_a) if (y - x) / FPS >= 0.3]
            # active but unvoiced and not already counted as wordless
            claimed = np.zeros(i1 - i0, bool)
            for x, y in vr:
                claimed[x - i0:y - i0] = True
            ur = [(x + i0, y + i0) for x, y in _runs(seg_a & ~seg_v & ~claimed) if (y - x) / FPS >= 0.4]
            unvoiced += ur
        tosec = lambda rr: [(x / FPS, y / FPS) for x, y in rr]
        out.update(wordless=tosec(wordless), silent=tosec(silent), unvoiced=tosec(unvoiced))
    else:
        act = np.where(active)[0]
        silent = []
        if len(act):
            seg = ~active[act[0]:act[-1] + 1]
            silent = [(x + act[0], y + act[0]) for x, y in _runs(seg) if (y - x) / FPS >= GAP_MIN]
        out.update(wordless=None, silent=[(x / FPS, y / FPS) for x, y in silent], unvoiced=[])
    tot = lambda rr: round(sum(b - a for a, b in rr), 2) if rr else 0.0
    out["wordless_s"] = tot(out["wordless"]) if out["wordless"] is not None else None
    out["silent_s"] = tot(out["silent"])
    out["unvoiced_s"] = tot(out["unvoiced"])
    return out


def gap_lines(g):
    wl, sl = g["wordless"], g["silent"]
    if wl is None:
        s = (f"SILENT: {g['silent_s']:.1f}s ({_spans(sl, 2)})" if sl else "SILENT: none")
        return [_fit("SILENT", [s[len('SILENT: '):], "no word times, so wordless voice can't be told from speech"])]
    w = (f"{g['wordless_s']:.1f}s of voice with no words ({_spans(wl, 3)})" if wl else "none")
    s = (f"{g['silent_s']:.1f}s ({_spans(sl, 2)})" if sl else "none")
    u = f"unvoiced sound {g['unvoiced_s']:.1f}s" if g["unvoiced_s"] >= 1.0 else ""
    line = f"  WORDLESS: {w} · SILENT: {s}" + (f" · {u}" if u else "")
    if len(line) <= MAX_LINE:
        return [line]
    # too long: fewer spans, then split in two
    w1 = f"{g['wordless_s']:.1f}s of voice with no words ({_spans(wl, 1)})" if wl else "none"
    s1 = f"{g['silent_s']:.1f}s ({_spans(sl, 1)})" if sl else "none"
    line = f"  WORDLESS: {w1} · SILENT: {s1}"
    if len(line) <= MAX_LINE:
        return [line]
    return [f"  WORDLESS: {w1}", f"  SILENT:   {s1}"]


# --------------------------------------------------------------------------- #
#  6b. CONTOUR  (what the pitch did inside a wordless stretch)
# --------------------------------------------------------------------------- #
# Huginn, 9 Nov 2026. WORDLESS says there was voice with no words in it. It never said
# what the voice DID: a hum that climbs a third and stays there read the same as a flat
# "mm". This describes the movement and nothing else. It does not say what the sound was
# (a hum, a held vowel, a moan, a laugh all arrive here as "voiced, no words") and it does
# not say what it meant. Semitone sizes are measured; the interval names are a convenience
# and say "about", because a speaking voice isn't tuned.
CONTOUR_MIN = 1.0           # seconds; shorter stretches are too brief to describe honestly
_INTERVAL = {1: "a semitone", 2: "a whole tone", 3: "a minor third", 4: "a major third", 5: "a fourth",
             6: "a tritone", 7: "a fifth", 8: "a minor sixth", 9: "a major sixth", 10: "a minor seventh",
             11: "a major seventh", 12: "an octave"}
_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_name(hz: float) -> str:
    midi = int(round(69 + 12 * math.log2(hz / 440.0)))
    return f"{_NOTE[midi % 12]}{midi // 12 - 1}"


def _step(st: float) -> str:
    n = max(1, int(round(abs(st))))
    return f"~{n} st (" + ("about " + _INTERVAL[n] if n <= 12 else "over an octave") + ")"


def describe_contour(f0) -> dict:
    """One wordless stretch's F0 frames (NaN where unvoiced) -> what the pitch did.

    Declines ("ok": False, with the reason) when under 60% of the frames carry pitch, or
    when the track jumps about too much to be one line of melody (octave errors, creak).
    """
    f0 = np.asarray(f0, dtype=float)
    n = len(f0)
    ok = np.isfinite(f0) & (f0 > 0)
    share = float(ok.mean()) if n else 0.0
    if n < 8 or share < 0.6:
        return {"ok": False, "reason": f"pitch unclear (voiced {share:.0%})", "voiced_share": round(share, 2)}
    hz = f0[ok]
    pos = np.where(ok)[0] / max(1, n - 1)
    ref = float(np.median(hz))
    st = _sem(hz, ref)
    if len(st) >= 5:                       # 80 ms median: removes single-frame octave blips
        from scipy.signal import medfilt
        st = medfilt(st, 5)
    jumps = float(np.mean(np.abs(np.diff(st)) > 6.0)) if len(st) > 1 else 0.0
    if jumps > 0.08:
        return {"ok": False, "reason": "pitch unclear (the track jumps)", "voiced_share": round(share, 2)}
    # Where it began and ended: the first and last ~150 ms of pitch. A wider window (I first
    # used a fifth of the run) reads a climb that is still moving as smaller than it was.
    edge = min(len(st), max(3, int(round(0.15 * FPS))))
    start = float(np.median(st[:edge]))
    end = float(np.median(st[-edge:]))
    lo, hi = float(np.percentile(st, 5)), float(np.percentile(st, 95))
    spread = lambda x: float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) else 0.0
    near = lambda m: _note_name(float(np.median(hz[m]))) if m.any() else _note_name(ref)
    out = {"ok": True, "voiced_share": round(share, 2), "median_hz": round(ref, 1),
           "start_st": round(start, 1), "end_st": round(end, 1), "range_st": round(hi - lo, 1)}
    delta = end - start
    peak_pos, trough_pos = float(pos[int(np.argmax(st))]), float(pos[int(np.argmin(st))])
    if hi - lo < 1.0:
        out.update(kind="held", phrase=f"held near {_note_name(ref)} (~{ref:.0f} Hz)")
    elif 0.2 <= peak_pos <= 0.8 and hi - max(start, end) >= 2.0:
        out.update(kind="arch", phrase=f"rises {_step(hi - start)}, then falls {_step(hi - end)}")
    elif 0.2 <= trough_pos <= 0.8 and min(start, end) - lo >= 2.0:
        out.update(kind="dip", phrase=f"dips {_step(start - lo)}, then rises {_step(end - lo)}")
    elif abs(delta) >= 1.5:
        verb = "rises" if delta > 0 else "falls"
        head, tail = pos <= 0.4, pos >= 0.6
        if spread(st[tail]) < 1.0 and spread(st[head]) >= 1.0:
            phrase = f"{verb} {_step(delta)}, then holds near {near(tail)}"
        elif spread(st[head]) < 1.0 and spread(st[tail]) >= 1.0:
            phrase = f"holds near {near(head)}, then {verb} {_step(delta)}"
        else:
            phrase = f"{verb} {_step(delta)}"
        out.update(kind="rise" if delta > 0 else "fall", phrase=phrase)
    else:
        out.update(kind="wavers", phrase=f"wavers within ~{hi - lo:.0f} st around {_note_name(ref)}")
    return out


def analyze_contour(fe, gaps):
    """Contour of every WORDLESS run of CONTOUR_MIN seconds or more. None without word times."""
    if not gaps or gaps.get("wordless") is None:
        return None
    runs = []
    for a, b in gaps["wordless"]:
        if b - a < CONTOUR_MIN:
            continue
        c = describe_contour(fe["f0"][int(round(a * FPS)):int(round(b * FPS))])
        c.update(start=round(a, 2), end=round(b, 2))
        runs.append(c)
    return {"runs": runs, "min_s": CONTOUR_MIN}


def contour_lines(c, k=3):
    """The k longest described stretches, in time order, one card line each (a second and
    third line are indented under the first). No lines when there is nothing to describe."""
    runs = (c or {}).get("runs") or []
    if not runs:
        return []
    top = sorted(sorted(runs, key=lambda r: r["start"] - r["end"])[:k], key=lambda r: r["start"])
    lines = []
    for n, r in enumerate(top):
        text = f"{_t(r['start'])} ({r['end'] - r['start']:.1f}s) " + (r["phrase"] if r["ok"] else r["reason"])
        if n == len(top) - 1 and len(runs) > k:
            text += f" · +{len(runs) - k} more"
        line = ("  CONTOUR: " if n == 0 else " " * 11) + text
        lines.append(line if len(line) <= MAX_LINE else line[:MAX_LINE - 1].rstrip() + "…")
    return lines


# --------------------------------------------------------------------------- #
#  4. SHAPE   and   7. PEAK
# --------------------------------------------------------------------------- #
@ST.stage('shape')
def analyze_shape(fe, words, p):
    active = np.where(fe["active"])[0]
    if len(active) == 0:
        return {"ok": False}
    t0, t1 = active[0] / FPS, active[-1] / FPS
    if t1 - t0 < 3.0:
        return {"ok": False, "reason": "note shorter than 3s"}
    edges = np.linspace(t0, t1, 4)
    th = []
    for k in range(3):
        sel = fe["active"] & (fe["times"] >= edges[k]) & (fe["times"] < edges[k + 1] + (1e-6 if k == 2 else 0))
        loud = float(np.percentile(fe["rms_db"][sel], 90)) if sel.any() else None
        vs = fe["voiced"] & (fe["times"] >= edges[k]) & (fe["times"] < edges[k + 1] + (1e-6 if k == 2 else 0))
        pitch = float(np.median(fe["f0"][vs])) if vs.sum() >= 8 else None
        wpm = None
        if words:
            cnt = sum(1 for _w, s, e in words if edges[k] <= (s + e) / 2 < edges[k + 1] + (1e-6 if k == 2 else 0))
            wpm = cnt / (edges[k + 1] - edges[k]) * 60.0
        th.append({"loud_db": loud, "pitch_hz": pitch, "wpm": wpm})
    return {"ok": True, "thirds": th, "span": (t0, t1)}


def shape_line(s):
    if not s.get("ok"):
        return None
    th = s["thirds"]
    pieces = []
    L = [t["loud_db"] for t in th]
    if all(v is not None for v in L):
        d, mid = L[2] - L[0], L[1] - (L[0] + L[2]) / 2
        word = ("soft→loud" if d >= 3 else "loud→soft" if d <= -3 else
                "swell" if mid >= 3 else "dip" if mid <= -3 else "steady")
        pieces.append(f"{word} ({L[0]:.0f}/{L[1]:.0f}/{L[2]:.0f} dB)")
    W = [t["wpm"] for t in th]
    if all(v is not None for v in W):
        pieces.append("→".join(f"{v:.0f}" for v in W) + " wpm")
    P = [t["pitch_hz"] for t in th]
    if all(v is not None for v in P):
        st = float(_sem(P[2], P[0]))
        arrow = "↑" if st >= 1 else "↓" if st <= -1 else "→"
        pieces.append(f"pitch {arrow} ({P[0]:.0f}→{P[1]:.0f}→{P[2]:.0f} Hz)")
    return _fit("SHAPE", pieces) if pieces else None


@ST.stage('peak')
def analyze_peak(fe, affect, dur):
    act = np.where(fe["active"])[0]
    if len(act) == 0 or dur < 4.0:
        return {"ok": False, "reason": "note shorter than 4s"}
    L = float(np.clip(0.2 * dur, 2.0, 5.0))
    step = 0.5
    rms_db, times, active = fe["rms_db"], fe["times"], fe["active"]
    starts = np.arange(0, max(dur - L, 0) + 1e-6, step)
    loud, frac = [], []
    for s in starts:
        sel = (times >= s) & (times < s + L)
        frac.append(active[sel].mean() if sel.any() else 0)
        loud.append(float(np.percentile(rms_db[sel], 90)) if sel.any() else -100.0)
    loud, frac = np.array(loud), np.array(frac)
    ok = frac >= 0.4
    if not ok.any():
        return {"ok": False, "reason": "nothing loud enough"}

    def z(x):
        s = np.nanstd(x[ok])
        return (x - np.nanmean(x[ok])) / s if s > 1e-6 else np.zeros_like(x)

    score = z(loud)
    aro = None
    if affect and affect.get("ok"):
        c, P = affect["centers"], affect["P"][:, 0]
        aro = np.interp(starts + L / 2, c, P)
        score = 0.5 * score + 0.5 * z(aro)
    score = np.where(ok, score, -np.inf)
    k = int(np.argmax(score))
    s = float(starts[k])
    return {"ok": True, "start": s, "end": s + L, "loud_db": round(float(loud[k]), 1),
            "note_loud_db": round(float(np.median(rms_db[active])), 1),
            "arousal": (round(float(aro[k]), 2) if aro is not None else None),
            "note_arousal": (affect["whole"]["arousal"] if aro is not None else None)}


def peak_line(pk):
    if not pk.get("ok"):
        return None
    pieces = [_span(pk["start"], pk["end"]), f"loud {pk['loud_db']:.0f} dB (note median {pk['note_loud_db']:.0f})"]
    if pk["arousal"] is not None:
        pieces.append(f"arousal {pk['arousal']:.2f} (note {pk['note_arousal']:.2f})")
    return _fit("PEAK", pieces)


# --------------------------------------------------------------------------- #
#  7b. EMPHASIS  (which words the speaker leaned on)
# --------------------------------------------------------------------------- #
EMPH_CONTEXT = 4           # neighbours each side that a word is measured against
EMPH_PITCH_ST = 3.0        # F0 peak this far above its neighbours' peaks
EMPH_LOUD_DB = 4.0         # word loudness this far above its neighbours
EMPH_LONG = 1.6            # duration this many times what the speaker's pace predicts
EMPH_MIN_S = 0.08          # Whisper times shorter words too roughly to trust
EMPH_LONG_MIN_S = 0.25     # "long" also needs to be audibly long, not just long for a "to"
EMPH_FUNC_X = 1.25         # function words must clear all three thresholds by this factor
EMPH_FUNCTION = {
    "a", "an", "the", "and", "or", "but", "so", "if", "of", "to", "in", "on", "at", "by", "for", "with",
    "as", "is", "are", "was", "were", "be", "been", "am", "it", "its", "i", "you", "he", "she", "we",
    "they", "me", "my", "your", "his", "her", "our", "their", "this", "that", "do", "did", "does", "than",
    "then", "just", "from", "up", "um", "uh", "oh", "mm", "hmm", "ah",
}


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1                                  # silent final e
    return max(1, n)


def emphasis_features(words, fe):
    """Per word: pitch_st (F0 peak vs the neighbours' peaks, None if unvoiced), loud_db (word
    loudness vs neighbours), dur_ratio (duration vs syllables x this clip's median s/syllable)."""
    n = len(words)
    times, f0, rms = fe["times"], fe["f0"], fe["rms_db"]
    peak_f0, peak_db = [], []
    for _w, s, e in words:
        idx = np.where((times >= s) & (times <= e))[0]
        v = f0[idx][np.isfinite(f0[idx])]
        peak_f0.append(float(np.percentile(v, 90)) if len(v) >= 3 else None)
        peak_db.append(float(np.percentile(rms[idx], 90)) if len(idx) else None)
    ok = [max(e - s, 0.0) for _w, s, e in words]
    rate = [d / _syllables(w) for (w, _s, _e), d in zip(words, ok) if d >= EMPH_MIN_S]
    rate = float(np.median(rate)) if rate else None
    feats = []
    for i, (w, s, e) in enumerate(words):
        nb = [j for j in range(max(0, i - EMPH_CONTEXT), min(n, i + EMPH_CONTEXT + 1)) if j != i]
        f = {"pitch_st": None, "loud_db": None, "dur_ratio": None}
        nf = [peak_f0[j] for j in nb if peak_f0[j]]
        if peak_f0[i] and len(nf) >= 2:
            f["pitch_st"] = float(_sem(peak_f0[i], float(np.median(nf))))
        nd = [peak_db[j] for j in nb if peak_db[j] is not None]
        if peak_db[i] is not None and len(nd) >= 2:
            f["loud_db"] = peak_db[i] - float(np.median(nd))
        if rate and ok[i] >= EMPH_MIN_S:
            f["dur_ratio"] = ok[i] / (_syllables(w) * rate)
        feats.append(f)
    return feats


def laugh_hint(words, feats):
    """Pitch/duration candidates only; no word text leaves this measurement.

    Reuse emphasis's neighbour-relative F0 peaks and pace-normalized duration.
    A one-syllable or filler word needs >=8 st, >=1.6x duration and >=0.25 s.
    These can also be pitch-tracking errors, emphasis or phrase-final lengthening.
    """
    if not words or len(words) < 4:
        return []
    found = []
    for (word, start, end), f in zip(words, feats):
        bare = re.sub(r"[^a-z]", "", word.lower())
        pitch, length = f.get('pitch_st'), f.get('dur_ratio')
        short = bare and (_syllables(bare) == 1 or bare in EMPH_FUNCTION or bare == 'like')
        if (short and pitch is not None and np.isfinite(pitch) and pitch >= 8
                and length is not None and np.isfinite(length) and length >= EMPH_LONG
                and end - start >= EMPH_LONG_MIN_S):
            found.append(dict(start=float(start), end=float(end), pitch_st=float(pitch), dur_ratio=float(length)))
    return found


def rank_emphasis(words, feats, top=None):
    """Words that stand out on >= 2 of pitch / loudness / duration, best first, only a few.
    Missing features (unvoiced words) simply cannot vote. Function words and very short words
    need all three, further past the line. Returns [{"i", "word", "start", "pitch_st",
    "loud_db", "dur_ratio", "hits", "score"}]. The last word before a pause is never judged on
    duration (phrases end long anyway)."""
    n = len(words)
    if n < 4:
        return []
    top = top if top is not None else min(5, max(1, round(n / 8)))
    found = []
    for i, ((w, s, e), f) in enumerate(zip(words, feats)):
        bare = re.sub(r"[^\w']", "", w.lower())
        weak = bare in EMPH_FUNCTION or len(bare) <= 2 or (e - s) < EMPH_MIN_S
        x = EMPH_FUNC_X if weak else 1.0
        final = i + 1 == n or words[i + 1][1] - e >= GAP_MIN
        hits, score = [], 0.0
        for key, thr in (("pitch_st", EMPH_PITCH_ST), ("loud_db", EMPH_LOUD_DB), ("dur_ratio", EMPH_LONG)):
            v = f.get(key)
            if v is None or (key == "dur_ratio" and (final or e - s < EMPH_LONG_MIN_S)):
                continue
            if v >= thr * x:
                hits.append(key)
                score += v / thr
        if len(hits) >= (3 if weak else 2):
            found.append({"i": i, "word": w, "start": s, "pitch_st": f.get("pitch_st"), "loud_db": f.get("loud_db"),
                          "dur_ratio": f.get("dur_ratio"), "hits": hits, "score": round(score, 2)})
    found.sort(key=lambda r: -r["score"])
    return found[:top]


@ST.stage('emphasis')
def analyze_emphasis(words, fe):
    if not words or len(words) < 4:
        return {"ok": False, "reason": "fewer than 4 words", "words": []}
    return {"ok": True, "words": sorted(rank_emphasis(words, emphasis_features(words, fe)), key=lambda r: r["i"])}


def emphasis_line(em):
    if not em or not em.get("ok"):
        return "  EMPHASIS: none (" + ((em or {}).get("reason") or "no words") + ")"
    ws = em["words"]
    if not ws:
        return "  EMPHASIS: none stands out, even delivery"
    texts = [re.sub(r"[^\w']", "", r["word"]) or r["word"] for r in ws]
    pieces = []
    for r, t in zip(ws, texts):
        why = []
        if "pitch_st" in r["hits"]:
            why.append(f"pitch +{r['pitch_st']:.0f} st")
        if "loud_db" in r["hits"]:
            why.append("loud")
        if "dur_ratio" in r["hits"]:
            why.append("long")
        at = f" at {_t(r['start'])}" if texts.count(t) > 1 else ""
        pieces.append(f'"{t}"{at} ({", ".join(why)})')
    pre, kept = "  EMPHASIS: ", pieces
    while len(kept) > 1 and len(pre + ", ".join(kept)) + 1 > MAX_LINE:
        kept = kept[:-1]
    more = f" +{len(pieces) - len(kept)}" if len(kept) < len(pieces) else ""
    return pre + ", ".join(kept) + more + "."


# --------------------------------------------------------------------------- #
#  8. WORDS  (the transcript, with the card's markers set inline)
# --------------------------------------------------------------------------- #
WORDS_FULL_S = 180.0       # clips shorter than this print every word line
PAUSE_SHOW = 1.0           # pauses this long are written into the text
INLINE_EVENTS = {          # AudioSet label -> marker text (Breathing/Whispering have their own lines)
    "Laughter": "laugh", "Giggle": "laugh", "Belly laugh": "laugh", "Chuckle, chortle": "laugh",
    "Baby laughter": "laugh", "Crying, sobbing": "cry", "Whimper": "whimper", "Wail, moan": "moan",
    "Sigh": "sigh", "Groan": "groan", "Grunt": "grunt", "Gasp": "gasp", "Pant": "pant",
    "Sniff": "sniff", "Snort": "snort", "Cough": "cough", "Sneeze": "sneeze", "Screaming": "scream",
    "Yell": "yell", "Shout": "shout", "Hiccup": "hiccup",
}


def _word_tokens(words, peak, events, gaps, emph=(), whisper=None):
    """Timeline of (time, order, text): peak open -1, events/wordless 0, pauses 1, words 2, peak close 3."""
    tok = []
    prev_end = None
    for i, (w, s, e) in enumerate(words):
        if prev_end is not None and s - prev_end >= PAUSE_SHOW:
            tok.append((prev_end, 1, f"…({s - prev_end:.1f}\u00a0s)"))
        if i in emph:                                   # *word*, punctuation left outside the stars
            m = re.match(r"^(\W*)(.*?)(\W*)$", w)
            w = f"{m.group(1)}*{m.group(2)}*{m.group(3)}" if m.group(2) else w
        tok.append((s, 2, w))
        prev_end = max(prev_end or 0.0, e)
    if events:
        last = {}
        for ev in sorted(events.get("events") or [], key=lambda x: x["start"]):
            name = INLINE_EVENTS.get(ev["label"])
            if not name:
                continue
            if name in last and ev["start"] - last[name] < 1.5:
                last[name] = ev["end"]
                continue
            last[name] = ev["end"]
            tok.append((ev["start"], 0, f"[{name}]"))
    for a, b in (gaps or {}).get("wordless") or []:
        tok.append((a, 0, f"(voice,\u00a0no\u00a0words\u00a0{b - a:.1f}\u00a0s)"))
    tok += RW.whisper_tokens(whisper, words)
    if peak and peak.get("ok"):
        ws = [t for t in tok if t[1] == 2]
        i0 = min(range(len(ws)), key=lambda i: abs(ws[i][0] - peak["start"]))
        i1 = max(min(range(len(ws)), key=lambda i: abs(ws[i][0] - peak["end"])), i0)
        tok.append((ws[i0][0], -1, "⟪PEAK⟫"))
        tok.append((ws[i1][0], 3, "⟪/PEAK⟫"))
    tok.sort(key=lambda x: (x[0], x[1]))
    return tok


def _word_rows(tok, words):
    """Group tokens into rows [(start_s, text)], a new row at each word gap >= GAP_MIN."""
    cut = {round(s, 3) for (_w, _s0, e0), (_w2, s, _e2) in zip(words, words[1:]) if s - e0 >= GAP_MIN}
    rows, cur = [], []
    for t, order, text in tok:
        if order in (-1, 0, 2) and round(t, 3) in cut and any(o == 2 for _t, o, _x in cur):
            rows.append(cur)
            cur = []
        cur.append((t, order, text))
    if cur:
        rows.append(cur)
    return [(r[0][0], " ".join(x for _t, _o, x in r)) for r in rows]


def _wrap_row(t, text, width=MAX_LINE):
    pre = f"  [{_t(t)}] "
    lines, cur = [], pre
    for w in text.split(" "):
        if len(cur) + len(w) + (0 if cur.endswith(" ") else 1) > width:
            lines.append(cur.rstrip())
            cur = " " * len(pre)
        cur += w if cur.endswith(" ") else " " + w
    lines.append(cur.rstrip())
    return [l.replace("\u00a0", " ") for l in lines]


FILLER_GAP_S = 1.0         # a tail phrase this long after the last spoken word counts as its own segment
FILLER_VOICED = 0.25       # fraction of voiced frames under which a tail phrase is "weak voicing"


def split_tail_filler(words, fe, dur):
    """(spoken, filler): Whisper's end-of-clip hallucinations ("Thank you.", "Thanks for watching") marked, not dropped.

    Reuses the song filter (lyrics_shared.is_hallucination, TAIL_S). Needs the trailing 1-4 words to be a known
    phrase, inside the last TAIL_S seconds, AND (weak voicing under them, or a gap before them, or a phrase no
    one says aloud such as "thanks for watching"). Mid-clip speech is never touched."""
    if not words or len(words) < 1:
        return words, []
    import lyrics_shared as LS
    spoken = list(words)
    filler = []
    while spoken:
        cut = None
        for n in (4, 3, 2, 1):
            run = spoken[-n:]
            if len(spoken) >= n and run[0][1] >= dur - LS.TAIL_S and LS.is_hallucination(" ".join(x[0] for x in run)):
                cut = n
                break
        if cut is None:
            break
        run, before = spoken[-cut:], spoken[:-cut]
        text = LS._norm(" ".join(x[0] for x in run))
        strong = any(h in text for h in LS._HALL_ANY)
        gap = (run[0][1] - before[-1][2]) >= FILLER_GAP_S if before else True
        weak = False
        if fe is not None:
            a, b = run[0][1], min(run[-1][2], run[0][1] + 1.5 * cut)
            i, j = int(a * SR / HOP), max(int(b * SR / HOP), int(a * SR / HOP) + 1)
            seg = np.asarray(fe["voiced"])[i:j]
            weak = len(seg) == 0 or float(np.mean(seg)) < FILLER_VOICED
        if not (strong or gap or weak):
            break
        filler = run + filler
        spoken = before
    return spoken, filler


def filler_line(filler):
    if not filler:
        return []
    return [f"  \u27ea\u2248 likely Whisper filler, not spoken\u27eb [{_t(filler[0][1])}] "
            + " ".join(x[0] for x in filler).strip()]


def words_lines(words, dur, peak=None, events=None, gaps=None, status=None, emphasis=None, whisper=None):
    """The transcript. emphasis: analyze_emphasis() result, its words get *stars*. status (used only when there are no words): 'failed: why' | 'off: why' | None."""
    if not words:
        if status and status.startswith("failed"):
            return [f"  WORDS FAILED: {status[7:].strip() or 'no reason given'} (NOT SILENCE: the words are unknown, not absent)"]
        if status and status.startswith("off"):
            return [f"  WORDS:   not transcribed ({status[4:].strip()})"]
        return ["  WORDS:   none heard (no speech found)"]
    rows = _word_rows(_word_tokens(words, peak, events, gaps, {r["i"] for r in (emphasis or {}).get("words", [])}, whisper), words)
    n_rows, omitted, keep = len(rows), 0, set(range(len(rows)))
    if dur >= WORDS_FULL_S and n_rows > 20:
        keep = set(range(6)) | set(range(n_rows - 4, n_rows))
        if peak and peak.get("ok"):
            keep |= {i for i, (t, _x) in enumerate(rows) if peak["start"] - 15 <= t <= peak["end"] + 5}
        omitted = n_rows - len(keep)
    out = [f"  WORDS:   {len(words)} words, {n_rows} phrases"]
    prev = -1
    for i, (t, text) in enumerate(rows):
        if i not in keep:
            continue
        if prev != -1 and i - prev > 1:
            out.append(f"  … ({i - prev - 1} phrases omitted)")
        out += _wrap_row(t, text)
        prev = i
    if omitted:
        out.append(f"  (long clip: {omitted} of {n_rows} phrases omitted; head, peak region and tail shown; --json has every word)")
    return out


# --------------------------------------------------------------------------- #
#  4b. VOICE QUALITY   (breathy / pressed / rough, voiced frames only)
# --------------------------------------------------------------------------- #
# Parselmouth (Praat in Python, GPL-3+) gives CPPS, HNR and jitter/shimmer. Without it the numpy fallback
# gives approximations only: an unsmoothed cepstral peak and an autocorrelation HNR; no jitter/shimmer.
QUALITY_KEYS = ("cpps_db", "hnr_db", "h1h2_db", "alpha_db", "jitter", "shimmer", "onset_ms")
# metric -> minimum spread for the baseline z-score (same idea as METRICS)
QUALITY_SCALE = {"cpps_db": 1.0, "hnr_db": 2.0, "h1h2_db": 2.0, "alpha_db": 2.0, "jitter": 0.004,
                 "shimmer": 0.02, "onset_ms": 40.0, "loud_db": 2.0}
Q_MIN_RUN = 14             # frames (~0.22 s) of continuous voicing before a stretch is measured
Q_MIN_THIRD_S = 0.5        # voiced seconds a third needs before it can be compared


def _parselmouth():
    try:
        import parselmouth
        from parselmouth.praat import call
        return parselmouth, call
    except Exception:
        return None, None


def _frame_spectra(y16, fe, idx):
    """H1-H2 (uncorrected for formants) and an alpha ratio (1-5 kHz minus 50-1000 Hz energy, dB) per frame."""
    n_win, nfft = 640, 2048                      # 40 ms Hann window
    win = np.hanning(n_win)
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    lo, hi = (freqs >= 50) & (freqs < 1000), (freqs >= 1000) & (freqs < 5000)
    h1h2, alpha = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        c = int(i * HOP + FRAME // 2)
        seg = y16[max(c - n_win // 2, 0):c + n_win // 2]
        if len(seg) < n_win:
            continue
        mag = np.abs(np.fft.rfft(seg * win, nfft)) + 1e-9
        f0 = fe["f0"][i]
        a = mag[(freqs >= 0.9 * f0) & (freqs <= 1.1 * f0)]
        b = mag[(freqs >= 1.8 * f0) & (freqs <= 2.2 * f0)]
        if a.size and b.size:
            h1h2[j] = 20 * np.log10(a.max() / b.max())
        pw = mag ** 2
        alpha[j] = 10 * np.log10(pw[hi].sum() / pw[lo].sum())
    return h1h2, alpha


def _np_hnr_cpp(y16, fe, idx):
    """Fallback without Parselmouth: autocorrelation HNR and an unsmoothed cepstral peak prominence per frame."""
    hnr, cpp = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    q = np.arange(1, 640) / SR
    for j, i in enumerate(idx):
        c = int(i * HOP + FRAME // 2)
        seg = y16[max(c - 512, 0):c + 512]
        if len(seg) < 1024:
            continue
        x = (seg - seg.mean()) * np.hanning(len(seg))
        ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2048)) ** 2)[:1024]
        lag = int(round(SR / fe["f0"][i]))
        if 0 < lag < 512 and ac[0] > 0:
            r = float(np.clip(ac[lag - 2:lag + 3].max() / ac[0], 1e-3, 0.999))
            hnr[j] = 10 * np.log10(r / (1 - r))
        cep = 20 * np.log10(np.abs(np.fft.irfft(np.log(np.abs(np.fft.rfft(x, 2048)) + 1e-9)))[1:640] + 1e-9)
        sel = (q >= 1 / 330) & (q <= 1 / 60)
        if sel.any():
            k = np.argmax(np.where(sel, cep, -1e9))
            fit = np.polyfit(q[sel], cep[sel], 1)
            cpp[j] = cep[k] - np.polyval(fit, q[k])
    return hnr, cpp


@ST.stage('quality')
def analyze_quality(y16, fe, notes=None, edges=None, f0_range=(60, 330)):
    """Voice quality over voiced frames only, whole note and per third of the sounding span. Music mode passes
    edges (section boundaries in seconds, n+1 of them) to get one pack per section instead, and a wider f0_range
    for the CPPS peak search (sung notes sit above 330 Hz); q['thirds'] then holds the n section packs."""
    vidx = np.where(fe["voiced"])[0]
    act = np.where(fe["active"])[0]
    if len(vidx) < 2 * Q_MIN_RUN or len(act) == 0:
        return {"ok": False, "reason": "too little voiced sound to measure"}
    pm, call = _parselmouth()
    optional_quality = pm is None
    if edges is None:
        edges = np.linspace(act[0] / FPS, act[-1] / FPS, 4)
    K = len(edges) - 1
    third_of = np.clip(np.searchsorted(edges, fe["times"], side="right") - 1, 0, K - 1)
    h1h2_f, alpha_f = _frame_spectra(y16, fe, vidx)
    hnr_f = np.full(len(vidx), np.nan)
    cpp_f = np.full(len(vidx), np.nan)
    if pm is not None:
        try:
            snd = pm.Sound(y16.astype(np.float64), sampling_frequency=SR)
            hn = snd.to_harmonicity_cc(time_step=1 / FPS, minimum_pitch=75.0)
            hv = hn.values[0].copy()
            hv[hv < -150] = np.nan
            tt = hn.xs()
            ok = np.isfinite(hv)
            if ok.sum() > 3:
                hnr_f = np.interp(fe["times"][vidx] + FRAME / 2 / SR, tt[ok], hv[ok])
        except Exception as e:
            pm = None
            if notes is not None:
                notes.append(f"QUALITY: parselmouth failed ({type(e).__name__}: {str(e)[:60]}); numpy approximations used")
    if pm is None:
        ST.mark("quality", "ran" if optional_quality else "degraded",
                "numpy approximations; optional praat-parselmouth: see INSTALL.md" if optional_quality else
                "parselmouth failed; numpy approximations used")
        hnr_f, cpp_f = _np_hnr_cpp(y16, fe, vidx)
    q = {"ok": True, "engine": "parselmouth" if pm else "numpy (approximate)"}
    # CPPS / jitter / shimmer need unbroken stretches: one Praat call per voiced run (cut at the third edges)
    runs = []
    for k in range(K):
        m = fe["voiced"] & (third_of == k)
        runs += [(a, b, k) for a, b in _merge_runs(_runs(m), 4) if b - a >= Q_MIN_RUN]
    runs.sort(key=lambda r: -(r[1] - r[0]))
    run_stats = []          # (third, seconds, cpps, jitter, shimmer)
    if pm is not None:
        for a, b, k in runs[:max(40, 12 * K)]:
            seg = pm.Sound(y16[int(a * HOP):int(b * HOP + FRAME)].astype(np.float64), sampling_frequency=SR)
            cp = jit = shim = None
            try:
                cg = call(seg, "To PowerCepstrogram", 60, 0.002, 5000, 50)
                cp = float(call(cg, "Get CPPS", True, 0.02, 0.0005, f0_range[0], f0_range[1], 0.05, "Parabolic",
                                0.001, 0, "Straight", "Robust"))
                if not np.isfinite(cp):
                    cp = None
            except Exception:
                pass
            try:
                pp = call(seg, "To PointProcess (periodic, cc)", 75, 600)
                if call(pp, "Get number of points") >= 8:
                    jit = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
                    shim = float(call([seg, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
                    jit = jit if np.isfinite(jit) else None
                    shim = shim if np.isfinite(shim) else None
            except Exception:
                pass
            run_stats.append((k, (b - a) / FPS, cp, jit, shim))

    def wmean(vals):            # duration-weighted mean of (seconds, value) pairs, None-safe
        v = [(s, x) for s, x in vals if x is not None]
        return float(sum(s * x for s, x in v) / sum(s for s, _ in v)) if v else None

    def med(a):
        return float(np.nanmedian(a)) if np.isfinite(a).any() else None

    def pack(sel, third):
        out = {"voiced_s": round(float(sel.sum() / FPS), 2),
               "hnr_db": med(hnr_f[sel]), "h1h2_db": med(h1h2_f[sel]), "alpha_db": med(alpha_f[sel])}
        rs = [r for r in run_stats if third is None or r[0] == third]
        if pm is not None:
            out["cpps_db"] = wmean([(r[1], r[2]) for r in rs])
            out["jitter"] = wmean([(r[1], r[3]) for r in rs])
            out["shimmer"] = wmean([(r[1], r[4]) for r in rs])
        else:
            out["cpps_db"], out["jitter"], out["shimmer"] = med(cpp_f[sel]), None, None
        return out

    vt = third_of[vidx]
    q["whole"] = pack(np.ones(len(vidx), bool), None)
    q["thirds"] = [pack(vt == k, k) for k in range(K)]
    q["whole"]["onset_ms"] = _onset_ms(fe)
    return q


def _onset_ms(fe):
    """Median time for loudness to climb from 10 dB under a voiced run's early peak to 3 dB under it (runs from quiet only)."""
    rd, act, out = fe["rms_db"], fe["active"], []
    for a, b in _merge_runs(_runs(fe["voiced"]), 4):
        if b - a < 18 or a < 8 or act[a - 8:a - 2].any():
            continue
        seg = rd[max(a - 4, 0):a + 20]
        pk = seg.max()
        lo, hi = np.where(seg >= pk - 10)[0], np.where(seg >= pk - 3)[0]
        if lo.size and hi.size and hi[0] >= lo[0]:
            out.append((hi[0] - lo[0]) / FPS * 1000.0)
    return float(np.median(out)) if len(out) >= 3 else None


# ---- sung voice, per SECTION of a song (music mode) --------------------------------------------------
# Sung voice is not speech: vibrato wobbles jitter, sustained notes flatter CPPS, and a Demucs stem carries bleed.
# So no speaker baseline, no jitter or "rough", wide limits, and only RELATIVE contrasts between sections of the
# same song, on the song's own median.
SUNG_F0 = (75.0, 1000.0)     # pyin range for singing
SUNG_CPPS_RANGE = (75, 700)  # CPPS peak search (Hz)
SUNG_MIN_S = 4.0             # voiced seconds a section needs before it is compared
SUNG_CPPS_GAP = 2.5          # dB under / over the song's median CPPS before a section is called different
SUNG_CORROB = 3.0            # dB a second measure (HNR, H1-H2, tilt) must also move
SUNG_BLEED_HNR = 6.0         # whole-song HNR under this (dB): the stem likely carries bleed, so read it as relative only


def sung_quality(y16, spans, bounds, notes=None):
    """Per-section voice quality of a vocal stem. spans = [(start_s, end_s)] where the voice is really present
    (absolute-level test, so a stem full of bleed in an instrumental never reaches here); bounds = section
    boundaries in seconds. Returns {'ok': False, ...} or {'ok': True, 'sections': [pack + start/end], 'whole': pack}."""
    if not spans or len(bounds) < 3:
        return {"ok": False, "reason": "no vocal spans or sections"}
    fe = _front_end(y16, fmin=SUNG_F0[0], fmax=SUNG_F0[1])
    present = np.zeros(fe["n"], bool)
    for a, b in spans:
        present[int(a * FPS):int(b * FPS) + 1] = True
    fe["voiced"] = fe["voiced"] & present
    fe["f0"] = np.where(fe["voiced"], fe["f0"], np.nan)
    q = analyze_quality(y16, fe, notes, edges=np.asarray(bounds, float), f0_range=SUNG_CPPS_RANGE)
    if not q.get("ok"):
        return q
    secs = [dict(t, start=bounds[i], end=bounds[i + 1]) for i, t in enumerate(q["thirds"])]
    return {"ok": True, "engine": q["engine"], "whole": q["whole"], "sections": secs}


def sung_contrast(sq):
    """Which sections are breathier / clearer than the song's own median. Returns (breathy, clear, median, bleed):
    two lists of (section, 'more pressed' flag), the median pack and whether the stem looks bleedy."""
    ok = [s for s in sq["sections"] if s["voiced_s"] >= SUNG_MIN_S and s.get("cpps_db") is not None]
    med = {m: float(np.median([s[m] for s in ok if s.get(m) is not None])) for m in ("cpps_db", "hnr_db", "h1h2_db", "alpha_db")
           if any(s.get(m) is not None for s in ok)} if ok else {}
    bleed = sq["whole"].get("hnr_db") is not None and sq["whole"]["hnr_db"] < SUNG_BLEED_HNR
    breathy, clear = [], []
    if len(ok) < 2:
        return breathy, clear, med, bleed
    for s in ok:
        d = {m: s[m] - med[m] for m in med if s.get(m) is not None}
        c = d.get("cpps_db", 0.0)
        h, hh, a = d.get("hnr_db"), d.get("h1h2_db"), d.get("alpha_db")
        if c <= -SUNG_CPPS_GAP and ((h is not None and h <= -SUNG_CORROB) or (hh is not None and hh >= SUNG_CORROB)
                                    or (a is not None and a >= SUNG_CORROB)):
            breathy.append((s, False))
        elif c >= SUNG_CPPS_GAP and ((h is not None and h >= SUNG_CORROB) or (hh is not None and hh <= -SUNG_CORROB)
                                     or (a is not None and a >= SUNG_CORROB)):
            clear.append((s, (hh is not None and hh <= -SUNG_CORROB) or (a is not None and a >= SUNG_CORROB)))
    return breathy, clear, med, bleed


def sung_quality_line(sq):
    """'VOCAL: breathier 0:00-0:58, 2:10-2:40; clearer and more pressed 1:20-1:58 [CPPS ...]'. None when there is
    nothing honest to say (no vocal, too few voiced seconds, a single section)."""
    if not sq or not sq.get("ok"):
        return None
    breathy, clear, med, bleed = sung_contrast(sq)
    if not med:
        return None
    ok = [s for s in sq["sections"] if s["voiced_s"] >= SUNG_MIN_S and s.get("cpps_db") is not None]
    if len(ok) < 2:
        return None
    rng = lambda ss: ", ".join(f"{_t(s['start'])}\u2013{_t(s['end'])}" for s, *_ in ss[:3]) + (f" +{len(ss) - 3}" if len(ss) > 3 else "")
    parts = []
    if breathy:
        parts.append("breathier " + rng(breathy))
    if clear:
        parts.append(("clearer and more pressed " if any(p for _, p in clear) else "clearer ") + rng(clear))
    head = "; ".join(parts) if parts else "no clear contrast in breathiness between sections"
    cp = [s["cpps_db"] for s in ok]
    nums = f"CPPS {min(cp):.1f}\u2013{max(cp):.1f}, HNR {med['hnr_db']:.0f} dB median" if "hnr_db" in med else f"CPPS {min(cp):.1f}\u2013{max(cp):.1f}"
    line = f"  {'VOCAL:':<10}{head} [{nums}]"
    if bleed:
        line += " (stem bleed likely, relative only)"
    return line if len(line) <= MAX_LINE else _fit("VOCAL", [head] + (["stem bleed likely, relative only"] if bleed else []) + [f"[{nums}]"])


def _qz(rows, k, x):
    """(z, usual) of x against the speaker's own history for key k, or None if there is not enough of it."""
    hist = np.array([r[k] for r in rows[-BASELINE_WINDOW:] if r.get(k) is not None], dtype=float)
    if x is None or len(hist) < 5:
        return None
    med = float(np.median(hist))
    scale = max(1.4826 * float(np.median(np.abs(hist - med))), QUALITY_SCALE[k])
    return (x - med) / scale, med


def _third_shift(q):
    """The one third that clearly differs from the other two, as words, or None."""
    th = q["thirds"]
    if any(t["voiced_s"] < Q_MIN_THIRD_S for t in th):
        return None
    at = ("at the start", "in the middle", "at the end")
    best = None
    for k in range(3):
        o = [t for j, t in enumerate(th) if j != k]
        d = {}
        for m in ("cpps_db", "hnr_db", "h1h2_db", "alpha_db"):
            if th[k][m] is not None and all(t[m] is not None for t in o):
                d[m] = th[k][m] - float(np.mean([t[m] for t in o]))
        c, h, s, a = (d.get(m) for m in ("cpps_db", "hnr_db", "h1h2_db", "alpha_db"))
        if c is not None and c <= -2.0 and ((h is not None and h <= -3.0) or (s is not None and s >= 3.0)):
            cand = (abs(c), "breathier " + at[k])
        elif c is not None and c >= 2.0 and ((h is not None and h >= 3.0) or (s is not None and s <= -3.0)):
            cand = (abs(c), "clearer " + at[k])
        elif s is not None and s <= -4.0 and a is not None and a >= 3.0:
            cand = (abs(s) / 2, "tenser " + at[k])
        else:
            continue
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1] if best else None


def quality_words(q, rec, rows):
    """Map the measures to words. Against the speaker's own history when it exists, else wide absolute limits."""
    w = q["whole"]
    z = {}
    if len(rows) >= BASELINE_MIN:
        for k in QUALITY_KEYS + ("loud_db",):
            r = _qz(rows, k, w.get(k) if k != "loud_db" else rec.get("loud_db"))
            if r:
                z[k] = r[0]
    basis = "baseline" if len(z) >= 3 else "none"
    g = lambda k: z.get(k, 0.0)
    words = []
    if basis == "baseline":
        cue = lambda *c: " (" + ", ".join(x for x in c if x) + ")"
        if g("cpps_db") <= -1.5 and (g("h1h2_db") >= 1.0 or g("hnr_db") <= -1.0):
            words.append(("breathy and soft" if g("loud_db") <= -1.5 else "breathy")
                         + cue("low CPPS", "high H1-H2" if g("h1h2_db") >= 1.0 else "low HNR"))
        elif g("jitter") >= 2.0 and (g("hnr_db") <= -1.0 or g("cpps_db") <= -1.0):
            words.append("rough" + cue("high jitter", "low HNR" if g("hnr_db") <= -1.0 else "low CPPS"))
        elif (g("h1h2_db") <= -1.5 and g("alpha_db") >= 1.0) or g("h1h2_db") <= -2.5:
            words.append("pressed/tense" + cue("low H1-H2", "bright tilt" if g("alpha_db") >= 1.0 else ""))
        elif g("cpps_db") >= 1.5 and g("hnr_db") >= 0:
            words.append("clearer than usual" + cue("high CPPS"))
        if g("onset_ms") >= 2.0:
            words.append("soft onsets")
        if g("alpha_db") <= -2.0 and g("loud_db") < 1.0:
            words.append("closer than usual (low-end lift)")
    else:
        c, h, j = w.get("cpps_db"), w.get("hnr_db"), w.get("jitter")
        if h is not None and h < 7.0:
            words.append("breathy or noisy (low HNR)")
        elif j is not None and h is not None and j > 0.025 and h < 12.0:
            words.append("rough (high jitter, low HNR)")
        elif c is not None and h is not None and c >= 6.0 and h >= 15.0:
            words.append("clear (high CPPS, high HNR)")
    return words, basis, z


def quality_line(q, rec=None, rows=None):
    if not q or not q.get("ok"):
        return None
    words, basis, _z = quality_words(q, rec or {}, rows or [])
    shift = _third_shift(q)
    head = ", ".join(words) if words else ("clear, nothing unusual" if basis == "baseline" else "nothing stands out")
    if shift:
        head += "; " + shift
    elif words and all(t["voiced_s"] >= Q_MIN_THIRD_S for t in q["thirds"]):
        head += ", steady"
    if basis != "baseline":
        head += " (no baseline yet)"
    w, nums = q["whole"], []
    if w.get("cpps_db") is not None:
        nums.append(f"CPPS {w['cpps_db']:.1f}" + ("~" if q["engine"].startswith("numpy") else ""))
    if w.get("hnr_db") is not None:
        nums.append(f"HNR {w['hnr_db']:.0f}")
    if w.get("h1h2_db") is not None:
        nums.append(f"H1-H2 {w['h1h2_db']:+.0f}")
    if w.get("alpha_db") is not None:
        nums.append(f"tilt {w['alpha_db']:+.0f}")
    nums = [", ".join(nums) + " dB"] if nums else []
    if w.get("jitter") is not None:
        nums.append(f"jitter {w['jitter'] * 100:.1f}%")
    line = f"  {'QUALITY:':<9}{head}"
    full = line + " [" + "; ".join(nums) + "]" if nums else line
    return full if len(full) <= MAX_LINE else _fit("QUALITY", [head])


# --------------------------------------------------------------------------- #
#  5. BASELINE   /   VS USUAL
# --------------------------------------------------------------------------- #
USUAL_MIN_VALUES = 5
USUAL_LIMIT = 2.0

# metric -> (display, unit, minimum scale, decimals)
METRICS = {
    "pitch_hz": ("pitch", " Hz", 8.0, 0),
    "range_st": ("range", " st", 1.5, 1),
    "voiced_frac": ("voiced", "", 0.08, 2),
    "loud_db": ("loudness", " dB", 2.0, 0),
    "dyn_db": ("dynamics", " dB", 3.0, 0),
    "arousal": ("arousal", "", 0.05, 2),
    "dominance": ("dominance", "", 0.05, 2),
    "valence": ("valence", "", 0.05, 2),
    "wpm": ("pace", " wpm", 15.0, 0),
    "wordless_s": ("wordless", "s", 1.0, 1),
    "silent_s": ("silence", "s", 1.0, 1),
    "laugh_s": ("laughter", "s", 0.5, 1),
    "breaths_pm": ("breaths", "/min", 3.0, 0),
}


def baseline_dir() -> Path:
    return Path(os.environ.get("RAVEN_EARS_HOME", str(Path.home() / ".local/share/raven-ears")))


def baseline_path(speaker: str) -> Path:
    """Speaker names are case-insensitive: the canonical history is lowercase (fox.jsonl). A lone legacy
    exact-case file (Fox.jsonl) is still read and appended to; with both, the lowercase one is used and
    baseline_conflict() lets the card say so. Nothing is merged or renamed."""
    import voice_book
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", speaker.strip())[:64] or "unknown"
    return voice_book.case_path(baseline_dir(), safe, ".jsonl")


def baseline_conflict(speaker: str) -> list:
    """Names of the history files when more than one letter case exists for this speaker, else []."""
    import voice_book
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", speaker.strip())[:64] or "unknown"
    found = voice_book.case_variants(baseline_dir(), safe, ".jsonl")
    return found if len(found) > 1 else []


def _record(d) -> dict:
    p, a, g, ev = d.get("pitch") or {}, d.get("affect") or {}, d.get("gaps") or {}, d.get("events") or {}
    w = (a.get("whole") or {}) if a.get("ok") else {}
    fe = d["_fe"]
    act = fe["rms_db"][fe["active"]]
    words = d.get("words")
    wpm = None
    if words and len(words) >= 2:
        span = max(words[-1][2] - words[0][1], 1e-6)
        wpm = len(words) / span * 60.0
    laugh = sum(e["end"] - e["start"] for e in (ev.get("events") or []) if e["label"] in LAUGH_LABELS) if ev else None
    br = (ev.get("breath") or {}) if ev else {}
    rec = {
        "pitch_hz": p.get("median_hz"), "range_st": p.get("range_st"), "voiced_frac": p.get("voiced_frac"),
        "loud_db": round(float(np.percentile(act, 90)), 1) if act.size else None,
        "dyn_db": round(float(np.percentile(act, 95) - np.percentile(act, 5)), 1) if act.size else None,
        "arousal": w.get("arousal"), "dominance": w.get("dominance"), "valence": w.get("valence"),
        "wpm": round(wpm, 1) if wpm else None,
        "wordless_s": g.get("wordless_s"), "silent_s": g.get("silent_s"),
        "laugh_s": round(laugh, 2) if laugh is not None else None,
        "breaths_pm": br.get("per_min"),
    }
    qw = ((d.get("quality") or {}).get("whole")) or {}
    for k in QUALITY_KEYS:
        rec[k] = round(float(qw[k]), 4) if qw.get(k) is not None else None
    return rec


def _read_baseline(path: Path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _usual_stats(rows, key, minscale):
    hist = np.array([r[key] for r in rows[-BASELINE_WINDOW:] if r.get(key) is not None], dtype=float)
    hist = hist[np.isfinite(hist)]
    if key == "pitch_hz":
        hist = hist[hist > 0]
    if len(hist) < USUAL_MIN_VALUES:
        return None
    med = float(np.median(hist))
    scale = max(1.4826 * float(np.median(np.abs(hist - med))), minscale)
    return med, scale, len(hist)


def _usual_pitch(rows):
    """The same comparison bounds as VS USUAL, around the median of note pitches."""
    if not rows:
        return {"reason": "no baseline"}
    if len(rows) < BASELINE_MIN:
        return {"reason": f"too few baseline notes ({len(rows)}/{BASELINE_MIN})"}
    stats = _usual_stats(rows, "pitch_hz", METRICS["pitch_hz"][2])
    if stats is None:
        return {"reason": f"fewer than {USUAL_MIN_VALUES} baseline notes with measured pitch"}
    med, scale, count = stats
    return {"low_hz": med - USUAL_LIMIT * scale, "high_hz": med + USUAL_LIMIT * scale, "notes": count}


def vs_usual(rec, rows):
    n = len(rows)
    if n < BASELINE_MIN:
        return f"  VS USUAL: still learning this voice ({n}/{BASELINE_MIN} notes so far)", []
    flagged = []
    for k, (disp, unit, minscale, dec) in METRICS.items():
        x = rec.get(k)
        if x is None:
            continue
        stats = _usual_stats(rows, k, minscale)
        if stats is None:
            continue
        med, scale, _ = stats
        z = (x - med) / scale
        if abs(z) >= USUAL_LIMIT:
            flagged.append((abs(z), k, disp, unit, dec, x, med, z))
    flagged.sort(reverse=True)
    items = []
    for _az, k, disp, unit, dec, x, med, z in flagged:
        arrow = "↑" if z > 0 else "↓"
        items.append(f"{disp} {arrow}{x:.{dec}f}{unit} (usual {med:.{dec}f})")
    head = f"VS USUAL (n={min(n, BASELINE_WINDOW)})"
    if not items:
        return f"  {head}: all within the usual range", []
    pieces = items[:5]
    line = f"  {head}: " + ", ".join(pieces)
    while len(line) > MAX_LINE and len(pieces) > 1:
        pieces = pieces[:-1]
        line = f"  {head}: " + ", ".join(pieces)
    return line, flagged


# --------------------------------------------------------------------------- #
#  orchestration
# --------------------------------------------------------------------------- #
def _separated_audio(path, samples, timings, scratch):
    import singer

    stem_dir, cached = singer.get_stems(path, timings, local_output=scratch)

    def load(name):
        y, _ = singer.load_mono(os.path.join(stem_dir, name + ".wav"), SR_SED)
        if abs(len(y) - samples) > 2 or not np.isfinite(y).all():
            raise ValueError("invalid or misaligned Demucs stem")
        return np.pad(y, (0, max(0, samples - len(y))))[:samples]

    vocals = load("vocals")
    if np.abs(vocals).max() < 1e-6:
        raise ValueError("Demucs vocals are silent")
    room = sum(load(name) for name in singer.SRC if name != "vocals")
    marker = Path(stem_dir) / "model.txt"
    model = marker.read_text().strip() if marker.is_file() else None
    return vocals, room, os.path.join(stem_dir, "vocals.wav"), dict(
        status="ran", model=model, cached=cached, voice="vocals", room="remainder")


@EP.speech_threads
@ST.engine
def analyze(audio_path, words=None, speaker=None, *, events=True, affect=True, use_whisper=True,
            update_baseline=True, keep_models=False, laughs=True, fast=False, words_skip_reason=None,
            identify_voice=True, capture_picture=False, capture_spectrum=False, laugh_model_default="off",
            separate=None, picture_read_only=True):
    """Run everything. Returns {"lines": [...], "data": {...}, "notes": [...], "timings": {...}}.

    words : optional [(word, start_s, end_s)] (or hear_core's wordTimestamps dicts, or a whole
            hear() result). If None and faster-whisper imports, it is run locally; otherwise the
            word-dependent parts (pace, WORDLESS) degrade and say so.
    fast : select small for speech (model env overrides win), skip affect, cap model threads at four.
    keep_models : in fast mode, retain Whisper and the CPU tagger for subsequent clips.
    capture_picture : return frame times/RMS/F0 and any usual pitch range for a local sheet,
                      outside the JSON data. Do not update the baseline unless picture_read_only=False.
    picture_read_only : the CLI passes False for every sheet except --by-ear, so drawing never freezes a baseline.
    capture_spectrum : include a spectrogram from the same decode when capturing a picture.
    separate : try local Demucs stems, or follow RAVEN_EARS_SEPARATE when None.
    """
    audio_path = local_audio_path(audio_path)
    affect = affect and os.environ.get("RAVEN_EARS_NO_AFFECT") != "1"
    update_baseline = update_baseline and os.environ.get("RAVEN_EARS_NO_BASELINE") != "1"
    if separate is None:
        separate = os.environ.get("RAVEN_EARS_SEPARATE", "").lower() in ("1", "true", "yes", "on")
    with ExitStack() as scratch:
        return _analyze(audio_path, words, speaker, events=events, affect=affect, use_whisper=use_whisper,
                        update_baseline=update_baseline and not (capture_picture and picture_read_only),
                        keep_models=keep_models, laughs=laughs, fast=fast,
                        words_skip_reason=words_skip_reason, identify_voice=identify_voice,
                        capture_picture=capture_picture, capture_spectrum=capture_spectrum,
                        laugh_model_default=laugh_model_default, separate=separate, scratch=scratch)


def _analyze(audio_path, words, speaker, *, events, affect, use_whisper, update_baseline,
             keep_models, laughs, fast, words_skip_reason, identify_voice, capture_picture,
             capture_spectrum=False, laugh_model_default="off", separate, scratch):
    from ears_handoff import finish_lines  # MORE EARS: listener handoff, including quiet cards.
    import jev_more
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    notes, clock = [], _Clock()
    affect_skip = ("--fast" if fast else "RAVEN_EARS_NO_AFFECT=1"
                   if os.environ.get("RAVEN_EARS_NO_AFFECT") == "1" else "--no-affect")
    if fast:
        affect = False
        selection = ("supplied word times" if words is not None else
                     "whisper " + EP.speech_model(True) if use_whisper else "Whisper disabled (--no-whisper)")
        message = selection + "; affect skipped; threads <=4"
        ST.mark("fast mode", "ran", message, kind="input")
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    _VRAM["peak_mb"] = 0.0
    with clock("decode"):
        y32 = load_audio(audio_path, SR_SED)
        from scipy.signal import resample_poly
        y16 = resample_poly(y32, 1, 2).astype(np.float32)
    dur = len(y16) / SR
    picture = {}
    if capture_picture and capture_spectrum:
        from listening_sheet import speech_spectrum
        picture = {"duration_s": dur, "spectrum": speech_spectrum(y16, SR)}
    silent = not len(y16) or float(np.abs(y16).max()) < 1e-6
    if dur < 0.5 or silent:
        reason = "audio is empty or digital silence" if silent else "too short to analyse (under 0.5 seconds)"
        ST.mark("decode")
        ST.mark("speech analysis", "skipped", reason)
        lines = [f"  EARS+:   ({reason})"] + jev_more.words_vs_voice_lines(None)
        result = {"lines": finish_lines(lines), "data": {"duration_s": dur},
                  "notes": notes, "timings": {}}
        if picture:
            result["picture_data"] = picture
        return result

    voice = None
    if not speaker and identify_voice:
        import voice_book
        voice = voice_book.identify(y16)
        if voice and voice['status'] == 'matched':
            speaker = voice['ranked'][0][0]
            update_baseline = False  # Guesses compare only; never teach a baseline.

    w = _parse_words(words)
    words_status = None
    if w is None and words is None and use_whisper:
        with clock("whisper"):
            if fast and keep_models:
                raw = _whisper_words(y16, notes, fast=True, keep_model=True)
            else:
                raw = _whisper_words(y16, notes, fast=True) if fast else _whisper_words(y16, notes)
        w = _parse_words(raw)
        if raw is None:
            why = [n[7:] for n in notes if n.startswith("words:") and ("failed" in n or "not importable" in n)]
            words_status = "failed: " + ("; ".join(why) or "whisper did not run")
        if not (fast and keep_models):
            release_models()
    elif w is None and not use_whisper:
        words_status = "off: " + (words_skip_reason or "--no-whisper, no word times given")
    elif w is None and words is not None:
        words_status = "off: the supplied word list was empty or unreadable"
    if w is not None:
        w, capped = _sanitize_words(w)
        if capped:
            notes.append(f"words: {capped} over-long word(s) capped so a stretched tail counts as a gap")
    else:
        notes.append("words: " + words_skip_reason if words_skip_reason else
                     "words: none available; pace and WORDLESS are skipped")

    voice_path, voice32, remainder, separation = audio_path, y32, None, None
    if separate:
        release_models()
        try:
            with clock("separation"):
                tmp = scratch.enter_context(tempfile.TemporaryDirectory(prefix="raven-speech-stems-"))
                vocals, room_audio, stem_path, provenance = _separated_audio(audio_path, len(y32), clock.t, tmp)
                y16 = resample_poly(vocals, 1, 2).astype(np.float32)
            voice_path, voice32, remainder, separation = stem_path, vocals, room_audio, provenance
            ST.mark("speech separation", source="Demucs")
        except Exception as e:
            why = _skip_why(e)
            separation = dict(status="failed", reason=why, voice="mix", room="mix")
            ST.mark("speech separation", "failed", why)

    with clock("front_end(pyin)"):
        fe = _front_end(y16)
    filler = []
    if w:
        try:
            w, filler = split_tail_filler(w, fe, dur)
        except Exception as e:
            notes.append(f"words: tail filler check skipped: {_skip_why(e)}")
    if filler:
        notes.insert(0, f"filler: '{' '.join(x[0] for x in filler)}' at {_t(filler[0][1])} looks like Whisper's end-of-clip filler, "
                        "not spoken; left out of PACE, EMPHASIS, sentiment")
        ST.mark("filler", "ran", f"'{' '.join(x[0] for x in filler)}' at {_t(filler[0][1])} looks like Whisper's end-of-clip "
                "filler, not spoken; left out of PACE, EMPHASIS, sentiment", kind="input")
    d = {"duration_s": round(dur, 2), "words": w, "_fe": fe, "filler_words": filler}
    if separation is not None:
        separation["words"] = ("supplied" if words is not None else "mix") if w else None
        d["separation"] = separation
    d["channel"] = None
    try:
        with clock("channel"):
            d["channel"] = analyze_channel(y32, _source_sample_rate(audio_path), fe)
        ST.mark("channel", "ran" if d["channel"]["rolloff_99_hz"] is not None else "skipped",
                "" if d["channel"]["rolloff_99_hz"] is not None else "no sounding-frame spectrum")
    except Exception as e:
        notes.append(f"CHANNEL skipped: {_skip_why(e)}")

    with clock("pitch"):
        d["pitch"] = analyze_pitch(fe, w)
    with clock("gaps"):
        d["gaps"] = analyze_gaps(fe, w, dur)
    try:
        d["contour"] = analyze_contour(fe, d["gaps"])
    except Exception as e:  # fail loud: a missing CONTOUR line must have its reason on the card
        d["contour"] = None
        notes.append(f"CONTOUR skipped: {type(e).__name__}: {str(e)[:100]}")

    d["events"] = None
    dog_passes = []
    room = None
    if events:
        try:
            with clock("events"):
                d["events"] = analyze_events(voice32, dur, notes, fe, w)
        except Exception as e:
            notes.append(f"EVENTS skipped: {_skip_why(e)}")
        mode = os.environ.get('RAVEN_EARS_ROOM_DOGS', 'raw')
        room = (d['events'] or {}).get('room')
        room32, room_fe = y32, fe
        if remainder is not None:
            room = None
            if os.environ.get('RAVEN_EARS_ROOM', '1') != '0':
                try:
                    with clock("room_remainder"):
                        room32 = remainder
                        room_fe = _front_end(resample_poly(room32, 1, 2).astype(np.float32))
                        centers, probs, labels, _ = _event_windows(room32, notes, room_fe, "ROOM")
                        room = RW.event_context(centers, probs, labels, dur, TAG_HOP,
                                                EVENT_LABELS['Whispering'][1])['room']
                except Exception as e:
                    notes.append(f"ROOM skipped: {_skip_why(e)}")
                    ST.mark("room", "failed", _skip_why(e))
            else:
                ST.mark("room", "skipped", "RAVEN_EARS_ROOM=0")
            if d["events"] is not None:
                d["events"]["room"] = room
        if (room is not None and os.environ.get('RAVEN_EARS_ROOM', '1') != '0'
                and mode in ('raw', 'voice-reduced')
                and any(label not in room['missing_labels'] for label in RW.DOG_LABELS)):
            for reduced in ((False, True) if mode == 'voice-reduced' else (False,)):
                source = 'voice-reduced' if reduced else 'raw'
                try:
                    with clock(f'room_dogs({source})'):
                        candidate = analyze_faint_dogs(room32, dur, room_fe, notes, reduced=reduced)
                    if candidate is not None:
                        dog_passes.append(candidate)
                except Exception as e:
                    notes.append(f"ROOM faint dogs ({source}) skipped: {_skip_why(e)}")
        if not fast:
            release_models()

    if remainder is None:
        RW.enrich(d, fe, w, notes, GAP_MIN, TAG_WIN, dur, dog_passes)
    else:
        RW.enrich(d, fe, w, notes, GAP_MIN, TAG_WIN, dur, dog_passes, room=room)

    d["laughs"] = None
    if laughs:
        try:
            import room_laugh
            with clock("laughs"):
                # Pass the instance explicitly: CLI execution is __main__, whereas
                # room_laugh imports raven_ears_plus as a separate module.
                kwargs = {"tagger": _load_tagger("cpu")} if fast else {}
                d["laughs"] = room_laugh.detect(voice_path, y=voice32, **kwargs)
                d["laughs"].pop("windows", None)
        except Exception as e:
            notes.append(f"LAUGHS skipped: {_skip_why(e)}")
        if not (fast and keep_models):
            release_models()

    d['laugh_model'] = None
    if laughs:
        import laugh_models
        try:
            laugh_model = laugh_models.selected(laugh_model_default)
            if laugh_model != 'off':
                release_models()  # The subprocess gets the machine's one heavy-model slot.
                with clock('laugh_model'):
                    d['laugh_model'] = laugh_models.run_detector(laugh_model, voice_path)
                result = d['laugh_model']
                ST.mark('laugh model', result['status'], result.get('reason', ''))
                if result['status'] != 'ran':
                    notes.append('laugh model skipped: ' + result['reason'])
        except ValueError as e:
            notes.append('laugh model skipped: ' + str(e))
            ST.mark('laugh model', 'failed', str(e))

    if d["events"] and d["laughs"]:
        # one story on the card: laughs the finer pass found that the 2 s tagger missed join EVENTS
        have = [(e["start"], e["end"]) for e in d["events"]["events"] if e["label"] in LAUGH_LABELS]
        for e in d["laughs"]["laughs"]:
            if not any(e["start"] < b and a < e["end"] for a, b in have):
                d["events"]["events"].append({"label": "Laughter", "name": "Laughter", "start": e["start"],
                                              "end": e["end"], "score": e["total"], "via": "room_laugh"})
        have = [(e["start"], e["end"]) for e in d["events"]["events"] if e["label"] == "Snort"]
        for e in d["laughs"]["snorts"]:
            if not any(e["start"] < b and a < e["end"] for a, b in have):
                d["events"]["events"].append({"label": "Snort", "name": "Snort", "start": e["start"],
                                              "end": e["end"], "score": e["total"], "via": "room_laugh"})
        d["events"]["events"].sort(key=lambda x: -x["score"])

    d["affect"] = None
    if affect:
        try:
            with clock("affect"):
                d["affect"] = analyze_affect(y16, fe, dur, notes)
        except Exception as e:
            notes.append(f"AFFECT skipped: {_skip_why(e)}")
        release_models()

    d["shape"] = analyze_shape(fe, w, d["pitch"])
    d["peak"] = analyze_peak(fe, d["affect"], dur)
    d["emphasis"] = analyze_emphasis(w, fe) if w else None
    d["laugh_hint"] = laugh_hint(w, emphasis_features(w, fe)) if w and len(w) >= 4 else []
    d["quality"] = None
    try:
        with clock("quality"):
            d["quality"] = analyze_quality(y16, fe, notes)
    except Exception as e:
        notes.append(f"QUALITY skipped: {_skip_why(e)}")
    qrows = _read_baseline(baseline_path(speaker)) if speaker and remainder is None else []
    if speaker and remainder is None and (clash := baseline_conflict(speaker)):
        notes.insert(0, f"BASELINE: {' and '.join(clash)} differ only by letter case; using {baseline_path(speaker).name}, not merged")
        ST.mark("baseline", "degraded", f"{' and '.join(clash)} differ only by letter case; using {baseline_path(speaker).name}, not merged")
    d["tune"] = analyze_tune(fe, w, d["pitch"], qrows)
    groups = list(_word_phrases(w))
    jev_speech = jev_more.read_speech([dict(t=a, end=b, text=" ".join(word[0] for word in group))
                                       for a, b, group in groups], notes)
    if jev_speech is not None:
        if all(row["jev"]["status"] == "ran" for row in jev_speech):
            for row, measurement in zip(jev_speech, _phrase_voice(d, fe, groups, qrows, dur)):
                row["voice"] = measurement
        d["jev_speech"] = jev_speech

    skipped = lambda key: next((n.split(": ", 1)[1] for n in notes if n.startswith(key)), "see notes")
    cl = channel_line(d["channel"])
    lines = ([cl] if cl else []) + [pitch_line(d["pitch"])]
    if remainder is not None:
        sources = [
            "  SEPARATE: Demucs vocals and remainder (drums + bass + other)",
            "  SOURCES: ROOM from remainder · CHANNEL and voice ID from mix",
            "  SOURCES: PITCH, TUNE, LAUGHS, QUALITY and other voice measures from vocals",
        ]
        if w:
            word_times = "supplied word times" if words is not None else "word times from mix"
            sources.append(f"  SOURCES: WORDS and PACE use {word_times} · WORDS annotations from vocals")
        lines[:0] = sources
    elif separation is not None:
        lines.insert(0, _skip_line("SEPARATE", "using original mix · " + separation["reason"]))
    if voice is not None:
        lines.append(voice_book.voice_line(voice))
        d['voice'] = voice
    if d["events"]:
        ev = d["events"]
        lines.append(events_line(ev))
        br = ev.get("breath")
        lines.append(breath_line(br) or _skip_line('BREATH', (
            (br or {}).get("skipped") or "no breath analysis ran (no Breathing label in the tagger)")))
    else:
        why = skipped("EVENTS skipped") if events else "--no-events"
        lines += [_skip_line('EVENTS', why), _skip_line('BREATH', f"events were not analysed ({why})" if not events else why)]
    lines += [line for line in (RW.room_line(d['room']), RW.whisper_line(d['whisper'])) if line]
    lines.append(_skip_line('PACE', words_skip_reason) if words_skip_reason and not w else pace_line(w))
    model_ran = (d.get('laugh_model') or {}).get('status') == 'ran'
    lines.append(laughs_line(d["laughs"], d["laugh_hint"], d.get('laugh_model')) if d["laughs"] or model_ran else _skip_line('LAUGHS', (
        skipped("LAUGHS skipped") if laughs else "--no-laughs")))
    if d["affect"] and d["affect"].get("ok"):
        lines.append(affect_line(d["affect"]))
    else:
        lines.append(_skip_line('AFFECT', skipped("AFFECT skipped") if affect else affect_skip))
    lines.append(shape_line(d["shape"]) or "  SHAPE:   none (" + (d["shape"].get("reason") or "no active sound") + ")")
    ql = quality_line(d["quality"], _record(d), qrows) if d["quality"] else None
    lines.append(ql or _skip_line('QUALITY', (d["quality"] or {}).get("reason") or skipped("QUALITY skipped")))
    lines += ([_skip_line('WORDLESS / emphasis', words_skip_reason)] if words_skip_reason and not w
              else gap_lines(d["gaps"]))
    lines += contour_lines(d.get("contour"))
    lines.append(peak_line(d["peak"]) or "  PEAK:    none (" + (d["peak"].get("reason") or "no peak found") + ")")
    if w:
        lines.append(emphasis_line(d["emphasis"]))
    lines += words_lines(w, dur, d["peak"], d["events"], d["gaps"], words_status, d["emphasis"], d['whisper'])
    lines += filler_line(filler)
    lines += tune_lines(d["tune"], w)
    lines += jev_more.words_vs_voice_lines(jev_speech, _t)
    d["words_status"] = words_status

    if speaker and remainder is not None:
        lines.append("  VS USUAL: skipped (separated voice uses a different input from the baseline)")
        ST.mark("VS USUAL", "skipped", "separated voice uses a different input from the baseline")
    elif speaker:
        path = baseline_path(speaker)
        rows = qrows
        rec = _record(d)
        vline, flagged = vs_usual(rec, rows)
        if voice is not None:
            vline = vline.replace('VS USUAL', 'VS USUAL (speaker from the voice book)', 1)
            vline += '; comparison only, guess never saved'
        lines.append(vline)
        d["vs_usual"] = [(f[1], f[5], f[6]) for f in flagged]
        if update_baseline:
            path.parent.mkdir(parents=True, exist_ok=True)
            rec_out = dict(rec, ts=int(time.time()), dur_s=round(dur, 1))
            with open(path, "a") as fh:
                fh.write(json.dumps(rec_out) + "\n")
        d["baseline_n_before"] = len(rows)
    else:
        lines.append("  VS USUAL: no speaker given, no comparison")
    for line in jev_more.speech_lines(jev_speech, speaker, voice):
        if line.startswith("  JOKING?"):
            at = max((i + 1 for i, text in enumerate(lines) if text.startswith(("  LAUGHS:", "  JOKING?"))), default=len(lines))
        elif line.startswith("  WHO"):
            at = next((i + 1 for i, text in enumerate(lines) if text.startswith("  VOICE:")), 2)
        else:
            at = len(lines)
        lines.insert(at, line)

    if not keep_models:
        release_models()
    _note_vram()
    timings = dict(clock.t)
    timings["vram_peak_mb(torch)"] = round(_VRAM["peak_mb"])
    picture_data = d.pop("_fe")
    d["pitch_ends"] = d["pitch"].get("phrase_ends")
    if d.get("affect"):
        d["affect"] = {k: v for k, v in d["affect"].items() if k not in ("centers", "P")}
    ST.collect(ST.speech_rows(d, [n for n in notes if not n.startswith("Jev")],
                              transcribed=words is None and use_whisper))
    if d["laughs"] is not None:
        ST.mark("laughs")
    if words is not None:
        ST.mark("Whisper", "skipped", "word times supplied", kind="input")
    elif not use_whisper:
        ST.mark("Whisper", "skipped", words_skip_reason or "--no-whisper", kind="input")
    for name, enabled, flag in (("events", events, "--no-events"), ("affect", affect, affect_skip),
                                ("laughs", laughs, "--no-laughs")):
        if not enabled:
            ST.mark(name, "skipped", flag)
    if words_status:
        ST.mark("words", "failed" if words_status.startswith("failed") else "skipped",
                words_status.partition(": ")[2].replace("whisper", "Whisper"), kind="input")
    if not w:
        ST.mark("pace / WORDLESS / emphasis", "skipped", words_skip_reason or "no word times")
    if d["quality"] and not d["quality"].get("ok"):
        ST.mark("quality", "skipped", d["quality"].get("reason", "no usable measurements"))
    if not speaker:
        ST.mark("VS USUAL", "skipped", "no speaker given")
    nl = notes_line(notes)
    if nl:
        lines.append(nl)
    d["listener"] = "and now, you"
    result = {"lines": finish_lines(lines), "data": d, "notes": notes, "timings": timings}
    if capture_picture:
        picture.update({k: picture_data[k] for k in ("times", "rms_db", "f0")})
        if speaker:
            picture["usual_pitch"] = (_usual_pitch(qrows) if remainder is None else
                                      {"reason": "separated voice uses a different input from the baseline"})
        result["picture_data"] = picture
    return result


# --------------------------------------------------------------------------- #
#  setup + CLI
# --------------------------------------------------------------------------- #
def _download(url, dest: Path, min_bytes=0, validate=None):
    require_online("Model download")
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest.is_file() and dest.stat().st_size >= min_bytes and dest.stat().st_size > 0
            and (validate is None or validate(dest))):
        print(f"ok   {dest}")
        return
    print(f"get  {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if (tmp.stat().st_size < min_bytes or tmp.stat().st_size == 0
                or (validate is not None and not validate(tmp))):
            raise ValueError(f"download incomplete or invalid: {dest.name}")
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _setup_labels_valid(path):
    """Validate the 527-class CSV that panns_inference.config reads on import."""
    import csv
    try:
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        return (rows[0] == ["index", "mid", "display_name"] and len(rows) == 528
                and all(len(row) == 3 and row[0] == str(i) and row[1] and row[2]
                        for i, row in enumerate(rows[1:])))
    except (OSError, UnicodeError, csv.Error, IndexError):
        return False


def _setup_checkpoint(check):
    if not check:
        _download(PANNS_CKPT_URL, PANNS_CKPT, min_bytes=int(3e8))
    if not PANNS_CKPT.is_file() or PANNS_CKPT.stat().st_size < 3e8:
        raise FileNotFoundError("PANNs checkpoint missing or incomplete")


def _setup_labels(check):
    upstream = Path.home() / "panns_data" / "class_labels_indices.csv"
    if not check:
        _download(PANNS_LABELS_URL, PANNS_LABELS, validate=_setup_labels_valid)
        if not _setup_labels_valid(upstream):
            upstream.parent.mkdir(parents=True, exist_ok=True)
            tmp = upstream.with_suffix(".csv.part")
            shutil.copyfile(PANNS_LABELS, tmp)
            tmp.replace(upstream)
    for path in (PANNS_LABELS, upstream):
        if not _setup_labels_valid(path):
            raise FileNotFoundError(f"PANNs labels missing or invalid: {path}")


def _setup_whisper(size, check):
    # Use the same resolver/cache as WhisperModel, without loading a model or GPU.
    from faster_whisper.utils import download_model
    root = Path(size) if Path(size).is_dir() else Path(download_model(size, local_files_only=check))
    required = ["config.json", "model.bin", "tokenizer.json"]
    missing = [name for name in required if not (root / name).is_file() or not (root / name).stat().st_size]
    if not any(p.is_file() and p.stat().st_size for p in root.glob("vocabulary.*")):
        missing.append("vocabulary.*")
    # Missing tokenizer.json lets upstream fetch a tokenizer even with local weights.
    if missing:
        raise FileNotFoundError("Whisper files missing or empty: " + ", ".join(missing))


def _setup_affect(check):
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(AFFECT_MODEL, local_files_only=check))
    config = root / "config.json"
    weights = [root / name for name in ("model.safetensors", "pytorch_model.bin")]
    if not config.is_file() or not config.stat().st_size or not any(
            p.is_file() and p.stat().st_size for p in weights):
        raise FileNotFoundError("AFFECT config or weights missing or empty")


def setup(target="full", *, check=False, affect=True):
    """Prepare selected local files, or inspect them without downloading or inference.

    Full retains the old PANNs/AFFECT set and adds the effective speech Whisper.
    Music also prepares large-v3 lyrics, Beat This! and RoBERTa sentiment.
    """
    if target not in ("fast", "speech", "music", "full"):
        raise ValueError(f"unknown setup target: {target}")
    if not check:
        require_online("Setup")
    size = "large-v3" if target == "music" else EP.speech_model(target == "fast")
    source = None if target == "music" else EP.speech_model_override()
    whisper_label = f"Whisper {size}" + (f" (from {source})" if source else "")
    tasks = [("PANNs checkpoint", _setup_checkpoint), ("PANNs labels (cache + upstream)", _setup_labels),
             (whisper_label, lambda local: _setup_whisper(size, local))]
    if target != "fast" and affect and os.environ.get("RAVEN_EARS_NO_AFFECT") != "1":
        tasks.append(("AFFECT", _setup_affect))
    if target == "music":
        from harmony import setup_sentiment
        from singer import setup_beat_checkpoint
        tasks.append(("Beat This! checkpoint (~81 MB)", setup_beat_checkpoint))
        tasks.append(("RoBERTa lyric sentiment", setup_sentiment))
    print(f"setup {target}: {'check only' if check else 'prepare files'}", flush=True)
    ready = True
    for name, action in tasks:
        try:
            if not check:
                size_note = (" (~330 MB)" if name == "PANNs checkpoint" else
                             " (~1.3 GB)" if name == "AFFECT" else "")
                print(f"prepare {name}{size_note}; downloads can take several minutes", flush=True)
            action(check)
            print(f"ok   {name}", flush=True)
        except Exception as e:
            ready = False
            print(f"{'missing' if check else 'failed'} {name}: {type(e).__name__}: {e}")
    if ready:
        print("setup files ready (inference not tested)")
    else:
        print(f"run --setup {target} while online to prepare missing files; check the errors above")
    if target == "music":
        print("music: Demucs and optional key models are not covered by this check")
    return ready


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def serve(paths, *, json_output=False, **options):
    """Local stdin paths -> one JSON response per line; EOF releases resident models.

    Load lazily on the first clip. Keep speech/PANNs weights, never clips or cards.
    A failed request gets an error response and does not stop the next request.
    """
    from contextlib import redirect_stdout
    options.update(fast=True, keep_models=True)
    try:
        for path in paths:
            path = path.rstrip('\r\n')
            if not path:
                continue
            try:
                # Third-party model loaders can print; stdout is the JSONL protocol.
                with redirect_stdout(sys.stderr):
                    result = analyze(path, **options)
                from ears_handoff import finish_json
                response = {"ok": True, "data" if json_output else "lines":
                            finish_json(result["data"]) if json_output else result["lines"]}
            except Exception as e:
                release_models()
                response = {"ok": False, "error": _skip_why(e)}
            print(json.dumps(response, default=_json_default), flush=True)
            del response
            result = None
    finally:
        release_models()
    return 0


@ST.engine
@ST.cli
def main(argv=None):
    ap = argparse.ArgumentParser(description="Extra ears: PITCH / EVENTS / AFFECT / SHAPE / WORDLESS / PEAK / BREATH / VS USUAL")
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--speaker", help="enables the per-speaker baseline and VS USUAL")
    book = ap.add_mutually_exclusive_group()
    book.add_argument('--enrol', nargs='+', metavar='NAME_OR_CLIP', help='replace a local voiceprint: --enrol NAME clip [clip…] (10+ retained seconds)')
    book.add_argument('--voices', action='store_true', help='list local voiceprints')
    book.add_argument('--forget', metavar='NAME', help='delete a voiceprint (leaves its separate baseline)')
    ap.add_argument("--words", help="JSON: [[word,start,end],...] | [{word,start,end}] | hear_core wordTimestamps | a hear() result")
    ap.add_argument("--no-whisper", action="store_true", help="never run faster-whisper for word times")
    ap.add_argument("--names", metavar="FILE", help="names/rare words for Whisper, one per line (default RAVEN_EARS_HOME/names.txt)")
    ap.add_argument("--verbatim", action="store_true", help="speech Whisper keeps fillers and stutters and stays silent on silence (also RAVEN_EARS_VERBATIM=1)")
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--no-affect", action="store_true")
    ap.add_argument("--fast", action="store_true", help="live speech: Whisper small, skip affect, cap model threads at 4 (model env overrides still win)")
    ap.add_argument("--serve", action="store_true", help="resident --fast listener: stdin paths, stdout JSONL responses; EOF exits")
    ap.add_argument("--no-laughs", action="store_true", help="skip the finer LAUGHS pass (room_laugh)")
    ap.add_argument("--separate", action="store_true", default=None,
                    help="try local Demucs stems (also RAVEN_EARS_SEPARATE=1)")
    ap.add_argument("--no-baseline-update", action="store_true", help="compare against the baseline (VS USUAL still prints) but do not append this note")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--json", action="store_true", help="print the full data dict as JSON instead of lines")
    import listening_sheet
    listening_sheet.add_picture_args(ap, help="draw a local listening sheet without updating the baseline")
    ap.add_argument("--picture-out", metavar="PATH", help="PNG destination (implies --picture; default beside input)")
    ap.add_argument("--by-ear", metavar="FILE", help="picture-only m:ss labels marked by a human (implies --picture)")
    ap.add_argument("--quiet", action="store_true", help="no notes/timings on stderr")
    ap.add_argument("--setup", nargs="?", const="full", choices=("fast", "speech", "music", "full"),
                    help="prepare selected weights, then exit (default full; fast skips AFFECT)")
    ap.add_argument("--check", action="store_true", help="with --setup: report local files without downloading; exit 1 if missing")
    a = ap.parse_args(argv)
    for audio in ([a.audio] if a.audio else []) + (a.enrol[1:] if a.enrol else []):
        local_audio_path(audio, ap)
    try:
        picture, picture_hint = listening_sheet.prepare_picture(
            a, text_card=not (a.json or a.serve or a.setup or a.check or a.enrol or a.voices or a.forget is not None),
            error="--picture needs a single audio text card; omit JSON, serve and management controls")
    except ValueError as error:
        ap.error(str(error))
    if a.enrol or a.voices or a.forget is not None:
        if a.audio or a.serve or a.setup or a.check or a.speaker:
            ap.error('voice-book management cannot be combined with audio, --serve, --setup, --check or --speaker')
        import voice_book
        try:
            if a.enrol:
                if len(a.enrol) < 2:
                    ap.error('--enrol requires NAME and at least one audio clip')
                row = voice_book.enrol(a.enrol[0], a.enrol[1:], load_audio,
                                       warn=lambda message: print(message, file=sys.stderr))
                print(f'Enrolled {a.enrol[0]}: {row["count"]} clips, {row["seconds"]:.1f}s retained voice; a resemblance, not a lock')
            elif a.voices:
                rows = voice_book.voices()
                for name, row in rows.items():
                    print(f'{name}: {row["count"]} clips, {row["seconds"]:.1f}s retained voice')
                if not rows:
                    print('No voices enrolled.')
            else:
                voice_book.forget(a.forget)
                print(f'Forgot voiceprint {a.forget}; any separate baseline is unchanged.')
        except Exception as error:
            ap.error(f'voice book: {error}')
        return 0
    if a.serve and (a.audio or a.setup):
        ap.error("--serve reads paths from stdin; do not combine with audio or --setup")
    if a.check and not a.setup:
        ap.error("--check requires --setup")
    if a.setup:
        return 0 if setup("fast" if a.fast and a.setup != "music" else a.setup, check=a.check, affect=not a.no_affect) else 1
    if not a.audio and not a.serve:
        ap.error("audio path required")
    if a.cpu:
        os.environ["RAVEN_EARS_DEVICE"] = "cpu"
    if a.verbatim:
        os.environ["RAVEN_EARS_VERBATIM"] = "1"
    words = None
    if a.names:
        import lyrics_shared
        lyrics_shared.configure(a.names)
    if a.words:
        words = json.load(open(a.words))
    options = dict(words=words, speaker=a.speaker, events=not a.no_events, affect=not a.no_affect,
                   use_whisper=not a.no_whisper, update_baseline=not a.no_baseline_update, laughs=not a.no_laughs,
                   separate=a.separate)
    if a.serve:
        return serve(sys.stdin, json_output=a.json, **options)
    if picture:
        options["capture_spectrum"] = True
        options["capture_picture"] = True
        # Drawing never decides what the baseline learns; only --by-ear (a human re-marking an old note) is read-only.
        options["picture_read_only"] = bool(a.by_ear)
    r = analyze(a.audio, fast=a.fast, **options)
    if a.json:
        from ears_handoff import finish_json  # MORE EARS: handoff inside valid JSON.
        print(json.dumps(finish_json(r["data"]), default=_json_default, indent=1))
    else:
        print("\n".join(r["lines"]))
        if picture:
            import listening_sheet
            output = Path(a.picture_out).expanduser() if a.picture_out else Path(a.audio).with_suffix(".listening-sheet.png")
            print(listening_sheet.attempt(listening_sheet.speech_sheet,
                  r["data"], r.get("picture_data", {}), output, title=Path(a.audio).name, by_ear=a.by_ear))
        elif picture_hint:
            print(picture_hint)
    if not a.quiet:
        for n in r["notes"]:
            _log(f"[note] {n}")
        _log("[timings] " + ", ".join(f"{k} {v}" for k, v in r["timings"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
