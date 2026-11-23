"""Model-free measurements of decoded mono audio; bass excess is not proof of editing."""
import numpy as np


# 2026-11-11: 43 shelf MP3s, ears mono decoder at 32 kHz. The labelled boosted
# copy has bass share .93206; all other 42 are <= .67497 (same-song normal .62368).
# .80 sits inside that gap. Clipping, crest, RMS and tilt overlap: no rules for
# those. This small shelf does not establish accuracy on other music/masters.
BASS_SHARE_WARNING = .80


def measure(y, sr):
    """Full-file mono statistics, without gain normalisation or silence gating.

    Near rail: abs(sample) >= .99; full scale: abs(sample) >= 1, including
    floating-point overshoots. Runs are same-sign near-rail sequences >=3 samples.
    Bass share: Welch power at 0 < f < 150 Hz / all power above DC.
    Tilt: least-squares slope of six octave bands' mean power density in dB,
    150--300 through 4800--9600 Hz, against log2(band centre), in dB/octave.
    Welch: periodic Hann, one-second windows, 50% overlap, constant detrend;
    the final incomplete window is discarded. Silent level/ratios are null.
    """
    x = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected nonempty finite mono audio')
    if not np.isfinite(sr) or int(sr) != sr or sr < 19200:
        raise ValueError('sample rate must be an integer >= 19200 Hz for tilt')
    if len(x) < sr:
        raise ValueError('at least one second of audio is required')
    from scipy.signal import welch
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    near = np.abs(x) >= .99
    # Split runs when the sign changes, even if both samples exceed the rail.
    rails = np.where(near, np.sign(x), 0)
    boundaries = np.flatnonzero(np.diff(np.r_[0, rails, 0]))
    starts, ends = boundaries[:-1], boundaries[1:]
    runs = (rails[starts] != 0) & (ends - starts >= 3)
    run_count = int(np.count_nonzero(runs))
    frequencies, power = welch(x, fs=sr, window='hann', nperseg=sr,
                               noverlap=sr // 2, detrend='constant', scaling='density')
    total = float(power[frequencies > 0].sum())
    bands = np.array([power[(frequencies >= low) & (frequencies < low * 2)].mean()
                      for low in (150, 300, 600, 1200, 2400, 4800)])
    tilt = (float(np.polyfit(np.arange(6), 10 * np.log10(bands), 1)[0])
            if np.all(bands > 0) else None)
    return dict(duration_s=len(x) / sr, peak=peak,
                full_scale_pct=100 * float(np.mean(np.abs(x) >= 1)),
                near_scale_pct=100 * float(near.mean()), near_runs=run_count,
                near_runs_per_s=run_count * sr / len(x),
                rms_dbfs=float(20 * np.log10(rms)) if rms else None,
                crest_db=float(20 * np.log10(peak / rms)) if rms else None,
                bass_share=float(power[(frequencies > 0) & (frequencies < 150)].sum() / total)
                           if total else None,
                tilt_db_per_oct=tilt)


def inspect(path):
    """Called once on a new listen, never while re-reading a saved bundle."""
    from raven_ears_plus import SR_SED, load_audio
    try:
        return dict(status='ran', sample_rate=SR_SED, **measure(load_audio(str(path), SR_SED), SR_SED))
    except Exception as exc:
        # Keep the song listen usable; unavailable is not evidence of clean audio.
        return dict(status='failed', reason=type(exc).__name__)


def warning(result):
    if not isinstance(result, dict) or result.get('status') != 'ran' or result.get('sample_rate') != 32000:
        return None
    share = result.get('bass_share')
    if (not isinstance(share, (int, float)) or isinstance(share, bool)
            or not np.isfinite(share) or not BASS_SHARE_WARNING <= share <= 1):
        return None
    # Calibrated on one boosted copy: some genres are genuinely this bass-heavy, so
    # the note states the measurement and offers the fix, never calls a file doctored.
    return (f"very bass-heavy: {share:.1%} of energy below 150 Hz (some genres are); "
            "if this is a bass-boosted copy, a cleaner one may read better")


def annotate(text, result):
    """Add one measurement note, preserving every existing card line verbatim."""
    note = warning(result)
    if not note:
        return text
    from stage_status import wrap_note
    # NOTE uses the existing continuation/wrapping contract and survives slim views.
    lines = '\n'.join(wrap_note('NOTE: ' + note)) + '\n'
    invitation = 'and now, you\n'
    if text.endswith(invitation):
        return text[:-len(invitation)] + lines + invitation
    return text + ('' if not text or text.endswith('\n') else '\n') + lines

