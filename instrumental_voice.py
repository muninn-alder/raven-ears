"""Presentation of vocal-stem measurements, including LRCLIB instrumentals.

No detector decision changes here. The saved measurements remain evidence;
the database designation only changes their description.
"""
import re


MELODY = "VOCAL-STEM MELODY: likely a lead instrument per LRCLIB (not verified by ear)"
AROUSAL = ("AROUSAL: not shown: speech-trained model on a likely lead instrument "
           "per LRCLIB; numbers retained in JSON")
WORDS = ("WORDS: transcription skipped: LRCLIB lists this as instrumental; "
         "stem activity is shown separately")
ATTRIBUTION = ("LRCLIB lists this as instrumental; likely a lead instrument in the "
               "vocal stem (not verified by ear)")


def window_pitch_text(f0):
    """Describe separated registers without turning their pooled median into a note.

    A >octave empty interval in log pitch, with >=25% of voiced frames on
    each side, is evidence of two registers, not proof of simultaneous voices.
    """
    import numpy as np
    def note(hz):
        midi = int(round(69 + 12 * np.log2(hz / 440.0)))
        names = ("C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B")
        return f"{names[midi % 12]}{midi // 12 - 1}"
    values = np.asarray(f0, dtype=float)
    values = np.sort(values[np.isfinite(values) & (values > 0)])
    if not len(values):
        return ""
    splits = [i for i in range(1, len(values))
              if min(i, len(values) - i) / len(values) >= 0.25
              and 12 * np.log2(values[i] / values[i - 1]) > 12]
    if splits:
        cut = max(splits, key=lambda i: values[i] / values[i - 1])
        low, high = note(np.median(values[:cut])), note(np.median(values[cut:]))
        return f"pitch there splits across octaves: ~{low} and ~{high} (layered vocals or two voices?)"
    return f"pitch there ~{note(np.median(values))}"


def saved_peak_pitch(text, data):
    """Refresh a saved peak-window note from captured voiced F0; no models."""
    if not isinstance(data, dict) or not isinstance(data.get("singer"), dict):
        return text
    peak = (data["singer"].get("data") or {}).get("peak") or {}
    inputs = data.get("melody_input") or {}
    if not peak.get("ok") or not inputs.get("f0") or not inputs.get("fps"):
        return text
    import numpy as np
    f0 = np.asarray(inputs["f0"], dtype=float)
    times = np.arange(len(f0)) / inputs["fps"]
    values = f0[(times >= peak["start"]) & (times <= peak["end"]) & np.isfinite(f0) & (f0 > 0)]
    if len(values) <= 10:
        return text
    return re.sub(r"(?m)^(\s*most intense moment:.*?)(pitch there ~[A-G][♭♯]?\d+)",
                  lambda m: m[1] + window_pitch_text(values), text)


def designated(data):
    """Only an explicit LRCLIB lyric resolution qualifies, never a low score."""
    if not isinstance(data, dict):
        return False
    lyrics = data.get("lyrics")
    source = data.get("lyrics_source")
    if not isinstance(source, dict) and isinstance(lyrics, dict):
        source = lyrics.get("lookup")
    return (isinstance(source, dict) and source.get("source") == "LRCLIB"
            and source.get("status") == "instrumental")


def saved_designation(cards, statuses):
    return any(designated(card.get("json")) for name, card in cards.items()
               if statuses.get(name, {}).get("status", "ran") == "ran")


def text_designation(text):
    """Text-only readers may use the explicit, attributed lyric label."""
    return bool(re.search(r"(?m)^\s*(?:NOTE:\s*)?LYRICS: instrumental \(LRCLIB\)", text))


def stem_text(text, instrumental):
    """Refresh fresh or saved prose without touching JSON or joining activity spans."""
    if not instrumental:
        return text
    stamp = r"\d+:\d\d"
    description = (r"voice present, words (?:unavailable \(transcription did not run or failed\)|"
                   r"not heard \(Whisper found none\))|no voice \(instrumental\)|"
                   r"words unavailable \(transcription did not run or failed\); voice presence unverified")
    claim = rf"{stamp}(?:–{stamp})?: (?:{description})"
    gap = re.compile(rf"\s*(?:NOTE:\s*|{stamp}\s+)?{claim}(?:; {claim})*\s*")
    out = []
    gap_summary = False
    arousal_summary = False
    in_stops = False
    skip_sections = False
    in_players = False
    for line in text.splitlines(keepends=True):
        body = line.strip()
        newline = "\n" if line.endswith("\n") else ""
        indent = line[:len(line) - len(line.lstrip())]
        if gap.fullmatch(line) or body.startswith(("voice present, words not heard", "voice with no words:")):
            if not gap_summary:
                out.append(indent + WORDS + newline)
                gap_summary = True
            continue
        if body == WORDS:
            gap_summary = True
        if body.startswith("arousal across the song:") or body.startswith("overall: arousal "):
            if not arousal_summary:
                out.append(indent + AROUSAL + newline)
                arousal_summary = True
            skip_sections = body.startswith("arousal across the song:")
            continue
        if skip_sections and body.startswith("by section:"):
            skip_sections = False
            continue
        skip_sections = False
        if body.startswith("AROUSAL:"):
            arousal_summary = True
        if body.startswith("SINGER:"):
            line = MELODY
            if body != "SINGER:":
                line += "\n  " + body.removeprefix("SINGER:").strip().replace("instrumental", "vocal stem")
            if not arousal_summary:
                line += "\n  " + AROUSAL
                arousal_summary = True
            line += newline
        elif body.startswith("VOCALS:"):
            line = line.replace("yes, voice present", "vocal stem active")
            line = line.replace("probably yes", "tagger vocal-like activity")
            line = line.replace("probably instrumental", "tagger vocal-like activity")
            line = line.replace("above voice level", "above the stem activity threshold")
            line = line.replace("instrumental (the vocal stem holds", "vocal stem (holds")
            if ATTRIBUTION not in line:
                line = line.rstrip("\n") + " · " + ATTRIBUTION + newline
        elif body.startswith("VOCAL:") and "vocal-stem texture" not in body:
            line = re.sub(r"(VOCAL:\s*)", r"\1vocal-stem texture · ", line, count=1)
        elif body.startswith(("pitch:", "pitch arc", "most intense moment:", "breathing:", "sounds the voice makes:")):
            line = line.replace("comfortable range", "central range")
            line = line.replace("the voice lifts", "the melody lifts").replace("the voice's median", "the vocal stem's median")
            line = re.sub(r" · arousal .*?\(speech-trained model; read as texture, not a verdict\)", "", line)
            line = line.replace("sounds the voice makes:", "vocal-stem event model (speech-trained; instrument sounds may be mislabelled):")
            line = line.replace("breathing:", "vocal-stem breath proxy (speech-trained; may be instrument sound):")
        if body.startswith("who is playing"):
            in_players = True
        elif body and not line[0].isspace():
            in_players = False
        if in_players:
            line = line.replace("+ voice", "+ vocal stem")
        # STOPS is also a stem-level measurement. Keep every diagnostic and number.
        if body.startswith("STOPS:"):
            in_stops = True
        elif body and not line[0].isspace():
            in_stops = False
        if in_stops:
            line = re.sub(r"\bvoice\b", "vocal stem", line)
            line = line.replace("singing", "stem activity")
        out.append(line)
    return "".join(out)
