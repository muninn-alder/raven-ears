"""Offline ACHE checks, called by the repository's self-test (also runnable alone)."""
import contextlib
import copy
import io
import json
from pathlib import Path
from unittest import mock

import numpy as np

import ache as A


def run_checks(check, tmp):
    def event(root, q, t=10, end=14, conf=0.8):
        return dict(root=root, q=q, t0=t, t1=end, conf=conf)

    def reading(*events, **kw):
        return dict(key_verdict={"key": "C major", "working": "C major", "state": "clear"}, evs=list(events), **kw)

    def text(r):
        return "\n".join(A.block(r))

    def kinds(r):
        return {e["kind"] for g in A.analyze(r)["moments"] for e in g["evidence"]}

    fixture = json.loads((Path(__file__).parent / "fixtures" / "ache-analysis.json").read_text())
    out = text(fixture)
    check("ache: fixture combines borrowed iv, three-second hush and placed lyric", 'borrowed iv (Fm in C major)' in out and 'after 3.0 s near-silence under ≈ "count bins" (lean)' in out, out)
    check("ache: fixture also finds deceptive V→vi", "deceptive V→vi (G7→Am)" in out)
    check("ache: analysis and selection are deterministic and do not mutate input", A.analyze(fixture) == A.analyze(copy.deepcopy(fixture)) and fixture == json.loads((Path(__file__).parent / "fixtures" / "ache-analysis.json").read_text()))

    for root, q, expected in [(5, "min", "iv"), (8, "maj", "bVI"), (10, "maj", "bVII")]:
        check("ache: modal mixture " + expected, "borrowed " + expected in text(reading(event(root, q))))
    minor = reading(event(5, "min"))
    minor["key_verdict"] = {"key": "C minor", "state": "clear"}
    check("ache: minor iv in a minor key is ordinary", A.analyze(minor)["moments"] == [])
    check("ache: diatonic material produces the explicit none line", text(reading(event(0, "maj"))).startswith("ACHE none found"))
    check("ache: absent input is unavailable, not none or silence", A.block({}) == ["ACHE unclear: analysis unavailable", "  ACHE NOTE: analysis unavailable"])
    check("ache: supplied empty timeline is a completed scan", text({"evs": []}).startswith("ACHE none found"))
    check("ache: a failed null timeline is unavailable, not an empty scan", "ACHE unclear: analysis unavailable" in text({"evs": None}))

    uncertain = reading(event(5, "min"))
    uncertain["key_verdict"] = {"key": None, "working": "C major", "state": "tonic"}
    check("ache: an ambiguous tonic stays a working reference and unclear", "working reference, tonic unclear" in text(uncertain) and "(unclear)" in text(uncertain))
    uncertain["key_verdict"] = {"key": "C major", "working": "C major", "state": "lean"}
    check("ache: a low-confidence claimed key cannot become certain", "tonic unclear" in text(uncertain) and "(unclear)" in text(uncertain))
    uncertain["key_verdict"] = {}
    uncertain["glob"], uncertain["kinfo"] = (0, "major"), {"key": "C major"}
    check("ache: never invents a key from the spectral pick or working tuple", "borrowed" not in text(uncertain) and "key verdict unavailable" in text(uncertain))
    strong = reading(event(5, "min"), spans=[(0, (9, "minor"))])
    check("ache: quoted key verdict outranks a stale initial working span", "borrowed iv (Fm in C major)" in text(strong))
    check("ache: low chord confidence is explicitly unclear", "(unclear)" in text(reading(event(5, "min", conf=0.1))))
    missing = event(5, "min")
    del missing["conf"]
    check("ache: missing chord confidence is never sure", "(unclear)" in text(reading(missing)))

    seq = [event(7, "maj", 8, 10), event(9, "min")]
    check("ache: consecutive V→vi is recognized", "deceptive" in text(reading(*seq)))
    gap = event(None, None, 9, 10)
    check("ache: never bridges unknown chords or missing time for a cadence", "deceptive" not in text(reading(seq[0], gap, seq[1])) and "deceptive" not in text(reading(event(7, "maj", 5, 7), seq[1])))
    check("ache: a phrase-end verdict supplies hanging-end evidence", "phrase ends on V and hangs" in text(reading(cads=[{"t": 20, "t_end": 22, "cadence": "half", "to": "V"}])))
    check("ache: unknown cadence is not unresolved evidence", "harmony" not in kinds(reading(cads=[{"t": 20, "cadence": "unclear"}])))
    change = reading(kinfo={"changes": [{"at": 40, "to": "D major"}]})
    check("ache: key changes keep their windowed-estimate qualification", "key area change to D major (windowed estimate) (lean)" in text(change))
    change["evs"] = [event(7, "min", 41, 45)]
    check("ache: mixture after a key change uses the existing local key", "borrowed iv (Gm in D major)" in text(change) and "(lean)" in text(change))

    # 0.1 s pitch frames: six seconds low, then one second held an octave up.
    mel = {"f0": [220.] * 60 + [440.] * 10, "vp": [0.95] * 70, "hop": 2205, "source": "vocal stem"}
    high = reading(mel=mel)
    check("ache: a sustained high vocal note is relative to this track", "voice near its highest sustained pitch in this track, held 1.0 s" in text(high) and "(lean)" in text(high))
    mixed = copy.deepcopy(high)
    mixed["mel"]["source"] = "full mix"
    check("ache: full-mix pitch is never asserted to be a voice", "full-mix melody" in text(mixed) and "voice unclear" in text(mixed))
    for name, f0 in [("spike", [220.] * 69 + [440.]), ("monotone", [220.] * 70)]:
        check("ache: no high-note claim from " + name, "pitch" not in kinds(reading(mel=dict(mel, f0=f0))))
    check("ache: low-probability pitch cannot become a held note", "pitch" not in kinds(reading(mel=dict(mel, vp=[0.1] * 70))))
    broken = reading(mel=dict(mel, vp=[]))
    check("ache: broken pitch backend gets a visible note", "pitch analysis unavailable" in text(broken) and "near-silence" not in text(broken))
    nanmel = dict(mel, f0=[float("nan")] * 70)
    check("ache: unvoiced/NaN frames are not high notes", "pitch" not in kinds(reading(mel=nanmel)))

    dip = {"t0": 7, "t1": 10, "depth": 15, "silent": False, "returns": True}
    check("ache: relative dips are never called silence", "level dip (15 dB)" in text(reading(dips=[dip])) and "silence" not in text(reading(dips=[dip])))
    check("ache: isolated swells are not swells after a hush", A.analyze(reading(swells=[{"t": 10, "rise": 12}]))["moments"] == [])
    swell = reading(dips=[dip], swells=[{"t": 12, "rise": 12}])
    check("ache: a swell after a dip keeps its measured source", any("swell +12 dB after 3.0 s level dip" in e["text"] for g in A.analyze(swell)["moments"] for e in g["evidence"]))
    check("ache: a dip never invents a solo voice", "texture" not in kinds(reading(dips=[dip])))
    check("ache: solo voice requires explicit stem activity", "texture" in kinds(reading(stem_events=[{"t": 10, "text": "everything drops out but vocals"}])))
    check("ache: standalone singer stem JSON is also accepted", "voice alone" in text({"stems": {"events": [{"t": 10, "text": "everything drops out but vocals"}]}}))

    rep = reading(event(0, "maj", 0, 4), event(0, "maj", 20, 22), event(5, "maj", 22, 24),
                  lines=[{"t": 0, "end": 4, "text": "count bins"}, {"t": 20, "end": 24, "text": "Count bins!"}])
    check("ache: repetition compares chords inside the line, not only its start", "repeated line, chord C→F" in text(rep))
    rep["lines"][0]["approx"] = True
    check("ache: an earlier placed repeat still qualifies the comparison", "≈ placed lyric timing" in text(rep))
    rep["evs"][-1] = event(0, "maj", 22, 24)
    check("ache: identical repeated harmony is not an anomaly", "repetition" not in kinds(rep))
    rep["evs"] = [event(0, "maj", 0, 2), event(5, "maj", 2, 4), event(0, "maj", 20, 24)]
    check("ache: a former chord change disappearing is also heard", "repeated line, chord F→C" in text(rep))

    many = reading(*(event(5, "min", t, t + 4) for t in [10, 30, 50, 70]), dips=[dict(dip, t0=27, t1=30)])
    moments = A.analyze(many)["moments"]
    check("ache: ranks combined evidence and returns at most three, chronologically", len(moments) == 3 and any(m["t"] == 30 and m["score"] == 4 for m in moments) and [m["t"] for m in moments] == sorted(m["t"] for m in moments))
    check("ache: nearby evidence is not printed as simultaneous", "at 0:10" in text(reading(event(5, "min"), dips=[dict(dip, t0=6, t1=8)])))
    check("ache: independent strong harmonic/dynamic facts may be sure", "(sure)" in text(reading(event(5, "min"), dips=[dip])))
    check("ache: malformed analysis cannot silently disappear from the card", "ACHE unclear: analysis failed" in text({"evs": [{}]}))
    check("ache: upstream failure remains visible", "tagger failed" in text(reading(notes=["tagger failed: offline"])))

    public = {"key_verdict": fixture["key_verdict"], "chords": [{"t0": 10, "t1": 14, "chord": "Fm"}]}
    check("ache: older harmony JSON can be read with unknown chord confidence", "borrowed iv" in text(public) and "(unclear)" in text(public))

    import harmony as H
    import music as M
    shared = dict(fixture, path="neutral.wav", dur=40, notes=[], spans=[(0, (0, "major"))],
                  tc={"step": 2, "T": []}, br=[], cads=[], con=[])
    names = ("block_sound", "block_chords", "block_tension", "block_resolves", "block_timeline", "block_lyric_timeline", "block_contrast", "block_surprise", "block_chills", "build_timeline")
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(mock.patch.object(H, name, return_value=[]))
        card, data = H.report(shared), H.to_json(shared)
    check("ache: harmony card adds exactly one block and ends with listener", card.count("borrowed iv") == 1 and card.endswith("and now, you"))
    check("ache: harmony JSON preserves the evidence and can be rendered again", data["ache"] == A.analyze(shared) and A.block(data) == A.block(shared) and data["listener"] == "and now, you")
    check("ache: music card also leaves the last line to the listener", M.format_card({"file": "neutral.wav", "duration": 40}).endswith("and now, you"))

    p = Path(tmp) / "ache.json"
    p.write_text(json.dumps(data))
    for flag in ([], ["--json"]):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = A.main([str(p)] + flag)
        result = stdout.getvalue()
        check("ache: CLI reads existing JSON " + str(flag), rc == 0 and (json.loads(result)["listener"] == "and now, you" if flag else result.rstrip().endswith("and now, you")))
    p.write_text("not json")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        rc = A.main([str(p)])
    check("ache: CLI read failures are explicit, nonzero, and still hand back to listener", rc == 1 and "analysis unavailable" in stdout.getvalue() and "analysis failed" in stdout.getvalue() and stdout.getvalue().rstrip().endswith("and now, you"))
    p.write_text('{"ache": null}')
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        rc = A.main([str(p)])
    check("ache: corrupt saved results stay inside the CLI failure boundary", rc == 1 and stdout.getvalue().rstrip().endswith("and now, you"))


if __name__ == "__main__":
    import tempfile
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + name)

    with tempfile.TemporaryDirectory(prefix="ache_test_") as tmp:
        run_checks(check, tmp)
    print(f"{sum(results)} passed, {len(results) - sum(results)} failed")
    raise SystemExit(0 if all(results) else 1)
