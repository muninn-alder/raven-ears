"""room_laugh: laugh / snort detector for the ears (2026-10-20).

PANNs CNN14 (AudioSet, 527 classes), the SAME checkpoint raven_ears_plus already loads
($RAVEN_PANNS_CACHE or ~/.cache/raven-panns, ~330 MB). No audio is uploaded.
1 s windows, 0.5 s hop, CPU. Laugh family and Snort are reported separately.

    import room_laugh; r = room_laugh.detect(path); print(room_laugh.format_line(r))
    CLI: python room_laugh.py <audio> [--thr 0.35] [--snort-thr 0.25] [--windows] [--json]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LAUGH = {"Laughter": "laugh", "Baby laughter": "baby laugh", "Giggle": "giggle", "Snicker": "snicker",
         "Belly laugh": "belly laugh", "Chuckle, chortle": "chuckle"}
SNORT = {"Snort": "snort"}
WIN, HOP, SR = 1.0, 0.5, 32000
LAUGH_THR, SNORT_THR = 0.20, 0.10
MERGE_GAP = 0.01   # windows that overlap or touch merge into one event


def _windows(y):
    W, H = int(WIN * SR), int(HOP * SR)
    if len(y) < W:
        y = np.pad(y, (0, W - len(y)))
    starts = list(range(0, len(y) - W + 1, H))
    if (len(y) - W) % H:
        starts.append(len(y) - W)
    return y, starts, W


def _probs(path, y=None, tagger=None):
    import raven_ears_plus as rep
    torch = rep._torch()
    if y is None:
        y = rep.load_audio(str(path), SR)
    dur = len(y) / SR
    y, starts, W = _windows(y)
    m, labels = tagger if tagger is not None else rep._load_tagger("cpu")
    out = []
    for i in range(0, len(starts), 32):
        x = np.stack([y[s:s + W] for s in starts[i:i + 32]]).astype(np.float32)
        with torch.no_grad():
            out.append(m(torch.from_numpy(x))["clipwise_output"].float().cpu().numpy())
    return dur, np.array(starts) / SR, np.concatenate(out), list(labels)


def _events(t0, family_p, names, thr, dur):
    """family_p [nW, k] -> merged events over windows whose summed prob >= thr."""
    tot = np.minimum(family_p.sum(1), 1.0)
    hit = tot >= thr
    evs, i = [], 0
    while i < len(hit):
        if not hit[i]:
            i += 1; continue
        j = i
        while j + 1 < len(hit) and hit[j + 1]:
            j += 1
        seg = family_p[i:j + 1]
        k = int(np.argmax(seg.max(0)))
        evs.append({"start": round(float(t0[i]), 2), "end": round(float(min(t0[j] + WIN, dur)), 2),
                    "label": names[k], "peak": round(float(seg[:, k].max()), 2),
                    "total": round(float(tot[i:j + 1].max()), 2)})
        i = j + 1
    return evs


def detect(path, laugh_thr=LAUGH_THR, snort_thr=SNORT_THR, y=None, tagger=None):
    """y: optional 32 kHz mono audio; tagger: optional already loaded CPU (model, labels)."""
    dur, t0, P, labels = _probs(path, y, tagger)
    ix = {l: i for i, l in enumerate(labels)}
    lk = list(LAUGH); sk = list(SNORT)
    LP = P[:, [ix[l] for l in lk]]; SP = P[:, [ix[l] for l in sk]]
    return {"duration": round(dur, 2),
            "laughs": _events(t0, LP, [LAUGH[l] for l in lk], laugh_thr, dur),
            "snorts": _events(t0, SP, [SNORT[l] for l in sk], snort_thr, dur),
            "windows": [{"t": round(float(t), 2), "laugh": round(float(min(a.sum(), 1)), 3), "snort": round(float(b.sum()), 3)}
                        for t, a, b in zip(t0, LP, SP)]}


def format_line(r):
    def part(evs):
        return "  ·  ".join(f"{e['start']:.2f}–{e['end']:.2f}s {e["label"]} {e["total"]:.2f}" for e in evs)
    L, S = r["laughs"], r["snorts"]
    s = f"😆 LAUGHS: {len(L)}" + (f"  ·  {part(L)}" if L else "")
    s += f"   |  SNORTS: {len(S)}" + (f"  ·  {part(S)}" if S else "")
    return s


def main():
    ap = argparse.ArgumentParser(prog="room-laugh")
    ap.add_argument("audio"); ap.add_argument("--thr", type=float, default=LAUGH_THR)
    ap.add_argument("--snort-thr", type=float, default=SNORT_THR)
    ap.add_argument("--windows", action="store_true", help="print per-window scores"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    from network_policy import local_audio_path
    local_audio_path(a.audio, ap)
    r = detect(a.audio, a.thr, a.snort_thr)
    if a.json:
        print(json.dumps(r)); return
    print(format_line(r))
    if a.windows:
        for w in r["windows"]:
            print(f"  {w['t']:6.2f}s laugh {w['laugh']:.2f} snort {w['snort']:.2f}")


if __name__ == "__main__":
    main()
