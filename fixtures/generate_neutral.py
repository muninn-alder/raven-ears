#!/usr/bin/env python3
"""Generate 32 seconds of voiced-like tones and gaps, using only the standard library.

Output is 44.1 kHz mono: harmonic tones for the "voice" plus a few fixed high sines
(9-13.5 kHz) standing in for fricative/air energy, so the CHANNEL check sees a full-band
recording rather than a telephone-style one.

No human voice, recording, speech text, model or network is used. Run from any cwd.
"""
from array import array
import math
from pathlib import Path
import wave

SR = 44100
AIR_HZ = (9000, 10500, 12000, 13500)  # deterministic stand-in for energy above 8 kHz
AIR_AMP = 450


def generate(path=None):
    path = Path(path) if path else Path(__file__).with_name("neutral-voiced.wav")
    samples = array("h")
    for burst in range(32):
        f0 = 125 + 5 * (burst % 6)
        for i in range(SR):
            t = i / SR
            if t < 0.58:
                envelope = min(1, t / 0.04, (0.58 - t) / 0.06)
                value = sum(math.sin(2 * math.pi * f0 * h * t) / h for h in range(1, 9))
                air = sum(math.sin(2 * math.pi * f * t + k) for k, f in enumerate(AIR_HZ))
                samples.append(int(6200 * envelope * value + AIR_AMP * envelope * air))
            else:
                samples.append(0)
    import sys
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(samples.tobytes())
    return path


if __name__ == "__main__":
    print(generate())
