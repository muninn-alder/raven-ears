"""
madmom_key.py: print madmom's CNN key probabilities for one audio file as JSON.  {"probs": [24 floats]}
Run it with the python of the SEPARATE madmom venv (./venv-madmom, see README); keyfuse.py calls it as a subprocess.
Order is madmom's: A, Bb, B, C ... G# major, then A minor ... G# minor. Model files are CC BY-NC-SA 4.0 (non-commercial).
"""
import json
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('file')
ap.add_argument('--start', type=float)
ap.add_argument('--end', type=float)
args = ap.parse_args()
from network_policy import local_audio_path
local_audio_path(args.file, ap)
from raven_ears_plus import load_audio
from madmom.features.key import CNNKeyRecognitionProcessor
from madmom.audio.signal import Signal
sr = 44100
y = load_audio(args.file, sr)
start = int(args.start * sr) if args.start is not None else 0
end = int(args.end * sr) if args.end is not None else len(y)
audio = Signal(y[start:end], sample_rate=sr)
probs = np.asarray(CNNKeyRecognitionProcessor()(audio)).reshape(-1)
print(json.dumps({"probs": [float(x) for x in probs]}))
