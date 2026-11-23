"""Optional local laugh workers. No model dependencies are imported by the caller.

Omine weights are research-use-only; use the separately installed checkout.
Every worker runs in a 10 GiB systemd scope, with four threads and no swap.
Only numeric measurements and fixed status strings cross the process boundary.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


MODELS = ('omine', 'gillick')
TASKS = ('panns', 'hint') + MODELS


def selected(default='off'):
    name = os.environ.get('RAVEN_EARS_LAUGH_MODEL', default).lower()
    if name not in ('off',) + MODELS:
        raise ValueError('RAVEN_EARS_LAUGH_MODEL must be omine, gillick or off')
    return name


def model_root(name):
    folder = {'omine': 'LaughterSegmentation', 'gillick': 'laughter-detection'}[name]
    return Path(os.environ.get('RAVEN_EARS_' + name.upper() + '_ROOT',
                               str(Path.home() / 'projects' / 'raven-ears-models' / folder)))


def gpu_ready():
    """Check before each heavy worker; never silently fall back after NVML failure."""
    try:
        p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return 'GPU availability check failed'
    if p.returncode:
        return 'GPU availability check failed'
    own = {os.getpid(), int(os.environ.get('RAVEN_EARS_LAUGH_PARENT_PID', os.getpid()))}
    if any(line.strip().isdigit() and int(line.strip()) not in own
           for line in p.stdout.splitlines()):
        return 'another GPU process is running; retry when idle'
    return None


def run_detector(name, audio, timeout=600):
    """Run sequentially. Scope failure and worker failure are never zero detections."""
    if name not in TASKS:
        raise ValueError('unknown laugh worker')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '4'
    env.update(RAVEN_EARS_NO_BASELINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               RAVEN_EARS_LAUGH_PARENT_PID=str(os.getpid()), RAVEN_EARS_OFFLINE='1')
    # Separate interpreters can be selected without installing into the main venv.
    python = env.get('RAVEN_EARS_' + name.upper() + '_PY', sys.executable)
    cmd = ['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=10G',
           '-p', 'MemorySwapMax=0', 'nice', '-n', '10', python,
           str(Path(__file__).resolve()), '--worker', name, str(Path(audio).resolve())]
    start = time.monotonic()
    result = dict(model=name, status='failed', reason='worker could not start')
    try:
        with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, start_new_session=True) as process:
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the scope launcher and all worker descendants in its process group.
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                result['reason'] = 'worker timed out'
            else:
                if process.returncode:
                    result['reason'] = 'worker/scope unavailable (exit %d)' % process.returncode
                else:
                    try:
                        data = json.loads(stdout)
                        if data.get('status') not in ('ran', 'failed'):
                            raise ValueError('invalid status')
                        if data['status'] == 'ran':
                            # Validate results before the card can treat them as findings.
                            data['spans'] = clean_spans(data.get('spans', []), data['duration'])
                        result.update(data)
                    except (TypeError, KeyError, ValueError):
                        result['reason'] = 'invalid worker response'
    except OSError:
        pass
    result['seconds'] = round(time.monotonic() - start, 2)
    return result


def clean_spans(spans, duration):
    """Clamp padding at the real audio end and merge overlapping detections."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid duration')
    cleaned = []
    for start, end in sorted(spans):
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError('invalid span')
        start, end = max(0., float(start)), min(float(duration), float(end))
        if end <= start:
            continue
        if cleaned and start <= cleaned[-1][1]:
            cleaned[-1][1] = max(end, cleaned[-1][1])
        else:
            cleaned.append([start, end])
    return [[round(a, 3), round(b, 3)] for a, b in cleaned]


@contextmanager
def _as_wav(audio):
    """Give upstream readers a seekable WAV in memory, with no copy on disk."""
    import numpy as np
    import soundfile
    import raven_ears_plus as R
    y = R.load_audio(str(audio), 16000)
    with os.fdopen(os.memfd_create('ears-laugh-wav'), 'w+b') as stream:
        soundfile.write(stream, np.asarray(y, dtype=np.float32), 16000, format='WAV')
        yield Path(f'/proc/self/fd/{stream.fileno()}')


def _omine(audio, duration):
    root = model_root('omine')
    sys.path[:0] = [str(root)]
    # Bind the namespace package `train` BEFORE inference.py appends root/train: a
    # plain module (train/train.py) anywhere on sys.path beats a namespace portion.
    import train.model  # noqa: F401
    import inference
    # Upstream inference calls a method that kills randomly selected children.
    # It has no acoustic role; never let a detector manage unrelated processes.
    inference.Model.kill_subprocess_randomly = lambda self: None
    with tempfile.TemporaryDirectory(prefix='laugh_times_') as output, _as_wav(audio) as wav:
        inference.main(str(wav), output, str(root / 'models' / 'model.safetensors'), batch_size=1)
        rows = json.loads((Path(output) / (wav.stem + '.json')).read_text())
    return clean_spans([(r['start_sec'], r['end_sec']) for r in rows.values()], duration)


def _gillick(audio, duration):
    root = model_root('gillick')
    sys.path[:0] = [str(root), str(root / 'utils')]
    import torch
    import numpy as np
    import configs
    import audio_utils
    import data_loaders
    import laugh_segmenter
    from functools import partial
    import librosa
    np.unicode_ = np.str_  # removed in numpy 2; upstream keras-style padding still names it
    # Upstream calls librosa 0.8 style (positional args); adapt in-process, leave librosa alone.
    mel, fix = librosa.feature.melspectrogram, librosa.util.fix_length
    librosa.feature.melspectrogram = lambda y, sr, **k: mel(y=y, sr=sr, **k)
    librosa.util.fix_length = lambda y, size, **k: fix(y, size=size, **k)
    config = configs.CONFIG_MAP['resnet_with_augmentation']
    device = torch.device('cuda')
    model = config['model'](dropout_rate=0., linear_layer_size=config['linear_layer_size'],
                            filter_sizes=config['filter_sizes'])
    model.set_device(device)
    checkpoint = root / 'checkpoints' / 'in_use' / 'resnet_with_augmentation' / 'best.pth.tar'
    # Local, separately installed checkpoint. Do not alter upstream loading policy.
    # Torch 2.6+ defaults to weights_only=True and this 2021 pickle holds numpy scalars.
    # The checkpoint is the user's own local download, so load it the way upstream meant.
    from torch_utils import load_checkpoint
    real_load = torch.load
    torch.load = lambda *a, **k: real_load(*a, **dict(k, weights_only=False))
    try:
        load_checkpoint(str(checkpoint), model)
    finally:
        torch.load = real_load
    model.eval()
    with _as_wav(audio) as wav:
        dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
            audio_path=str(wav), feature_fn=config['feature_fn'], sr=8000)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=8, shuffle=False,
        collate_fn=partial(audio_utils.pad_sequences_with_labels,
                           expand_channel_dim=config['expand_channel_dim']))
    probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            probs.extend(model(torch.from_numpy(inputs).float().to(device)).cpu().numpy().reshape(-1))
    probs = laugh_segmenter.lowpass(np.asarray(probs))
    return clean_spans(laugh_segmenter.get_laughter_instances(
        probs, threshold=.5, min_length=.2, fps=len(probs) / duration), duration)


def _measure(name, audio):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    if name in MODELS and not torch.cuda.is_available():
        return dict(status='failed', reason='CUDA unavailable; no CPU model fallback')
    import raven_ears_plus as R
    y = R.load_audio(str(audio), R.SR_SED)
    duration = len(y) / R.SR_SED
    base = dict(status='ran', duration=duration)
    if name == 'panns':
        import numpy as np
        import room_laugh as L
        dur, times, probs, labels = L._probs(str(audio), y=y)
        ix = {lab: i for i, lab in enumerate(labels)}
        family = [lab for lab in L.LAUGH if lab in ix]
        values = probs[:, [ix[lab] for lab in family]]
        detections = {}
        default_rows = []
        for threshold in (.20, .15, .10):
            rows = L._events(times, values, [L.LAUGH[lab] for lab in family], threshold, dur)
            detections[str(threshold)] = [[e['start'], e['end']] for e in rows]
            if threshold == .20:
                default_rows = rows
        from scipy.signal import resample_poly
        fe = R._front_end(resample_poly(y, 1, 2).astype(np.float32))
        ev = R.analyze_events(y, duration, [], fe)
        gaps = R.analyze_gaps(fe, None, duration)
        room = R.RW.room_result(ev, gaps)
        room_variants = {}
        dog_passes = []
        for method, reduced in (('raw', False), ('voice-reduced', True)):
            try:
                candidate = R.analyze_faint_dogs(y, duration, fe, [], reduced=reduced)
            except Exception as error:
                room_variants[method] = dict(status='failed', reason=type(error).__name__)
                continue
            if candidate is not None:
                dog_passes.append(candidate)
            elif not reduced or fe['voiced'].any():
                room_variants[method] = dict(status='failed', reason='insufficient windows or missing dog labels')
                continue
            if reduced and room_variants['raw']['status'] != 'ran':
                room_variants[method] = dict(status='failed', reason='raw pass unavailable')
                continue
            trial = R.RW.room_result(ev, gaps, dog_passes)
            room_variants[method] = dict(status='ran', pass_data=candidate, room_line=R.RW.room_line(trial),
                                         dogs=[s for h in trial['hits'] if h['name'] == 'dog' for s in h['spans']])
        return dict(base, spans=detections['0.2'], thresholds=detections,
                    dogs=[s for h in room['hits'] if h['name'] == 'dog' for s in h['spans']],
                    room_line=R.RW.room_line(room), room_variants=room_variants, laughs_line=R.laughs_line(dict(
                        laughs=default_rows, snorts=L._events(times, probs[:, [ix['Snort']]], ['snort'], L.SNORT_THR, dur))),
                    sneezes=[[e['start'], e['end']] for e in ev['events'] if e['label'] == 'Sneeze'])
    if name == 'hint':
        from scipy.signal import resample_poly
        import numpy as np
        y16 = resample_poly(y, 1, 2).astype(np.float32)
        notes = []
        # Keep the speech engine's selected model while using its explicit thread cap.
        # This fast flag changes execution limits, not the already selected model.
        os.environ['RAVEN_EARS_WHISPER'] = R.EP.speech_model(False)
        raw = R._whisper_words(y16, notes, fast=True)
        words = R._parse_words(raw)
        R.release_models()
        if not words:
            return dict(status='failed', reason='hint requires available word timestamps')
        words, _ = R._sanitize_words(words)
        fe = R._front_end(y16)
        hints = R.laugh_hint(words, R.emphasis_features(words, fe)) if len(words) >= 4 else []
        return dict(base, spans=[[h['start'], h['end']] for h in hints],
                    whisper_model=os.environ['RAVEN_EARS_WHISPER'],
                    line=R.laughs_line(dict(laughs=[], snorts=[]), hints))
    spans = _omine(audio, duration) if name == 'omine' else _gillick(audio, duration)
    return dict(base, spans=spans)


def worker(name, audio):
    # Silence even native/library output; no private speech or raw exceptions escape.
    saved = os.dup(1)
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            with tempfile.TemporaryDirectory(prefix='laugh_home_') as home:
                # Direct raven_ears_plus imports do not implement listen's disposable home.
                # Never read recognition hints or a baseline during evaluation.
                os.environ['RAVEN_EARS_HOME'] = home
                why = gpu_ready()
                result = dict(status='failed', reason=why) if why else _measure(name, Path(audio))
        except Exception as error:
            reason = type(error).__name__
            if isinstance(error, ModuleNotFoundError):
                reason += ': missing module ' + str(error.name)
            result = dict(status='failed', reason=reason)
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=TASKS, required=True)
    parser.add_argument('audio')
    args = parser.parse_args()
    from network_policy import local_audio_path
    local_audio_path(args.audio, parser)
    worker(args.worker, args.audio)
