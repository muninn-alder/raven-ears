"""Opt-in CPU policy. Import before numerical libraries so their pools see the cap."""
from contextlib import contextmanager
from functools import wraps
import os


THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')


def cpu():
    return os.environ.get('RAVEN_EARS_PROFILE', '').strip().lower() == 'cpu'


def cpu_threads():
    try:
        return max(1, min(4, int(os.environ.get('OMP_NUM_THREADS', '4'))))
    except ValueError:
        return 4


def _cap_environment():
    for name in THREAD_VARS:
        try:
            count = max(1, min(4, int(os.environ.get(name, '4'))))
        except ValueError:
            count = 4
        os.environ[name] = str(count)


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def speech_model_override():
    return next((key for key in ('RAVEN_EARS_WHISPER', 'WHISPER_MODEL') if key in os.environ), None)


def speech_model(fast=False):
    default = 'small' if fast else 'large-v3-turbo' if cpu() else 'large-v3'
    override = speech_model_override()
    return os.environ[override] if override else default


@contextmanager
def _fast_threads():
    previous = {name: os.environ.get(name) for name in THREAD_VARS}
    _cap_environment()
    torch = _torch()
    old_threads = torch.get_num_threads() if torch is not None else None
    try:
        if torch is not None:
            torch.set_num_threads(cpu_threads())
        yield
    finally:
        if torch is not None:
            torch.set_num_threads(old_threads)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def speech_threads(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if kwargs.get('fast', False):
            with _fast_threads():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapped


if cpu():
    os.environ['RAVEN_EARS_DEVICE'] = 'cpu'
    _cap_environment()
    _t = _torch()
    if _t is not None:
        _t.set_num_threads(cpu_threads())
        try:
            _t.set_num_interop_threads(1)
        except RuntimeError:
            pass  # An embedding application may already have started its interop pool.
