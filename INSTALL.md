# Install

Windows users need WSL (Ubuntu recommended); native Windows is not supported. Run these commands inside WSL.

Use Python 3.12 and FFmpeg (both `ffmpeg` and `ffprobe` on PATH). Python 3.14 is
**unverified** for this dependency set; a system with only 3.14 can use the
no-sudo route below. No GPU is required. See PRIVACY.md before using song metadata.
The Linux CPU route below was installed from scratch on 21 November 2026, from these
docs alone, in a new venv with an empty model cache (torch 2.8.0+cpu, Python 3.12):
every step ran, and both a voice card and a song card came out clean. All macOS
recipes remain **unverified**.

On Debian/Ubuntu, install system tools with `sudo apt install python3-venv ffmpeg`.
On macOS with Homebrew: `brew install python@3.12 ffmpeg` (unverified).
From this checkout:

```sh
python3.12 -m venv venv
. venv/bin/activate
python -m pip install -r requirements.txt
RAVEN_EARS_OFFLINE=1 python raven_ears_plus.py fixtures/neutral-voiced.wav \
  --cpu --no-whisper --no-events --no-affect --no-laughs
```

Without sudo on Linux x86_64, [uv](https://docs.astral.sh/uv/getting-started/installation/)
can supply Python 3.12. If uv is already installed, skip its installer:

```sh
curl -LsSf https://astral.sh/uv/install.sh -o uv-install.sh
sh uv-install.sh
rm uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 venv
uv pip install --python venv/bin/python pip
. venv/bin/activate
```

uv venvs have no pip by default; the second uv command adds it for the recipes
here. If FFmpeg is also missing, a user-space static build supplies both tools.
For example, from the checkout (this download recipe is unverified here):

```sh
mkdir -p tools-local
curl -fL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o tools-local/ffmpeg.tar.xz
tar -xJf tools-local/ffmpeg.tar.xz -C tools-local
export PATH="$(find "$PWD/tools-local" -maxdepth 1 -type d -name 'ffmpeg-*-static' | head -n 1):$PATH"
ffmpeg -version
ffprobe -version
python -m pip install -r requirements.txt
```

Keep that PATH setting in the shell used for cards. The static build above is
for Linux x86_64, including WSL; macOS uses its own FFmpeg build.

This first card uses pitch, energy, gaps and approximate voice quality. It does
not need model weights. For words, prepare a speech target below before switching
offline; large-v3 on CPU is slow and can use several GB of RAM. `--words FILE`
accepts local word timestamps without Whisper.

Bring your own music files; the ears do not download songs. A copy whose length
doesn't match the studio track gets flagged when existing LRCLIB, MusicBrainz or
Discogs metadata establishes a length (tolerance: five seconds or 3%, whichever
is larger). Saved views keep the warning without new requests.

For optional speech EVENTS/AFFECT and music, install CPU Torch first on Linux:

```sh
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-music.txt
RAVEN_EARS_DEVICE=cpu python music.py /path/to/audio.wav --no-lyrics --no-credits
```

Keep matching torch/torchaudio versions. Transformers 5 requires Torch >=2.6;
the old 2.5.1 pair disables its Torch backend and breaks AFFECT and sentiment.
The 2.8.0 CPU pair installed cleanly on Python 3.12 in the 21 November fresh install. The [official wheel recipes](https://pytorch.org/get-started/previous-versions/)
include CPU builds. Installing music requirements directly from Linux PyPI can
select CUDA dependencies: install the CPU pair first. See the CPU measurements
below for complete speech and song timings.

## CPU only

You can keep all the ears on a machine without a GPU. For Linux, start in a
fresh venv with the CPU wheels above, then enable one opt-in profile:

```sh
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-music.txt
export RAVEN_EARS_PROFILE=cpu
python raven_ears_plus.py --setup speech
python raven_ears_plus.py /path/to/voice-note.wav
python raven_ears_plus.py --setup fast
python raven_ears_plus.py /path/to/voice-note.wav --fast
python raven_ears_plus.py --setup music
python listen.py /path/to/song.wav --quick
```

`--setup music` includes lyrics' large-v3 (~3.1 GB), even when speech uses turbo
or small, plus PANNs, AFFECT, Beat This! and RoBERTa sentiment. Run it while
online before the first song. Setup prints each model before fetching it. The
~330 MB PANNs download alone can take several minutes on a slow connection.

`RAVEN_EARS_PROFILE=cpu` forces CPU even on a GPU machine. It caps OpenMP, MKL,
OpenBLAS, Numba and Torch at four threads (smaller environment limits are kept),
with one Torch interop thread and up to four Whisper CPU threads. Set the profile
before starting Python so numerical libraries see those limits when they load.
Unset it to return to the main setup. With neither the profile nor `--fast`,
model choices and card text are unchanged, including on GPU.

Speech uses **large-v3-turbo** in CPU int8: on this machine it matched large-v3's
accuracy in half the time and less memory. `RAVEN_EARS_WHISPER` still wins;
the legacy `WHISPER_MODEL` is next, then the selected default. Lyrics default to
**large-v3**, including in this profile: turbo on sung vocals is unvalidated.
For a lyrics comparison, set `RAVEN_LYRICS_WHISPER=large-v3-turbo`; unset it
or set it to `large-v3` to restore the lyrics default. Speech overrides do not
change lyrics. Shared lyric caches separate model, device and compute type;
older untagged entries are left untouched and transcribed again. A device or
precision fallback is reported and is not cached under the requested configuration.

`RAVEN_EARS_LANG` always selects the lyric transcription language when set.
Otherwise LRCLIB or supplied lyrics can select the dominant language using
conservative whole-song text cues (English, French, Spanish, German, Italian,
Portuguese and Dutch). Short, mixed or unsupported text keeps English and says
why in the lyric notes. This is text inference, not audio language detection;
short foreign phrases do not change the main language. The cache also separates
the selected language and lyric text used to protect genuine sung filler phrases.

For live conversation, `--fast` selects **small** and skips AFFECT. It keeps
words, pitch, pace, WORDLESS/SILENT, events, laughs and voice quality. The card
states the selected model, the affect skip and thread cap; PEAK uses loudness
alone. Model overrides still win, so `RAVEN_EARS_WHISPER=large-v3-turbo` with
`--fast` keeps the more accurate transcript while skipping affect. Supplied
`--words` and `--no-whisper` keep their meanings and are named on the fast card.
Fast mode also works without the profile, preserving automatic device selection;
it caps Torch/Whisper threads during the call. Fast EVENTS and LAUGHS now reuse
one CPU PANNs model. Pitch stays pyin: faster trackers changed phrase endings
or WORDLESS on the twelve comparison clips. With caches populated but models
loaded fresh, the 15-second card took **8.4 s** in analysis (**10.6 s** including
CLI startup), peak **1.7 GiB**. The original code took 10.5 s in analysis in the
same session; the earlier 17.3 s baseline was not reproduced.

For repeated live clips, `python raven_ears_plus.py --serve` implies `--fast`:
send one local audio path per stdin line and keep stdin open between requests.
Models load on the first clip and stay resident; EOF releases them. Each response
is one JSON line with `ok` and the card's `lines` (`--json` returns `data` instead);
a bad path returns an error and the next request continues. This local stdin
interface has no authentication. At four CPU threads, the warm 15-second request
took **5.7 s**; ten phone clips took **3.1–12.5 s**, with **1.9 GiB** idle RAM.
All thirteen measured clips produced the same card bytes as one-shot `--fast`,
including a repeated reading after other speakers. Use `--no-baseline-update`
for repeatable comparisons when supplying `--speaker`.

First-use speech weights are roughly **1.6 GB** for turbo, **0.48 GB** for small,
or **3.1 GB** for large-v3 (local CTranslate2 model-file sizes, before CPU int8
loading). EVENTS and LAUGHS share PANNs (~330 MB); AFFECT is ~660 MB of active
weights. `--setup fast` fetches PANNs, both labels copies and the selected fast
Whisper, skipping AFFECT. `--setup speech` (also the default `--setup` / `full`)
adds the AFFECT repository (~1.3 GB if both weight formats download) and uses
the selected ordinary speech Whisper instead. Model selection follows the policy
above, including both overrides. AFFECT is **non-commercial, research-only**;
see THIRD-PARTY.md. For a minimal words/pitch install, core requirements suffice;
missing optional events/affect/laugh backends are explicit skips on the card.
Install the optional backends for complete fast/speech setup. Prepare while online,
then check without downloading (exit 1 means missing/invalid files):

```sh
python raven_ears_plus.py --setup fast
RAVEN_EARS_OFFLINE=1 python raven_ears_plus.py --setup fast --check
RAVEN_EARS_OFFLINE=1 python raven_ears_plus.py /path/to/voice-note.wav --fast
```

Setup validates the 527-class labels in both `RAVEN_PANNS_CACHE` and the upstream
import location `~/panns_data/class_labels_indices.csv`. The check inspects files;
it does not prove inference or a working dependency installation. Real downloads were checked in the 21 November fresh install; an offline card from a
fresh home still needs checking.

Allow **8 GB system RAM** for ordinary CPU speech; the old large-v3 whole card
peaked near 5 GB. Turbo's Whisper-only peak was 2.7 GB, small's 1.7 GB. Plan for
at least **4 GB free RAM** for turbo or fast speech plus room for the OS; a
4 GB total machine's full card is unverified. Peak RAM for longer fast clips
remains unmeasured on the i9 below. Songs with large-v3 lyrics need at least the same 8 GB planning budget, plus room for the OS.
Keep one heavy job at a time.

Measured on 15 November 2026: Intel Core i5-11400H laptop, Ubuntu 26.04 in WSL,
CPU profile (GTX 1650 forced off). These are complete card times, with weights
already downloaded; the full song also reused cached stems. The revised pins
have not yet been timed on this laptop.

| Card | Audio length | Time | Peak process RAM |
|---|---:|---:|---:|
| Speech, turbo CPU profile | 7 s | 19 s | 2.4 GB |
| Speech, `--fast` | 7 s | 11 s | 1.3 GB |
| Song, *Purple Haze*, `--quick` | 2:52 | 3:40 | 2.0 GB |
| Song, *Purple Haze*, full, stems cached | 2:52 | 6:44 | 7.7 GB |

AFFECT and sentiment failed in that laptop install because of the old Torch
pin. These are measurements of those cards, not proof that every stage ran.

These are **this machine's** numbers (12 November 2026): Intel Core i9-9900K
@ 3.60 GHz, WSL2, CPU, four threads per run. The Whisper comparison below used
a 6 GB memory box; the new fast/resident timings above did not. The
9:31 LibriVox reading of *The Raven* was scored strictly against Gutenberg
#1065; this is one English reading, not an accuracy guarantee for other voices.

| Speech Whisper model | Word error rate | Time for 9:31 | Peak process RAM |
|---|---:|---:|---:|
| large-v3 (main default) | 5.6% | 307 s | 4.7 GB |
| large-v3-turbo (CPU profile) | 5.4% | 146 s | 2.7 GB |
| medium | 7.4% | 173 s | 2.9 GB |
| small (`--fast`) | 8.0% | 59 s | 1.7 GB |
| distil-large-v3.5 | 14.5% | 123 s | 2.6 GB |
| base | 23.5% | 24 s | 1.5 GB |

The full 60-second speech card with large-v3 took 70 s at four threads,
99 s at eight and 153 s at sixteen. At four, Whisper took 43 s, affect 12.8 s,
events 5.2 s, laughs 2.8 s, quality 2.5 s and the front end 2.2 s. This is why
we cap threads and fast mode skips affect. A names list (README's `names.txt`)
cut base's error rate from 23.5% to 13.1% on the same reading.

Songs still take minutes. Fresh four-stem separation uses **htdemucs** (~84 MB)
in the CPU profile, with one status saying the CPU profile chose it instead of htdemucs_ft.
On the 4:09 studio *Strawberry Fields Forever*, separation alone took **107 s /
2.4 GB** versus htdemucs_ft's **434 s / 3.4 GB** (four models, ~336 MB weights).
The card-quality difference is **UNMEASURED**; speed is the measured gain.
Existing complete cached stems are reused in either profile; htdemucs provenance
still warns. `listen.py` also reuses saved cards; use `--fresh` to rebuild one
(complete cached stems are still reused). The optional guitar/piano second opinion stays htdemucs_6s (~55 MB;
88 s / 2.8 GB here). Music setup prepares Beat This! (~81 MB), PANNs/AFFECT,
large-v3 lyrics and RoBERTa sentiment. Optional key models download separately.
No song inference or lyrics accuracy policy is otherwise reduced.

On Apple Silicon, the proposed equivalent is `python -m pip install
torch==2.8.0 torchaudio==2.8.0` before music requirements; this uses CPU here.
On Intel macOS, the proposed pair is `torch==2.2.2 torchaudio==2.2.2`, with
NumPy `<2` and Transformers `>=4.40,<4.50` to retain the older Torch backend;
platform markers in both requirements files preserve these limits
required by [Demucs 4.1.0](https://pypi.org/pypi/demucs/4.1.0/json).
Neither macOS architecture nor its wheel availability has been tested here.

Optional higher quality speech metrics: `python -m pip install praat-parselmouth`
(GPL-3.0-or-later dependency). Without it, voice quality uses approximate metrics
and the card says so as an ordinary note. A failure in an installed parselmouth
still reports degradation.
`mir_eval` is optional for the key-scoring cross-check in tests.

`python raven_ears_plus.py --setup music` prepares PANNs (~330 MB), both labels
copies, AFFECT, lyrics' **large-v3** (speech overrides do not change lyrics), the
Beat This! final0 checkpoint (~81 MB), and the RoBERTa sentiment weights and tokenizer.
PANNs goes to `~/.cache/raven-panns` (or `RAVEN_PANNS_CACHE`). If you already have
the checkpoint somewhere else, point `RAVEN_PANNS_CACHE` at it; otherwise the music
card's INSTRUMENTS line reports that setup is needed.
`--setup music --check` checks those files locally. Demucs and the optional key
models aren't part of setup and download on first use, so run your first song while
online; after that, cached models run offline. Read PRIVACY.md and THIRD-PARTY.md first.
**AFFECT is non-commercial, research-only**.

Beat This! setup and first use try the official checkpoint server, then the
[oulianov/beat_this mirror](https://huggingface.co/oulianov/beat_this), then the
[teamup-tech/Beat-This-weights mirror](https://huggingface.co/teamup-tech/Beat-This-weights).
Every source must match **81058141 bytes** and SHA256
`8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`.
Downloads use a temporary file beside the checkpoint and are renamed only after
verification. Progress names the successful source and confirms verification;
if all fail, setup and the card list every URL and error. TEMPO then falls back
to a librosa estimate and METER is skipped, with the cause on the card.

A changed upstream file is rejected, including from the official server. For a
legitimate update, independently verify its checksum, then set
`RAVEN_BEAT_THIS_SHA256` to the new SHA256 and, if needed, `RAVEN_BEAT_THIS_SIZE`
to its exact byte count. These overrides also apply to cached files.

You can still fetch the checkpoint on another machine and copy it into Torch's
cache. With the venv active, place it at the following path and verify locally
(the check does not download or run inference):

```sh
BEAT_DIR=$(python -c 'import torch; print(torch.hub.get_dir() + "/checkpoints")')
mkdir -p "$BEAT_DIR"
cp /path/to/final0.ckpt "$BEAT_DIR/beat_this-final0.ckpt.part" &&
  python -c 'from pathlib import Path; from singer import _verify_beat_checkpoint; import sys; _verify_beat_checkpoint(Path(sys.argv[1]))' "$BEAT_DIR/beat_this-final0.ckpt.part" &&
  mv "$BEAT_DIR/beat_this-final0.ckpt.part" "$BEAT_DIR/beat_this-final0.ckpt"
python raven_ears_plus.py --setup music --check
```

A verified local checkpoint is loaded directly without contacting any server.

Optional madmom CNN key voter: separate environment, not in the main requirements.
madmom builds from source, so it needs a C compiler and the Python headers
(Debian/Ubuntu: `sudo apt install build-essential python3.11-dev`).
`venv-madmom` may live beside the code or beside the venv you run; `RAVEN_MADMOM_PY` takes priority.
The KEY line warns when the CNN voter did not run, including when that environment was not found.
The source documentation records abbreviated revision `27f032e`; its fetch/build
has not been verified here. The code is BSD-2-Clause; models are CC BY-NC-SA 4.0.

```sh
python3.11 -m venv venv-madmom
venv-madmom/bin/python -m pip install 'numpy>2' cython setuptools wheel scipy mido
venv-madmom/bin/python -m pip install --no-build-isolation 'git+https://github.com/CPJKU/madmom.git@27f032e'
export RAVEN_MADMOM_PY="$PWD/venv-madmom/bin/python"
```

The optional meter fallback imports madmom in the **main** environment. Installing
the key voter environment does not enable that fallback. Beat This! is preferred;
if neither tracker works the card records that meter is unavailable. Avoid adding
madmom to the core environment just to silence this warning.

Optional CUDA users choose appropriate matching GPU Torch wheels from the official
selector and install `requirements-gpu.txt` for faster-whisper's CUDA 12 runtime.
That file is never included by the CPU or music requirements.

GPU users should prepare the desired setup target before the first card,
with network access enabled and the venv active. Other song models can still
download on first use, as described above.
On Linux, faster-whisper also needs the venv's NVIDIA cuBLAS/cuDNN libraries on
the loader path **before launching Python**. In the same shell used for cards:

```sh
python raven_ears_plus.py --setup
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Without these loader paths, Whisper can fall back to CPU int8 even when Torch
sees a GPU. The card records that fallback as `Whisper: degraded` in NOTES.
GPU inference and this CUDA recipe have not been verified for this release.

## Optional listening sheets

New text listens produce a card and listening sheet by default. Drawing needs
optional Matplotlib on either CPU or GPU installs:
`python -m pip install -r requirements-picture.txt`. Then use
`python listen.py song.mp3` (also works on saved views) or
`python raven_ears_plus.py note.wav`. Use `--no-picture` or
`RAVEN_EARS_PICTURE=0` for just the card; `RAVEN_EARS_PICTURE=1` enables the sheet.
CLI flags win over the environment, and `--picture` remains an explicit on.
`--quick` keeps the sheet on. Speech accepts
`--picture-out sheet.png` to choose a destination. Without Matplotlib the card
still prints, followed by the install hint. Drawing is local and starts no
models or lookups. Song PNGs and the cached full-mix spectrum live beside
`song.json`; speech PNGs default to `note.listening-sheet.png` beside the input.
Old song bundles show their saved section loudness means; a fresh listen drawing
a sheet also retains the existing frame RMS. No fresh listen is needed to
draw an older bundle. JSON, management commands and speech server mode keep
their existing output; an explicit picture request still requires a text card.
Automatic speech sheets keep normal speaker-baseline updates; explicit
`--picture`, `--picture-out` and `--by-ear` remain comparison-only.

Both commands accept `--by-ear marks.txt` (implies `--picture`): UTF-8 text,
one `m:ss label` per line, optionally fractional seconds and blank lines. Times
must fall within the recording. Invalid annotations leave the text card intact
and produce a picture error. These marks are never saved in the song bundle.
Song sheets include a spectral-return colour lane (pale for weak grouping,
grey for no displayed match) and explicit gap/cut bands. Headers wrap above the
lanes; long headers increase image height while width stays 2400 pixels.
Optional stem energy reads only existing WAVs in `RAVEN_STEMS_CACHE`
(default `~/.cache/raven-stems`), using the song hash's first 16 characters and
the `-6s` sibling for guitar/piano or missing four-stem files (labelled “6s”).
Missing, unreadable or duration-mismatched
stems are skipped silently; no separation or model upgrade runs. Stem energy
does not establish instrument identity or absence.

For a conversation, pass speech clips in order:

```sh
python conversation_sheet.py first.webm reply.webm --speakers voice-a voice-b --output conversation.png
```

Use one `--speakers` value per clip. Use `-` for a clip or omit `--speakers`
to use the local voice book, which falls back to “unknown”. Specify one ISO start time per clip
with `--timestamps`, for example `2026-01-01T12:00:00Z 2026-01-01T12:04:30Z`.
Include offsets on every timestamp or use the same local clock for all of
them. Gaps longer than 10 seconds use a 5-second hatched band with the true
length printed on it. Without timestamps the sheet marks inserted half-second
spaces. The spectrum, RMS and pitch are measured once per note. Laugh spans
use Omine when it runs and fall back to the speech card's CNN14 laugh pass.
Model loading is offline and no speaker baseline is updated. Add `--words`
to include labels from locally cached Whisper.

## Disk and caches

Models and saved cards live outside the checkout. The laptop used about **9.5 GB**
including its venv and caches; about 6.3 GB was Hugging Face downloads and 313 MB
was PANNs. Allow more space for other models, songs and cached stems. Set cache
variables before setup and keep them set for cards; moving `RAVEN_EARS_HOME`
alone does not relocate model weights.

| Default location | Contents | Relocate with |
|---|---|---|
| `~/.cache/huggingface/hub` | Whisper, AFFECT, sentiment and any HF-backed Demucs weights | `HF_HOME` (parent), or `HF_HUB_CACHE` (hub itself) |
| `~/.cache/raven-panns` | PANNs checkpoint and labels | `RAVEN_PANNS_CACHE` |
| `~/panns_data` | Upstream PANNs labels copy | No dedicated variable; upstream uses `HOME` |
| `~/.cache/torch/hub/checkpoints` | Beat This! and Torch-backed Demucs weights | `TORCH_HOME` (uses its `hub/checkpoints`) |
| `~/.cache/raven-stems` | Separated audio and per-song analysis | `RAVEN_STEMS_CACHE` |
| `~/.cache/raven-lyrics` | Shared word timestamps | `RAVEN_LYRICS_CACHE` |
| `~/.cache/harmony` | Feature and voter caches | `RAVEN_HARMONY_CACHE` |
| `~/.cache/raven/cards/songs` | Saved song bundles | `RAVEN_CARDS_HOME` or `RAVEN_CARDS_DIR` (uses `songs`), or `RAVEN_EARS_HOME` (uses `songs`, takes priority) |
| `~/.local/share/raven-ears` | Speech settings and speaker baselines | `RAVEN_EARS_HOME` |

Hugging Face and Torch also honour `XDG_CACHE_HOME` when their own variables are
unset; the explicit Raven cache paths above do not. `panns_data` holds labels,
not a second checkpoint. `RAVEN_EARS_NO_BASELINE=1` prevents speech baseline
updates, including direct `analyze()` calls; comparisons can still read existing
baselines. Only `listen.py` additionally creates a disposable home, including
disposable models and song storage. Standalone speech/song engines keep their
normal caches. Use the cache overrides above to relocate them.

Validation performed (21 November 2026): a fresh Linux CPU install from these docs
alone, in a new venv with empty caches. Timings on that machine: core pip about 5 min,
the CPU torch pair about 7 min, music requirements about 5 min, `--setup speech` about
4 min. About 10 GB on disk in all. A 3:41 song took about 13 minutes for its first full
listen, peaking at 5.3 GB RAM; a 30-second voice note took about 2.5 minutes.

**Running the tests:** `python -m unittest discover -p "test_*.py"` from this folder
(about 1,000 tests, several minutes on a CPU). They pin their own environment, so
they pass with or without the CPU profile set. On a Windows-mounted drive (WSL
`/mnt/c`, `/mnt/d`), the voice-book permission checks are skipped with a reason,
because that filesystem can't make files private.
