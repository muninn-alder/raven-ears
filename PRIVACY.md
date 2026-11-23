# Network and retention

Audio decoding, separation, transcription and inference run locally. No engine
uploads audio for remote inference. By default nothing uploads supplied text,
transcripts or analysis results either. Text readings are opt-in: if you set
`TYPESAFE_API_KEY`, speech transcripts, lyrics and song filenames can go to TypeSafe's Jev (see the table and
`JEV_READS.md`). The default song helpers can send identifying metadata over HTTPS:

| Service | What is sent | When |
|---|---|---|
| MusicBrainz (`musicbrainz.org/ws/2/`) | artist and cleaned title in a recording-search query; candidate recording/release IDs and relation requests | automatic credits lookup; positive cached answers may avoid requests |
| Discogs (`api.discogs.com`) | artist and title in release search, then candidate release IDs; authorization token in the request header | credits fallback when MusicBrainz is insufficient and `DISCOGS_TOKEN` is configured |
| LRCLIB (`lrclib.net/api/get`, `/api/search`) | artist and title; exact lookup also sends duration when known and album only if explicitly supplied; search sends artist + title | automatic song word lookup unless supplied words bypass it |
| AcoustID (`api.acoustid.org/v2/lookup`) | **opt-in only.** Locally computed Chromaprint fingerprint and whole-file duration; application key (`client`), `meta=recordings` and JSON format in an HTTPS POST body. Never audio, filename, tags or paths | song commands only, with `ACOUSTID_API_KEY`, online mode and optional local `fpcalc`; only when artist and title are both absent from flags, tags and split filename; `--no-acoustid` disables |
| Last.fm (`ws.audioscrobbler.com/2.0/`) | **opt-in only.** Artist, title (`track`), API key, `method=track.getTopTags`, JSON format and `autocorrect=0` in the query. No audio, lyrics, user listening history, account name or paths | song commands only, with `LASTFM_API_KEY` and online mode; `--no-lastfm` disables; never feeds detector decisions |
| Genius (`api.genius.com/search`, `/songs/:id`, `/referents`) | **opt-in only.** Artist + title search query, matched song ID, `song_id`, `text_format=plain`, page and page-size controls; access token in the Authorization header. No audio, full lyrics, card text, measurements or paths | only `listen.py song.mp3 --lore` after printing/flushing an already saved card, with `GENIUS_ACCESS_TOKEN` and online mode; never during analysis or ordinary saved reads |
| TypeSafe Jev (`api.typesafe.ai/v1/systemone`) | **opt-in only.** Lyric lines with two lines before/after and song title for CONTRAST/LONGING; speech phrases with two phrases before/after for stated feeling, longing, contradictions in the wording, joking and explicit self-identification; writer/Whisper text pairs for MISHEARD; song basename and candidate artist/title splits derived from it for filename reading. Requests include question wording/criteria and the key in the Authorization header. Never audio, voiceprints, acoustic measurements, speaker history, enrolled-name lists or directory paths | analysis only, with `TYPESAFE_API_KEY` and online mode; the filename reading fills missing metadata when flags/tags are incomplete; MISHEARD reads only non-approximate writer-file pairs; saved views make no requests |

Artist/title can come from CLI flags, file tags or an `Artist - Title` filename,
or from an accepted optional AcoustID match on a mystery song. Fingerprint-derived
names then follow the existing credits/lyrics metadata path, including its requests.
The fingerprint, audio bytes and API keys are not stored in the optional response
cache or the saved card. Accepted artist/title, score, recording ID and
linked-recording agreement counts are saved; Last.fm's filtered tags are saved
separately from detector results.
AcoustID, Last.fm and Genius are never called by the speech engine.
The optional cache is `RAVEN_EARS_HOME/song-extras` (under the default ears home
when unset). Positive fingerprint identities have no expiry; filtered tags and
trimmed lore are reused for seven days, then fetched again (stale files remain
until replaced or removed). Only the reduced lore is cached: at most
1000 description characters, three annotations of 500 characters each, and
80-character referent fragments with contributor attribution. Lore is never
written into `song.json` or any card. Removing `song-extras` clears these caches;
the usual saved-song deletion leaves metadata caches intact. Failed/empty answers
are not cached. Service requests are spaced across processes; HTTP 429 persists
a cooldown of at least 60 seconds and respects a longer `Retry-After` (seconds or
HTTP date). Last.fm's JSON rate-limit error also starts a 60-second cooldown.
With Jev enabled, a confident reading of the filename can fill missing artist/title
at credits and lyric lookup boundaries. Local title detection alone makes no Jev
filename request. Explicit flags and tags always win. The card marks
this `(artist/title read from the file name by Jev)`. A process reuses its filename
reading; a fresh process can ask again. No key or offline mode adds none of these
new readings or requests. Existing offline CONTRAST skip notes remain unchanged.
Speech sends supplied `--words` text or locally transcribed words, cut into phrases
at punctuation, pauses or 32 words (first 64 phrases, with a note if capped).
The self-identification questions name only candidate spans found in those words.
Local enrolled names and `RAVEN_EARS_HOME/names.txt` help filter candidates; the
rosters stay local. Unknown names need a capital and must clear a small built-in
common-word list.
Lyrics can be supplied, fetched or locally transcribed. Text inside a song or note
is not necessarily the singer's or speaker's own claim; these are model readings.
Protect output JSON as transcript/lyric text: it saves exact requests, raw answers
and interpreted probabilities. Speech keeps these in its returned/CLI JSON, not
the numeric baseline; listen-once keeps music readings in `song.json`. Neither
the API key nor voice identity inferred from words is written into a baseline.
MusicBrainz may be queried again without a version marker when no match is found.
Requests also reveal the client's IP address and ordinary HTTP headers, including
a generic application User-Agent. Discogs can read a token from the explicit
`RAVEN_DISCOGS_ENV` file; no implicit personal credential file is read.

Disable credits with `--no-credits` or `RAVEN_NO_CREDITS=1`. Disable LRCLIB with
`--no-lyrics-lookup` or `RAVEN_NO_LYRICS_LOOKUP=1`. They are independent.
`--no-lyrics` in music/harmony skips the word stage, including LRCLIB and Whisper.
`--lyrics FILE` supplies local text and bypasses LRCLIB; `--credits FILE` replaces
automatic credit lookup. Explicit `--no-credits` takes precedence over a file.

For a shared offline setting, set **`RAVEN_EARS_OFFLINE=1` before launching**.
It blocks all metadata and lore services (and Jev) even with a warm in-process credits cache,
and disables setup/downloads. `--setup TARGET --check` remains available without
downloading. Whisper and AFFECT request only cached local files.
PANNs tagging/laughs run with a complete local checkpoint in `RAVEN_PANNS_CACHE`
and the upstream labels file at `~/panns_data/class_labels_indices.csv`. If that
separate labels copy is absent, cached PANNs weights are not used offline because
importing the library would download labels (the card says so). Setup now validates
and prepares both labels copies; it does not bypass this offline guard. Beat This! runs
with `checkpoints/beat_this-final0.ckpt` in the Torch hub cache (`TORCH_HOME/hub`,
normally `~/.cache/torch/hub`); it receives an open local checkpoint and does not
try another backend if loading or inference fails offline. Fresh Demucs stem
separation uses cached weights from Torch or Hugging Face, and optional text
sentiment uses the cached Hugging Face model and tokenizer. Existing
stem/features/downbeat caches remain usable. Missing stages stay visibly unavailable. Saved-card views work offline;
`--lore` prints the saved card and skips Genius offline. External song
resolvers and external audio launchers are blocked because they cannot enforce
this policy. This setting does not control independent programs you launch yourself.

Opt-in speech separation reuses matching local stems or loads Demucs weights from
the local model cache. Fresh speech stems live in an owner-only temporary directory
and are deleted after the analysis. A successful separated analysis skips the
mixed-audio speaker baseline.

Without offline mode, model first-use/setup downloads contact Hugging Face
(including its storage/CDN endpoints), Zenodo, Google storage (AudioSet labels),
Meta model hosting and the Beat This! model host. Model identifiers and ordinary
HTTP connection information are sent, not the audio being analysed. Third-party
library endpoints may change. Hugging Face uses its normal configured cache;
`HF_HOME` can relocate it. Torch uses `TORCH_HOME`.

During Beat This! setup or first use, if the official checkpoint server fails
(including a size or checksum mismatch), the downloader tries Hugging Face's
`oulianov/beat_this`, then `teamup-tech/Beat-This-weights` mirrors. Each request
is a plain GET of a public `final0.ckpt` file, sending ordinary HTTP connection
information (including IP address and headers), never audio or song metadata.
All sources must match the pinned size and SHA256 before use. A verified cached
checkpoint or offline mode makes no mirror request; `--setup music --check`
only verifies local files.

`--setup fast` downloads PANNs, labels and the effective fast speech Whisper;
it does not fetch AFFECT. `--setup speech` and the default `--setup` also fetch
AFFECT, using the effective ordinary speech Whisper. `--setup music` fetches
PANNs/labels, AFFECT, lyrics' large-v3, Beat This! and the RoBERTa lyric sentiment
weights and tokenizer. These commands send model identifiers and keep the audio
local. Other song models can download separately on first use.

Local storage is plaintext and has no automatic deletion:

| Default | Contents | Override |
|---|---|---|
| `~/.cache/raven-lyrics` | transcribed words and timestamps keyed by audio hash and recognition hints | `RAVEN_LYRICS_CACHE` |
| `~/.cache/raven-stems` | reconstructable separated audio, four/six-stem variants | `RAVEN_STEMS_CACHE` |
| `~/.cache/harmony` | feature/pitch/downbeat/key caches; names include input basenames | `RAVEN_HARMONY_CACHE` |
| `~/.cache/raven-panns` | PANNs checkpoint and label CSV | `RAVEN_PANNS_CACHE` |
| `~/panns_data` | upstream panns_inference uses a separate labels copy on import; setup prepares it | upstream library behavior; this copy must already exist for offline tagging |
| `~/.local/share/raven-ears` | numeric per-speaker baseline history, names hints, credits and LRCLIB cache subdirectories; `song-extras` holds positive fingerprint identities, filtered tags and trimmed Genius lore, plus service rate-limit timestamps | `RAVEN_EARS_HOME` |
| `~/.local/share/raven-ears/voices/NAME.json` | private local voiceprint: mean embedding, clip count and retained enrolment seconds only; mode 600; never sent anywhere | `RAVEN_EARS_HOME/voices`; delete with `python raven_ears_plus.py --forget NAME` (separate baseline remains) |
| `~/.cache/raven/cards` | saved cards, queue results and failed-run diagnostics | `RAVEN_CARDS_DIR` (queue), `RAVEN_CARDS_HOME` (wrapper) |
| `~/.cache/raven/cards/songs/<sha256>/song.json` | listen-once bundle: three card texts, lyrics, analyses, diagnostics, input hashes and manifest | `RAVEN_EARS_HOME/songs` when set; otherwise `RAVEN_CARDS_HOME/songs` or `RAVEN_CARDS_DIR/songs` |
| `~/.cache/raven/listens` | editable listening sessions and notes | `RAVEN_LISTENS_HOME` or `--state-home`; if set, `RAVEN_EARS_HOME/listens` |
| `~/.local/state/raven-ears/queue` | queue, checkpoints and run records | `RAVEN_QUEUE_DIR`, or `XDG_STATE_HOME` parent |

Delete the relevant directories to remove retained data; stop running engines
first. Supplied text and `--keep-stems` copies remain where you put them. The home
setting does not relocate every cache. Engine JSON/card outputs and shell redirects
can also retain text, filenames and identifiers; choose their destinations yourself.
No personal configuration or populated caches ship in this release.

Listen-once automatically retains one plaintext `song.json` per audio hash. It
holds all three full card texts (including supplied/retrieved lyrics, timed rows
and retained transcription/alignment evidence), available captured analyses,
complete stderr diagnostics and failure reasons. The manifest holds audio SHA-256,
duration/reason, code fingerprint, requested options, artist/title/album flags,
possibly the metadata basename, hashes of supplied lyrics/names/credits, selected
output-affecting environment/model values and lookup settings. Each pass records
its status/reason, date, options actually forwarded (text-file hashes rather than
paths; stem destination basename), first/last listened dates and count. Counts
cover completed pass attempts, including failed outcomes and completed passes
from an interrupted refresh; reads and passes that never started do not count.
A refresh updates these counters while keeping the previous cards/statuses until
all replacement passes finish. An interrupted pass without a returned outcome
is not counted. Status/reason/date describe the saved card, which can precede
the last completed refresh attempt. Older bundles without this record show
unknown dates/counts rather than invented history. `--history` reads this record; there is no separate history
file. `--names` and `--no-lyrics` reach music/harmony only; singer can still use
its default hints and resolve/transcribe words. Other flag scopes are in `--help`.

Absolute paths in captured analysis values (including singer's `file` and
`stems_dir`) are reduced to basenames before saving. This is not a path-free
bundle guarantee: full card text, diagnostics, failure reasons and environment
model identifiers may contain absolute audio, supplied-text, dependency or cache
paths. Supplied hint names, credited performer names and matched metadata can
appear verbatim in cards, analyses and diagnostics.

For listen-once bundles, `python listen.py song.mp3 --delete` removes that song's
validated bundle and its listening record; `python listen.py --delete-all`
removes validated bundles in direct hash directories. Symlinks and unrelated or
invalid files are left intact; directories are removed only when empty. Both
leave the separate stems, lyrics, features and metadata caches described above
intact. Interrupted refreshes leave the previous published bundle readable.
`RAVEN_EARS_NO_BASELINE=1 python listen.py ...` instead uses a disposable home
under the checkout, redirects engine/model/temp caches there and removes it on
exit, ignoring persistent home/cache overrides. It cannot reuse persistent songs;
`--keep-stems` is rejected in this mode. Shell redirects remain your responsibility.
