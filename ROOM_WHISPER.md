# ROOM and WHISPER

ROOM's ambient-class candidates use the gated two-second PANNs windows from
EVENTS. These candidates need at least two above-threshold windows. Related labels
use their maximum score. The card shows up to four entries, with the strongest
run for each regular class. Data in `result["data"]["room"]` includes all retained
runs and whether regular hits cover more than 60% of windows.

The faint-dog pass uses the same model on one-second windows. Its threshold
uses the clip's own median dog score and spread, and it requires unvoiced energy
above the clip's acoustic floor. A single isolated score needs extra margin.
The card shows candidate peak times as `dog, faint`. Set
`RAVEN_EARS_ROOM_DOGS=voice-reduced` to add a pass that attenuates voiced frames
by about 15 dB. Set `RAVEN_EARS_ROOM_DOGS=off` to use only the regular pass.
The thresholds are preliminary.

A hit points to a time to listen. The scores provide no source distance.
Humming and singing can trigger Music, so ROOM suppresses a lone Music hit when
WORDLESS reports voice. An empty scan says `nothing stands out`. An unavailable
event stage uses its existing NOTES reason.

Exact names were checked against the cached 527-class AudioSet label CSV.
“Bird song”, “Footsteps”, “Telephone ringing” and “Frying” map respectively to
`Bird vocalization, bird call, bird song`, `Walk, footsteps`,
`Telephone bell ringing` and `Frying (food)`. Other full names and families are
listed in `ROOM_CLASSES`. `Tick` and `Tick-tock` cover clock ticks; generic
`Clock` alone does not establish a tick. Kettle whistle has no class and is omitted.

WHISPER uses the existing front-end active/voiced masks by either of two routes.
The word route uses existing word times. Each word
must be at least 80% active and under 20% voiced; consecutive qualifying words
must span at least 0.6 seconds. Masks must cover the words at the front end's
frame cadence; sparse or invalid supplied evidence is diagnosed, not extrapolated.
Gaps at least GAP_MIN break the run. AudioSet Whispering supports “whispered”
only with at least two windows above EVENTS' threshold, at least 60% of eligible
windows, and at least 60% of the run covered by their union in time. An eligible
window overlaps at least half the shorter of the candidate and the model window.
These are initial safeguards, not calibrated accuracy. The card reports the
supporting/eligible window counts and time coverage (“AS support 2/2, 100% time”);
overlapping windows are not independent votes. Without that support it says
“≈likely whispered”. Both labels include the measured share of voiced frames
within the candidate's words (“13% voiced”), never a claim of physical absence
of pitch. Missing event evidence is labelled “AS unavailable”. A
narrowband CHANNEL result always caps it at likely.

The second route needs no words: a stretch of active, unvoiced frames
must last at least 2.0 seconds AND pass the same Whispering window count, share
and union-in-time coverage checks above, using EVENTS' existing 0.40 threshold.
Two seconds matches one model context and excludes brief consonants and rustles;
it is an initial conservative minimum, not a calibrated whisper detector. Unvoiced
runs separated by under 0.4 s of other frames join into one stretch (a whisper
pauses between words and catches its voice now and then), which must still hold
at least 1.5 s of active, unvoiced sound. The supplied
masks must have the front end's cadence, and run boundaries never extend beyond
those masks or the audio duration. Sparse/invalid evidence is diagnosed even
when no words were transcribed. Active unvoiced sound without overlapping model
support produces no candidate from this route: fans, breath and rustle qualify
acoustically too. Missing or disabled EVENTS cannot support this route.

These spans always say “≈likely whispered” (or “≈likely half-whispered (partly voiced)”
when at least 20% of the stretch is voiced), including when words are present.
They report measured active, unvoiced seconds, the supporting Whispering window
count and peak score, and the number of overlapping transcribed words. For example:
`WHISPER: 0:02–0:14 (≈likely whispered; 11.4s unvoiced; Whispering 5 windows, peak 0.83; 1 word transcribed)`
That count does not establish how many words were spoken or why transcription
missed them. Model scores cannot distinguish speakers or establish a whisper.
Overlapping word-route spans merge into this span; the enlarged interval must
still pass all model support checks. If it fails, only the original word-route
spans remain. Distinct stretches remain separate. All evidence remains in JSON;
`unvoiced_s` identifies spans accepted through the new route. Inline markers
bracket overlapping transcribed words only; a span with none needs no markers.

Music/television hits, and the listed continuous maskers heard throughout,
withhold candidates without sufficient model support only when their timed
ROOM spans overlap the candidate. This does not establish foreground/background
placement. Model-supported narrowband candidates remain likely even with a
masker. Withheld times and masker names remain in JSON. Nonfinite or out-of-range
event probabilities fail the event stage with a diagnostic; they never silently
become a negative ROOM scan or support a whisper.

The card chooses at most three longest spans, then displays them in time order;
the 118-character budget can reduce this further. All candidates, supporting
scores and reasons remain in `result["data"]["whisper"]`. Inline WORDS markers
use the same token ordering as PEAK and retain existing emphasis/events. Every
accepted whisper span is removed from the unvoiced-sound total, including spans
omitted from the compact line. Word times and pitch errors can cause false
candidates; the model cannot prove which speaker whispered. Neither line
describes intent, emotion or why the voice changed.

Set `RAVEN_EARS_ROOM=0` or `RAVEN_EARS_WHISPER=0` to disable a new line and its
inline markers; NOTES records the switch. `--no-events` also disables ROOM and
leaves only the word route's frame evidence available to WHISPER. This switch does not disable
the faster-whisper transcription engine (`--no-whisper` controls that engine).

Tests use hand-built probability arrays, mocked tagging and invented word times.
The fast suite uses supplied frame masks. Running `test_room_whisper.py` directly
also checks the real front end on resonated synthetic noise and a voiced vowel.
No recorded voices, songs, model inference or network are used by these tests.

Unworded whispers came in on 10 Nov 2026, after Fox's two whisper notes: the first
build had no bridging and missed the quiet one. Whispering is a spectrum (Fox).
