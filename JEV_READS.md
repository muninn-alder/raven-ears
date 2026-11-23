# Jev readings

Jev is TypeSafe's text model. Its optional CONTRAST reading runs only when
`TYPESAFE_API_KEY` is set. With no key, cards keep their existing output.
`RAVEN_EARS_OFFLINE=1` skips Jev and records the skip in NOTES and stage status.

Each request sends one lyric line, its two preceding and two following lines,
and the song title to `api.typesafe.ai`. It never sends audio. NOTES reports
the number of lines sent/attempted, including failed requests whose receipt
cannot be confirmed. Repeated lines get independent requests with their own
context. Up to eight requests run concurrently, with a 20-second timeout each.

Questions, criteria, thresholds and HTTP transport live in `jev_reads.py`.
The stated-feeling score maps from 0–4 to −1..+1. Noul probabilities of at
least 0.7 mean yes; at most 0.3 mean no. The band between is **not settled**,
stored in JSON but never shown as a finding. These thresholds are
initial choices, not calibrated accuracy.

The third question, `longing`, is a separate Noul judgment in the same per-line
request, with the same state as stated feeling and masking; no extra requests:

```python
"longing": {
    "type": "noul",
    "instructions": "In `line`, does the singer say they want someone or something they do not have yet, or no longer have?",
    "criteria": {
        "true": "The singer wants a person, a life or a moment that is out of reach right now.",
        "false": "The singer describes what they already have, or the line states no want.",
    },
},
```

CONTRAST keeps roberta's words, music lift and existing flags. It adds Jev's
reading beside them, shows masking candidates, and shows opposite-sign text
readings differing by more than 0.6 without choosing a winner. Text models
cannot establish the singer's intent. Compact views select the strongest
examples; all readings remain in JSON.

For each extra Noul, when any line is yes, CONTRAST adds one summary with the
number of qualifying lines, total lines and earliest qualifying example. Both counts use
all timed line occurrences, including chorus repeats, rather than distinct
texts. Unavailable lines and older readings without that answer stay in the total
but do not qualify. The example is shortened to keep the line within 118
characters for ordinary labels. No qualifying lines means no summary. Summaries
follow the explanation line in `QUESTIONS` order.

An extra Noul alone never adds a timed CONTRAST row. On a row already shown for an
existing flag, masking or disagreement, it appends `<label> (0.77)` to the reason.
The compact default view keeps these tags on its selected row; summaries use
only spare space in the existing headline budget, after other selections have
been placed, in question order. Otherwise summaries remain on the full card.

## Add your own question

Adding an entry to `QUESTIONS` in `jev_reads.py` is the whole job. For example:

```python
"mentions_a_place": {
    "type": "noul",
    "instructions": "Does `line` name a place?",
    "criteria": {"true": "Names a place.", "false": "No place named."},
},
```

This displays `Jev hears mentions a place in N of M lines (first <time> “<line>”)`
and `mentions a place (0.80)` tags using the rules above, just like `longing`.
Optional `"label": "places"` and `"verb": "finds"` change the summary to
`Jev finds places in …` and tags to `places (0.80)`. Defaults are the key with
underscores replaced by spaces, and `hears`. Keep labels and verbs short for
compact cards. These display settings stay local: payloads send only `type`,
`instructions` and `criteria` from each entry.

There are no extra requests: every question goes in the same request per line.
The question wording and criteria are sent to TypeSafe along with the lyric text
and context.
Extra `score` questions are also asked and saved, but only the built-in
`stated_feeling` Score has a display; extra Scores do not render yet.

Exact request payloads, raw answers and interpreted values stay on each
contrast row in the saved song bundle. Rendering saved views makes no new
calls; use a fresh listen (`--fresh`) to ask again after adding a question.
Saved views with CONTRAST JSON use the current question entries: answers for
removed questions are ignored, and new questions without saved answers stay
absent. An answer saved for different wording of a question (same name, edited
instructions or criteria) also stays absent until a fresh listen asks again. Rendering does not rewrite the bundle. Removing the key does not
erase readings already saved. Protect bundles as lyric text, including their
context and title. The key is never saved.

Timeouts, HTTP errors and malformed or missing answers are errors, not answers.
Affected lines are unavailable, with one failure-count/reason note and an
explicit stage outcome. Other lines keep their readings. A failure is never
converted into zero, no, or not settled.
New responses must answer every question in `QUESTIONS`: any missing or malformed
answer makes the whole line unavailable. Older saved readings without an answer still
render; the missing judgment stays absent, rather than becoming a no.

## Speech and the other music witnesses

`jev_more.py` contains the new question wording and conservative display rules.
It calls `jev_reads.read_tasks`, reusing the transport, bounded concurrency,
strict answer validation and error policy above. It introduces no new client or
dependency. `QUESTIONS` and its extensible CONTRAST summaries are unchanged.

These readings run only with a key and online mode. Speech shows one WORDS vs
VOICE status line when Jev is off or any phrase request fails. All new Nouls
require .85 to print yes, or .15 or below to print the opposite MISHEARD
reading. This is an initial quiet threshold, not measured
accuracy. Filename choices and competing identity claims also require a .15 lead
over the runner-up; ties remain silent. Errors remain unavailable, never negative votes.

Speech uses TUNE's word runs, split at gaps of at least half a second. Each
phrase gets one request with up to two phrases of context on either side.
Jev scores the speaker's stated feeling from -1 to +1 and checks for explicit
longing and contradictions between feeling statements. The request also asks
about explicit claims of feeling fine, excited or tired/low, and playful joking.
Candidate names from self-introductions and sign-offs get separate
self-identification questions.
Candidate extraction accepts an enrolled voice-book name, a name from
`RAVEN_EARS_HOME/names.txt` (case-insensitive), or a capitalised token outside a
small built-in common-word/verb/adjective list. Explicit introductions and
sign-offs still qualify; ordinary predicates such as "I'm on headphones" and
"I'm committing it" do not. Only the candidate span goes into its question;
the model receives no name roster or acoustic information.

WORDS vs VOICE shows up to three contrasting phrases. It compares text scores
of magnitude 0.50 or more with opposite vocal valence at 0.40 or below, or
0.60 or above. A negative text score can also be flagged when there is a laugh
detection and at least two raised measures. The cutoffs are pitch 3 semitones
above the note median, pace 25% above the median phrase pace, loudness 4 dB above
the note's active-frame 90th percentile, and arousal 0.60.
This second rule applies when vocal valence is neutral or unavailable. A
summary appears if a majority of all phrases have matching directions.
Affect uses the existing audeering windows with at least one second and half
the window inside the phrase. Pitch, pace, loudness, TUNE endings and laugh
sources are local measurements. Prior notes can supply pitch and range
references. Display thresholds have been exercised with synthetic input.

JOKING? appears by LAUGHS with the original phrase time. WHO from words appears by VOICE,
only if identity is missing or differs, and says both witnesses when they differ.
It never sets a speaker or changes baseline behavior. Exact requests, answers
and phrase voice measurements live under `jev_speech` in speech CLI JSON.

MISHEARD sends each non-approximate writer-file/Whisper text pair. Its question
asks whether the words sound plausibly alike when sung. The row and both texts
remain; the label is a plausible mishearing or possible changed words, explicitly
text only. Neither proves what the recording contains. Existing exclusions of
database lyrics and approximate placement are unchanged.

Filename reading proposes contiguous splits (including reversed roles) from
the basename after removing track numbers and upload noise. Jev confirms a split
instead of inventing metadata. Explicit flags and embedded tags win; only missing
metadata is filled at the shared credits/LRCLIB lookup boundaries; local title
detection alone never triggers a filename request. Recording/version
markers (including Remastered) stay. Trailing house notes containing edit, not
album, cut, copy, clip or upload are removed; recording markers such as Live,
Acoustic, Remix and Radio Edit remain. A process memoizes these filename readings
and prints the required provenance annotation when used; a fresh process may
ask again.

Hook by meaning was tried on three real songs and returned no winner on all
three. It has been removed: no hook question, request or `meaning_hook` field is
created. Old trial fields are ignored and their saved melody opinion line is
hidden. The existing hook selection, notes and timing remain.

The small live text check found one sensible JOKING hit and missed two light
teases; its quiet threshold is unchanged. MISHEARD correctly read the supplied
"ballgowns / ball gowns" pair as a plausible mishearing (0.93); this still does
not establish what was sung. WHO produced two ordinary-word false claims before
the local name filter was tightened. These observations are qualitative, not
accuracy estimates; the revised filters have not had a fresh live run.

All new music request/answer rows use the `jev` envelope, so the worker's
`analysis_basenames` scrubber preserves exact text payloads through `song.json`.
Filename candidates/answers are saved under `filename_reading`; MISHEARD evidence
stays beside its pair. The listen-once bundle is the persistent cache. Refresh
with `--fresh` to ask again. Protect these files as transcript/lyric text.
