# Neutral fixtures

`neutral-voiced.wav` is deterministic mathematical synthesis from
`generate_neutral.py`: 32 one-second cycles, each with 0.58 seconds of harmonic
tones (plus fixed 9-13.5 kHz sines for air energy) and 0.42 seconds of silence,
mono 44.1 kHz PCM, so the CHANNEL check does not flag it as narrowband. It has no spoken text or real
voice. The release energy detector measures 32 separate runs (minimum needed: 8).
Regenerate with `python fixtures/generate_neutral.py`.

`ache-analysis.json` is a synthetic chord/timing example with invented neutral
text. Its input values describe a test, not a measured song or household note.
Except for the public metadata regression fixtures below, text in test cards and mocked
HTTP payloads is invented for tests; credits use placeholder players. Hash tests
use arbitrary temporary bytes.

`discogs-original.json` contains projected public Discogs API responses fetched
on 12 November 2026 with artist/title queries for The Dave Brubeck Quartet's
"Take Five". Only fields used for release ranking, track matching and credits
remain; tracklists retain only the requested track. Pagination counts describe
the full search, whose selected rows include the original and the wrong live
release. There are no request headers, tokens or account data. The public sources
are https://api.discogs.com/masters/34081 (Time Out),
https://api.discogs.com/masters/110102 (the edited single),
https://api.discogs.com/masters/664785 (Experiments In Time - Vol. 2), and
https://api.discogs.com/masters/270654 (Concord On A Summer Night), plus each
master's `main_release`. Tests explicitly mutate copies for ambiguous cases.

`musicbrainz-performers.json` projects public MusicBrainz search and recording
responses captured on 12 November 2026. It retains the generic Take Five group
relation and the instrument/vocal relations for Strawberry Fields Forever,
Twilight of the Thunder God and Tolerate It. Expected entries are the first
pass's saved results, to check that role-bearing answers remain unchanged.
Sources (recording IDs also identify the retained search rows):
https://musicbrainz.org/ws/2/recording/3aa9460f-9c0f-4b2a-85b1-71b14c3f131f?inc=artist-rels%2Bwork-rels&fmt=json,
https://musicbrainz.org/ws/2/recording/6c322706-aa5d-4c18-b43d-13a5e8deb0ac?inc=artist-rels%2Bwork-rels&fmt=json,
https://musicbrainz.org/ws/2/recording/40681ecd-21c6-4ae8-9de2-2f693bcf8e67?inc=artist-rels%2Bwork-rels&fmt=json,
https://musicbrainz.org/ws/2/recording/6eec701e-a0a2-4532-8708-14ecc08fdb8f?inc=artist-rels%2Bwork-rels&fmt=json.
The APIs supply no publication date; this is the observation date. Tests mutate
the generic relation's artist type/modifiers and mock empty release enrichment.

The synthetic fixtures and generator use the project MIT license.
