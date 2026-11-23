# first-listen-song.json

A saved-song bundle (schema 1) derived from one real listen. Every lyric word,
in card text, lyric rows and quoted excerpts, was replaced with an invented
placeholder word of the same length; timestamps, line counts, the approximate
and (unsure...) markers and all non-lyric measurements are unchanged. File
names and paths were neutralised, and JSON the first-listen view does not read
was dropped. `test_listen.py` copies it into a disposable home and renders it.

# uhuru-title-song.json

A trimmed saved-song bundle (schema 1) with the structure of one real listen of a
song whose chorus lines all repeat equally and whose title word sits in one chorus
line. Every lyric word, in the lyric rows, the timed words and the card text, was
replaced with a consistent invented word of the same length (so repeats still
repeat); the title word "Uhuru" and bracketed section labels were kept. The
audio file name and the lyric-bearing card sections (timelines, quoted lines, notes)
were removed or neutralised. The pitch track is invented: only which frames are voiced
was kept; the tune (C minor, G3 to F4) was made up, with the title word on the highest
note of its line. The singer card lines that summarised the real contour (pitch arc,
highest and lowest held notes, phrase endings) were dropped.
The saved melody record was dropped: `test_listen.py` derives it with
`listen_melody.build`, then checks the title-based hook and renders the default
view with every engine blocked.
