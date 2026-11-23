# Third-party notices

Project code is MIT (LICENSE). Separately downloaded software, model weights,
training data and user-supplied audio/text retain their own terms. No weights,
training corpora or service-returned words are distributed in this snapshot.
This inventory covers all download paths selected by the shipped engines;
transitive dependency licenses remain in the installed distributions.

## AI_Ears

Derived from [Ace's AI_Ears](https://github.com/menelly/AI_Ears), MIT. Its notice
is retained here for derived portions. The upstream
[license](https://raw.githubusercontent.com/menelly/AI_Ears/main/LICENSE) states:

Copyright (c) 2026 Ace (ace@sentientsystems.live)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Models and datasets

| Downloaded asset / use | Attribution and terms | Primary source |
|---|---|---|
| Whisper large-v3 converted for faster-whisper; speech words and sung words | OpenAI Whisper / SYSTRAN conversion, MIT model and code. No training audio is downloaded by these engines. | [model card](https://huggingface.co/Systran/faster-whisper-large-v3), [Whisper license](https://github.com/openai/whisper/blob/main/LICENSE) |
| PANNs `Cnn14_mAP=0.431.pth`; events, breathing, instruments and laughs | Qiuqiang Kong and collaborators; PANNs code MIT. Weight license needs direct confirmation: the accessible Zenodo record exposed no readable license value during this build. No weight redistribution is included. Treat this as unresolved before bundling weights. | [weight record](https://zenodo.org/records/3987831), [code license](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/LICENSE.MIT) |
| AudioSet `class_labels_indices.csv`; class names downloaded by setup/library import | Google AudioSet dataset/annotations: CC BY 4.0. The complete ontology is separately CC BY-SA 4.0. These engines download class labels, not the AudioSet audio corpus. | [AudioSet terms](https://research.google.com/audioset/download.html) |
| **AFFECT** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | **audEERING GmbH; CC BY-NC-SA 4.0, non-commercial, attribution and share-alike; model card says research purposes only.** Fine-tuned on MSP-Podcast v1.7; training corpus is not downloaded and has separate access terms. MIT project licensing does not grant commercial rights to this model. Disable with speech/song `--no-affect` or `RAVEN_EARS_NO_AFFECT=1` (also honoured by setup); offline mode uses only a previously cached copy. | [model card and terms](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| Demucs `htdemucs_ft`, fallback `htdemucs`, and `htdemucs_6s`; separated stems | Meta Platforms and affiliates / Demucs contributors; MIT upstream project. Training uses MUSDB and additional songs; training audio is not downloaded. Six-stem piano output may contain bleed. | [Demucs license](https://github.com/facebookresearch/demucs/blob/main/LICENSE), [models](https://github.com/facebookresearch/demucs#pre-trained-models) |
| Beat This! default `final0`; beats and downbeats | CPJKU / Beat This! authors; code and published weights MIT. Upstream notes that training recordings have mixed restrictions; training recordings are not downloaded by inference. | [license statement](https://github.com/CPJKU/beat_this#license) |
| madmom CNN key ensemble and optional RNN downbeat models | Department of Computational Perception / madmom contributors; code BSD-2-Clause. **All included model/data files CC BY-NC-SA 4.0: non-commercial, attribution, share-alike.** Separate environment does not remove these terms. Source docs record revision `27f032e`. | [madmom license](https://github.com/CPJKU/madmom/blob/main/LICENSE), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `cardiffnlp/twitter-roberta-base-sentiment-latest`; optional local word/music contrast | Cardiff NLP; model CC BY 4.0 (attribution). Trained on tweets and TweetEval; no training corpus is downloaded. An unavailable model omits the contrast result with a recorded failure. | [model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Optional Omine laughter segmentation; `model.safetensors` | Taisei Omine, Kenta Akita and Reiji Tsuruno. Code: MIT (2024). Published weights: research use only, separately restricted by the upstream README. | [code license](https://github.com/omine-me/LaughterSegmentation/blob/main/LICENSE), [weight terms](https://github.com/omine-me/LaughterSegmentation#license), [weights](https://huggingface.co/omine-me/LaughterSegmentation) |
| Optional Gillick laughter detection; `resnet_with_augmentation` checkpoint | Jon Gillick and collaborators. Code: MIT (2018). Weights are shipped in that repository; a separate weight-specific grant is **UNCONFIRMED**. The training corpus has separate terms. | [code license](https://github.com/jrgillick/laughter-detection/blob/master/LICENSE), [checkpoint and training provenance](https://github.com/jrgillick/laughter-detection#overview) |
| Optional Resemblyzer voice encoder; voice-book embeddings | Resemble AI. Code: Apache 2.0. The pretrained encoder is distributed with the package; a separate weight-specific grant is **UNCONFIRMED**. | [code license](https://github.com/resemble-ai/Resemblyzer/blob/master/LICENSE), [encoder provenance](https://github.com/resemble-ai/Resemblyzer#additional-info) |

The three optional entries above were checked against upstream on 23 November
2026. Their README pages are undated; the code license dates are stated above
where supplied. Repository code terms are not a separately confirmed weight licence.

Fetched text is separate from downloaded models: MusicBrainz and Discogs metadata
remain subject to their service/data terms; LRCLIB results do not transfer rights
to song words. This release ships only invented neutral text fixtures, never
retrieved song text. See PRIVACY.md for request and cache behavior.

Optional song extras use [AcoustID's lookup service](https://acoustid.org/webservice),
[Last.fm's track tags API](https://www.last.fm/api/show/track.getTopTags) and the
[official Genius API](https://docs.genius.com/). Their returned metadata and
contributor text retain their service/data terms; no fetched text ships in this
release. AcoustID's free service is for non-commercial use (service documentation,
undated, checked 16 Nov 2026). Optional `fpcalc` is a separately installed
[Chromaprint](https://acoustid.org/chromaprint) tool, not bundled here.
Genius excerpts are trimmed and attributed; no lyrics pages are scraped and no
full lyrics are retrieved. API accounts and tokens do not transfer rights to text.

## Other acknowledgments and dependencies

Optional listening sheets use Matplotlib (PSF-based Matplotlib license) and its
bundled DejaVu Sans font. Their license and font notices remain in the installed
Matplotlib distribution; neither library nor font files are vendored here.

Saved-card listening modes retain source attribution to seven-ears by Seven and
Sunny (MIT), described by the source documentation as an independent implementation
of the idea. No upstream seven-ears package or history is included; that attribution
and its exact upstream provenance have not been independently verified.

Optional Parselmouth/Praat is GPL-3.0-or-later; its presence does not make the
project's MIT grant a license to redistribute that dependency under MIT. FFmpeg's
license depends on its build options (LGPL/GPL). NumPy/SciPy/librosa/soundfile,
Torch/torchaudio, Transformers/Hugging Face Hub, faster-whisper/CTranslate2,
panns_inference, Demucs, Beat This!, mutagen, mir_eval and their transitive
packages keep the notices delivered in their wheels/source distributions.

The neutral audio fixture and generator are original mathematical tone synthesis,
covered by the project MIT license. They contain no real voice, speech words,
lyrics, pretrained model output or third-party recording.
