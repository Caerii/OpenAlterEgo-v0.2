# EMG-UKA session-eight virtual-program lock

This assay uses the freely downloadable [EMG-UKA trial corpus](https://www.kaggle.com/datasets/xabierdezuazo/emguka-trial-corpus),
described by [Wand, Janke and Schultz](https://www.isca-archive.org/interspeech_2014/wand14b_interspeech.pdf).
The local archive hash and subset/document hashes are in the
[source manifest](data-sources/emg-uka-trial-kaggle.json). Kaggle reports the
mirror's license as **Unknown**, so the raw corpus stays outside Git.

The publisher's documentation specifies six facial EMG channels plus a
synchronization marker, interleaved 16-bit little-endian samples at 600 Hz.
Speaker 008 has eight audible recording sessions. Each session's official
test subset contains the same ten prompts, recorded on that day; their
utterance IDs can change. We match by transcript text. These are
**closed-prompt audible EMG** data, not unrestricted or silent speech.

## Frozen method before scoring session 008

The [runner](../../software/python/scripts/emg_uka_session_program.py)
checks the publisher subset hash, reads only sessions 001–007 in selection
mode, crops each ADC file using its EMG offsets, excludes the marker, and
extracts per-channel log RMS and log mean absolute adjacent difference in
five relative-time bins after per-utterance centering. The classifier is a
train-fold-standardized ridge model (`alpha=10`) for the ten prompts.

All 20 three-of-six recorded-channel programs are scored by
leave-one-session-out accuracy over sessions 001–007. Selection maximizes
mean accuracy, breaking ties by minimum fold accuracy and then channel
indices. The frozen [selection record](results/emg-uka-session-008-selection-v1.json)
chooses **zero-based channels (1, 3, 5)**. Its seven-fold mean is 0.83
and its minimum is 0.60. Fixed (0, 1, 2) and seed-42 random (0, 1, 3)
are the same-cost controls. All six recorded channels are a costlier
reference. Every candidate and its seven fold accuracies are retained.

Next, fit the same classifier on sessions 001–007 and score the **ten**
session-008 prompts exactly once for each frozen program. Report correct
counts and per-prompt paired outcomes for all four choices, including any
losses. The primary comparison is selected versus fixed three channels;
random three is a search-effort control, and all six measures the cost
tradeoff. A selected gain must be at least one additional correct prompt
to be a positive observed result. Ten prompts make any such estimate
imprecise; it cannot by itself validate a physical montage or generalize
to new people.

Before this lock, we inspected the shape, offset format and transcript of
one session-008 file to verify corpus parsing. We did **not** compute its
features or recognition outcomes. The complete public corpus and its
sessions are available to the researcher, so this is a locally frozen
comparison rather than independent hidden evaluation.
