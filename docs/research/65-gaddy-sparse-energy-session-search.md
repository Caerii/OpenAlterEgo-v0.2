# Sparse-channel search on real facial speech EMG

The [Gaddy corpus](https://zenodo.org/records/4064409) records eight
facial EMG channels during vocalized and silent speech. We used only
its **vocalized parallel** sessions because the objective here is
log acoustic energy, which requires simultaneous audio. This is a
retrospective method diagnostic: every session used below had been
exposed in earlier research, and the corpus has one recorded speaker.
Nothing here establishes silent intended prosody, a new electrode
position, or a physical contact-level design improvement.

## Assay

The [batched regression scorer](../../software/python/openalterego/research/dense_regression.py)
solves each channel subset from shared ridge sufficient statistics.
Its predictions agree with independent scikit-learn ridge fits in a
focused parity test. The [search runner](../../software/python/scripts/gaddy_channel_search.py)
uses four **past-only** frame lags of two per-channel EMG features,
ridge alpha 100, and all 28 six-of-eight recorded-channel subsets.
The source-derived Rust feature caches remain outside Git. Model
fitting uses session 5-5; selection uses 5-6 after its first ten
utterances supply unlabeled gain calibration. The chosen program
`(0, 1, 3, 4, 6, 7)` was written to an
[exclusive selection record](results/gaddy-six-channel-energy-posthoc-v1-selection.json)
before the runner loaded the 5-9 query session. This local ordering
does not undo the sessions' prior exposure in other experiments.
Each query session uses its first ten utterances for **unlabeled**
normalization and scores the remaining utterances. All models are
fitted only on 5-5. The fixed six comparison is the declared index
subset `(0, 1, 2, 4, 5, 7)`. All eight is an observed-channel reference.
The local run used the verified source manifest, the existing
`F:\Datasets\OpenAlterEgo\Gaddy\feature-cache-v1` cache, and the Rust
extension in `F:\Datasets\OpenAlterEgo\accel-wheel-parity` under the
dataset Python 3.14 environment. The runner recomputes the cache key
from manifest, source file metadata, feature code, and extension binary;
a miss recomputes features from the source files. Paths and the
first-100-utterance subset are machine-local inputs, not checked-in data.

| Query session | Selected six MAE | Fixed six MAE | All eight MAE | Selected rank among 28 six-channel programs |
| --- | ---: | ---: | ---: | ---: |
| 5-9 | 0.484 | 0.525 | 0.480 | 1 |
| 5-10 | 0.477 | 0.529 | 0.469 | 5 |
| 5-11 | 0.693 | 0.724 | 0.683 | 20 |

The [first outcome](results/gaddy-six-channel-energy-posthoc-v1.json),
[all-program rank audit](results/gaddy-six-channel-rank-posthoc-v1.json),
and [frozen later-session replay](results/gaddy-six-channel-frozen-replay-posthoc-v1.json)
retain session provenance and per-utterance or per-program evidence.
On 5-9 the selected six beat the fixed six by 0.051 mean
**utterance** MAE; an utterance bootstrap 95% interval was
0.036–0.067. All eight still beat the selected six slightly.
That interval describes this exposed session and does not convert
it into an independent population estimate.

The 5-11 failure is more instructive than the nearby gains. The
best six-channel program on that session scored 0.506, far better
than the frozen choice's 0.693. The chosen channel ranking had not
transported. A simple [unlabeled marginal-drift diagnostic](results/gaddy-channel-shift-posthoc-v1.json)
did not isolate the harmful channels: one of its largest-shift
channels was retained by the 5-11 oracle. A per-channel amplitude
or distribution alarm cannot yet be treated as a solution.

## Multi-session selection

We next scored the same 28 programs on 5-6, 5-9 and 5-10 and
chose by either mean or worst session MAE. Both choices were saved
in a [local selection record](results/gaddy-multisession-channel-posthoc-v1-selection.json)
before this runner loaded 5-11. This remains post hoc because 5-11
had been inspected in prior work and above. The **mean** objective
chose `(0, 1, 2, 3, 4, 6)` and scored **0.538** on 5-11, compared
with 0.693 for the original 5-6-only choice, 0.683 for all eight,
and 0.506 for the 5-11 oracle. The **worst** objective retained the
original choice and its 0.693 result. The
[multi-session result](results/gaddy-multisession-channel-posthoc-v1.json)
records the exact choices. A [rolling-origin replay](results/gaddy-channel-rolling-posthoc-v1.json)
shows the mean rule made the same choice as the earliest-session
rule on 5-9 and 5-10, then changed it for 5-11 after accumulating
three earlier selection sessions.

This makes a concrete next hypothesis: multiple recording sessions
can expose channel complementarity and failures that a single
selection session misses. It is still an eight-channel vocalized
energy task, with no measured contact map. A decisive extension
needs a **new session not used to shape this rule**, then the
38-channel facial-array corpus and a speech-decoding endpoint.
