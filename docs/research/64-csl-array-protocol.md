# CSL array protocol before corpus access

The [CSL-EMG_Array paper](https://www.isca-archive.org/interspeech_2020/diener20b_interspeech.pdf)
specifies **40 contacts**, in a 4×8 cheek array and an eight-contact
chin strip. Chained differential derivation yields 31 cheek and seven
chin differences, or **38 nominal EMG channels**, plus a marker signal.
The [official corpus page](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
offers the CC BY 4.0 data through an emailed download link. The archive
has not been obtained here. Channel order and any exceptions must be
checked against its metadata before treating the nominal pair map as
the recorded map.

The acquisition protocol has ordered blocks. Block 1 contains 250
training, 50 development, and 40 evaluation utterances. Block 2 has
20 training and 20 each development/evaluation. Blocks 3, 5, and 7
record later development/evaluation utterances; blocks 4 and 6 have
training utterances for adaptation. Some sentence texts recur across
blocks. A random utterance split would erase the online time structure
and could leak evaluation sentences into fitting.

The [protocol module](../../software/python/openalterego/research/csl_protocol.py)
defines a **first fixed-model, within-session assay**:

| Role | Included recordings | Purpose |
| --- | --- | --- |
| Fit | Block 1 training plus block 2 training | Learn model and calibration from early speech |
| Selection | Block 3 development | Select a feasible recorded-channel program |
| Final | Blocks 5 and 7 evaluation | Inspect later decoding only after selection freezes |

Block 0 alignment, block 1 development/evaluation, block 2
development/evaluation, and blocks 4/6 are excluded from this assay.
The protocol requires unique recording identities, one speaker and
session, complete expected block counts, correct audible/silent modes,
and disjoint sentence identities across fit, selection, and final roles.
The [tests](../../software/python/tests/test_csl_protocol.py) check
these rules with constructed metadata. This validates the *protocol
logic*, not the uninspected archive's JSON schema or a model result.

The search now enforces a physical contact limit on every partial
recorded-channel program. It also checks that a partial program can
still be completed within the limit. That second check matters:
without it, beam search can keep attractive early channels that make
all six-channel completions impossible. The exact contact-completion
bound exploits the two published differential chains and was checked
against small exhaustive cases. With a six-AFE-channel, eight-contact
budget, the [compute-only smoke artifact](results/csl-contact-search-scaling-local-v1.json)
evaluated 282 partial/full candidates and completed in a median
0.104 seconds over three warm local repetitions. The Gaussian scores
have no physiological or recognition interpretation; feasibility is
nominal until the downloaded channel map is verified.

For silent-testing sessions, the training utterances used here are
audible while selected development and final evaluation utterances are
silent. The silent utterances have transcripts but no simultaneous
spoken reference audio. Text error can be measured; direct acoustic
fidelity or prosody against a truly simultaneous silent target cannot.
The paper provides a separate audible alignment block, which requires
an explicitly different paired-audio evaluation protocol.

This split tests **within-session drift over ordered blocks**. It does
not prove transfer across separate recording sessions or electrode
reseating. A later cross-session assay must group by the actual session
and speaker IDs and reserve entire target sessions before tuning.
Likewise, searching subsets of 38 recorded differential channels is
only a virtual montage intervention. The contact incidence and AFE
budget must be checked for each candidate, and no result can establish
performance at an unrecorded contact location.
