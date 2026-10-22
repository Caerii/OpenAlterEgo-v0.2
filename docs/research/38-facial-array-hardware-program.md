# Contact-level hardware programs and the facial-array validation target

The seven-person [Hyser calibration result](37-hyser-fewshot-heldout-result.md)
is strong evidence for a **bounded labeled adaptation method on forearm
gestures**. It leaves two central wearable questions open: whether the
method transfers to facial silent speech, and whether a proposed layout is
physically implementable and predictably better than another layout.

## The closest public facial-array test

The [CSL-EMG_Array paper](https://www.csl.uni-bremen.de/cms/images/documents/publications/Diener_IS2020_CSLEMGArray.pdf)
describes a 4×8 electrode array on the left cheek and an eight-contact strip
below the chin. The electrodes use **chained differential derivations**:
a recorded channel is the voltage difference between a specified pair of
contacts. The data include audible and silent speech, 12 sessions from
eight speakers, and ordered recording blocks. Initial training is followed
by adaptation blocks 2, 4, and 6 and later evaluation blocks 3, 5, and 7.
That order directly addresses the chronology problem in Hyser's
class-grouped repetitions. The official [corpus page](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
says the CC BY 4.0 download link is sent after email entry. We have not
downloaded or inspected its files, so no numerical CSL result is claimed.

Its paper also reports that a session-dependent EMG-to-speech baseline can
degrade on later blocks and explores running EMG normalization. Our first
CSL evaluation should therefore hold the utterance split and physical
channel mapping fixed, fit on the initial block, allow only the prescribed
adaptation blocks, and score later blocks in recording order. Silent blocks
have no direct acoustic reference; text recognition and audio reconstruction
must use distinct endpoints and respect the paper's reference limitations.

## A program must count contacts and channels separately

For differential channels, a program is a set of ordered contact pairs.
Eight consecutive channels on a chain use nine active contacts, while eight
disjoint pairs use sixteen. Other topologies have different counts. Either
program also consumes eight
analog front-end channels. The new
[`DifferentialProgram`](../../software/python/openalterego/research/differential_hardware.py)
validates both budgets and exposes the incidence matrix `B`, where recorded
channel voltages are `B` times contact potentials. It does **not** assume
that the array's physical contacts map one-to-one to model features.

An idealized observation model is

```text
sources → uncertain lead field G → contact potentials + contact artifacts
        → differential map B → front-end artifacts → measured channels
```

Shared contact artifacts produce **correlated** differential-channel noise.
The implementation computes that covariance as
`B diag(contact variance) Bᵀ + diag(front-end variance)`. It also exposes a
linear-Gaussian information score for a specified source covariance and
lead field. This score can help compare *hypotheses* under fixed assumptions;
it is not validated facial anatomy or proof that a layout improves speech
decoding. The correct CSL contact-pair map must be taken from its actual
metadata before using those recordings in a layout search.

## What would count as a genuine design advantage

Search should optimize **future-block decoding or reconstruction after a
declared calibration budget**, with the physical contact count, front-end
count, placement region, and latency held feasible. The baseline is a fixed
implementable layout. A model may abstain and choose that layout when its
predicted advantage is uncertain. Selection and uncertainty intervals must
be recorded before later evaluation blocks are opened. Report realized
paired gains, ranking agreement, interval coverage, and decision regret.

The simulator becomes an engineering instrument when it predicts the
*effect of changing a feasible contact program* on data it has not used for
fit or selection. Signal resemblance alone cannot establish that. With the
current public data, the next practical step is a chronologically valid
facial-array benchmark; testing an unrecorded electrode placement or
reseating intervention still requires future physical measurements.
