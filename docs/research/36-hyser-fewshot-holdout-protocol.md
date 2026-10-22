# Frozen seven-person calibration method check

**Status:** protocol written after development on subjects 01–05 and 09–16,
before acquiring any raw records for subjects 06–08 and 17–20. This is a
method check on the [Hyser forearm dataset](https://physionet.org/content/hd-semg/2.0.0/).
It cannot prove silent facial-speech performance or a wearable design.

## Primary comparison

For each of seven reserved subjects, fit a ridge classifier from their day-1
whole-trial log RMS and waveform-length features at the same fixed eight
recorded contacts: grid positions (2,2) and (6,6) on each of four grids.
Choose alpha from 1, 10, 100, 1000 using only three stratified folds on day
1. On day 2, take the **first recorded repetition of each gesture class** as
a labeled support set. Remaining repetitions of each class are query trials.
Compare query balanced accuracy with and without support on the **identical
query trials**. Support receives equal *per-class mass* to day-1 training
(weight 1). Report each person's paired gain and their mean. A positive mean
with at least six of seven people improving and no loss worse than five
percentage points would justify further work; it would not establish a
population effect or a deployed interaction gain.

The secondary comparison uses two support repetitions per class at per-class
weight 4, against the corresponding no-support model on the same query set.
The secondary outcome cannot rescue a failed primary claim. No temporal
feature family, electrode search, or model weight will be changed based on
these seven people's results. The site choices and per-person alpha will be
written to an exclusive selection file and pushed to Git **before day-2
acquisition**. The scoring command checks its code hash and source hashes.

## Interpretive limit

Hyser records all repetitions of a gesture consecutively. Support trials
for later gestures occur after query trials of earlier gestures in the
original timeline. Thus this is an **episodic within-class support/query
replay**, not a chronological online calibration test. Query trials are
disjoint from support trials, but proximity within each class block may
benefit the method. An actual wearable validation needs a session where all
34 prompted calibration gestures are completed first and subsequent,
temporally separated gestures are scored. One-shot support costs at least
34 one-second signal recordings, plus prompts, rests, and fitting; two-shot
support doubles the signal-recording minimum.

The seven-subject gate tests whether the large development gain reproduces
under the same dataset and split protocol. It does not test new electrode
positions, contact impedance, reseating, physiology simulation fidelity,
voice prosody, or unrestricted silent speech. Those remain separate goals.
