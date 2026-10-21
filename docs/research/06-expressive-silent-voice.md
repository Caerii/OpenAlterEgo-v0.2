# Expressive silent voice: parallel offline research track

Status: proposed research track, no demonstrated OpenAlterEgo expressive voice
output. No new physical recordings are required for the first experiments.

## Product hypothesis and signal boundary

The desired interaction is person-to-person silent articulation converted to
low-latency speech in the listener's earbud, with the sender's chosen voice
identity and expressive timing. Separate four variables:

1. **Content:** phones, words, repairs, and uncertainty inferred from EMG.
2. **Timing:** onset, pauses, stress, and duration estimated from motor activity.
3. **Expression:** pitch, intensity, affect, and speaking style, inferred only
   where evidence supports it or explicitly selected by the user.
4. **Voice identity:** a consented target-voice profile for synthesis. The
   target voice is a generative prior, not a property directly contained in
   silent EMG.

The device must not label an inferred emotion as the user's true internal
state. Its controllable output may include neutral, questioning, gentle,
excited, and emphatic styles, with a neutral fallback. Silent sEMG can still
require articulation and muscle recruitment; a claim of speech with **no mouth
movement at all** is a separate sensing problem and is not established by
the current surface-EMG approach.

## First paired-data probe, completed

The bounded extractor imported IDs 0–99 from one voiced Gaddy session outside
Git. The archive matched Zenodo's MD5 before extraction; 24 publisher
reference records (`sentence_index: -1`) were excluded. The remaining 76
utterances were split by ID with seed 42: 60 training and 16 test utterances.
Within each utterance, 100 ms EMG windows produced eight-channel RMS and
waveform-length features; a separate 40 ms audio window supplied log energy
and a simple autocorrelation F0 estimate. A ridge model was compared with a
training-mean prior and shuffled-training-EMG control.

| Held-out target | EMG MAE | Mean-prior MAE | Shuffled-EMG MAE | EMG R² |
| --- | ---: | ---: | ---: | ---: |
| Log audio RMS, all frames | 0.669 | 0.910 | 0.895 | 0.338 |
| Log audio RMS, voiced frames only | 0.680 | 0.968 | 0.960 | 0.392 |
| F0, Hz, voiced frames only | 79.24 | 88.69 | 87.52 | 0.121 |

The voiced-only energy result suggests a relation beyond simple speech/rest
detection in this single recorded session. F0 prediction is weak and its
hand-built pitch target may contain octave and voicing errors. This is not
evidence that individual emotional nuance or a person's intended silent
pitch has been recovered. Cross-session, cross-speaker, and silent conditions
remain untested. See the [complete result](results/gaddy-voiced-prosody-first100.json).

Reproduce after acquiring the verified archive and installing `ffmpeg`:

```bash
uv run --frozen python scripts/extract_gaddy_voiced_subset.py \
  --archive /data/Gaddy/emg_data.tar.gz --out /data/Gaddy/voiced-first100 --count 100
uv run --frozen python scripts/gaddy_prosody_probe.py \
  --subset /data/Gaddy/voiced-first100 --output /runs/gaddy-prosody.json
```

## Evidence available now

- [Gaddy and Klein's open corpus](https://zenodo.org/records/4064409)
  pairs facial EMG with audio on vocalized trials and includes silent trials.
  The 3.9 GB archive was downloaded outside Git and matched its published MD5
  `7f97d2182b896652999b1b2d0c69fd7b` on 2026-10-21. Silent trials do not
  provide a simultaneous voiced target for their intended pitch contour.
- [CSL-EMG_Array](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
  offers multi-speaker, multi-session paired EMG/audio and silent trials under
  CC BY 4.0. Download requires an emailed link, so import is pending.
- [F0/intensity prediction from face and neck sEMG](https://pmc.ncbi.nlm.nih.gov/articles/PMC9592063/)
  provides evidence that continuous prosodic correlates can be learned during
  vocal tasks. Those results do not establish exact F0 recovery from silent
  articulation.
- [Affect decoding from phonated and silent sEMG](https://arxiv.org/abs/2603.11715)
  reported a frustration AUC up to 0.845 in its studied setting, while its
  spontaneous cross-speaker condition was about 0.63 AUC. This supports a
  constrained affect probe, not reliable open-world emotion reading. A public
  download of that study's raw data was not confirmed.

## Offline experiment sequence

1. **Paired-data audit.** Safely extract a small, hashed Gaddy subset outside
   Git. Verify session, trial, EMG/audio sync, text, speaker identity and
   vocalized/silent status. Preserve the publisher's held-out splits.
2. **Predict acoustics where targets exist.** On vocalized trials, compare a
   text-only duration/F0/energy prior with EMG-conditioned predictions. Score
   F0 in semitones on voiced frames, energy error, timing error, calibration,
   and performance by future session. Include a shuffled-EMG negative control.
3. **Silent transfer.** Freeze that model, test content and timing on silent
   trials, and report which acoustic quantities lack ground truth. Compare
   direct EMG-to-acoustic synthesis, EMG-to-phones plus duration/style controls,
   and a text-to-speech baseline at matched voice and latency budgets.
4. **Expressive generation.** Condition a streaming speech generator on
   content, estimated timing, optional style command, and a consented voice
   profile. Evaluate intelligibility, voice similarity, naturalness, unwanted
   expression, onset-to-audio latency, and correction/abort behavior. Synthetic
   listener studies may test the interface offline; subjective claims still
   require actual human listening evidence later.

Treat amplitude and rate as possible effort or timing cues, not direct proof of
emotion. Pre-register the primary metric and untouched test corpus before
model selection. Report an abstention or neutral rendering when the input does
not support an expressive inference. Keep target-voice enrollment and playback
under the voice owner's control.

## Architecture hypothesis

`EMG → causal content/timing encoder → uncertain phones/text + prosody controls
→ consented speaker representation → streaming acoustic generator → earbud`.

A direct EMG-to-acoustic path is a competing research model, not a prerequisite
for a first conversational prototype. Both paths must distinguish what was
measured from what the generator supplied. The listener should be able to
understand, interrupt, and correct an utterance before accidental expressive
or lexical content is committed.
