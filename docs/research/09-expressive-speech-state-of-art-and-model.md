# Expressive silent speech: evidence and model decision (October 2026)

## The claim we want to test

Can a low-channel wearable infer *content, timing, and intended expression* from silent articulation across recording sessions, then render intelligible speech in a consented personal voice? Each italicized term is a separate inference problem. A plausible voice generated from text is not evidence that the wearer intended its prosody or emotion. Earbud playback is an output channel; a dialogue system also needs turn detection, latency, privacy, and listening tests. No present result in this repository demonstrates the complete loop.

## Closest primary work

| Work | Relevant result | Limit for our target |
| --- | --- | --- |
| [Gaddy and Klein, ACL 2021](https://aclanthology.org/2021.acl-short.23/) and [official code](https://github.com/dgaddy/silent_speech) | Parallel voiced and silent EMG; speech synthesis and phoneme-aided modeling. Official preprocessing includes notch/drift filtering and resampling; its raw model input has a separate six-sample crop. | Our small voiced probe does not yet reproduce official acoustic-feature alignment, split, or synthesis baseline. |
| [Gowda et al., ACL 2026](https://aclanthology.org/2026.acl-long.750/) | Maps EMG into self-supervised speech representation for synthesis, including a silent-articulation demonstration with one participant with ALS. | This does not establish personal voice, expressive intent, or cross-person wearable robustness. |
| [Scheck et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/scheck25_interspeech.html) | Diffusion-based multi-voice EMG-to-speech with speaker-independent targets and selectable voices. | Selectable voice is distinct from recovering the wearer's original voice or intended moment-to-moment expression; no end-to-end earbud latency is established here. |
| [Vojtech et al., JSLHR 2021](https://pubmed.ncbi.nlm.nih.gov/33979177/) | In an eight-person proof of concept, silent phrase content and phrase-level stress were classified and rendered through personalized synthetic voices. | Bounded phrase corpus and stress classes, not unrestricted expressive speech. The reported 96.3% word and 91.2% stress accuracy must retain that setting. |
| [Vojtech et al., Vibration 2022](https://pubmed.ncbi.nlm.nih.gov/36299552/) | Eight facial/neck sensors in ten participants predicted acoustic F0 and intensity for voiced tasks; reported F0 precision 0.56 semitones and intensity precision 3.25 dB SPL. | Voiced acoustic targets are observable; silent intended F0 is not directly labeled. Neither error number is a silent-speech result. |
| [Pistrosch et al., arXiv 2026](https://arxiv.org/abs/2603.11715) | Facial/neck EMG captured affect-related signal in phonated and silent tasks; controlled frustration classification reached 0.845 AUC on 2,780 utterances from 12 people. | Task emotion labels do not prove reliable inference of a person's private feeling or the expression they want to transmit. Dataset access and cross-session replication remain to verify. |
| [SilentWear, arXiv 2026](https://arxiv.org/abs/2603.02847) | 14-channel wearable command classification: 77.5% silent cross-validation accuracy and 59.3% inter-session accuracy; 2.47 ms MCU inference. | Eight-command classification and inference time do not establish continuous speech or articulation-to-earbud latency. Session shift is a major design target. |
| [Myovox, arXiv 2026](https://arxiv.org/abs/2609.17548) | 31-channel vocalized EMG, bidirectional Conformer with audio representation distillation, 18.53% WER after 7B-model reranking on a single-subject held-out corpus. | Vocalized, high-density, offline/bidirectional and language-model-aided; not a wearable silent streaming comparison. The paper reports acoustic phone error as a limiting factor. |

These are different tasks and corpora. The numbers must not be ranked as if they measured the same device or generalization setting.

## What we can learn from free data now

- **Gaddy**: paired voiced/silent EMG and audio allow content, alignment, acoustic-latent, and voiced prosody studies. Use the official book/sentence split and preprocessing as a parity baseline. Silent samples have no simultaneously spoken acoustic target; paired text is insufficient to recover intended F0.
- **SilentWear**: repeated-day EMG allows session, reseating, channel-subset and calibration studies for command content. It has no paired acoustic waveform for voice fidelity. Existing S04 layout ranking missed the prespecified regret threshold [here](03-public-holdout-s04-outcome.md); the model is not validated for hardware choice.
- **Somach**: alternate montages offer an observational geometry contrast. Recording time and montage are confounded [here](04-public-cross-configuration.md), so do not present a causal electrode effect.
- **[CSL-EMG Array](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)**: multi-speaker, multi-session paired recordings appear valuable for transfer checks. Access is by a published request form; verify terms, provenance, and full split before use.

Our [locked third-session voiced probe](08-prosody-third-session-outcome.md) reduced log-energy MAE after full-session unlabeled normalization; F0 did not beat its constant prior. It is a single-speaker diagnostic. A subsequent metadata audit of the non-reference first-100 trials found **zero overlapping book/sentence keys and zero identical text strings** between 5-5, 5-6 and 5-8. This corrects our earlier generic warning about repeated prompts for these particular subsets. The direct clock-time alignment still differs from Gaddy's official preprocessing and offset; a shared speech-activity envelope or session artifacts could explain some of the gain. The added calibration-only and normalized-shuffle controls in `gaddy_prosody_probe.py` still need a prospective reserved-session test.

### Exploratory control run on already examined 5-8

After the 5-8 outcome was known, we reran the fixed ridge model with the first 10 valid 5-8 utterances supplying **unlabeled past-only calibration**; 57 subsequent utterances supplied evaluation labels. The [full machine-readable result](results/gaddy-prosody-5-8-causal-exploratory.json) includes trial IDs and data provenance. This is a post hoc diagnostic and cannot serve as a new holdout result.

| Voiced-frame target | Past-only calibration MAE | Whole-evaluation-session normalization MAE | Normalized shuffled-training MAE | Mean prior MAE |
| --- | ---: | ---: | ---: | ---: |
| Log audio RMS | **1.826** | 0.744 | 0.833 | **0.823** |
| F0, Hz | 58.35 | 70.16 | 72.74 | 71.99 |

The energy gain disappears and reverses under this calibration prefix; full-session normalization still beats its shuffled control. The F0 ordering reverses, but the autocorrelation target is crude, the choice of a 10-utterance prefix was made after prior analysis, and the data were already used. Neither pattern establishes silent expression. The failure of deployable energy calibration makes alignment parity, session-gain diagnostics, and prospective calibration-length selection higher priority than scaling the renderer.

### Filter comparison on development session 5-6

With the same ten-trial calibration prefix, 5-5 training and 5-6 evaluation, forward-only filtering improved **all-frame energy** MAE to 0.587 versus a 0.812 mean prior. The published-style offline filtering scored 0.558 versus 0.812. For oracle-selected voiced frames, the corresponding MAEs were 0.524 and 0.501 versus 0.790. These are development outcomes on a session already examined in earlier work; the new causal-filter result has no new holdout status. The [causal-filter output](results/gaddy-prosody-5-6-causal-filter-dev.json) and [offline output](results/gaddy-prosody-5-6-official-offline-dev.json) preserve trial IDs and settings. Both use our own 100 ms feature windows rather than Gaddy's official acoustic feature alignment. Their unshuffled energy improvement is compatible with useful EMG information, but voiced EMG can contain phonation-related activity or artifacts that silent speech lacks.

## Model to build, in order

1. **Signals and nuisance state.** Explicitly record montage, channel mask, gain, contact/noise features, session and allowable calibration history. Build causal 1000 Hz preprocessing with parity checks against official Gaddy transforms. Train session-disjoint baselines first; prohibit future-session statistics in streaming claims.

The probe now exposes `raw`, `official-offline`, and `causal-filter` EMG preprocessing. `official-offline` reproduces the published notch/high-pass/resample operations as a reference but uses forward-backward filtering and adjacent trials; it is unsuitable for a live-stream claim. `causal-filter` uses forward-only notch/high-pass filters at 1000 Hz, with no adjacent-trial data. This is a first causal signal baseline, not a claim of exact official feature-frame or audio alignment. The six-sample crop in Gaddy's source belongs to its raw-model input path, not this probe's hand-engineered features.
2. **Shared causal content encoder.** Multi-resolution EMG convolution plus a streaming Conformer or state-space temporal model. Supervise phone/CTC content and, on paired voiced recordings only, distill a frozen self-supervised audio representation. Compare against a small causal CNN and text-only prior with matched splits and compute. Keep an offline bidirectional upper bound separate.
3. **Expression heads with calibrated abstention.** Predict articulatory onset, duration, energy/stress and question/statement contour where labeled. Estimate F0 only in a voiced-trained head and report its uncertainty and silent transfer gap. Emotion is optional, coarse, user-adjustable and abstains when uncertain; never equate a classifier's output with the person's internal feeling.
4. **Voice rendering.** A consented voice enrollment creates a private speaker embedding or adapter. A lightweight streaming acoustic model receives decoded content, timing and permitted expression controls; a vocoder renders audio. Compare natural prosody prior, EMG-conditioned prosody, and explicit user control in blinded listening. Multi-voice diffusion is an offline quality upper bound until latency and compute meet the device budget.
5. **Simulation hierarchy.** First test electrode geometry with empirically calibrated low-dimensional mixing and placement/contact perturbations, preserving uncertainty. Use the biophysical motor-unit generator as an ablation and waveform stressor. Only add facial volume-conductor FEM when a measurable design-ranking failure points to missing geometry. [emgforge](https://arxiv.org/abs/2609.17216) offers a limb-focused reference architecture, not a validated facial digital twin.

For a causal model, the prediction at time *t* can use EMG and adaptation data observed through *t* only. Speaker identity comes from a separate consented enrollment. Text-only TTS prosody is an explicit prior, never mislabeled as recovered silent expression.

### A falsifiable forward and inverse model

For electrode channel *c* and time *t*, start with

`y[c,t] = gain[c,s] * sum_m lead[c,m](geometry, placement[s], conductivity) * muscle[m,t] + motion[c,t] + mains[c,t] + amplifier[c,t]`.

Here *s* is the session. The muscle term is driven by articulatory gesture and motor-unit activity; both articulation and session-specific contact/placement are latent. The amplifier term includes input noise, filtering, ADC limits, and common-mode rejection. The acoustic target is produced by articulation **plus phonation**, and phonation is absent or altered in silent trials. Thus voiced F0 is not a direct ground-truth label for silent intended F0. Uncertainty over placement, contact, and anatomy must be propagated into the predicted signal and design ranking.

Fit an inverse encoder to `p(content, timing, stress | y[<=t], past_calibration)` rather than claiming unique anatomy. Couple it to a renderer `p(audio | content, timing, chosen_style, enrolled_voice)`; permit EMG-conditioned style only where held-out evidence beats the text/style prior. Compare predictive likelihood and *counterfactual ranking* for channel masking or recorded montage differences. Simulation parameters are useful only if they predict interventions under held-out sessions; multiple parameter sets may explain the same waveform. A hierarchical session model can estimate nuisance drift from a fixed past-only calibration prefix while sharing content weights across sessions. The present data do not identify a high-resolution personal anatomy model.

## Gates before a stronger claim

1. **Parity:** compare filtering/resampling against official Gaddy source, then reproduce its feature-frame alignment and split; inspect train/test text overlap. Its six-sample raw-input crop must not be applied blindly to our hand-engineered features. Run lag sweeps and within-utterance temporal-shuffle controls so timing gains cannot be explained by speech activity alone.
2. **Transfer:** lock new sessions and speakers before tuning. Report content WER/PER, frame and utterance energy/stress metrics, uncertainty calibration, and no-calibration versus past-only calibration. Preserve poor results.
3. **Silent expression:** use actual silent stress/intonation labels where public data support them; separate recognition of instructed intent from inference of latent emotion. If no suitable public data exist, mark this gate untestable offline instead of generating synthetic labels and claiming success.
4. **Rendering:** compare content-fixed, voice-fixed listening trials for intelligibility, speaker similarity and expression control. Record whether listeners can tell *intended* expression apart from generic TTS prior. This requires human judgment eventually; offline objective metrics alone cannot close it.
5. **Interaction:** measure from articulation onset through stable decode, synthesis, transport and earbud playback. MCU inference alone is one component.

The next bounded experiment is utterance-level uncertainty and temporal negative controls on a reserved voiced session, alongside exact Gaddy feature-alignment parity. These checks can falsify the energy-cue interpretation efficiently, while leaving the silent prosody claim explicitly open.

**Progress update, 2026-10-21:** The [5-9 locked test](11-gaddy-causal-filter-5-9-outcome.md) passed its all-frame vocalized energy criterion with forward-only filtering: 0.590 MAE against 0.816 mean-prior and 0.807 shuffled-control MAE. Exact published feature alignment and the official book/sentence benchmark split remain open; no silent expression result follows from this energy test.

The [5-10 locked temporal audit](13-gaddy-temporal-audit-5-10-outcome.md) also passed three utterance-level controls, including a one-second within-utterance feature shift. This strengthens the vocalized timing association but does not separate articulatory expression from actual phonation or speech activity. Session 5-11 remains unexamined for a prespecified residual-expression check.
