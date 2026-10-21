# Measured wearable research program

Status: protocol and engineering plan, not a claim of a validated device.

The current work phase uses simulation and freely downloadable recordings.
See [the public-data lane](01-public-data-simulation.md) and its locked S04
check. Electrical and participant studies below are future gates; they are
not prerequisites for the present offline iterations.

## Decisive question

Can a model trained on measured acquisition behavior predict which change to an
electrode layout or wearable frame improves **held-out, next-session** silent
command performance? A simulation is credible for this use only when its
prospective ranking of candidate designs agrees with measurements. Synthetic
accuracy, visual plausibility, and one matched SNR target do not answer it.

## Evidence ladder

1. **Reproduction:** freeze the existing Gowda split, code commit, corpus
   hashes, simulator configuration, seeds, and complete transfer output. Report
   real-only, simulator-labeled, and sim-plus-real results without selecting on
   test. The current `sim_only` lane still fits its SPD feature basis on real
   **training** trials; it is not a no-real-data zero-shot experiment.
   The documented phase-6 and M1 numbers are historical observations, not a new
   result from this protocol. Use `scripts/transfer_experiment.py` for a pinned
   run; keep its generated workspace outside the input sessions.
2. **Electrical V0:** record injected and phantom signals through the actual
   AFE, electrode leads, reference, packetizer, and receiver. Measure input
   noise, gain, clipping, common-mode rejection, filter response, effective
   bandwidth, clock drift, loss, power, and full interaction latency. Store raw
   samples, configuration, calibration, and instrument uncertainty.
3. **Human V0:** with appropriate consent and oversight, record a fixed command
   set from several participants over several days. Include deliberate
   electrode reseating, measured displacement, quiet and motion conditions,
   skin/contact observations, and independent held-out sessions. Record
   negative examples and non-speech activity. Freeze the final participant and
   session evaluation before simulator fitting.
4. **Predictive design:** register two or more feasible montage/frame variants.
   Predict each variant's next-session performance and uncertainty using
   training measurements only. Freeze the predictions before outcomes are
   revealed. Randomize A/B order within participant, reseat between blocks,
   measure placement and contact, and repeat on held-out future sessions.
   Compare against a simple empirical electrode-selection baseline. Score
   pairwise design ranking, decision regret, and interval coverage before
   retaining or revising the model and design.
5. **Wearable V1/V2:** publish editable schematic, PCB, firmware, and mechanical
   sources; repeat the prediction experiment with untethered operation,
   self-reseating, comfort, power, and multi-hour stability measured on-device.

## Multiscale simulator

Keep a fast stochastic motor-unit generator for corpus production, but make
each layer independently measurable and replaceable:

| Layer | Inputs and validation target |
| --- | --- |
| Motor drive | Per-muscle activation, timing, coarticulation, fatigue and participant variation; compare labeled real envelopes and activation relationships. |
| Volume conductor | 3D electrode, muscle, skin and fat geometry with conductivity uncertainty; compare spatial amplitude, phase and cross-channel structure. Use selected finite-element solves to check a fast surrogate. |
| Contact and mechanics | Pressure, electrode displacement, impedance, jaw and frame movement, sweat and reseating; compare repeated physical perturbations. |
| AFE and transport | Measured transfer function, noise, quantization, anti-aliasing, timing and packet behavior; compare injected-signal and phantom tests. |
| Interaction | Segmentation, abstention, correction and acquisition-to-decision time; compare recorded end-to-end traces. |

The current 1D exponential pickup model remains an L0 baseline. An empirically
calibrated spatial surrogate is the next fidelity level. A 3D finite-element
model is an optional escalation when the simpler model fails a predeclared
design comparison and the extra fidelity improves a held-out decision. Fit
parameter ranges on training sessions and carry uncertainty through design
predictions. Different anatomical parameterizations can yield similar signals;
judge them by predictions under interventions that can actually be tested.
Never fit a simulator to held-out recordings used to claim transfer.

## Prospective scientific decision

Use `scripts/design_experiment.py register` to freeze a hypothesis, feasible
candidate design hashes, predicted command accuracy and intervals, baseline,
fit-data/model/split hashes, budget, independent reviewer, and the human study
protocol. The protocol names the randomized within-participant comparison,
reseating, placement precision, contact measurement, blinding, and sealed
future-session evaluation. Register before revealing measurements. Then use
`scripts/design_experiment.py score` with a separate measurement manifest and
held-out outcomes. The score reports pairwise ranking accuracy on decisive
pairs, regret against the best observed candidate, regret of the baseline
candidate, and interval coverage. A small study will have uncertain estimates;
publish participant-level uncertainty and all failed predictions too.

The local JSON hash and exclusive file creation prevent accidental overwrite
but do not prove that a preregistration predates measurement. Bind the file to
an external timestamped immutable evidence store before physical testing. A
signed SIG execution receipt verifies what ran, not whether randomization,
measurement, consent, or inference were sound. Human approval controls any
protocol change or participant work. Reusing held-out participants to tune
later models consumes that holdout; register a new untouched evaluation.

SIG may propose the next experiment by estimated information gain minus
cost, but only among human-approved feasible experiments. Compare predictions
from competing explanations of placement, contact, and physiology; spend a
physical measurement where those predictions differ enough to change a design
decision. Report information gain and actual decision improvement per dollar
and participant hour, rather than simulation run count.

## Required result card

Each experiment records: question and predicted sign; protocol version; code,
input, spec and model hashes; participant/session/device split; fit data;
parameter ranges; comparator; compute budget; full failures/skips; waveform
and feature discrepancies; WER or command accuracy with uncertainty; measured
interaction latency; and a retain/revise/reject decision by a reviewer who did
not generate the result. Human-data artifacts use restricted storage and
publish only consented aggregates.

## Immediate next run

1. Import licensed Gowda data into a private, immutable session directory.
2. Generate one Gowda-shaped corpus and record its actual engine and realism
   configuration. The wearable JSON preset currently selects
   `heuristic`/`field`, while simulation guidance calls `biophysical`/`tang`
   the target; do not silently treat them as the same run.
3. Commit this harness, then execute `scripts/transfer_experiment.py` from a
   clean checkout with an output directory outside both sessions.
4. Review the manifest, complete logs, split and test metrics. Compare with a
   real-only baseline on exactly the same held-out split before changing the
   physics model.

See [hardware tiers](../../hardware/01-architecture-tiers.md),
[current physics implementation](../../hardware/10-neurobiophysical-emg.md),
and [historical M1 results](../gowda/validation/11-phoneme-synth-m1.md).
