# Where deep learning and neural operators fit

This is an experiment decision, not a claim that either model family has solved
silent speech. The [cross-subject channel-prior pilot](25-cross-subject-channel-prior.md)
suggests that smoothing 6,006 noisy program scores to transferable channel
effects may help. The gain is concentrated in one of three development people.
It motivates learning transferable structure, while warning against choosing a
larger model by the same exposed session-3 outcomes.

## Two different learning problems

| Model | Input → output | Evidence currently available | First falsifiable test |
| --- | --- | --- | --- |
| Small temporal encoder | Recorded EMG windows → content representation and class/phone probabilities | SilentWear, Gaddy and existing repo CNN/CTC code | Future-session accuracy and calibration against logistic/SPD at matched splits and compute |
| Geometry-conditioned neural operator | Conductivity, anatomy, source and electrode coordinates → lead field or surface potential | Existing simplified simulator; limb FEM/analytic data from [emgforge](https://github.com/dhalatsis/emgforge) | Predict held-out geometries, electrode sites and downstream layout ranking against a solver and analytic/low-rank surrogates |

The first model addresses measured content and session invariance. The second
could accelerate repeated **forward** physics solves when many electrode
positions or anatomies are queried. An operator does not, by itself, infer a
person's articulatory intent, identify their anatomy, or prove the best wearable
layout. It must be connected to a decoder and a measured decision objective.

For quasi-static volume conduction, the operator's physical target is the
solution map of an elliptic conductivity problem, conceptually
`∇·(σ(x)∇φ(x)) = source(x)`, with domain geometry, boundary conditions and
electrode coordinates as inputs. The exact EMG source and recording pipeline
add motor-unit timing, fibre trajectories, contact and electronics. A useful
factorization is **lead-field operator → motor-unit/action-potential synthesis
→ sensor chain → decoder**. It lets each layer face a separate validation test.
Amortized speed must be measured across the whole chain: making lead-field
queries cheap does not automatically accelerate motor-unit synthesis or model
selection, and generating a large solver-labeled training set has a cost.

## Why an operator is plausible, and why it is premature as a result

A 2026 [geometry-conditioned ECG lead-field surrogate](https://arxiv.org/abs/2602.22367)
predicts fields from anatomical shape and electrode coordinates. Geometry-aware
neural-operator research also addresses varying domains and parameters
([diffeomorphism neural operator](https://www.nature.com/articles/s42005-024-01911-3)).
[emgforge](https://arxiv.org/abs/2609.17216) releases an MRI-based **limb**
volume-conductor pipeline, analytical checks, and ground-truth data. These
sources support an operator *research program*, not facial-EMG validation.
I inspected its public `arxiv-v1` source at commit
`2a3a4f840fbcbf2211dfcaf968c71ab3959040cf`. Its dataset manifests and
generation scripts are present, but the three reference `.npz` datasets are
not tracked in that checkout. The cylinder atlas generator calls a FEM lead
field cache; regenerating its paired analytical/FEM labels needs the external
FEM stack and cache or an equivalent solver run. The small checked-in test
fixtures are regression cases, not a training corpus. These acquisition and
compute costs belong in the operator benchmark budget.

Our current `diffuse_electrode_mix` and `montage_geometry` functions use an
index-based matrix or hand-assigned **one-dimensional** site positions. Their
outputs cannot serve as ground truth for an anatomical operator. Training a
powerful network to mimic them would only produce a faster copy of their
assumptions. SilentWear's 14 fixed channels also lack the counterfactual field
values at unrecorded sites needed to identify such an operator from this
dataset alone.

## Proposed comparison, in order

1. **Real-signal encoder.** Freeze train/selection/evaluation units before
   fitting. Use a small causal convolutional model on raw windows with channel
   masking and measured session perturbations. Compare to the existing feature
   logistic model and SPD/CNN controls. Score per-person future-session content,
   uncertainty, compute, and channel-choice regret. A larger model earns its
   place only if it improves these decisions across people, not merely same-day
   accuracy.
2. **Operator method benchmark.** Import provenance-pinned analytic and FEM
   lead-field examples from emgforge outside Git. Compare an analytic baseline,
   low-rank/POD surrogate, coordinate MLP or DeepONet, and a geometry-aware
   operator. Split by *anatomy and electrode positions*, not random samples
   from the same mesh. Report field error, action-potential error, held-out
   layout ranking/regret, uncertainty, solver-data generation cost, training
   cost and amortized query speed. Limb success is a method check only.
3. **Facial forward model.** Transfer the method only after a facial geometry,
   conductivity, electrode-coordinate and validation source can be obtained or
   built. Calibrate uncertainty against actual facial recordings or controlled
   montage contrasts. If the rank of physical layouts fails to transfer, use
   the discrepancy to revise the physics, not just increase network capacity.

The operator should first be a **surrogate of a validated solver**, with a
baseline that could beat it. The temporal encoder should first be a **competitor
on real future-session data**. Both eventually feed the same criterion: whether
their chosen wearable design performs better after reseating or on a new person.
