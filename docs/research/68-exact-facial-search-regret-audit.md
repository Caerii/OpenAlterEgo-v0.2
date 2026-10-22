# Exact search audit on the 20-channel facial command assay

Before committing more effort to optimizer speed, we compared the existing
beam-4 six-channel search with **all 38,760** six-of-twenty recorded-channel
programs. The [audit runner](../../software/python/scripts/faceemg11_exact_search_audit.py)
uses the same two forward selection folds for both methods: fit blocks 1–10,
validate 11–15; fit 1–15, validate 16–20. It then fits on blocks 1–20 and
inspects the already exposed blocks 21–30. These are FaceEMG-11 facial
**command** labels, one recording per person, not silent sentences or
wearable reseating. Every reported accuracy is retrospective.

| Subject | Beam exact rank | Beam selection regret | Later-block beam | Later-block exact selection | Later-block all 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 1 | 0.000 | 0.955 | 0.955 | 0.991 |
| 02 | 3 | 0.009 | 0.927 | 0.945 | 0.982 |
| 03 | 1 | 0.000 | 0.864 | 0.864 | 0.964 |
| 04 | 1 | 0.000 | 0.855 | 0.855 | 0.882 |
| 05 | 1 | 0.000 | 0.927 | 0.882 | 0.982 |
| 06 | 7 | 0.018 | 0.764 | 0.791 | 0.891 |
| Mean | — | — | **0.882** | **0.882** | **0.948** |

The per-subject [records](results/faceemg11-exact-search-audit-sub01-v2.json)
through `sub06-v2.json` contain source hashes, programs, quantiles, and
timings. Beam search scored 338–342 partial/full programs per subject;
the exact comparator scored 38,760 full programs. On this local Windows
machine, mean search times were 0.063 versus 10.81 seconds, roughly
171× different. This is a **compute and selection-score** comparison;
the timings are not portable and do not establish design transfer.

Subject 05 reveals a more important limitation. There are **13 distinct
programs tied for the top forward-selection accuracy of 1.0**. Their
later-block accuracies range from **0.827 to 0.927** in the
[subject 05 audit](results/faceemg11-exact-search-audit-sub05-v2.json).
The beam choice reaches the upper end, while the lexicographically first
exact optimum reaches 0.882. Thus an exact optimization of this selection
score does not resolve which equally ranked program will transport.

The six-channel choices also trail the all-20 reference by 0.067 mean
later-block accuracy. That is a channel-count/accuracy tradeoff, not a
fair equal-cost contest. More exhaustive search alone cannot repair
the coarse selection objective or guarantee out-of-block performance.
The research priority is now to improve and validate **decision quality**:
uncertainty over tied programs, multi-session evidence, contact cost,
and truly new session/speaker evaluation. A dense facial speech corpus
would let the same audit test physically meaningful contact-constrained
programs; the current command dataset cannot establish that claim.
