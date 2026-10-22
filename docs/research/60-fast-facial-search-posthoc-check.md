# Fast compositional facial search: exposed-subject check

The [development comparison](59-facial-temporal-and-fast-search-development.md)
selected beam width 16. We replayed that fixed rule for subjects 07–12
and independently scored all 38,760 six-channel programs with the same
temporal train-CV objective. Query blocks 21–30 had already been viewed
for other program choices, so this is a **post hoc method check**, not
an independent confirmation.

| Search | Mean train CV | Mean later-block query | Mean search wall time | Program evaluations |
| --- | ---: | ---: | ---: | ---: |
| Exact temporal | 0.9374 | 0.9152 | 26.85 s | 38,760 full |
| Stepwise beam 16 | 0.9354 | 0.9136 | 0.57 s | about 1,134 partial/full |
| Coarse screen + 128 temporal finalists | 0.9323 | 0.9303 | about 8 s | 38,760 cheap + 128 rich |

Beam search was about **47× faster** than exact temporal search on
these six subjects in this local run. Its mean train-CV score was
0.0020 lower, and mean query balanced accuracy was 0.0015 lower.
Five of six beam choices matched or exceeded the exact-CV choice on
query, while subject 11 lost 0.018. Those query results do not make
beam optimal: exact search maximizes *training CV*, and either choice
may overfit it. The two-stage method produced higher mean query
accuracy here but took much longer than beam search.

The [beam result](results/faceemg11-stepwise-method06-posthoc-v1.json)
records selected channels, each partial-stage evaluation count, source
hashes, times, and query scores. The
[exact comparison](results/faceemg11-exact-temporal-method06-posthoc-v1.json)
records per-subject train-CV regret and timings. These exploratory
wall-clock figures include scoring/search but exclude data loading;
they are not controlled hardware benchmarks.

This establishes a useful **search-computation** advantage for the
facial command benchmark. It does not establish a 47× recognition
gain, an electrode-placement improvement after reseating, silent
speech, or a wearable. The next performance work should preserve the
fast search while replacing the task proxy with facial speech array
recordings and evaluating across independent sessions.
