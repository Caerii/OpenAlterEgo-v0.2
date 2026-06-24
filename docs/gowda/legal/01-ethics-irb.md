# Ethics & IRB (Gowda 2025 Paper)

Source: [arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2) — Ethical statement & Appendix B.

---

## Institutional review

| Field | Value |
|-------|-------|
| Institution | University of California, Davis |
| IRB protocol | **2078695-1** |
| Framework | Declaration of Helsinki |
| Consent | Written informed consent for participation **and** deidentified data publication |

---

## Participant inclusion

- Healthy volunteers; any gender; all ethnic and racial groups represented in recruitment.
- Age **18+**.
- Fluent spoken and written **English**; able to follow task instructions.
- No skin conditions or wounds at electrode sites.

---

## Participant exclusion

- Uncorrected vision problems preventing task performance.
- Neuromotor disorders preventing speech articulation.
- Children, adults unable to consent, prisoners.

---

## Data collection conduct

- Silent speech: participants articulate **naturally but inaudibly**.
- **Mouse clicks** mark sentence start/end (self-paced articulation).
- Controlled environment to reduce AC electrical interference.
- LSL (Lab Streaming Layer) used for time synchronization.

---

## Implications for OpenAlterEgo

| Use case | Requirement |
|----------|-------------|
| **Train/eval on public OSF data** | No new IRB for analysis-only use of deidentified releases; still cite dataset + paper. |
| **Collect new EMG from people** | Your institution’s IRB/ethics review required. |
| **Clinical / patient deployment** | Full regulatory path beyond research code — not covered by this repo. |

See also [02-data-license-osf.md](02-data-license-osf.md), [04-attribution-citation.md](04-attribution-citation.md).
