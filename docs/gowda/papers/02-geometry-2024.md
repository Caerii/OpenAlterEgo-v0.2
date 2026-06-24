# Paper Summary: Geometry of Orofacial EMG (2024)

**Authors:** Harshavardhana T. Gowda, Zachary D. McNaughton, Lee M. Miller  
**Link:** [arXiv:2411.02591](https://arxiv.org/abs/2411.02591) · [Journal of Neural Engineering](https://arxiv.org/abs/2411.02591)  
**Data:** Same OSF ecosystem [YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD)

---

## Relation to 2025 paper

| 2024 (geometry) | 2025 (neuroprosthesis) |
|-----------------|------------------------|
| SPD matrices on manifold; **classification** | SPD σ(τ) sequences + **GRU + CTC** |
| 16 subjects, word/phoneme **classification** | Continuous silent speech → **phoneme sequences** |
| Establishes “change of basis” across users | Exploits fixed Q from Fréchet mean for efficient GRU inputs |

The 2025 paper explicitly **builds on** gowda2024geometry and adds sequence decoding + sparse spectral domain.

---

## Key ideas for OpenAlterEgo

1. **Inter-channel covariance** carries articulatory information — not just per-channel amplitude.
2. **Riemannian / SPD geometry** — Fréchet mean, Cholesky decomposition (Lin 2019).
3. **Per-user eigenbasis** differs → domain shift ≈ rotation of sensor space → **calibration required**.
4. Small networks (10k–150k params) can work with right representation.

---

## emg2speech notebook alignment

`smallVocabEuclidean.ipynb` in [emg2speech repo](https://github.com/HarshavardhanaTG/emg2speech):

- Per-trial z-normalize `DATA` along time per channel.
- Word chunks: `0:10000`, `10000:20000`, `20000:30000`, `30000:` (2 s / 2 s / 2 s / 3 s).
- 50 ms SPD windows → GRU → phoneme CTC.
- Split: first **1480** word events train (370×4), next **120** val (30×4), remainder test.

OpenAlterEgo `--split-by gowda` implements the **370/30 sentence** split at event level.

---

## Citation

See [../legal/04-attribution-citation.md](../legal/04-attribution-citation.md).
