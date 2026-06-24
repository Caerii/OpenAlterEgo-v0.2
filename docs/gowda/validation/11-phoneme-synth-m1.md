# Phoneme synthesizer M1 — real-data templates

**Status:** In progress (June 2026)  
**Goal:** Replace equal random phone splits + random motor synergies with **duration priors** and **per-phone templates** fit from real OSF Gowda EMG.

## Phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **M1a** | ARPABET duration priors + weighted `partition_event_to_phones` | ✅ |
| **M1b** | `fit_phone_templates` from real session (pseudo-align within word events) | ✅ |
| **M1c** | Template-driven motor channel weights + rate scale in biophysical drive | ✅ |
| **M1f** | Coarticulation — raised-cosine phone overlap at boundaries | ✅ |
| **M1d** | CLI `analyze fit-phone-templates`; Gowda sim `--phone-templates` | ✅ |
| **M1e** | Separability eval + sim-transfer with template corpus | ✅ grid complete |
| **M2** | STE-GAN / HuBERT-conditioned generator (future) | ❌ |
| **M3** | Articulatory P2A trajectories (future) | ❌ |

## Commands

```bash
cd software/python

# Fit templates from real OSF session (train events, pseudo phone align)
uv run openalterego analyze fit-phone-templates \
  --session ./sessions/gowda_sv_full \
  --out ./sessions/gowda_sv_full/phone_templates.json

# Corpus-duration align (TTS-informed ms priors, lower jitter)
uv run openalterego analyze fit-phone-templates \
  --session ./sessions/gowda_sv_full \
  --align corpus_duration \
  --out ./sessions/gowda_sv_full/phone_templates_corpus.json

# Generate Gowda sim with phoneme drive + templates
uv run openalterego sim-dataset --scenario gowda_sv --out ./corpus/gowda_sim_m1 \
  --trials 100 --realism wearable \
  --phone-templates ./sessions/gowda_sv_full/phone_templates.json

# Disable coarticulation ablation
uv run openalterego sim-dataset --scenario gowda_sv --out ./corpus/gowda_sim_m1_nocoart \
  --trials 100 --realism wearable --no-coarticulation \
  --phone-templates ./sessions/gowda_sv_full/phone_templates.json

# Full M1 ablation grid (500-trial + 100-trial variants + transfer)
uv run openalterego analyze m1-grid \
  --real ./sessions/gowda_sv_full \
  --corpus-root ./corpus/m1_grid \
  --device cuda

# Evaluate phone separability (sim vs real template geometry)
uv run openalterego analyze phone-separability \
  --session ./sessions/gowda_sv_full \
  --sim ./corpus/gowda_sim_m1
```

## Design

1. **Pseudo-alignment** — CMUdict phone sequence per word; sample boundaries via duration priors (no MFA yet).
2. **Templates** — per ARPABET phone: channel RMS profile, `diag_delta` SPD centroid, rate scale, duration weight.
3. **Sim drive** — blend MU channel weights toward template profile; scale token firing rate per phone.
4. **Coarticulation (M1f)** — raised-cosine overlap at phone boundaries; adjacent motor synergies superpose during crossfade (default 28% overlap, ≥10 ms). `phonemes.csv` keeps nominal center boundaries for labels. Disable via `coarticulation_enabled=False` on `BiophysicalSimStreamConfig`.

## First results (June 2026)

**Templates:** 32 phones fit from `gowda_sv_full` train pseudo-align → `phone_templates.json`

**Separability** (`between/within` SPD diag_delta ratio, higher = more separable phones):

| Source | Ratio |
|--------|-------|
| Real OSF (400 events) | **0.53** |
| Sim M1 (20 trials, wearable + templates) | **0.42** |

Sim is not yet matching real phone geometry on the naive hard-partition metric; coarticulation requires **dominant-phone** eval (see `sim_coarticulation_eval` in report).

### Sim-transfer (100 trials, wearable, templates + coarticulation)

Report: `sessions/gowda_sv_full/ablations/sim_transfer_m1_coart_report.json`

| Stage | Real test WER | Word acc |
|-------|---------------|----------|
| Sim only | 93.4% | 5.8% |
| Sim pretrain → real anchor | **70.0%** | **30.0%** |

Compare prior v2 tang corpus anchor: **82.6%** WER → M1+coart anchor **−12.6 pp** (relative improvement on transfer).

### M1 ablation grid (June 2026)

Run: `openalterego analyze m1-grid --real ./sessions/gowda_sv_full --corpus-root ./corpus/m1_grid`

| Variant | Trials | Coart | Realism | Templates | Anchor WER | Sim-only WER |
|---------|--------|-------|---------|-----------|------------|--------------|
| **`m1_nocoart_100`** | 100 | **off** | wearable | pseudo | **64.4%** | 92.4% |
| `m1_off_100` | 100 | on | off (raw) | pseudo | 66.9% | 93.4% |
| `m1_coart_500` | 500 | on | wearable | pseudo | 68.4% | 94.2% |
| `m1_coart_100` | 100 | on | wearable | pseudo | 70.0% | 93.4% |
| `m1_coart_100_corpus_tpl` | 100 | on | wearable | corpus_duration | 70.2% | 93.4% |

**Takeaways:**
- M1 templates + wearable realism beat prior tang v2 anchor (**82.6%**) by **12–18 pp** across all variants.
- **Disabling coarticulation** (`m1_nocoart_100`) yields the best anchor (**64.4%**) — overlap blending may blur phone boundaries for CTC; revisit coart fraction or eval metric before enabling by default.
- Scaling 100 → 500 trials (`m1_coart_500`) did not improve anchor vs 100-trial coart (68.4% vs 70.0%); more data alone is not the bottleneck yet.
- Corpus-duration template align (`corpus_tpl`) ≈ pseudo align on transfer (70.2% vs 70.0%).

Aggregate report: `sessions/gowda_sv_full/ablations/m1_transfer_grid_report.json`

---

- [10-realism-ablations.md](./10-realism-ablations.md) — sensor realism presets  
- [09-sim-transfer.md](./09-sim-transfer.md) — transfer harness  
- [../../19-open-vocab-and-sim2real.md](../../19-open-vocab-and-sim2real.md) — north star
