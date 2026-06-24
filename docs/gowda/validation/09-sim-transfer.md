# Sim→real transfer (first scoreboard)

**Date:** June 2026  
**Harness:** `openalterego analyze sim-transfer`  
**Sim:** `corpus/gowda_sim` (500 trials, biophysical Tang, 31 ch @ 5 kHz)  
**Real:** `sessions/gowda_sv_full`  
**Baseline (real-trained):** Phase 6 trial LM **6.8% test WER**

Report: `sessions/gowda_sv_full/ablations/sim_transfer_report.json`

---

## Run v1 (initial — sim-fitted SPD basis, small model)

| Training mix | Real test WER | Word acc |
|--------------|---------------|----------|
| Sim only | 92.7% | 5.8% |
| Sim + 10% real | 93.7% | 6.3% |
| Sim + 50% real | 85.9% | 14.1% |
| Sim + 100% real train | 81.8% | 18.2% |

**Issues identified:**
1. Sim training used **sim-fitted Q** while eval used **real Q** → feature domain mismatch.
2. Harness used **2×192** GRU vs Phase 6 **3×256**.
3. No **anchor fine-tune** on real after sim pretrain (doc 19 recipe step 3).

---

## Run v2 (fixes in code — re-run with GPU)

| Change | Module |
|--------|--------|
| Train sim with **real OSF SPD basis** | `ensure_gowda_spd_basis(basis_session_dir=real)` |
| Match Phase 6 architecture (256 hidden, 3 layers) | `sim_transfer.py` |
| **sim_pretrain_real_anchor** run after sim-only | `anchor_epochs` fine-tune on real train |
| CLI `--no-anchor`, `--anchor-epochs` | `cli.py` |

**Command (full):**
```bash
uv run openalterego analyze sim-transfer \
  --sim ./corpus/gowda_sim --real ./sessions/gowda_sv_full --device cuda
```

**Command (quick smoke):**
```bash
uv run openalterego analyze sim-transfer \
  --sim ./corpus/gowda_sim --real ./sessions/gowda_sv_full --device cuda \
  --pretrain-epochs 5 --finetune-epochs 3 --anchor-epochs 5 --real-fracs 0
```

---

## Interpretation

- Large v1 gap is **expected** for naive sim pretrain; sim is a **pipeline validator**, not yet a real-data replacement.
- Monotonic improvement with more real data in the mix confirms the harness and merged split logic.
- Next levers: realism SNR/montage sweeps, longer sim pretrain, cross-modal audio (MONA), per-user Q (M3 onboarding).

---

## Related

- **Realism ablations:** [`10-realism-ablations.md`](./10-realism-ablations.md)  
- Design: [`../../19-open-vocab-and-sim2real.md`](../../19-open-vocab-and-sim2real.md)  
- Gap analysis: [`../openalterego/01-gap-analysis.md`](../openalterego/01-gap-analysis.md)
