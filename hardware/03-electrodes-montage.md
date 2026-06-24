# Electrodes and montage

Muscle targets, channel wiring, dry vs wet trade-offs, and literature-backed placement for silent speech EMG.

> **Not medical advice.** This documents research montages for prototyping. Clinical use requires qualified supervision and regulatory compliance.

---

## Montage philosophy

Silent speech EMG is **not** generic limb EMG. Signals come from **orofacial and neck muscles** involved in articulation when the user **subvocalizes** (moves articulators without phonation).

**Design goals:**
1. Capture discriminable patterns per command/token
2. Maximize SNR at moderate sample rates (250–1000 Hz)
3. Minimize motion sensitivity (jaw, head turn, cable pull)
4. Support per-user calibration (anatomy varies — [Gowda 2024](https://arxiv.org/abs/2411.02591))

---

## Literature montages (summary)

| Study | # Ch | Sites (abbrev.) | Electrode type | Reference |
|-------|------|-----------------|----------------|-----------|
| **Kapur 2018** | 7 | Laryngeal, hyoid, LAO, OO, platysma, digastric, mentum | Au-plated Ag or dry Ag/AgCl + Ten20 | [IUI 2018](https://dl.acm.org/doi/10.1145/3172944.3172977) |
| **Kapur 2020** | 8 | 4 face + 4 neck | — | Earlobe ref + bias | [PMLR](https://proceedings.mlr.press/v136/kapur20a.html) |
| **Wang 2021** | 4 | LAO, DAO, BUC, ABD | Tattoo-like flexible | [npj Flex Elec](https://doi.org/10.1038/s41528-021-00119-7) |
| **Tang 2025** | 4 | Headphone earmuff (peri-oral / jaw) | Textile graphene/PEDOT:PSS | [arXiv:2504.13921](https://arxiv.org/abs/2504.13921) |
| **Lai 2023** | 3 | LAO, DAO, ZM | Conventional | [arXiv:2308.06533](https://arxiv.org/abs/2308.06533) |
| **Deng 2023** | 8 (from HD) | ZYG, RIS, DAO, SCM, ABD, PLT, … | HD-guided sparse | [IEEE TIM](https://doi.org/10.1109/TIM.2023.3276540) |
| **MDPI 2025 pilot** | 8 | Digastric, DAO, risorius, LLS, masseter, ZYG, DLI, stylohyoid | Bipolar pairs | [Sensors 25(3):781](https://www.mdpi.com/1424-8220/25/3/781) |
| **Ji 2021 HD** | 10 optimal | Neck > face for digits | HD sEMG | [JNE](https://doi.org/10.1088/1741-2552/abca14) |
| **SilentWear 2025** | 10–14 diff | Neckband rows, overlapping diff | Dry SoftPulse textile | [arXiv:2603.02847](https://arxiv.org/abs/2603.02847) |
| **Gowda 2024** | 22 | Neck, jaw, chin, cheek, lips | Research cap | [OSF](https://osf.io/ym5jd/) |

---

## OpenAlterEgo default 8-channel map (AlterEgo-aligned)

Bipolar differential pairs — signal between two electrodes ~1–2 cm apart over the muscle belly:

| Ch | Muscle (anatomy) | Region | AlterEgo 2018 name |
|----|------------------|--------|-------------------|
| 1 | **Laryngeal / thyroid cartilage area** | Front neck | Laryngeal region |
| 2 | **Hyoid / suprahyoid** | Upper front neck | Hyoid region |
| 3 | **Levator anguli oris** | Upper lip / nasolabial | LAO |
| 4 | **Orbicularis oris** | Lip margin | OO |
| 5 | **Platysma** | Lower jaw / neck skin | Platysma |
| 6 | **Anterior digastric** | Under jaw, submental | Digastric |
| 7 | **Mentum / genioglossus area** | Chin | Mentum |
| 8 | **Masseter or SCM** (alternate) | Jaw angle or lateral neck | Supplementary |

**Reference / BIAS:** bilateral earlobe (2020 paper) or wrist (2018).

**Minimum viable 4-ch subset** (Wang / Tang layout): LAO, DAO, buccinator area, abdomen of digastric — for cost-reduced prototypes.

---

## Wiring configurations

### Bipolar differential (recommended V0)

```
CH_n = E+ over muscle − E− nearby (same muscle or tendon)
```

- Rejects common-mode noise
- Standard in Kapur, Wang, EMG-UKA

### Overlapping differential (V2 neckband)

Adjacent rows share a middle electrode → more channels with fewer physical sites ([SilentWear](https://arxiv.org/abs/2603.02847)).

### Referential (less common for us)

Single electrode vs common reference — EMG-UKA uses mastoid reference; higher CM noise sensitivity.

---

## Dry vs wet electrodes

| | **Wet Ag/AgCl + gel** | **Dry / textile** |
|---|----------------------|-------------------|
| **Impedance** | ~kΩ | ~100 kΩ+ ([Sensors 2023.16](https://www.oaepublish.com/articles/ss.2023.16)) |
| **SNR** | Higher, more stable | Lower; motion-sensitive |
| **Setup time** | Skin prep, gel, cleanup | Snap on and go |
| **Wear duration** | Gel dries 2–4 h | Multi-hour / multi-day ([Wang 2021](https://doi.org/10.1038/s41528-021-00119-7)) |
| **Literature tier** | V0 benchtop, clinical | V1+ wearable (Tang, SilentWear) |
| **Software implication** | Standard + clinical modes | Wide bandpass + SE channel weighting |

### Dry electrode engineering tactics

| Tactic | Source |
|--------|--------|
| Conformal / moss-stitch textile | [PMC9460933](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460933/) |
| Towel-like micro-fiber protrusions | Tang 2025 — improves skin coupling |
| Conductive polymer elastomer + Ag/AgCl coating | SilentWear comparison table |
| Tattoo / epidermal electronics | Wang 2021 — extreme conformal |

**OpenAlterEgo path:** Wet gel for V0 validation → iterate dry only after software pipeline proven.

---

## Placement procedure (V0)

1. **Clean skin** with alcohol wipe; light abrasion if dry electrodes
2. **Landmarks:** user subvocalizes "yes" / "no" while palpating jaw/neck for muscle activation
3. **Apply pairs** with 1–2 cm inter-electrode distance along muscle fiber direction
4. **Reference** on earlobe (avoid jaw motion)
5. **Impedance check** (if hardware supports lead-off): target stable contact before collection
6. **Rest recording** 30 s → baseline SNR per channel (`dsp/quality.py`)
7. **Token rehearsal** → `openalterego collect ble` with `events.csv` labels
8. **Photo diagram** saved to session metadata for repeatability

---

## Channel reduction guidance

From HD optimization literature ([Ji 2021](https://doi.org/10.1088/1741-2552/abca14); [Deng 2023](https://doi.org/10.1109/TIM.2023.3276540)):

1. **Neck channels carry more information than face** for many vocabularies
2. **10 well-placed channels** can match 120 HD channels for digit/word tasks
3. If reducing from 8 → 4, keep: **DAO, LAO/OO region, digastric, one neck (SCM or laryngeal)**

Software: future per-channel importance from trained model (Tang SE-ResNet motivation).

---

## EMG-UKA differences (historical note)

[Wand et al. 2014](https://www.isca-archive.org/interspeech_2014/wand14b_interspeech.html): 6 channels, 600 Hz, mastoid/ear reference, continuous speech — useful for preprocessing comparisons but different montage than AlterEgo.

---

## Session metadata (collect)

Record in `session.json` (supported by `DataCollectionSession`):

```json
{
  "electrode_type": "Ag/AgCl gel",
  "electrode_placement_notes": "2020 layout, earlobe ref",
  "channels_map": {"1": "laryngeal", "2": "hyoid", "...": "..."}
}
```

Enables cross-session comparison and literature-aligned documentation.

---

## Related docs

- Mechanical repeatability: [06-mechanical-wearable.md](06-mechanical-wearable.md)
- AFE input impedance / bias: [02-analog-front-end.md](02-analog-front-end.md)
- Full paper list: [07-references.md](07-references.md)
