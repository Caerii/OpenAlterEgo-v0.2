# Hardware references

Bibliography for electrode, AFE, wearable, and safety decisions. General ML/DSP papers are in [`docs/12-references.md`](../docs/12-references.md).

---

## A. Surveys and clinical context

| ID | Citation | Hardware relevance |
|----|----------|-------------------|
| H-S1 | González-López et al. (2020). *Silent Speech Interfaces for Speech Restoration: A Review.* IEEE Access. [DOI](https://doi.org/10.1109/ACCESS.2020.3026579) | Modalities overview; EMG for laryngectomy; lab vs real-world gap |
| H-S2 | Tang et al. (2026). *Sensing technologies for silent speech interfaces.* Nature Sensors 1:16–26. [Nature](https://www.nature.com/articles/s44460-025-00010-2) · [PDF](https://www.nature.com/articles/s44460-025-00010-2.pdf) · [Offline extract](../docs/literature/archive/tang-2026-nature-sensors-sensing-technologies.md) | Off/on/in-body taxonomy; on-body EMG deployability; multimodal fusion; edge AI |

---

## B. AlterEgo foundation (hardware parameters)

| ID | Citation | Key hardware specs |
|----|----------|-------------------|
| H-A1 | Kapur, Kapur, Maes (2018). *AlterEgo.* IUI. [ACM](https://dl.acm.org/doi/10.1145/3172944.3172977) | 7 ch; 250 Hz; 24× gain; Au/Ag + Ten20; head band + brass arms; BLE |
| H-A2 | Kapur et al. (2020). *Dysphonic MS.* ML4H. [PMLR](https://proceedings.mlr.press/v136/kapur20a.html) | 8 ch (4 face + 4 neck); 24-bit; earlobe ref/bias; 250 Hz |
| H-A3 | US10878818B2 — *Methods and Apparatus for Silent Speech Interface.* [Patents](https://patents.google.com/patent/US10878818B2) | System blocks, isolation, wireless relay, safety |

---

## C. Wearable form factors and electrodes

| ID | Citation | Form factor | Electrodes | fs |
|----|----------|-------------|------------|-----|
| H-W1 | Wang et al. (2021). Tattoo-like electronics. npj Flex Elec. [DOI](https://doi.org/10.1038/s41528-021-00119-7) | Epidermal patch | 4 flexible | 500 Hz |
| H-W2 | Tang et al. (2025). Headphone textile EMG. IEEE TIM / [arXiv:2504.13921](https://arxiv.org/abs/2504.13921) | Headphone earmuff | 4 dry textile | 1000 Hz |
| H-W3 | Meier et al. (2025). SilentWear. [arXiv:2603.02847](https://arxiv.org/abs/2603.02847) | Textile neckband | 14 diff dry | 500 Hz |
| H-W4 | Meier et al. (2025). Fully-dry neckband (conference). [arXiv:2509.21964](https://arxiv.org/abs/2509.21964) | Neckband | 14 diff | Ultra-low power |

---

## D. Electrode placement and HD optimization

| ID | Citation | Finding |
|----|----------|---------|
| H-E1 | Ji et al. (2021). HD electrode optimization. J. Neural Eng. [DOI](https://doi.org/10.1088/1741-2552/abca14) | Neck > face; 10 ch ≈ 86%+ |
| H-E2 | Deng et al. (2023). Few sites from HD guidance. IEEE TIM. [DOI](https://doi.org/10.1109/TIM.2023.3276540) | 8 sparse sites from HD arrays |
| H-E3 | MDPI Sensors (2025). Electrode setup pilot. [Link](https://www.mdpi.com/1424-8220/25/3/781) | 8-muscle Spanish laryngectomy montage |
| H-E4 | Zhang et al. (2023). HD-sEMG hybrid nets. IEEE THMS. [DOI](https://doi.org/10.1109/THMS.2022.3226197) | 64 ch HD; anomaly patterns |

---

## E. Dry electrode materials

| ID | Citation | Topic |
|----|----------|-------|
| H-D1 | Li et al. (2023). Soft dry electrodes review. Soft Science. [Link](https://www.oaepublish.com/articles/ss.2023.16) | Impedance; conformal contact |
| H-D2 | Kim et al. (2022). Textile dry electrode fabrication. [PMC9460933](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460933/) | Moss-stitch vs plain; skin impedance |
| H-D3 | Repository.cam.ac.uk — Tang headphone SSI. [Link](https://www.repository.cam.ac.uk/items/1f3ece2b-c12c-4158-b80e-59a1f22e3bb8) | Graphene/PEDOT:PSS textile |

---

## F. Analog front-end and platforms

| ID | Citation | Topic |
|----|----------|-------|
| H-F1 | Texas Instruments. ADS1299 datasheet. [TI](https://www.ti.com/lit/ds/symlink/ads1299.pdf) | 8-ch 24-bit; 250 SPS–16 kSPS; PGA; BIAS |
| H-F2 | OpenBCI Cyton documentation. [openbci.com](https://docs.openbci.com/) | ADS1299 dev platform; AlterEgo lineage |
| H-F3 | BioGAP-Ultra platform (cited in SilentWear). | 2× ADS1298; nRF5340; 16 diff ch |
| H-F4 | Wand et al. (2014). EMG-UKA Corpus. INTERSPEECH. [Link](https://www.isca-archive.org/interspeech_2014/wand14b_interspeech.html) | 6 ch; 600 Hz; Varioport recorder |

---

## G. Acquisition parameters (cross-reference)

| Parameter | Papers supporting range | OpenAlterEgo doc |
|-----------|-------------------------|------------------|
| 250 Hz fs | H-A1, H-A2 | [02-analog-front-end.md](02-analog-front-end.md) |
| 500–1000 Hz fs | H-W1, H-W2, H-W3 | [02-analog-front-end.md](02-analog-front-end.md) |
| 4 ch minimum | H-W1, H-W2, Lai 2023 (software bib) | [03-electrodes-montage.md](03-electrodes-montage.md) |
| 7–8 ch | H-A1, H-A2 | [03-electrodes-montage.md](03-electrodes-montage.md) |
| 24× gain | H-A1 | [02-analog-front-end.md](02-analog-front-end.md) |
| 18.9 / 12.7 dB SNR | H-W2 | [02-analog-front-end.md](02-analog-front-end.md) |
| BLE transport | H-A1, H-A3 | [04-ble-firmware-protocol.md](04-ble-firmware-protocol.md) |

---

## H. Standards (reference only)

| Standard | Topic |
|----------|-------|
| IEC 60601-1 | Medical electrical equipment safety |
| IEC 60601-2-26 | EMG/EP equipment particular standard |
| IEC 60601-1-11 | Home healthcare environment |
| ISO 13485 | Medical device QMS |
| Bluetooth SIG Core 5.x | BLE compliance for product |

See [05-power-safety.md](05-power-safety.md).

---

## BibTeX (hardware-focused)

```bibtex
@inproceedings{kapur2018alterego,
  title={AlterEgo: A Personalized Wearable Silent Speech Interface},
  author={Kapur, Arnav and Kapur, Shreyas and Maes, Pattie},
  booktitle={IUI},
  year={2018}
}

@article{wang2021tattoo,
  title={All-weather, natural silent speech recognition via tattoo-like electronics},
  author={Wang, Youhua and others},
  journal={npj Flexible Electronics},
  volume={5},
  number={20},
  year={2021}
}

@article{tang2025headphone,
  title={Wireless Silent Speech Interface Using Multichannel Textile EMG in Headphones},
  author={Tang, Chenyu and others},
  journal={IEEE TIM},
  year={2025},
  note={arXiv:2504.13921}
}

@article{meier2025silentwear,
  title={SilentWear: Ultra-Low Power Wearable EMG Silent Speech},
  author={Meier, F. and others},
  journal={arXiv:2603.02847},
  year={2025}
}

@article{ji2021hd,
  title={Towards optimizing electrode configurations for SSR based on HD-sEMG},
  author={Ji, Lin and others},
  journal={Journal of Neural Engineering},
  year={2021}
}

@article{deng2023few,
  title={Silent Speech Recognition Using Few Electrode Sites Under Guidance From HD Arrays},
  author={Deng, Zhihang and others},
  journal={IEEE TIM},
  volume={72},
  year={2023}
}

@misc{ti_ads1299,
  title={{ADS1299} datasheet},
  author={{Texas Instruments}},
  howpublished={\url{https://www.ti.com/product/ADS1299}},
  year={2024}
}
```

---

## Parameter decision tree (quick lookup)

```
Need fastest real EMG?
  → V0 OpenBCI Cyton (H-F2, H-A1)

Need discreet daily wear?
  → Neckband (H-W3) OR headphone (H-W2)

Need minimum channels?
  → 4 ch (H-W1, H-W2) + software SE weighting

Need AlterEgo-compatible fs/montage?
  → 8 ch @ 250 Hz, earlobe ref (H-A1, H-A2)

Need wide DSP band (20–450 Hz)?
  → fs ≥ 500 Hz (H-W1, H-W2, H-W3)

Need ultra-low power?
  → SilentWear / BioGAP pattern (H-W3, H-W4)
```
