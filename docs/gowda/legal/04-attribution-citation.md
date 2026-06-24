# Attribution & Citation

Cite the **paper**, **dataset DOI**, and **software** when using Gowda materials in publications, demos, or derivative datasets.

---

## BibTeX (2025 — primary)

```bibtex
@article{gowda2025neuroprosthesis,
  title={Non-invasive electromyographic speech neuroprosthesis: a geometric perspective},
  author={Gowda, Harshavardhana T. and Miller, Lee M.},
  journal={arXiv preprint arXiv:2502.05762},
  year={2025},
  url={https://arxiv.org/abs/2502.05762}
}
```

## BibTeX (2024 — geometry / SPD foundation)

```bibtex
@article{gowda2024geometry,
  title={Geometry of orofacial neuromuscular signals: speech articulation decoding using surface electromyography},
  author={Gowda, Harshavardhana T. and McNaughton, Zachary D. and Miller, Lee M.},
  journal={Journal of Neural Engineering},
  year={2024},
  url={https://arxiv.org/abs/2411.02591}
}
```

## Dataset

```bibtex
@misc{gowda_osf_ym5jd,
  author = {Gowda, Harshavardhana T. and Miller, Lee M.},
  title = {Orofacial EMG Silent Speech Data (OSF)},
  year = {2024},
  doi = {10.17605/OSF.IO/YM5JD},
  url = {https://doi.org/10.17605/OSF.IO/YM5JD}
}
```

## Software

```bibtex
@misc{emg2speech_github,
  author = {Gowda, Harshavardhana T.},
  title = {emg2speech},
  year = {2025},
  url = {https://github.com/HarshavardhanaTG/emg2speech}
}
```

---

## In-text attribution examples

- “We evaluate on the Gowda small-vocabulary OSF corpus (Gowda & Miller, 2025, App. C.1).”
- “Preprocessing follows Gowda et al. (2025, App. B): reference subtraction, 80–1000 Hz bandpass, per-channel z-normalization.”
- “OpenAlterEgo word-classification results are **not** comparable to reported PER/WER without implementing the SPD+CTC pipeline.”

---

## OpenAlterEgo validation docs

When citing **this repository’s** benchmark numbers, point to:

- [../validation/02-top30-corrected.md](../validation/02-top30-corrected.md)
- Code: `software/python/openalterego/ml/datasets/gowda.py`
