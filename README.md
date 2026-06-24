# KARAT: Knowledge-Aware Resilience via Adaptive Teacher-Student Distillation

> **Maintaining cloud service priority ranking precision under adversarial label-drift using CKA-triggered teacher-student correction.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange)]()

---

## Overview

KARAT is a resilience framework for **knowledge-distilled cloud service triage models** deployed under adversarial drift. A lightweight student model (MLP, 16 hidden units) is continuously monitored via **Centred Kernel Alignment (CKA)** against a frozen teacher ensemble. When representation-space fidelity drops below a learned threshold θ, KARAT blends student and teacher scores to recover ranking precision.

### Key Results (Precision@5K, 5 attack seeds)

| Scenario | B2 Student | B3 L2 | **KARAT** | Oracle |
|---|---|---|---|---|
| S1 Targeted | 0.858 ± 0.090 | 0.921 ± 0.099 | **0.909 ± 0.038** | 0.977 ± 0.033 |
| S2 Coordinated | 0.909 ± 0.030 | 0.909 ± 0.030 | **0.914 ± 0.024** | 0.984 ± 0.019 |
| S3 Cascading | 0.912 ± 0.022 | 0.912 ± 0.022 | **0.915 ± 0.020** | 0.990 ± 0.007 |

**Cross-domain (CIC-IDS2018):** KARAT recovers **+0.057 Precision@5K** (triggered timesteps), exceeding the oracle at peak drift (t=10).

---

## Repository Structure

```
KARAT/
├── src/
│   ├── dataset.py          # Synthetic KATS dataset generator
│   ├── model.py            # Teacher ensemble + student MLP builder
│   ├── attack.py           # Adversarial injection (S1/S2/S3 scenarios)
│   ├── metrics.py          # Precision@K, CKA, L2 divergence
│   └── karat.py            # Core KARAT correction logic
├── experiments/
│   ├── E2_multiseed.py     # Main 5-seed × 3-scenario experiment
│   ├── E5_cic_validation.py# CIC-IDS2018 cross-domain validation
│   └── E6_cka_vs_l2.py    # CKA vs L2 trigger ablation
├── results/
│   ├── E2_multiseed_all.csv
│   ├── E5_CIC_IDS2018_validation.csv
│   ├── E6_cka_vs_l2_ablation.csv
│   └── wilcoxon_significance.csv
├── notebooks/
│   └── KARAT_full_pipeline.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Test set size | 15,000 |
| High-label services | 4,500 (30%) |
| K (primary, 5%) | 750 (16.7% of High) |
| K (secondary, 10%) | 1,500 (33.3% of High) |
| Attack seeds | 42, 7, 13, 99, 2024 |
| Timesteps | 0, 2, 4, 6, 8, 10 |
| Teacher | Gradient Boosting ensemble (α=5) |
| Student | MLP (16 hidden, 8% training data) |
| CKA θ | 0.00279 (val p20, trigger rate 0.80) |
| L2 θ | 0.000988 (positive val drops only) |

### Attack Scenarios

- **S1 Targeted:** 75% of High-label services corrupted, noise scale 0.40
- **S2 Coordinated:** 70% of all services, noise scale 0.55
- **S3 Cascading:** High-dependency seed nodes → cascade to downstream_critical services

---

## Sanity Check: PAK Degradation under S1 (seed=42)

```
t  | B2_frozen  live_s   KARAT   oracle   cka_d
------------------------------------------------
 0 |   0.9387   0.9387  0.9387  0.9973  0.0000
 2 |   0.9387   0.9293  0.9333  0.9960  0.0075
 4 |   0.9387   0.9133  0.9333  0.9947  0.0190
 6 |   0.9387   0.8680  0.9227  0.9920  0.0392
 8 |   0.9387   0.8120  0.8987  0.9747  0.0672
10 |   0.9387   0.6907  0.8360  0.9133  0.0909
```

---

## E5: CIC-IDS2018 Cross-Domain Validation

```
t  | B2_stud   KARAT   Oracle  Gain_B2   cka_d  trig
----------------------------------------------------
 0 |  0.8010  0.8010  0.9430  +0.0000  0.0000     ·
 2 |  0.7920  0.8000  0.9420  +0.0080  0.0046     ▲
 4 |  0.7900  0.8250  0.9320  +0.0350  0.0178     ▲
 6 |  0.7670  0.8240  0.9160  +0.0570  0.0361     ▲
 8 |  0.7250  0.8130  0.8670  +0.0880  0.0510     ▲
10 |  0.6550  0.7530  0.7300  +0.0980  0.0826     ▲
```

> **Notable:** At t=10, KARAT (0.753) exceeds the oracle (0.730), indicating CKA detects representational drift before teacher accuracy fully degrades.

---

## E6: CKA vs L2 Trigger Ablation

Under **S1 targeted** attack, L2 divergence provides competitive signal at moderate drift (t=2–8) because teacher and student co-move positively on corrupted features. At extreme drift (t=10), CKA outperforms L2 by **+0.145 PAK**. Under **S2/S3**, L2 divergence turns negative (co-movement), disabling the L2 trigger entirely — CKA continues to trigger and correct.

---

## Installation

```bash
git clone https://github.com/hamidborkot/KARAT.git
cd KARAT
pip install -r requirements.txt
```

## Quickstart

```python
from src.dataset import generate_kats_syn
from src.model import build_kats_ensemble
from src.karat import KARATCorrector

df = generate_kats_syn(n=75000, seed=42)
teacher = build_kats_ensemble(alpha=5, seed=42)
# ... fit teacher, train student, then:
corrector = KARATCorrector(teacher, student, scaler, theta=0.00279)
corrected_scores = corrector.predict_proba(df_test)
```

## Run Experiments

```bash
python experiments/E2_multiseed.py
python experiments/E5_cic_validation.py
python experiments/E6_cka_vs_l2.py
```

---

## Citation

```bibtex
@misc{tulla2026karat,
  title  = {KARAT: Knowledge-Aware Resilience via Adaptive Teacher-Student
             Distillation for Cloud Service Priority Ranking},
  author = {Tulla, MD Hamid Borkot},
  year   = {2026},
  url    = {https://github.com/hamidborkot/KARAT}
}
```

## License

MIT — see [LICENSE](LICENSE).
