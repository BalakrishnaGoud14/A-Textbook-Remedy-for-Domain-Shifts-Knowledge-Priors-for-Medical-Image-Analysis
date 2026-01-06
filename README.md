# A Textbook Remedy for Domain Shifts: Knowledge Priors for Medical Image Analysis

This repository contains our **re-implementation and extension** of the *Knowledge-Enhanced Bottleneck (KnoBo)* framework for **robust and interpretable medical image analysis under domain shift**.

The project demonstrates how **explicit medical knowledge**, **concept bottlenecks**, and **structured priors** significantly improve **out-of-distribution (OOD) generalization** compared to black-box deep learning models.

---

##  Project Overview

Deep learning models achieve high accuracy on in-domain medical imaging tasks but often **fail catastrophically under domain shift**, caused by:
- Demographic differences (age, sex, race)
- Hospital and scanner variations
- Acquisition protocols and artifacts

To address this, we build upon **KnoBo**, a **knowledge-guided concept bottleneck model**, and introduce:

-  **Medical concept bottlenecks** grounded in textbooks and PubMed
-  **Knowledge-based parameter priors**
-  **Spatial & channel attention mechanisms**
-  **Robust preprocessing (CLAHE) and augmentation**
-  **Optuna-based hyperparameter optimization**

We evaluate the approach across **20 datasets** covering:
- **Chest X-rays**
- **Skin lesion images**
- **Confounded & unconfounded distributions**

---

##  Key Contributions

1. Reproduced KnoBo results with strong alignment to the original paper  
2. Achieved **20%+ improvement in OOD accuracy** over ViT and DenseNet baselines  
3. Reduced **domain gap by ~41.8%** on chest X-ray tasks  
4. Added **attention mechanisms** for better clinical focus  
5. Preserved **interpretability via concept-level reasoning**  

---

##  Architecture

The model consists of three core components:

1. **Structure Prior**
   - Retrieval-augmented medical concepts generated from:
     - PubMed
     - Medical textbooks
     - Clinical resources
   - Concepts include:
     - Ground-glass opacity
     - Lesion asymmetry
     - Border irregularity
     - Trachea deviation

2. **Concept Bottleneck Predictor**
   - Maps images → interpretable clinical concepts
   - Concepts are predicted using weak supervision from reports/captions

3. **Parameter Prior**
   - Enforces medically meaningful concept–label relationships
   - Regularizes classifier weights using knowledge-based sign constraints

 **Enhanced Architecture**
- Added **spatial + channel attention**
- Integrated **CLAHE preprocessing**
- Strong data augmentation pipeline

---

##  Datasets

### Chest X-ray
- NIH-CXR
- CheXpert
- Pneumonia
- COVID-QU
- Open-i
- VinDr-CXR

### Skin Lesion
- ISIC
- HAM10000
- BCN20000
- PAD-UFES-20
- Melanoma
- UWaterloo

Each dataset includes:
- **In-Domain (ID) splits**
- **Out-of-Domain (OOD) splits** with flipped confound correlations

---

##  Results Summary

| Model | Avg OOD Gain |
|------|-------------|
| ViT-L/14 |  Large drop |
| DenseNet-121 |  Large drop |
| LSL / PCBM-h |  Moderate |
| **KnoBo (Ours)** |  **+20% OOD accuracy** |

- Minimal ID–OOD performance gap
- Consistent gains across modalities
- Strong interpretability with concept-level explanations

---

##  Experimental Enhancements

### Attention Mechanism
- Improves localization of diagnostically relevant regions
- Especially effective for chest X-rays

### CLAHE + Augmentation
- Improves contrast and robustness
- Stabilizes concept predictions across domains

### Hyperparameter Optimization
- Optuna-based search
- Tuned learning rate, regularization, augmentation strength

---

##  Implementation Details

- Language: **Python**
- Frameworks: **PyTorch**
- Training:
  - Single GPU (16–24 GB)
  - Deterministic seeds for reproducibility
- Modular design:
  - Backbone
  - Concept bottleneck
  - Attention module (optional)
  - Knowledge-guided classifier

---

## Contributors

- Balakrishna Goud Malela
- Bharathi Thanikonda
- Korukanti Deekshitha
- Nithin Kumar Jada
- Vaishnavi Peddaboina



##  How to Use

```bash
git clone https://github.com/BalakrishnaGoud14/A-Textbook-Remedy-for-Domain-Shifts-Knowledge-Priors-for-Medical-Image-Analysis.git
cd A-Textbook-Remedy-for-Domain-Shifts-Knowledge-Priors-for-Medical-Image-Analysis
