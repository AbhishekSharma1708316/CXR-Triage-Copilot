# CXR-Triage Copilot

**Explainable chest X-ray triage with a guideline-grounded clinical assistant.**
A research prototype built on the RSNA Pneumonia Detection Challenge data.

> **Not a medical device.** This project is educational and experimental. It does not diagnose, prescribe, or replace a clinician. India's Telemedicine Practice Guidelines (2020) do not allow AI platforms to counsel patients or prescribe medicines, so the language-model component of this project only *explains* computed results and never suggests drugs or doses.

---

## Overview

Chest X-ray queues are long, and urgent cases can wait behind routine ones. This project builds a triage assistant that:

1. classifies a chest X-ray into a triage level (**Routine / Review soon / Urgent**),
2. shows *where* the model looked (Grad-CAM) and checks that against radiologist annotations,
3. combines the image result with patient intake details through a transparent, rule-based clinical layer,
4. uses an LLM to write a clinician hand-off summary and a plain-language explanation, grounded in public guidelines and verified by a safety layer.

The LLM never makes the decision. Decisions come from deterministic, unit-tested code; the LLM communicates them.

## Status

| Component | Status |
|---|---|
| Data audit and leak-free split | Done |
| Baselines (view-only, small CNN) | Done |
| DenseNet121 transfer learning + view-balanced ablation | Done (validation set) |
| Threshold selection and calibration | Next |
| Grad-CAM + localization check against radiologist boxes | Planned |
| Error analysis and final test-set evaluation | Planned |
| External validation on a second dataset | Planned |
| Clinical rules layer (CRB-65 + red flags) | Planned |
| Guideline RAG + LLM explainer + output verifier | Planned |
| Evaluation harness (safety, faithfulness, retrieval) | Planned |
| API and demo UI | Planned |

## Triage mapping

| Dataset class | Triage level |
|---|---|
| Normal | Routine |
| No Lung Opacity / Not Normal | Review soon |
| Lung Opacity | Urgent |

## Planned architecture

```
Intake form (age, symptoms, RR, SpO2, BP, confusion...)  +  chest X-ray
        |                                                      |
        v                                                      v
 Clinical rules layer                          DenseNet121 -> class probabilities
 (red flags + CRB-65)                          + Grad-CAM + lung zone
        +--------------------------+--------------------------+
                                   v
            Final priority = the MORE urgent of image / clinical
                                   v
              RAG over public pneumonia guidelines
                                   v
     LLM writer (JSON schema, every claim cites a retrieved passage)
       -> clinician hand-off summary
       -> plain-language explanation + questions to ask a doctor
       -> warning signs / when to seek emergency care
                                   v
     Verifier: blocks drug names/doses, checks text matches the computed
     priority, validates citations, falls back to a template on failure
```

Design rules: red flags can only escalate, never de-escalate; missing intake fields count as "unknown", never as normal; no diagnoses, drugs or doses in any output.

## Data

- **Source:** RSNA Pneumonia Detection Challenge (Kaggle), derived from NIH ChestX-ray14. The data is **not included** in this repository; see *Reproducing* below.
- **Size:** 26,684 frontal radiographs, one per patient ID; DICOM, 1024x1024, 8-bit.
- **Classes:** Normal 8,851 | No Lung Opacity / Not Normal 11,821 | Lung Opacity 6,012 (9,555 radiologist bounding boxes).
- **Split:** 70/15/15, stratified by class, seed 42 -> 18,678 train / 4,003 val / 4,003 test. Every image has a unique patient ID, so the split is leak-free at patient level.
- **Preprocessing:** resized to 256x256, stored as one `uint8` array (1.75 GB), ImageNet normalization, grayscale copied to 3 channels. Training augmentation: rotation up to 7 degrees, random resized crop (scale 0.9-1.0), brightness/contrast jitter; no flips.

### Data audit: the AP/PA view is a confounder

| Class | AP | PA |
|---|---|---|
| Lung Opacity | 0.78 | 0.22 |
| No Lung Opacity / Not Normal | 0.50 | 0.50 |
| Normal | 0.18 | 0.82 |

AP (often bedside, portable) films come disproportionately from sicker patients, so a model can partly "cheat" by learning the view. A classifier using **only** the AP/PA flag reaches macro-F1 0.376 (accuracy 0.444). All results below are therefore also reported per view.

Other audit notes: all images are MONOCHROME2 (no inversion needed); sex is balanced across classes; the age field contains invalid values (e.g. 153, 155) and some young children (minimum age 1), so age is not used as a model input.

## Method

| Model | Description |
|---|---|
| View-only baseline | Logistic regression on the AP/PA flag only (no image) |
| Small CNN | 4 conv blocks, trained from scratch, 8 epochs, lr 1e-3 |
| DenseNet121 | ImageNet-pretrained, new head (dropout 0.2 + linear), class-weighted cross-entropy |
| DenseNet121 + view-balanced | Same, but sampling weights equalise every (class, view) group instead of class weights |

DenseNet training: AdamW (weight decay 1e-4), OneCycle learning rate (peak 3e-4), mixed precision, batch 64, up to 10 epochs, early stopping (patience 4) on validation macro-AUROC, best checkpoint restored. Seed 42. About 118 s/epoch on a Kaggle T4.

## Results (validation set)

| Model | Macro-F1 | Macro-AUROC | Urgent (Lung Opacity) recall | AP macro-F1 | PA macro-F1 | PA Urgent recall |
|---|---|---|---|---|---|---|
| View-only (AP/PA) | 0.376 | - | 0.785 (precision 0.379) | - | - | - |
| Small CNN | 0.620 | 0.828 | 0.532 | 0.594 | 0.512 | 0.144 |
| **DenseNet121** | 0.693 | **0.882** | **0.757** | 0.628 | 0.677 | **0.521** |
| DenseNet121 + view-balanced | **0.703** | 0.862 | 0.561 | 0.647 | 0.673 | 0.366 |

DenseNet121 per class (argmax decision): Normal P 0.762 / R 0.911; Not Normal P 0.767 / R 0.509; Lung Opacity P 0.551 / R 0.757; accuracy 0.698.

![DenseNet121 confusion matrix](results/densenet121_plain_confusion.png)
![DenseNet121 training curves](results/densenet121_plain_curves.png)

**All numbers above are validation results, and the best epoch was also chosen on validation, so they are slightly optimistic. The test set has not been used yet and will be evaluated once, for the final model.**

### Findings

1. **Pretraining helps.** DenseNet121 beats the small CNN by +0.073 macro-F1 and +0.054 macro-AUROC, and raises Urgent recall from 0.53 to 0.76.
2. **The AP/PA shortcut is real, and a simple fix did not work.** View-balanced sampling was meant to improve PA performance. It did not: PA Urgent recall fell from 0.521 to 0.366 and macro-AUROC from 0.882 to 0.862 (run-to-run noise is about 0.005). Its slightly higher macro-F1 (+0.010) is within epoch-to-epoch variation. One possible explanation, not yet tested, is that the view carries legitimate clinical context (sicker patients are more often imaged at the bedside), so removing it discards useful signal. This is reported as a negative result.
3. **Overfitting after about epoch 5.** Training loss keeps falling while validation loss rises; early stopping and best-checkpoint restoring handle this.
4. **"Not Normal" is the hardest class** (recall 0.51). It is a catch-all label covering many different findings.
5. **Argmax comparisons are incomplete for triage.** The next step compares models at a matched Urgent recall (threshold selection on validation data), which is what a triage system actually uses.

## Limitations

- Single source dataset; no external validation yet.
- Mostly adult patients, with some children; labels are coarse and "Not Normal" is heterogeneous.
- No real patient intake data exists for this dataset, so the clinical-rules layer will be validated on synthetic cases built from published criteria. That is engineering validation, not clinical validation.
- Residual dependence on the AP/PA view remains (PA Urgent recall is lower than AP).
- Research prototype only; not for clinical use.

## Reproducing

1. Create a Kaggle account and join the RSNA Pneumonia Detection Challenge (accept the rules).
2. Run `notebooks/01_data_prep.ipynb` on Kaggle (CPU). It produces `images_256.npy`, `train.csv`, `val.csv`, `test.csv`, `boxes.csv`. Save the output as a private dataset (`rsna-png256-splits`).
3. Run `notebooks/02_training.ipynb` on Kaggle (GPU T4, Internet on) with that dataset attached. It writes metrics, curves, confusion matrices and weights to `results/`.

The competition data is subject to its terms of use and is not redistributed here.

## Repository structure

```
cxr-triage-copilot/
├── README.md
├── notebooks/
│   ├── 01_data_prep.ipynb
│   └── 02_training.ipynb
├── results/                 (metrics JSON, curves, confusion matrices, comparison.csv)
├── src/                     (planned: imaging, clinical, rag, llm, eval)
├── app/                     (planned: API + demo UI)
└── tests/                   (planned)
```

## Acknowledgements

RSNA and Kaggle for the challenge data; NIH for ChestX-ray14; DenseNet (Huang et al., 2017); Grad-CAM (Selvaraju et al., 2017).

## Author

Abhishek Sharma, B.Tech Artificial Intelligence & Data Science, IIIT Raichur.

## License

Code: MIT (to be confirmed). Data and trained weights follow the terms of the source dataset.
