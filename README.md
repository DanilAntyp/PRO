# Diabetic Retinopathy Severity Classification

A deep-learning pipeline that classifies retinal fundus images into the five clinical grades of diabetic retinopathy (DR) using an ensemble of two convolutional neural networks. The project addresses severe class imbalance and is evaluated with Quadratic Weighted Kappa (QWK), the standard ordinal metric for DR grading.

---

## Overview

Diabetic retinopathy is a leading cause of preventable blindness. Early detection through retinal screening is critical, but manual grading does not scale. This project automates severity grading on the five-point International Clinical Diabetic Retinopathy scale:

| Grade | Class | Description |
|-------|-------|-------------|
| 0 | No DR | No abnormalities |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than microaneurysms, less than severe |
| 3 | Severe | Extensive haemorrhages / venous beading |
| 4 | Proliferative | Neovascularization; most advanced stage |

The grades are **ordinal**, so the project optimizes for QWK (which penalizes errors by their distance on the scale) rather than raw accuracy.

---

## Results

Test-set performance across models and inference strategies:

| Model / Strategy | Accuracy | QWK |
|------------------|----------|-----|
| Baseline (EfficientNet-B4) | 83.06% | 0.8966 |
| Model 1 (EffNet-B4 + improvements + TTA) | 82.24% | 0.8833 |
| Model 2 (ResNet50 + improvements + TTA) | 80.60% | 0.8765 |
| Ensemble (mean softmax) | 83.33% | 0.8928 |
| Ensemble (OR rule) | 82.24% | 0.8965 |
| **Ensemble (threshold-optimized)** | **78.96%** | **0.9020** |

The **threshold-optimized ensemble** is selected as the final model: it achieves the highest QWK, the clinically appropriate metric for ordinal severity grading.

---

## Pipeline

1. **Preprocessing** — Ben Graham normalization (Gaussian-blur subtraction to enhance vascular contrast) + CLAHE (adaptive histogram equalization).
2. **Augmentation** — Albumentations: flips, shift/scale/rotate, brightness/contrast, hue/saturation, Gaussian blur, CoarseDropout.
3. **Imbalance handling** — Focal Loss (γ=2) + Weighted Random Sampler (weights ∝ 1/√class_count).
4. **Regularization** — MixUp (p=0.3, α=0.2), dropout, weight decay, early stopping on validation QWK.
5. **Architectures** — EfficientNet-B4 and ResNet50, both pretrained on ImageNet.
6. **Inference** — Test-Time Augmentation (×4 flips) + threshold optimization (ordinal-regression decoding).

---

## Repository Structure

```
.
├── Untitled1.ipynb              # Full pipeline: preprocessing → training → evaluation → interpretation
├── README.md                    # This file
├── DR_Thesis.docx               # Final thesis document
├── model_comparison.csv         # Results table
└── figures/
    ├── dataset_distribution_splits.png
    ├── confusion_matrices_grid.png
    ├── learning_curves.png
    ├── model_comparison_chart.png
    ├── gradcam_gallery.png
    ├── misclassified_grid.png
    └── error_analysis.png
```

Trained model checkpoints (`best_baseline.pth`, `best_model.pth`, `best_resnet.pth`) are saved to Google Drive during training.

---

## Setup

This project is designed to run in **Google Colab with a GPU runtime**.

1. Open `Untitled1.ipynb` in Google Colab.
2. Set the runtime to GPU: **Runtime → Change runtime type → GPU**.
3. Mount Google Drive (the notebook does this in an early cell) and place the following under `/content/drive/MyDrive/`:
   - `train_1.csv`, `valid.csv`, `test.csv` (image IDs + diagnosis labels)
   - the corresponding image folders (`train_images/`, `valid_images/`, `test_images/`)
4. Required packages are installed by the first cells:
   ```python
   !pip install albumentations timm scipy
   ```
   (`torch`, `torchvision`, `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas` are pre-installed in Colab.)

---

## How to Run

Run the notebook top to bottom: **Runtime → Run all**.

Execution order:

1. Imports and device setup
2. Google Drive mount
3. Preprocessing functions (Ben Graham, CLAHE) and Dataset class
4. Augmentation transforms and DataLoaders (with weighted sampler)
5. Dataset statistics visualization
6. Model class definitions (EfficientNet-B4, ResNet50)
7. Focal Loss definition
8. `train_model` function
9. Baseline training
10. Model 1 and Model 2 training
11. Per-model and ensemble evaluation (confusion matrices, per-class metrics)
12. OR-rule and threshold-optimized evaluation
13. Learning curves and comparison chart
14. Grad-CAM interpretation and error analysis

> **After a runtime restart:** rerun the model-reload cell to load saved checkpoints into memory before running the evaluation/interpretation cells, otherwise `model_eff`/`model_res` will be undefined.

---

## Methodology Notes

- **Why QWK over accuracy?** DR grades are ordinal. Misclassifying Severe as Moderate (one grade off) is far less serious than misclassifying it as No DR (four grades off). QWK weights errors by squared distance; accuracy treats all errors equally.
- **Why an ensemble of two architectures?** EfficientNet-B4 (compound scaling, depthwise-separable convolutions) and ResNet50 (residual standard convolutions) have different inductive biases and make uncorrelated errors, which average out when combined.
- **Fair comparison.** All models share the same preprocessing, augmentation, optimizer, and learning-rate schedule. Only the deliberately-varied factors differ (loss/sampler/MixUp/TTA between baseline and improved models; backbone between Model 1 and Model 2). The test set is used only once, for final evaluation.

---

## Limitations

- Small, single-source, severely imbalanced dataset (only ~17 Severe and ~33 Proliferative test samples → wide confidence intervals).
- No cross-dataset validation; performance may drop under domain shift to other clinics or cameras.
- The model detects disease presence reliably but tends to under-grade severity. It is a triage aid, not an autonomous diagnostic system.

---

## Future Work

- Incorporate external datasets (Messidor, IDRiD, EyePACS) for more minority-class samples and cross-dataset validation.
- Replace the softmax head with a dedicated ordinal-regression head (CORAL/CORN).
- Train at higher resolution to capture small neovascular lesions.
- Model calibration and prospective cross-site validation.

---

## Dataset

APTOS 2019 Blindness Detection — Asia Pacific Tele-Ophthalmology Society. Retinal fundus photographs labeled by clinical experts with DR severity grades.

## References

1. Gulshan et al., *Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy*, JAMA, 2016.
2. Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML, 2019.
3. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
4. Lin et al., *Focal Loss for Dense Object Detection*, ICCV, 2017.
5. Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR, 2018.
6. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV, 2017.
7. Cohen, *Weighted Kappa*, Psychological Bulletin, 1968.
