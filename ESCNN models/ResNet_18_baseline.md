##  The Dataset: PatchCamelyon (PCam)
To audit the geometric stability of these architectures, we utilized the **PatchCamelyon (PCam)** dataset. 

* **The Data:** 327,680 color images (96x96 pixels) extracted from histopathologic scans of breast cancer lymph node sections.
* **The Objective:** Binary classification. The network must predict whether the central 32x32 pixel region contains at least one pixel of malignant tumor tissue.
* **Why PCam?** In clinical pathology, tissue orientation is arbitrary. A malignant cell cluster remains malignant whether the slide is viewed at 0°, 90°, or upside down. PCam contains perfectly unoriented images, making it the ideal benchmark to test if a neural network possesses intrinsic geometric memory ($D_4$ symmetry) or if it relies on brittle, translation-only memorization.

### Phase 1: The Baseline Vulnerability (Standard ResNet-18)
* **Architecture:** An off-the-shelf ResNet-18, serving as the industry-standard CNN baseline.
* **The Goal:** Establish a benchmark for predictive capacity on the PCam dataset and precisely quantify the geometric blindspot (rotational instability) of standard convolutions.
* **The Internal Algorithm (Translation vs. Rotation):** Standard CNNs are built for flat Euclidean spaces ($Z^2$).
  1. **The Input Space:** Biopsy patches enter as standard tensors `[B, 3, 96, 96]`.
  2. **Standard Convolutions:** The network applies standard sliding-window filters. These filters are *translationally equivariant* (they can track a tumor shifting left or right) but fundamentally lack geometric memory for rotations or reflections.
  3. **Brute-Force Memorization:** Because it cannot mathematically rotate its internal weights, the ResNet attempts to learn orientations by memorizing them individually during training.
* **The Clinical Result (The Geometric Blindspot):** The baseline audit proved that standard CNNs are clinically unsafe for unoriented pathology slides. While the model achieved a high aggregate accuracy, its geometric stability collapsed under test-time augmentation (TTA). 

#### Baseline vs. Worst-Case Orientation
When the exact same biopsies were rotated 180°, the network became slightly more conservative, predicting "Healthy" more often. While this slightly boosted specificity, it resulted in a catastrophic drop in recall, missing over 1,200 additional cancer cases purely due to the angle of the slide.

| Metric | Standard Evaluation (0°) | Worst-Case Angle (180°) | Degradation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 84.90% | 80.44% | - 4.46% |
| **AUC-ROC** | 0.9317 | 0.9091 | - 0.0226 |
| **Sensitivity (Recall)** | 79.90% | 72.49% | **- 7.41%** |
| **Specificity** | 92.79% | 93.72% | + 0.93% |
| **Missed Cancers (FN)** | 3,291 missed | 4,505 missed | **+ 1,214 errors** |

#### Full $D_4$ Manifold Audit
By testing the network across all 8 spatial orientations of the Dihedral Group ($D_4$), we isolated the network's **Flip Rate**—the percentage of times it changed its diagnosis for the exact same patient based solely on spatial orientation.

| Angle | Accuracy | AUC | Recall | Missed Cancers | Flip Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0° (Baseline)** | 0.8490 | 0.9317 | 0.7990 | 3,291 | 0.00% |
| **0° Flipped** | 0.8193 | 0.9171 | 0.7490 | 4,111 | 10.31% |
| **90°** | 0.8157 | 0.9185 | 0.7457 | 4,164 | 10.96% |
| **90° Flipped** | 0.8362 | 0.9210 | 0.7824 | 3,563 | 10.21% |
| **180°** | 0.8044 | 0.9091 | 0.7249 | 4,505 | **11.30% (Peak)**|
| **180° Flipped**| 0.8285 | 0.9224 | 0.7659 | 3,834 | 9.51% |
| **270°** | 0.8351 | 0.9228 | 0.7794 | 3,613 | 10.38% |
| **270° Flipped**| 0.8199 | 0.9133 | 0.7473 | 4,138 | 10.17% |
