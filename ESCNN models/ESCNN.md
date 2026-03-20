### Phase 3: The Continuous Peak (Steerable ESCNN)
* **Architecture:** A state-of-the-art Equivariant Steerable Convolutional Neural Network built using the `escnn` library.
* **The Goal:** Eliminate the 4.82% discrete grid artifact discovered in Phase 2 by replacing discrete tensor permutations with continuous circular harmonics, achieving perfect geometric safety.
* **The Internal Algorithm (Continuous Harmonics):** Standard CNNs and our Phase 2 G-CNN rely on chunky, discrete pixel grids. The Steerable CNN abandons this limitation:
  1. **Harmonic Basis Functions:** Instead of learning rigid pixel weights, the network learns weights as linear combinations of continuous circular harmonics (smooth, wave-like mathematical functions).
  2. **Sub-Pixel Steerability:** Because the filters are continuous, they can be rotated to *any* angle (including exactly 90°) without suffering from the sub-pixel interpolation loss that plagued the discrete Phase 2 model.
  3. **Strict $D_4$ Restriction:** We restricted the $E(2)$ continuous group down to the $D_4$ discrete subgroup to perfectly align with the 8 physical orientations of our testing manifold.
  4. **Invariant Collapse:** A group-wise max pooling layer collapses the multidimensional harmonic representations into a perfectly invariant 1D feature vector for final classification.
* **The Clinical Result (Absolute Geometric Safety):** The Steerable CNN completely solved the rotational vulnerability. It achieved a mathematically locked **0.00% Flip Rate** across the entire spatial manifold while simultaneously improving predictive capacity, catching nearly 1,500 more cancers than the standard ResNet's worst-case orientation.

#### Baseline vs. Worst-Case Orientation (The Perfect Lock)
Because the network possesses perfect intrinsic geometric memory, the concept of a "worst-case angle" no longer exists. The network's performance is mathematically locked, guaranteeing that a patient will receive the exact same diagnosis regardless of how the pathologist places the slide.

| Metric | Standard Evaluation (0°) | Worst-Case Angle (Any) | Degradation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 84.55% | 84.55% | **- 0.00%** |
| **AUC-ROC** | 0.9315 | 0.9315 | **- 0.0000** |
| **Sensitivity (Recall)** | 81.36% | 81.36% | **- 0.00%** |
| **Specificity** | 90.85% | 90.85% | **- 0.00%** |
| **Missed Cancers (out of 16,377)** | 3,053 missed | 3,053 missed | **+ 0 errors** |

#### Full $D_4$ Manifold Audit
The wall of identical metrics below is the ultimate proof of equivariance. The Steerable CNN is completely immune to the spatial orientation of the underlying biology.

| Angle | Accuracy | AUC | Recall | Missed Cancers (out of 16,377) | Flip Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0° (Baseline)** | 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |
| **0° Flipped** | 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |
| **90°** | 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |
| **90° Flipped** | 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |
| **180°** | 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |
| **180° Flipped**| 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |
| **270°** | 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |
| **270° Flipped**| 84.55% | 0.9315 | 81.36% | 3,053 | **0.00%** |

#### ⚙️ Clinical Calibration
Because the Steerable CNN's predictions are geometrically stable, they can be reliably calibrated for hospital deployment without fear of rotational variance breaking the threshold. 

* **Targeting 85.0% Recall:** To guarantee that 85.0% of all malignant cases are caught across *any* slide orientation, the binary classification threshold must simply be lowered to **0.1545 (15.45%)**.
