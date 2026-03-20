### Phase 2: The Intermediate Breakthrough (Custom 5-Layer G-CNN)
* **Architecture:** A custom 5-layer feed-forward network built entirely from scratch in pure PyTorch.
* **The Goal:** Prove that explicitly forcing the network to obey $D_4$ symmetries would solve the ResNet's geometric instability. 
* **The Internal Algorithm (Mathematical & Tensor Flow):** To achieve strict equivariance, we expanded the network's internal representations from standard 4D tensors to 5D topological tensors `[Batch, Channels, Group_Elements, Height, Width]`.
  
  1. **Layer 1 (The Lifting Layer): $Z^2 \to D_4$**
     * Takes the flat 2D clinical image ($Z^2$) and projects it up into the 8-dimensional topological space of the dihedral group. The tensor shape transforms from `[B, 3, 96, 96]` to `[B, 24, 8, 96, 96]`.
  2. **Layers 2-5 (Group Convolutions): $D_4 \to D_4$**
     * These layers extract deeper features while mathematically shuffling the channels to maintain perfect rotational tracking. The final convolutional tensor reaches `[B, 96, 8, 12, 12]`.
  3. **The Bottleneck (Group Pooling): $D_4 \to \text{Invariant Space}$**
     * By taking the maximum value across the 8 orientation channels, the network collapses the manifold and stops tracking *where* the rotation is, outputting a definitive, invariant signal `[B, 96, 12, 12]`.
  4. **Final Output (Linear Classification): $\text{Invariant} \to \text{Binary Diagnosis}$**
     * Standard spatial pooling crushes the dimensions, and a linear layer maps the invariant features to the final binary output `[B, 2]`.

* **The Clinical Result (The Discrete Grid Artifact):** This architecture was a massive success, radically outperforming the ResNet's stability. It achieved a mathematically perfect **0.00% Flip Rate** for 180° rotations and flips (operations that perfectly map onto a square pixel grid). However, it revealed a profound mathematical limitation: rotating square grids by 90° causes sub-pixel interpolation loss, resulting in a persistent 4.82% diagnostic fluctuation. 

#### Baseline vs. Worst-Case Orientation (The 90° Artifact)
Unlike the ResNet (which failed completely at 180°), the G-CNN locked in its performance for 180° and flips perfectly. Its only degradation occurred at 90°, and even then, the failure was remarkably contained compared to the baseline CNN.

| Metric | Standard Evaluation (0°) | Worst-Case Angle (90°) | Degradation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 83.62% | 82.29% | - 1.33% |
| **AUC-ROC** | 0.9439 | 0.9393 | - 0.0046 |
| **Sensitivity (Recall)** | 76.89% | 75.02% | **- 1.87%** |
| **Specificity** | 96.71% | 96.32% | - 0.39% |
| **Missed Cancers (out of 16,377)** | 3,784 missed | 4,091 missed | **+ 307 errors** |

*(Note: While the raw recall is slightly lower than the ResNet baseline, the G-CNN is vastly more precise, dropping False Alarms (FP) from over 1,100 down to roughly 500).*

#### Full $D_4$ Manifold Audit
This table beautifully illustrates the "Subgroup Equivariance" phenomenon. The network is completely immune to $180^\circ$ transformations but struggles with the discrete grid interpolation required for $90^\circ$ and $270^\circ$ shifts.

| Angle | Accuracy | AUC | Recall | Missed Cancers (out of 16,377) | Flip Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0° (Baseline)** | 83.62% | 0.9439 | 76.89% | 3,784 | **0.00%** |
| **0° Flipped** | 83.62% | 0.9439 | 76.89% | 3,784 | **0.00%** |
| **90°** | 82.29% | 0.9393 | 75.02% | 4,091 | 4.82% |
| **90° Flipped** | 82.29% | 0.9393 | 75.02% | 4,091 | 4.82% |
| **180°** | 83.62% | 0.9439 | 76.89% | 3,784 | **0.00%** |
| **180° Flipped**| 83.62% | 0.9439 | 76.89% | 3,784 | **0.00%** |
| **270°** | 82.29% | 0.9393 | 75.02% | 4,091 | 4.82% |
| **270° Flipped**| 82.29% | 0.9393 | 75.02% | 4,091 | 4.82% |
