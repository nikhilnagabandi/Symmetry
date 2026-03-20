# Symmetry-Preserving Neural Networks for Computational Pathology

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#)

## 1. Project Abstract

While standard Convolutional Neural Networks (CNNs) like ResNet achieve high baseline accuracy on medical imaging tasks, they suffer from a fundamental architectural flaw: **a lack of intrinsic geometric memory**. In clinical histopathology, where tissue slides are arbitrarily rotated or flipped under a microscope, standard CNNs exhibit dangerous diagnostic instability, frequently altering their predictions based purely on spatial orientation.

This repository explores the intersection of geometric deep learning and clinical safety by integrating group theory directly into neural network architectures. In this project, we aim to isolate and solve this vulnerability by working on the PatchCamelyon (PCam) metastasis dataset, which contains over 220,000 histopathology images from lymph node biopsies with a pre-determined train/test split.

We structured this research in three evolutionary phases:

1. **The Baseline (ResNet-18/50):** Established a standard $Z^2$ Euclidean CNN to expose the hidden topological vulnerabilities present in widely accepted medical architectures.
2. **The Intermediate Breakthrough (Custom G-CNN):** Built a 5-layer network from scratch in pure PyTorch, engineering custom "Lifting Layers" to mathematically map standard image pixels into the discrete $D_4$ Dihedral Group. 
3. **The Ultimate Solution (Steerable ESCNN):** Deployed a state-of-the-art continuous harmonic network to resolve the interpolation limits of discrete pixel grids, achieving perfect, full-manifold geometric stability.

### Key Clinical Findings
* **The Hidden Flaw:** While all architectures achieve a highly performant baseline AUC of **~0.93 - 0.94**, a topological audit reveals that the standard ResNet-18 is structurally fragile, unpredictably altering ~11% of its diagnoses based purely on spatial orientation.
* **The Clinical Danger:** Under a strict 25% clinical safety threshold, simply rotating a biopsy slide $180^\circ$ caused the standard CNN to miss **1,214 additional cancers**. 
* **The Discrete vs. Continuous Grid Problem:** Our custom discrete G-CNN successfully forced stability across $180^\circ$ and reflection subgroups (0.00% Flip Rate), but suffered a minor 4.82% error rate at $90^\circ$ shifts due to the limitations of discrete pixel grids. The continuous harmonic filters of the ESCNN entirely resolved this, proving the necessity of steerable bases.
* **The Mathematical Guarantee & Superior Sensitivity:** The Steerable ESCNN achieved a perfect **0.00% Flip Rate**. Not only did it permanently lock its False Negative count across all orientations, but its invariant recall (**81.36%**) strictly outperformed the standard CNN's absolute best-case orientation. 
* **Rescued Calibration:** Enforcing geometric priors fundamentally improved feature separability. When targeting an 85% guaranteed recall rate, the standard CNN suffered total probability collapse (requiring a near-zero 0.88% threshold), whereas the ESCNN maintained a highly usable 15.45% confidence threshold.

---

## 2. The Physics & Clinical Problem: $D_4$ Symmetry

In standard clinical workflows, a histopathology slide has no "correct" up or down. A pathologist or a microscope camera might arbitrarily rotate a biopsy by $90^\circ$ or flip it horizontally. The underlying biology (the presence of metastasis) remains absolutely unchanged, meaning our neural network's diagnosis should be perfectly **invariant** to these physical transformations.

### The Math: The $D_4$ Dihedral Group
Standard Convolutional Neural Networks (CNNs) only share weights across the $Z^2$ Euclidean plane (translations). If a feature (like a tumor cell) appears in the top-left or bottom-right, the CNN recognizes it. However, standard CNNs do *not* share weights across rotations or reflections. They must brute-force learn what a tumor looks like at every possible angle.

We solve this by explicitly hardcoding the **$D_4$ Dihedral Group** into the network's layers. The $D_4$ group consists of 8 geometric actions:
* **4 Rotations:** $0^\circ, 90^\circ, 180^\circ, 270^\circ$
* **4 Reflections:** Horizontal and vertical flips.

By upgrading standard convolutions to **Group Convolutions ($G$-Convs)**, the network mathematically applies its filters across all 8 spatial orientations simultaneously. The internal feature maps are no longer just scalar grids; they become structured vector spaces that rotate predictably with the input image (Equivariance).

### The Challenge of Discrete Grids ($Z^2$) vs. True Equivariance
A core finding of this project is the limitation of applying discrete group operations to square pixel grids. 

When you physically rotate a square $Z^2$ pixel grid by exactly $180^\circ$, or mirror it horizontally, the pixels map perfectly 1-to-1. Because of this perfect alignment, our Custom Discrete G-CNN achieved a mathematically perfect **0.00% Flip Rate** for these specific subgroups. 

However, rotating a discrete square grid by $90^\circ$ or $270^\circ$ introduces subtle interpolation artifacts and alignment shifts at the corners of the tensor. This physical limitation of standard computer vision caused our Custom G-CNN to exhibit a **4.82% diagnostic fluctuation** at $90^\circ$ angles. 

**The Solution:** To achieve true, full-manifold equivariance, we cannot rely on rotating discrete grids. We must use **continuous circular harmonics** (via the Steerable ESCNN). By parameterizing the convolutional filters as continuous waves rather than discrete pixels, the ESCNN understands $D_4$ symmetries at a sub-pixel level, effectively eliminating the $90^\circ$ grid artifact and dropping the full-manifold error rate to an absolute **0.00%**.

## 3. The Progression of Architectures

To definitively prove the clinical necessity of geometric priors, we approached the problem in three distinct evolutionary phases. This progression allowed us to isolate the exact mathematical bottlenecks in standard computer vision.

### Phase 1: The Baseline (ResNet based CNN models)

#### Phase 1a: ResNet50 plus two fully connected layers

- **Architecture:** A ResNet residual network augmented with two additional fully connected layers.  
- **The Goal:** Establish a strong, clinically relevant baseline using a widely adopted CNN architecture on the PCam metastasis dataset.  
- **The Result:** The model achieved strong classification performance (AUC ~0.93; F1 ~0.82) on standard validation data. However, under rotational perturbations it exhibited a critical failure mode: predictions were not invariant to orientation, leading to an ~11% diagnostic fluctuation across equivalent inputs. This exposes a fundamental lack of geometric robustness, rendering the model clinically unreliable despite high headline accuracy.  

#### Phase 1b: Test-Time Augmentation (TTA)

- **Architecture:** The same as 1a, but with test-time augmentation applied during inference. TTA is a technique where multiple transformed versions of the same input (e.g. rotations, flips) are passed through the model at inference, and their predictions are aggregated (typically averaged) to produce a more stable final output.  
- **The Goal:** Mitigate orientation sensitivity without modifying the underlying architecture, by enforcing approximate rotational invariance at inference time.
- **The Method:** Each image is rotated into 15 distinct orientations, passed independently through the model, and the resulting predictions are averaged to produce a final classification.  
- **The Result:** TTA substantially improves robustness and overall performance (F1: 81.74% → 87.20%, Accuracy: 84.89% → 89.98%), reducing orientation-induced variance. However, this comes at a significant computational cost (15× inference time) and remains a workaround rather than a principled architectural solution.

### Phase 2: The Intermediate Breakthrough (Custom 5-Layer G-CNN)
* **Architecture:** A custom 5-layer feed-forward network built entirely from scratch in pure PyTorch.
* **The Goal:** Prove that explicitly forcing the network to obey $D_4$ symmetries would solve the ResNet's instability. 
* **The Internal Algorithm (Mathematical & Tensor Flow):** To achieve strict equivariance, we expanded the network's internal representations from standard 4D tensors to 5D topological tensors `[Batch, Channels, Group_Elements, Height, Width]`. The data flows through four distinct mathematical phases:
  
  1. **Layer 1 (The Lifting Layer): $Z^2 \to D_4$**
     * *The Math:* Takes the flat 2D clinical image ($Z^2$) and projects it up into the 8-dimensional topological space of the dihedral group.
     * *The Tensors:* Standard biopsy patches enter as `[B, 3, 96, 96]`. A base filter is explicitly rotated and reflected 8 times before application. The tensor shape transforms to `[B, 24, 8, 96, 96]`.
  
  2. **Layers 2, 3, 4, and 5 (Group Convolutions): $D_4 \to D_4$**
     * *The Math:* These four layers keep the data strictly inside that 8-dimensional space. They extract deeper and deeper features (from basic edges to complex cancer textures) while mathematically shuffling the channels to maintain perfect rotational tracking.
     * *The Tensors:* Custom G-Conv layers permute their internal weights to account for spatial rotations. The final convolutional tensor reaches `[B, 96, 8, 12, 12]`.
  
  3. **The Bottleneck (Group Pooling): $D_4 \to \text{Invariant Space}$**
     * *The Math:* This is where the manifold collapses. By taking the maximum value across the 8 orientation channels, the network stops tracking *where* the rotation is and just outputs a definitive, invariant signal.
     * *The Tensors:* To make a final binary diagnosis, the network applies a Max Pool across the 3rd dimension (the 8 group states), collapsing the tensor into a mathematically invariant state `[B, 96, 12, 12]`.
  
  4. **Final Output (Linear Classification): $\text{Invariant} \to \text{Binary Diagnosis}$**
     * *The Math:* The fully connected layer translates that invariant signal into the final prediction.
     * *The Tensors:* Standard spatial pooling (`AdaptiveAvgPool2d`) crushes the spatial dimensions, and a linear layer maps the invariant features to the final binary output `[B, 2]`.

* **The Clinical Result (Subgroup Equivariance):** This intermediate step was highly successful but revealed a profound mathematical limitation of square pixel grids. It locked False Negatives and achieved a perfect **0.00% Flip Rate** for 180° rotations and horizontal/vertical flips. However, it still exhibited a **4.82% fluctuation** at 90° and 270° angles due to discrete tensor alignment. This perfectly proved the geometric prior works, but set the stage for a continuous solution.

### Phase 3: The Ultimate Solution (Steerable ESCNN)
* **Architecture:** Steerable networks using the `escnn` library.
* **The Goal:** Achieve the perfect geometric stability of our custom model without the interpolation limitations of discrete pixel grids.
* **The Mechanism:** Instead of manually rotating discrete filter grids (which caused the 4.82% artifact in Phase 2), Steerable CNNs parameterize their filters using **continuous circular harmonics**. By restricting the learned weights to a mathematically steerable basis, the network inherently understands $D_4$ symmetries at a sub-pixel level.
* **The Result:** The ultimate architecture. It completely resolved the discrete grid problem to achieve a mathematically guaranteed **0.00% Flip Rate across the entire manifold**, alongside elite predictive power (0.9315 AUC) in just 5 epochs of training.

### The Mathematics of Steerable Convolutions
To understand how the Steerable CNN (ESCNN) entirely eliminated the 4.82% error found in the discrete G-CNN, we have to look at how it redefines the convolutional filter.

* **The Interpolation Wall (Phase 2's Limit):** Discrete networks learn filters as rigid, square pixel matrices. Rotating a square grid by $90^\circ$ or $270^\circ$ forces the system to estimate sub-pixel values. This lossy interpolation creates a hard mathematical ceiling on diagnostic stability.
* **Continuous Basis Functions:** The ESCNN abandons the discrete grid. Instead, it parameterizes its filters as a linear combination of continuous **circular harmonics** (smooth wave functions over the $SO(2)$ rotation group). A filter $f$ at a given radius $r$ and angle $\theta$ is defined by its angular frequency $m$:
  $$f(r, \theta) = R(r) e^{i m \theta}$$
* **Perfect Steerability via Phase Shift:** Because the filters are continuous mathematical functions, rotating them by an arbitrary angle $\alpha$ does not require physically moving pixels. The rotation is achieved through an exact analytic phase shift:
  $$f(r, \theta - \alpha) = R(r) e^{i m (\theta - \alpha)} = e^{-i m \alpha} f(r, \theta)$$
* **Bypassing the Grid:** The network calculates the rotated filter in continuous mathematical space *first*, and only then samples the resulting wave onto the discrete clinical image. By applying the rotation analytically before interacting with the pixel grid, the network completely bypasses interpolation loss, resulting in strict geometric safety.

## 4. Results & Clinical Evaluation

To rigorously evaluate the clinical reliability of standard CNNs versus Steerable architectures, we audited all three models across the complete **$D_4$ Dihedral Manifold** (4 physical rotations + 4 physical reflections). 

### 4.1 Robustness Audit (The "Flip Rate" & AUC Stability)
In standard clinical workflows, histopathology slides are arbitrarily rotated and flipped under the microscope. We measured the **Flip Rate**—the percentage of biopsies where the model changed its diagnosis purely because the image was physically manipulated from its $0^\circ$ anchor.

| Network Architecture | Symmetry Prior | Peak AUC | 0° Anchor | 90° Rot | 180° Rot | 270° Flip | Manifold Stability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** (Phase 1) | None | 0.9317 | 0.00% | 10.96% | 11.30% | 10.17% | Unstable |
| **Custom G-CNN** (Phase 2) | Discrete $D_4$ | **0.9439** | 0.00% | **4.82%** | **0.00%** | **4.82%** | Partial (Grid Limits) |
| **ESCNN Steerable** (Phase 3) | Harmonic $D_4$ | 0.9315 | 0.00% | **0.00%** | **0.00%** | **0.00%** | Equivariant |

**Analysis:** The standard CNN suffers from a dangerous lack of geometric memory (~11% fluctuation). The Custom G-CNN successfully forced stability across the $180^\circ$ and reflection subgroups (0.00% flip error), but revealed the limitations of discrete square grids with a 4.82% flip error at $90^\circ$. By employing continuous harmonics, the ESCNN mathematically guarantees a full-manifold 0.00% Flip Rate while maintaining peak predictive power.

### 4.2 Clinical Sensitivity & Specificity (25% Safety Threshold)

To simulate a real-world triage environment, we set a strict **25% Safety Threshold** (if the network is even 25% confident a biopsy is malignant, it flags it for human review). There are exactly **16,377 positive cancer cases** in the test set.

**A. The Baseline Clinical Profile (Standard 0° Slides)**
Before introducing physical rotations, we evaluated the baseline trade-off between Sensitivity (catching cancer) and Specificity (correctly clearing healthy patients) on standard, un-rotated slides:

| Metric | ResNet-18 | Custom G-CNN | ESCNN Steerable |
| :--- | :--- | :--- | :--- |
| **Recall (Sensitivity)** | 79.90% | 76.89% | **81.36%** (Superior) |
| **Specificity** | 92.79% | **96.71%** | 90.85% (Highly Viable) |
| **Missed Cancers (FN)** | 3,291 / 16,377 | 3,784 / 16,377 | **3,053 / 16,377** |

**B. The Rotational Crash (Diagnostic Fluctuation)**
We then rotated the slides across the $D_4$ manifold to simulate real-world microscope handling. We tracked how the Recall and False Negatives fluctuated based entirely on physical orientation.

| Slide Orientation | ResNet-18 (Recall / Missed) | Custom G-CNN (Recall / Missed) | ESCNN Steerable (Recall / Missed) |
| :--- | :--- | :--- | :--- |
| **0° (Standard Anchor)** | 79.90% *(3,291 / 16,377)* | 76.89% *(3,784 / 16,377)* | **81.36%** *(3,053 / 16,377)* |
| **90° Rotation** | 74.57% *(4,164 / 16,377)* | 75.02% *(4,091 / 16,377)* | **81.36%** *(3,053 / 16,377)* |
| **180° Rotation** | 72.49% *(4,505 / 16,377)* | 76.89% *(3,784 / 16,377)* | **81.36%** *(3,053 / 16,377)* |
| **270° Reflection** | 74.73% *(4,138 / 16,377)* | 75.02% *(4,091 / 16,377)* | **81.36%** *(3,053 / 16,377)* |

**Analysis:** The ESCNN demonstrates the ultimate clinical profile. At the baseline, it achieves the highest absolute Recall while successfully maintaining a highly viable >90% Specificity. However, the most critical finding is the rotational fragility of standard models. If a technician accidentally inserts a slide upside-down ($180^\circ$), the ResNet-18's recall collapses to **72.49%**, resulting in **1,214 additional missed cancers**. The ESCNN Steerable Network mathematically guarantees safety, permanently locking its superior recall and specificity across all 8 geometric states.

### 4.3 High-Recall Calibration
Finally, we tested the absolute confidence of the models by searching for the clinical probability threshold required to guarantee high Recall rates on standard, un-rotated slides.

* **ResNet-18 (Targeting 95% Recall):** Experienced severe probability decay. To catch 95% of cancers, the model's confidence was so low that the threshold had to be dropped to a microscopic **0.88%**—resulting in 6,757 False Alarms just to hit the target.
* **ESCNN Steerable (Targeting 85% Recall):** Maintained strong, defined class separability. The network achieved a highly reliable 85% guaranteed Recall rate with a highly usable **15.45% probability threshold**.

**Analysis:** Hardcoding geometric priors does not just prevent rotational errors; it fundamentally improves feature separability and model calibration. The Steerable network retains highly functional probability scores even under heavily constrained, 5-epoch training loops, whereas standard CNNs suffer from total probability collapse when pushed to clinical sensitivity targets.

## 8. Conclusion & The Next Frontier

This repository demonstrates a fundamental truth in computational pathology: **predictive power means nothing without geometric stability.** By auditing a standard ResNet-18 across the $D_4$ manifold, we exposed a critical architectural flaw. The standard CNN's lack of intrinsic geometric memory caused it to miss over 1,200 additional cancers simply because a biopsy slide was physically rotated, rendering standard architectures clinically unsafe for deployable triage. 

### The Evolution of Geometric Priors & Data Efficiency
By systematically upgrading the architecture, we isolated the exact mathematical bottlenecks causing this rotational fragility:
* **The Discrete Grid Limitation:** Our intermediate Custom 5-Layer G-CNN successfully stabilized 180° rotations and reflections (achieving a 0.00% flip rate for those subgroups). However, it still exhibited a ~4.8% diagnostic fluctuation at 90° angles. This perfectly highlighted the fundamental limitation of applying discrete $D_4$ group operations to square $Z^2$ pixel grids, where 90° rotations inevitably introduce subtle interpolation artifacts.
* **The Continuous Solution:** By upgrading to a continuous Steerable Network (ESCNN), we bypassed the discrete grid entirely. Using continuous circular harmonics, the ESCNN achieved a mathematically guaranteed **0.00% Flip Rate** across the entire spatial manifold. 
* **The 5-Epoch Advantage:** Crucially, these results were achieved under a highly constrained 5-epoch training loop. Standard CNNs must brute-force learn rotational invariance by observing thousands of augmented examples over massive training cycles. By contrast, equivariant models inherently possess this geometric knowledge at initialization. Locking in superior clinical sensitivity (81.36%) and specificity (90.85%) in just 5 epochs proves that hardcoding structural symmetries produces models that are not only mathematically safer, but radically more data-efficient.

## References & Acknowledgements

This project heavily utilizes the mathematical frameworks and PyTorch libraries pioneered by Taco Cohen, Max Welling, Maurice Weiler, Gabriele Cesa, and Leon Lang. Their foundational work on Group Equivariant and Steerable CNNs made the $D_4$ manifold audits in this repository possible.

If you are exploring the theory behind the equivariant models used in this repository, please refer to their original papers:

1. **Group Equivariant Convolutional Networks** *(The Foundation of Phase 2)*
   * Cohen, T., & Welling, M. (2016). *International Conference on Machine Learning (ICML).*

2. **Learning Steerable Filters for Rotation Equivariant CNNs** *(The Bridge to Continuous Math)*
   * Weiler, M., Hamprecht, F. A., & Welling, M. (2018). *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).*

3. **General E(2)-Equivariant Steerable CNNs** *(`e2cnn` Core Theory)*
   * Weiler, M., & Cesa, G. (2019). *Conference on Neural Information Processing Systems (NeurIPS).*

4. **A Program to Build E(N)-Equivariant Steerable CNNs** *(`escnn` Implementation for Phase 3)*
   * Cesa, G., Lang, L., & Weiler, M. (2022). *International Conference on Learning Representations (ICLR).*

### Beyond CNN: The Vision Transformer (ViT)
While continuous steerable networks represent the absolute pinnacle of *convolutional* safety, convolutions are still fundamentally restricted by their local receptive fields. Clinical histopathology often requires understanding global tissue context—for example, how a cluster of malignant cells in one corner of a slide relates to the surrounding stroma in another.
 
 
## The Model: ViT-Tiny
 
Unlike CNNs, ViT doesn't use convolutional filters at all. It splits the input image into a grid of patches, treats each patch as a token (like a word in a sentence), and processes the entire sequence through transformer layers using self-attention.
 
Here's what that looks like concretely for our setup:
 
| Spec | Value |
|------|-------|
| Total parameters | ~5.7M |
| Transformer layers | 12 |
| Attention heads | 3 |
| Embedding dimension | 192 |
| FFN hidden dimension | 768 |
| Patch size | 16×16 pixels |
| Input tokens | 37 (36 patches + 1 [CLS] token) |
| Pretraining | ImageNet-21K |
 
The [CLS] token is a special learnable token prepended to the sequence. Its job is to aggregate information from all patches through the attention layers, and it's the only token used for the final cancer/non-cancer prediction.
 
**The rotation problem:** ViT's spatial awareness comes entirely from learned positional embeddings — there are no architectural inductive biases for position like there are in CNNs. Each grid position gets its own embedding vector. When the image is rotated, each patch lands at a different grid position and receives a different positional embedding than it had during training. The model learned specific content-position associations on unrotated images, and rotation breaks those associations.
 
---
 
## Experimental Design
 
We ran four experiments varying two things: **how much of the model is trainable** and **what augmentation is applied during training**. Everything else was held constant.
 
**Shared training config:**
- Optimizer: Adam
- LR schedule: Cosine Annealing (decays from initial LR to 0 over 6 epochs)
- Batch size: 32
- Loss: BCEWithLogitsLoss
- Input: 96×96 native resolution (no resize)
 
**D4 augmentation** (when used) applies `RandomChoice(0°/90°/180°/270°)` + `RandomHorizontalFlip` + `RandomVerticalFlip` + `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`. This covers all 8 symmetries of the dihedral group D₄.
 
---
 
## Results
 
### Experiment A: Frozen Backbone — No Augmentation
 
All 12 transformer blocks and embeddings are frozen. Only a single `Linear(192, 1)` head is trainable — **193 parameters**. Initial LR: 1e-3. No augmentation beyond normalization.
 
**Confusion matrix (0° test set):**
 
|  | Pred − | Pred + |
|--|--------|--------|
| **True −** | 11,454 | 1,637 |
| **True +** | **2,026** | 6,886 |
 
**Per-angle results:**
 
| Angle | AUC | Accuracy | Recall | Flip Rate |
|-------|-----|----------|--------|-----------|
| 0° | 0.9082 | 0.8335 | 0.7727 | — |
| 90° | 0.9084 | 0.8345 | 0.7791 | 9.03% |
| 180° | 0.9073 | 0.8347 | 0.7773 | 8.60% |
| 270° | 0.9083 | 0.8345 | 0.7776 | 9.24% |
 
**Precision:** 0.8079 · **Specificity:** 0.8750 · **F1:** 0.8329 · **Avg Flip Rate:** 8.95%
 
The rotation gap is essentially zero — AUC varies by only 0.0011 across all angles. But overall performance is weak. A recall of 0.77 means the model missed 2,026 out of 8,912 cancers. With only 193 trainable parameters, the model is fitting a simple linear boundary on frozen ImageNet features. It's not expressive enough to learn orientation-specific patterns, which is why it's rotationally stable — but it's also not expressive enough to be clinically useful.
 
---
 
### Experiment B: Frozen Backbone + D4 Augmentation
 
Same architecture and frozen backbone as A, but training now includes D4 symmetry augmentation. Initial LR: 1e-3.
 
**Confusion matrix (0° test set):**
 
|  | Pred − | Pred + |
|--|--------|--------|
| **True −** | 10,902 | 2,189 |
| **True +** | **1,846** | 7,066 |
 
**Per-angle results:**
 
| Angle | AUC | Accuracy | Recall | Flip Rate |
|-------|-----|----------|--------|-----------|
| 0° | 0.8933 | 0.8166 | 0.7929 | — |
| 90° | 0.8955 | 0.8182 | 0.8046 | 10.07% |
| 180° | 0.8939 | 0.8150 | 0.7968 | 9.75% |
| 270° | 0.8945 | 0.8158 | 0.8012 | 9.95% |
 
**Precision:** 0.7635 · **Specificity:** 0.8328 · **F1:** 0.8171 · **Avg Flip Rate:** 9.92%
 
This is the worst-performing experiment. D4 augmentation actually *hurt* the model — AUC dropped from 0.9082 to 0.8933, precision fell from 0.8079 to 0.7635, and the flip rate went *up* from 8.95% to 9.92%.
 
What happened? D4 augmentation presents each image in 8 different orientations during training. That's a much harder learning problem, and 193 parameters simply aren't enough to handle it. The augmentation added difficulty without adding capacity. Recall did tick up slightly (0.77 → 0.79), but at the cost of a lot more false positives.
 
**Takeaway:** augmentation without capacity is counterproductive.
 
---
 
### Experiment C: Partial Fine-Tuning — Color Jitter Only
 
Now we unfreeze the last 2 transformer blocks (10 and 11), the final LayerNorm, and replace the head with a deeper MLP: `Linear(192, 128) → ReLU → Dropout(0.3) → Linear(128, 1)`. That gives us **922,625 trainable parameters** — a 4,780× increase over the frozen experiments. Initial LR: 1e-4 (10× lower to avoid destabilizing pretrained weights). Augmentation is ColorJitter only — no spatial transforms.
 
**Confusion matrix (0° test set):**
 
|  | Pred − | Pred + |
|--|--------|--------|
| **True −** | 12,237 | 854 |
| **True +** | **862** | 8,050 |
 
**Per-angle results:**
 
| Angle | AUC | Accuracy | Recall | Flip Rate |
|-------|-----|----------|--------|-----------|
| 0° | 0.9749 | 0.9220 | 0.9033 | — |
| 90° | 0.9662 | 0.9091 | 0.8756 | 7.34% |
| 180° | 0.9693 | 0.9158 | 0.8845 | 6.83% |
| 270° | 0.9670 | 0.9091 | 0.8788 | 7.54% |
 
**Precision:** 0.9041 · **Specificity:** 0.9348 · **F1:** 0.9220 · **Avg Flip Rate:** 7.24%
 
This is a massive jump. AUC went from 0.908 to 0.975 (+7.4%), recall from 0.77 to 0.90, and missed cancers dropped from 2,026 to 862 — a 57% reduction. The model is now both detecting cancer well *and* avoiding false alarms (specificity 0.93).
 
But the rotation gap came back. AUC at 90° (0.9662) is 0.87% lower than at 0° (0.9749). With enough capacity, the attention layers learned position-dependent features through the positional embeddings, and those features are disrupted by rotation.
 
More concerning: the flip direction is asymmetric. At 90°, 912 predictions flipped from Cancer → Non-Cancer, but only 703 flipped the other way. Rotation is systematically biasing the model toward *missing cancers* — the most dangerous failure mode in a screening tool.
 
---
 
### Experiment D: Partial Fine-Tuning + D4 Augmentation
 
Same architecture as C, but training now includes D4 symmetry augmentation before ColorJitter. Everything else identical.
 
**Confusion matrix (0° test set):**
 
|  | Pred − | Pred + |
|--|--------|--------|
| **True −** | 12,111 | 980 |
| **True +** | **832** | 8,080 |
 
**Per-angle results:**
 
| Angle | AUC | Accuracy | Recall | Flip Rate |
|-------|-----|----------|--------|-----------|
| 0° | 0.9734 | 0.9176 | 0.9066 | — |
| 90° | 0.9737 | 0.9201 | 0.9100 | 6.07% |
| 180° | 0.9731 | 0.9193 | 0.9087 | 6.10% |
| 270° | 0.9736 | 0.9189 | 0.9098 | 6.38% |
 
**Precision:** 0.8918 · **Specificity:** 0.9251 · **F1:** 0.9193 · **Avg Flip Rate:** 6.18%
 
This is the best overall result. The rotation gap is gone — AUC varies by only 0.0006 across all angles (compared to 0.0087 in Experiment C). The flip rate dropped from 7.24% to 6.18%. And critically, the flip direction is now balanced: at 90°, it's 665 Cancer→Non-Cancer vs 670 the other way. No more systematic bias toward missing cancers.
 
The cost? AUC at 0° dipped slightly from 0.9749 to 0.9734 (−0.15%), and specificity dropped from 0.9348 to 0.9251 (slightly more false positives: 980 vs 854). But recall *improved* from 0.9033 to 0.9066, and missed cancers hit the absolute lowest at 832. That's a trade-off worth making.
 
---
 
## Comparative Summary
 
| | A. Frozen | B. Frozen+D4 | C. Unfrozen | D. Unfrozen+D4 |
|---|---|---|---|---|
| **Trainable params** | 193 | 193 | 922,625 | 922,625 |
| **AUC @ 0°** | 0.9082 | 0.8933 | **0.9749** | 0.9734 |
| **AUC gap** | ~0% | ~0% | 0.87% | **~0%** |
| **Recall** | 0.7727 | 0.7929 | 0.9033 | **0.9066** |
| **Specificity** | 0.8750 | 0.8328 | **0.9348** | 0.9251 |
| **Precision** | 0.8079 | 0.7635 | **0.9041** | 0.8918 |
| **Missed cancers** | 2,026 | 1,846 | 862 | **832** |
| **Avg flip rate** | 8.95% | 9.92% | 7.24% | **6.18%** |
| **Flip direction (90°)** | 947 vs 1039 | 1020 vs 1195 | 912 vs 703 | **665 vs 670** |
 
---
 
## Key Findings
 
**1. Capacity matters more than augmentation.** Going from 193 to 922K trainable parameters (A → C) improved AUC by 7.4%, recall by 17 percentage points, and cut missed cancers by 57%. In contrast, adding D4 augmentation to the frozen model (A → B) made things worse. Capacity sets the ceiling; augmentation refines within it.
 
**2. D4 augmentation closes the rotation gap — but only with sufficient capacity.** On the unfrozen model (C → D), D4 reduced the AUC gap from 0.87% to ~0% and flip rate from 7.24% to 6.18%. On the frozen model (A → B), D4 *increased* the flip rate and *decreased* AUC. The lesson: don't throw augmentation at an underpowered model.
 
**3. D4 fixed the flip direction imbalance.** Without D4 (Exp C), rotation at 90° caused 912 Cancer→Non-Cancer flips vs only 703 in the reverse — a 23% asymmetry biased toward missing cancers. With D4 (Exp D), this balanced to 665 vs 670 (0.7% asymmetry). In a screening context, this is the difference between a model that systematically misses cancers when rotated and one that fails symmetrically.
 
**4. ViT is inherently more rotation-robust than ResNet.** Under comparable conditions (partial fine-tuning, color jitter only), ViT's rotation gap was 0.87% versus ResNet-18's 1.67% — roughly half. Self-attention gives every patch a global receptive field from layer 1, which partially compensates for positional embedding mismatch.
 
**5. The accuracy–stability trade-off.** Our best ViT (Exp D) achieves the highest AUC (0.9734) and recall (90.66%) of any architecture tested, including ResNet-18, Custom G-CNN, and ESCNN. But ESCNN achieved a perfect 0.00% flip rate through built-in harmonic equivariance. ViT wins on accuracy; ESCNN wins on stability.
 
---
 
## Best Configuration
 
**Experiment D (Unfrozen + D4):**
 
| Metric | Value |
|--------|-------|
| AUC | 0.9734 |
| Recall (Sensitivity) | 0.9066 |
| Specificity | 0.9251 |
| Precision | 0.8918 |
| F1 Score | 0.9193 |
| Missed Cancers | 832 / 8,912 (9.3%) |
| Rotation Gap | ~0% |
| Avg Flip Rate | 6.18% (balanced direction) |
 
---
 
## Notebooks
 
| Notebook | Description |
|----------|-------------|
| `ViT_Frozen_no_augmentation.ipynb` | Experiment A — frozen backbone, no augmentation |
| `ViT_Frozen_D4_augmentation.ipynb` | Experiment B — frozen backbone + D4 augmentation |
| `ViT_UnFrozen_last_two_transformer_blocks_and_custom_head.ipynb` | Experiment C — partial fine-tuning, color jitter only |
| `ViT_D4_augmentation.ipynb` | Experiment D — partial fine-tuning + D4 augmentation |
 
---
 
## Reference for ViT
 
- Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* arXiv:2010.11929.

To break past the limitations of local convolutions, our team implemented a **Vision Transformer (ViT)** architecture to replace the sliding window paradigm entirely. By utilizing global self-attention, the ViT can capture long-range biological dependencies from the very first layer. 

Early clinical benchmarking indicates that our ViT architecture strictly outperforms both the standard ResNet and the $D_4$ Steerable Network. 
