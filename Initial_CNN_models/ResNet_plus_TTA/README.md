This model adds test-time augmentation (TTA) to the more basic, ResNet-50 based CNN model.

We found that image rotations introduce a non-zero flip rate. Therefore, by predicting a classification after passing the image through the model at multiple orientations, we may improve our model performance.

Each image is transformed with 15 rotations, passed through the model. The predictions are averaged to produce a final prediction.

This improves performance by reducing sensitivity to orientation, as judged by e.g. F1, AUC.

There is a longer inference period due to multiple forward passes per image.

Metrics:

Accuracy: 89.98% | Precision: 89.91% | Recall: 84.67% | Specificity: 93.57% | F1: 87.20%
