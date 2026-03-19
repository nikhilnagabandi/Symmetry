**ResNet50 plus two fully connected layers**:

/ResNet_plus_2_FC_layers

Our deep learning approach began with ResNet, a residual neural network which has been optimised for pattern detection in images.

We added two fully connected layers and performed a grid search over hyperparameters, such as learning rate and batch size. Limiting training to 10 epochs, on validation data we achieve:

Accuracy: 84.89% | Precision: 80.20% | Recall: 83.34% | Specificity: 85.95% | F1: 81.74%

**TTA**:

/ResNet_plus_TTA

This model adds test-time augmentation (TTA) to the more basic, ResNet-50 based CNN model.

We found that image rotations introduce a non-zero flip rate. Therefore, by predicting a classification after passing the image through the model at multiple orientations, we may improve our model performance.

Each image is transformed with 15 rotations, passed through the model. The predictions are averaged to produce a final prediction.

This improves performance by reducing sensitivity to orientation, as judged by e.g. F1, AUC.

There is a longer inference period due to multiple forward passes per image.

Metrics:

Accuracy: 89.98% | Precision: 89.91% | Recall: 84.67% | Specificity: 93.57% | F1: 87.20%

**Bagging**:

/ResNet_plus_bagging

This model adds bagging (bootstrap aggregation) to the base ResNet-50 CNN model.

We found that training a single model can lead to variance depending on the specific training data seen. Therefore, by training multiple models on different bootstrapped samples of the dataset, we may improve robustness and generalisation.

Each model is trained on a dataset sampled with replacement from the original training set. During inference, predictions from all models are averaged to produce a final prediction.

This improves performance by reducing variance and overfitting, as judged by e.g. F1, AUC.

There is increased training and inference cost due to maintaining multiple models. However, it does not improve the model performance within a reasonable training time.
