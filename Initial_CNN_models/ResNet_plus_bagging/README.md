This model adds bagging (bootstrap aggregation) to the base ResNet-50 CNN model.

We found that training a single model can lead to variance depending on the specific training data seen. Therefore, by training multiple models on different bootstrapped samples of the dataset, we may improve robustness and generalisation.

Each model is trained on a dataset sampled with replacement from the original training set. During inference, predictions from all models are averaged to produce a final prediction.

This improves performance by reducing variance and overfitting, as judged by e.g. F1, AUC.

There is increased training and inference cost due to maintaining multiple models. However, it does not improve the model performance within a reasonable training time.
