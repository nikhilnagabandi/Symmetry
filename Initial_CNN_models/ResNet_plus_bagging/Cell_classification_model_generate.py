import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


#ENSEMBLE SETTINGS
NUM_MODELS = 7
NUM_TTA_ROTATIONS = 7

#Device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print("Using device:", device)


#Transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(180), #random rotation from -180 to 180
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05), #slight random colour alteration
    transforms.Resize(96),
    transforms.CenterCrop(64), #remove outer parts of image for faster training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_base_transform = transforms.Compose([
    transforms.Resize(96),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])


#Dataset
class HistopathologicCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["id"]
        label = self.df.iloc[idx]["label"]
        img_path = os.path.join(self.img_dir, f"{image_id}.tif")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


#Train / Val Split

train_full = HistopathologicCSVDataset(
    csv_file="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train_labels.csv",
    img_dir="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train",
    transform=train_transform
)

val_full = HistopathologicCSVDataset(
    csv_file="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train_labels.csv",
    img_dir="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train",
    transform=None
)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size

indices = torch.randperm(len(train_full))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda batch: (
        [item[0] for item in batch],
        torch.tensor([item[1] for item in batch])))

#Model
def create_model():
    resnet50_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )
    resnet50_model.fc = nn.Identity()
    for param in resnet50_model.parameters():
        param.requires_grad = False
    resnet50_model.eval()
    resnet50_model = resnet50_model.to(device)

    fc_model = nn.Sequential(
        nn.Linear(2048,1024),
        nn.ReLU(),
        nn.Linear(1024,1),
    ).to(device)

    model = nn.Sequential(resnet50_model, fc_model).to(device)
    optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.0002)

    return model, resnet50_model, optimizer


#Create ensemble

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

models = []
optimizers = []
best_accuracies = []

for i in range(NUM_MODELS):
    model, resnet, optimizer = create_model()
    path = os.path.join(save_dir, f"model_{i+1}_best.pth")

    if os.path.exists(path):
        print(f"Loading existing model {i+1}")
        model.load_state_dict(torch.load(path, map_location=device))
        best_accuracies.append(0)
    else:
        best_accuracies.append(0)

    models.append((model, resnet))
    optimizers.append(optimizer)

loss_fn = nn.BCEWithLogitsLoss()


#print metrics function
def print_metrics(preds, labels):

    tp = ((preds==1)&(labels==1)).sum()
    tn = ((preds==0)&(labels==0)).sum()
    fp = ((preds==1)&(labels==0)).sum()
    fn = ((preds==0)&(labels==1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(f"[[{tn} {fp}]")
    print(f" [{fn} {tp}]]\n")

    print(f"Accuracy:    {accuracy*100:.2f}%")
    print(f"Precision:   {precision*100:.2f}%")
    print(f"Recall:      {recall*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")

    return accuracy


#training loop

for epoch in range(50):

    print(f"EPOCH: {epoch}")

    #train and validate each model

    for m in range(NUM_MODELS):

        print(f"\nTraining model {m+1}/{NUM_MODELS}")

        bootstrap_indices = np.random.choice(
            len(train_dataset),
            size=len(train_dataset),
            replace=True)

        bootstrap_subset = Subset(train_dataset, bootstrap_indices)
        train_dataloader = DataLoader(
            bootstrap_subset,
            batch_size=128,
            shuffle=True)

        model, resnet50_model = models[m]
        optimizer = optimizers[m]

        model.train()
        resnet50_model.eval()

        for X, y in tqdm(train_dataloader,
                         desc=f"Model {m+1}",
                         leave=False):

            X = X.to(device)
            y = y.to(device).float().reshape(-1,1)

            outputs = model(X)

            optimizer.zero_grad()
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

        #validate model

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_raw, y in val_dataloader:

                y = y.to(device).float().reshape(-1,1)
                images = [val_base_transform(img) for img in X_raw]
                X = torch.stack(images).to(device)

                outputs = model(X)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.append(preds.cpu().int().view(-1))
                all_labels.append(y.cpu().int().view(-1))

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        print(f"\nModel {m+1} Validation:")
        accuracy = print_metrics(all_preds, all_labels)

        if accuracy > best_accuracies[m]:
            print(f"New best model {m+1} — saving.")
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"model_{m+1}_best.pth")
            )
            best_accuracies[m] = accuracy

    #Ensemble validation

    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():

        for X_raw, y in tqdm(val_dataloader, desc="Validation"):

            batch_size = len(y)
            y = y.to(device).float().reshape(-1,1)
            ensemble_prob_sum = torch.zeros(batch_size,1).to(device)

            for m in range(NUM_MODELS):

                model, _ = models[m]
                model.eval()

                tta_prob_sum = torch.zeros(batch_size,1).to(device)

                for _ in range(NUM_TTA_ROTATIONS):

                    rotated_images = []

                    for img in X_raw:
                        img = transforms.RandomRotation(90)(img)
                        img = val_base_transform(img)
                        rotated_images.append(img)

                    X = torch.stack(rotated_images).to(device)
                    outputs = model(X)
                    probs = torch.sigmoid(outputs)
                    tta_prob_sum += probs

                ensemble_prob_sum += (tta_prob_sum / NUM_TTA_ROTATIONS)

            avg_ensemble_prob = ensemble_prob_sum / NUM_MODELS
            final_predictions = (avg_ensemble_prob > 0.5).float()

            val_correct += (final_predictions == y).sum().item()
            val_total += y.size(0)

            all_val_preds.append(final_predictions.cpu().int().view(-1))
            all_val_labels.append(y.cpu().int().view(-1))

    print("\nEnsemble Validation accuracy:",
          np.round((val_correct/val_total)*100,2), "%")

    all_val_preds = torch.cat(all_val_preds).numpy()
    all_val_labels = torch.cat(all_val_labels).numpy()

    print_metrics(all_val_preds, all_val_labels)