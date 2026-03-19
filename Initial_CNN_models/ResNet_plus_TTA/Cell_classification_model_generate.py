import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# TTA SETTINGS

NUM_TTA_ROTATIONS = 15  #rotate image multiple times and take vote during validation

# Device

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print("Using device:", device)


# Transforms

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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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


#Create Independent Train / Val Datasets

train_full = HistopathologicCSVDataset(
    csv_file="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train_labels.csv",
    img_dir="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train",
    transform=train_transform
)

val_full = HistopathologicCSVDataset(
    csv_file="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train_labels.csv",
    img_dir="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train",
    transform=None  # keep PIL images for TTA
)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size

train_dataset, _ = random_split(train_full, [train_size, val_size])
_, val_dataset = random_split(val_full, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

#Custom collate_fn so PIL images aren't auto-stacked
val_dataloader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda batch: (
        [item[0] for item in batch],
        torch.tensor([item[1] for item in batch])
    )
)


#Model

resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)

resnet50_model.fc = nn.Identity()

for param in resnet50_model.parameters():
    param.requires_grad = False

resnet50_model.eval()
resnet50_model = resnet50_model.to(device)

fc_model = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1),
).to(device)

model = nn.Sequential(
    resnet50_model,
    fc_model
).to(device)

optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.00025)
loss_fn = nn.BCEWithLogitsLoss()


#Training loop

for epoch in range(100):

    print(f"\n--- EPOCH: {epoch} ---")

    
    # TRAINING
    

    model.train()
    resnet50_model.eval()

    loss_sum = 0
    train_accurate = 0
    train_sum = 0

    for X, y in tqdm(train_dataloader):

        X = X.to(device)
        y = y.to(device).type(torch.float).reshape(-1, 1)

        outputs = model(X)

        optimizer.zero_grad()
        loss = loss_fn(outputs, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        predictions = torch.sigmoid(outputs) > 0.5
        train_accurate += (predictions == y).sum().item()
        train_sum += y.size(0)

    print("Training loss: ", np.round(loss_sum / len(train_dataloader), 4))
    print("Training accuracy: ", np.round((train_accurate / train_sum) * 100, 2), "%")

    torch.save(fc_model.state_dict(), f"fc_model_{epoch}.pth")

    
    #VALIDATION WITH TTA

    model.eval()

    val_loss_sum = 0
    val_accurate = 0
    val_sum = 0

    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():

        for X_raw, y in tqdm(val_dataloader):

            batch_size = len(y)
            y = y.to(device).type(torch.float).reshape(-1, 1)

            vote_sum = torch.zeros(batch_size, 1).to(device)

            for _ in range(NUM_TTA_ROTATIONS):

                rotated_images = []

                for img in X_raw:
                    img = transforms.RandomRotation(90)(img)
                    img = val_base_transform(img)
                    rotated_images.append(img)

                X = torch.stack(rotated_images).to(device)

                outputs = model(X)
                probs = torch.sigmoid(outputs)

                vote_sum += (probs > 0.5).float()

            predictions = (vote_sum >= (NUM_TTA_ROTATIONS / 2)).float()

            avg_probs = vote_sum / NUM_TTA_ROTATIONS
            loss = loss_fn(avg_probs, y)
            val_loss_sum += loss.item()

            all_val_preds.append(predictions.detach().cpu().int().view(-1))
            all_val_labels.append(y.detach().cpu().int().view(-1))

            val_accurate += (predictions == y).sum().item()
            val_sum += y.size(0)

    print("Validation loss: ", np.round(val_loss_sum / len(val_dataloader), 4))
    print("Validation accuracy: ", np.round((val_accurate / val_sum) * 100, 2), "%")

    
    #Confusion matrix + metrics   

    all_val_preds = torch.cat(all_val_preds).numpy()
    all_val_labels = torch.cat(all_val_labels).numpy()

    tp = ((all_val_preds == 1) & (all_val_labels == 1)).sum()
    tn = ((all_val_preds == 0) & (all_val_labels == 0)).sum()
    fp = ((all_val_preds == 1) & (all_val_labels == 0)).sum()
    fn = ((all_val_preds == 0) & (all_val_labels == 1)).sum()

    eps = 1e-12
    accuracy    = (tp + tn) / (tp + tn + fp + fn + eps)
    precision   = tp / (tp + fp + eps)
    recall      = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"[[TN FP]\n [FN TP]]")
    print(f"[[{tn} {fp}]\n [{fn} {tp}]]")

    print(f"Accuracy:    {accuracy*100:.2f}%")
    print(f"Precision:   {precision*100:.2f}%")
    print(f"Recall:      {recall*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%\n")