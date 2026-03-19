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

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

preprocess = transforms.Compose([
    transforms.RandomRotation(180), #increase variation of training data with random rotation
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    #only minimal changes to colour because unsure if this is an important property to classification
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),

    transforms.Resize(96), #resize images to match training dataset, useful if later images don't match
    transforms.CenterCrop(64), #use because outer parts of images have other cells/are empty and they waste time, note that kaggle says use central 32 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

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

dataset = HistopathologicCSVDataset(
    csv_file="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train_labels.csv",
    img_dir="/Users/tomrose/Google_Drive/PyTorch/cancer_cells/histopathologic-cancer-detection/train",
    transform=preprocess
)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2]) #keep 20% for validation

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True) #don't train all at once
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = nn.Identity()
for param in resnet50_model.parameters(): #don't train ResNet
    param.requires_grad = False
resnet50_model.eval()
resnet50_model = resnet50_model.to(device)

fc_model = nn.Sequential( #add a couple of layers on after ResNet
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1), #1 output for binary classification
)
fc_model = fc_model.to(device)

model = nn.Sequential(
    resnet50_model,
    fc_model
)
model = model.to(device)

optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.00025) #a slow learning rate
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):
    print(f"--- EPOCH: {epoch} ---")
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
        accurate = (predictions == y).sum().item()
        train_accurate += accurate
        train_sum += y.size(0)
    print("Training loss: ", np.round(loss_sum / len(train_dataloader), 4))
    print("Training accuracy: ", np.round(((train_accurate / train_sum) * 100), 2), "%")

    torch.save(fc_model.state_dict(), f"fc_model_{epoch}.pth")

    model.eval()
    val_loss_sum = 0
    val_accurate = 0
    val_sum = 0

    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for X, y in tqdm(val_dataloader):
            X = X.to(device)
            y = y.to(device).type(torch.float).reshape(-1, 1)

            outputs = model(X)
            loss = loss_fn(outputs, y)
            val_loss_sum += loss.item()

            predictions = torch.sigmoid(outputs) > 0.5

            all_val_preds.append(predictions.detach().cpu().int().view(-1))
            all_val_labels.append(y.detach().cpu().int().view(-1))

            accurate = (predictions == y).sum().item()
            val_accurate += accurate
            val_sum += y.size(0)
    print("Validation loss: ", np.round(val_loss_sum / len(val_dataloader), 4))
    print("Validation accuracy: ", np.round(((val_accurate / val_sum) * 100), 2), "%")

    all_val_preds = torch.cat(all_val_preds).numpy()
    all_val_labels = torch.cat(all_val_labels).numpy()

    #Confusion matrix components for binary classification (positive class = 1)
    tp = ((all_val_preds == 1) & (all_val_labels == 1)).sum()
    tn = ((all_val_preds == 0) & (all_val_labels == 0)).sum()
    fp = ((all_val_preds == 1) & (all_val_labels == 0)).sum()
    fn = ((all_val_preds == 0) & (all_val_labels == 1)).sum()

    #Metrics (add small eps to avoid divide-by-zero)
    eps = 1e-12
    accuracy    = (tp + tn) / (tp + tn + fp + fn + eps)
    precision   = tp / (tp + fp + eps)
    recall      = tp / (tp + fn + eps) #aka sensitivity
    specificity = tn / (tn + fp + eps)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"[[TN FP]\n [FN TP]]")
    print(f"[[{tn} {fp}]\n [{fn} {tp}]]")

    print(f"Accuracy:    {accuracy*100:.2f}%")
    print(f"Precision:   {precision*100:.2f}%")
    print(f"Recall:      {recall*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%\n")
