'''
A code summary with comments explaining its functionality at necessary points:

This code trains a modified VGG19 neural network for sound classification, using Mel-spectrogram image representations of sounds. 
It preprocesses and loads the dataset, defining transformations, a training loop, and validation checks. 
Key operations include customizing a PyTorch Dataset, setting up data transformations, freezing VGG19 convolution layers, and defining a classifier for sound category output. 
The training loop iterates through epochs, storing metrics for each epoch, and ends with visualization of accuracy, loss, and a confusion matrix to evaluate performance.
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import parameters  # This imports all predefined parameters

# Checking if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset for Mel-Spectrogram Images
class SoundDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, num_samples=1000):
        self.labels = pd.read_csv(csv_file)  # Load CSV with filenames and labels
        self.labels = self.labels.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)  # Randomly sample 1000 samples
        self.img_dir = img_dir  # Directory for images
        self.transform = transform  # Image transformations (resize, normalize)

        # Creating label-to-index mapping based on CLASS_LIST in parameters
        self.label_to_idx = {label: idx for idx, label in enumerate(parameters.CLASS_LIST)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])  # Image filename from CSV
        try:
            image = Image.open(img_name).convert('RGB')  # Open and convert image to RGB
        except FileNotFoundError:
            return None  # Handle missing image case

        label = self.labels.iloc[idx, 1]  # Getting label for image
        label = self.label_to_idx[label]  # Mapping label to corresponding index
        label = torch.tensor(label)  # Convertting label index to tensor

        if self.transform:
            image = self.transform(image)  # Applying transformations

        return image, label

# DataLoader function to skip None entries
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Skipping missing images
    if len(batch) == 0:
        return None, None

    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(labels, dim=0)

# Transformations for images (Resizing and Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for VGG19
    transforms.ToTensor(),  # Convertting image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizing based on ImageNet values
])

# Define paths for CSV and images
train_csv = "data/train_labels.csv"
img_dir = "data/Mel_Spectrogram_images"

# Dataset for Mel-Spectrogram Images
dataset = SoundDataset(csv_file=train_csv, img_dir=img_dir, transform=transform, num_samples=1000)

# Splitting dataset into 80% train and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders for train and validation
train_loader = DataLoader(train_dataset, batch_size=parameters.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=parameters.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Loadded pre-trained VGG19 model and modify for classification task
vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)  # Loadded VGG19 pretrained on ImageNet
for param in vgg19.features.parameters():
    param.requires_grad = False  # Freeze convolutional layers

vgg19.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(0.61),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.61),
    nn.Linear(4096, parameters.NUM_CLASSES),  # Output layer with NUM_CLASSES as output
    nn.LogSoftmax(dim=1)
)

# Moving model to device (GPU or CPU)
vgg19 = vgg19.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
LEARNING_RATE = 0.0002  # Set learning rate directly
optimizer = torch.optim.Adam(vgg19.parameters(), lr=LEARNING_RATE)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).float().sum()
    return correct / len(labels)

# Training loop
def train_model():
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    NUM_EPOCHS = 20  # Setting number of epochs
    for epoch in range(0, NUM_EPOCHS):

        vgg19.train()  # Enable training mode
        running_loss = 0.0
        correct_train = 0.0
        total_train = 0.0

        for i, batch in enumerate(train_loader):
            if batch is None:
                continue  # Skip empty batches

            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = vgg19(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_train += calculate_accuracy(outputs, labels) * len(labels)
            total_train += len(labels)

        train_accuracy = correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # Validation phase
        vgg19.eval()
        val_loss = 0.0
        correct_val = 0.0
        total_val = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if batch is None:
                    continue  # Skip empty batches

                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = vgg19(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                correct_val += calculate_accuracy(outputs, labels) * len(labels)
                total_val += len(labels)

                # Collect predictions and labels for confusion matrix
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = correct_val / total_val
        val_loss /= len(val_loader)

        # Store losses and accuracies
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy.item())
        val_accuracies.append(val_accuracy.item())

        # Printting metrics
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Printting final training and validation accuracy
    final_training_accuracy = train_accuracies[-1]
    final_validation_accuracy = val_accuracies[-1]

    print(f"\nFinal Training Accuracy: {final_training_accuracy * 100:.2f}%")
    print(f"Final Validation Accuracy: {final_validation_accuracy * 100:.2f}%")

    # Plotting training and validation accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plotting final confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=parameters.CLASS_LIST, yticklabels=parameters.CLASS_LIST)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Final Confusion Matrix')
    plt.show()

# Run the training
train_model()
