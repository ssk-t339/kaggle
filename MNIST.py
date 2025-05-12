import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from google.colab import drive
drive.mount('/content/drive')

# データを読み込む
with open("/content/drive/MyDrive/data/sampled_mnist.pkl", "rb") as f:
    sampled_data, sampled_labels = pickle.load(f)

# MNISTのテストデータを読み込む準備（評価のため）
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./content/drive/MyDrive/data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# サンプリングデータ用のカスタムデータセット
class SampledDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0).float() / 255.0, self.labels[idx]

# サンプリングデータのデータローダー
sampled_dataset = SampledDataset(sampled_data, sampled_labels)
sampled_loader = DataLoader(sampled_dataset, batch_size=64, shuffle=True)

# CNNモデルの定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# モデル、損失関数、オプティマイザの初期化
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in sampled_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(sampled_loader)
    epoch_accuracy = correct / total

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# 学習したモデルを保存
model_path = "trained_cnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# テストデータでの評価
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(y_true, y_pred)

class_correct = conf_matrix.diagonal()
class_total = conf_matrix.sum(axis=1)
class_accuracy = class_correct / class_total

for i, accuracy in enumerate(class_accuracy):
    print(f"Class {i}: Accuracy = {accuracy:.2%}")

ConfusionMatrixDisplay(conf_matrix, display_labels=list(range(10))).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()