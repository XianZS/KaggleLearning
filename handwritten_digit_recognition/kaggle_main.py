import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from torchvision import transforms
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 1. 自定义 Dataset ----------
class CSVMNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None, train=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.train:
            label = self.df.iloc[idx, 0]
            pixels = self.df.iloc[idx, 1:].values.astype(np.uint8)
        else:
            # 测试集用不到 label，随便给个 0（只要不参与 loss 计算）
            label = 0
            pixels = self.df.iloc[idx].values.astype(np.uint8)
        image = pixels.reshape(28, 28)
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------- 2. 预处理 ----------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# ---------- 3. 加载数据并划分验证集 ----------
full_train_dataset = CSVMNISTDataset(
    "./digit-recognizer/train.csv", transform=transform, train=True
)
test_dataset = CSVMNISTDataset(
    "./digit-recognizer/test.csv", transform=transform, train=False
)

# 划分训练集和验证集（9:1）
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ---------- 4. 模型定义（你的 CNN，尺寸正确） ----------
class CNN(nn.Module):
    def __init__(self, output=10):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 28→26
            nn.ReLU(),
            nn.MaxPool2d(2),  # 26→13
            nn.Conv2d(32, 64, kernel_size=3),  # 13→11
            nn.ReLU(),
            nn.MaxPool2d(2),  # 11→5
        )
        self._fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, output),
        )

    def forward(self, x):
        x = self._conv(x)
        x = x.flatten(start_dim=1)
        x = self._fc(x)
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

# ---------- 5. 训练 + 验证 ----------
for epoch in range(1, epochs + 1):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum().item()
        train_total += labels.size(0)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # 这里 labels 是真实的 0~9

            val_loss += loss.item()
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    train_acc = train_correct / train_total * 100
    val_acc = val_correct / val_total * 100
    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )


# ---------- 6. 对测试集进行预测并生成提交文件 ----------
def predict(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:  # 丢弃 label
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().tolist())
    return all_preds


test_preds = predict(model, test_loader, device)

# 生成 Kaggle submission.csv
submission = pd.DataFrame(
    {"ImageId": np.arange(1, len(test_preds) + 1), "Label": test_preds}
)
submission.to_csv("submission.csv", index=False)
print("预测结果已保存至 submission.csv")

# 保存模型（推荐保存 state_dict）
torch.save(model.state_dict(), "./save/kaggle_model.pth")
