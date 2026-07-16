import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


# 加载数据
train_dataset = datasets.MNIST(
    root="./data", download=True, train=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", download=True, train=False, transform=transform
)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)


# 定义模型
class CNN(nn.Module):
    def __init__(self, output=10):
        super().__init__()
        self._conv = nn.Sequential(
            # --- first layer
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # --- second layer
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, output),
        )

    def forward(self, X):
        X = self._conv(X)
        X = X.flatten(start_dim=1)
        X = self._fc(X)
        return X


model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
_train, _test = 0.0, 0.0


for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    train_private = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 参数统计
        train_loss += loss.item()
        _, pred_index = torch.max(outputs.data, dim=1)
        train_total += pred_index.size(0)
        train_private += (pred_index == labels).sum().item()
    model.eval()
    test_loss = 0.0
    test_private = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, pred_index = torch.max(outputs, dim=1)
            test_private += (pred_index == labels).sum().item()
            test_total += pred_index.size(0)
    print(f"=== {epoch} / {epochs} ===")
    print(
        f"训练准确率:{train_private / train_total * 100:.4f}%\t|\t训练损失:{train_loss}"
    )
    print(f"测试准确率:{test_private / test_total * 100:.4f}%\t|\t测试损失:{test_loss}")
    _train += train_private / train_total * 100
    _test += test_private / test_total * 100
    # break
print(f"训练平均准确率:{_train / epochs}")
print(f"测试平均准确率:{_test / epochs}")

torch.save(model, "./save/cnn_model.pth")
