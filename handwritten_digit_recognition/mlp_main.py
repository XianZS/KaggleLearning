# === 多层感知机实现 ===
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}\ttype:{type(device)}")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_dim=784, output1=256, output2=128, output3=10):
        super().__init__()
        self.__fc = nn.Sequential(
            nn.Linear(input_dim, output1),
            nn.ReLU(),
            nn.Linear(output1, output2),
            nn.ReLU(),
            nn.Linear(output2, output3),
        )

    def forward(self, X):
        # print(X)
        X = X.flatten(start_dim=1)
        return self.__fc(X)


model = MLP().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
_train, _test = 0, 0

epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    train_private = 0
    train_total = 0
    for images, labels in train_loader:
        # 前向传播
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 统计训练指标
        train_loss += loss.item()
        _, pred_index = torch.max(outputs, dim=1)
        train_total += pred_index.size(0)
        train_private += (pred_index == labels).sum().item()
    model.eval()
    test_loss = 0.0
    test_private = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 前向传播
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, pred_index = torch.max(outputs, dim=1)
            test_total += pred_index.size(0)
            test_private += (pred_index == labels).sum().item()

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

torch.save(model, "./save/mlp_model.pth")
