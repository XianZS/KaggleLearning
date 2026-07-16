import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- base config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# --- data loader
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# --- set model
class SoftMaxModel(nn.Module):
    def __init__(self, input_dim=28 * 28, output_dim=10):
        super().__init__()
        self.Linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = X.flatten(start_dim=1)
        X1 = self.Linear(X)
        X2 = self.softmax(X1)
        return X2


model = SoftMaxModel().to(device=device)

# --- set loss function
criterion = nn.CrossEntropyLoss()

# --- set optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- train model
# 10轮，每轮拿60000个数据，每单次使用64个数据
epochs = 10
_train, _test = 0.0, 0.0
for epoch in range(1, epochs + 1):
    # ========= 训练阶段 =========
    model.train()
    # 训练总损失
    train_loss = 0.0
    # 训练得到的正确样本数
    train_private = 0
    # 训练总样本数
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # --- 前向传播
        # print(labels)
        # tensor([1, 8, 3, 1, 9, 8, 1, 7, 4, 4, 2, 4, 2, 8, 3, 0, 8, 4, 3, 2, 2, 5, 5, 4,
        # 2, 5, 6, 5, 7, 6, 3, 4, 4, 7, 6, 2, 2, 3, 4, 3, 3, 9, 1, 6, 7, 4, 5, 7,
        # 5, 0, 1, 7, 5, 2, 6, 2, 5, 3, 0, 7, 6, 8, 8, 7], device='cuda:0')
        outputs = model(images)
        # print(len(outputs))
        # 64
        loss = criterion(outputs, labels)
        # print(loss)
        # --- 反向传播 + 优化器参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # --- 统计指标
        train_loss += loss.item()
        # print(len(outputs.data), len(outputs.data[0]))
        # 64 * 10
        _, pred_index = torch.max(outputs.data, dim=1)
        # print(pred_index)
        # 更新训练总样本数
        train_total += len(pred_index)
        # 更新正确样本数
        res = pred_index == labels
        train_private += res.sum().item()
        # print(train_private)
    # print(train_private, "/", train_total)
    # ========= 验证阶段 =========
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
            _, pred_index = torch.max(outputs.data, dim=1)
            test_total += len(pred_index)
            res = pred_index == labels
            test_private += res.sum().item()
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
torch.save(model, "./save/softmax_model.pth")


def test1():
    """
    查看预处理之前的数据集对象
    """
    from random import randint
    import matplotlib.pyplot as plt

    pre_dataset = datasets.MNIST(train=True, download=True, root="./data")
    # 随机输出三个（图像信息，标签）
    for index in range(3):
        rand_index = randint(1, 1000)
        raw_img, label = pre_dataset[rand_index]
        print(type(raw_img), label)
        plt.figure(figsize=(4, 4))
        plt.imshow(raw_img)
        plt.title(f"[{index}]-[{label}]")
        plt.show()


def test2():
    """
    查看预处理之后的数据集对象
    """
    # 取第 0 个样本，dataset 会自动执行你定义的 transform 预处理
    sample_img, sample_label = train_dataset[0]

    print("=== 单样本基础信息 ===")
    print(f"输入图片形状：{sample_img.shape}")  # 格式：[通道数, 高度, 宽度]
    print(f"输入数据类型：{sample_img.dtype}")
    print(f"输入数值范围：[{sample_img.min():.4f}, {sample_img.max():.4f}]")
    print(f"对应标签：{sample_label}")
    print(f"标签类型：{type(sample_label)}")

    # 查看数据集总样本数
    print(f"\n训练集总样本数：{len(train_dataset)}")
    print(f"测试集总样本数：{len(test_dataset)}")


if __name__ == "__main__":
    # test2()
    pass
