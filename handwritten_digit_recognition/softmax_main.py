from sys import deactivate_stack_trampoline

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
loss = nn.CrossEntropyLoss()

# --- set optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- train model
epochs = len(train_dataset) // 64


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
    test2()
