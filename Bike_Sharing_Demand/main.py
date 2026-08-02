import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

train_csv = pd.read_csv("./data/competitions/bike-sharing-demand/train.csv")
test_csv = pd.read_csv("./data/competitions/bike-sharing-demand/test.csv")
# print(train_csv.columns, len(train_csv.columns))
# print(test_csv.columns, len(test_csv.columns))
val_label = test_csv["datetime"].copy()
"""
feature:
日期时间、季节、假日、工作日、天气、温度、体感温度、湿度、风速
pred:
普通、注册、计数
"""


def pr():
    print("-" * 60)


def data_transformation(data_csv):
    # --- 用众数补全nan
    data_csv = data_csv.fillna(list(data_csv.mode())[0])

    # --- 特殊类型处理
    # data_csv_head5 = data_csv.head(5)
    data_csv["datetime"] = pd.to_datetime(data_csv["datetime"])
    data_csv["datetime"] -= data_csv["datetime"][0]
    data_csv["datetime"] = data_csv["datetime"].dt.total_seconds()

    # --- 归一化
    scaler = MinMaxScaler()
    data_csv[["datetime", "temp", "atemp", "humidity", "windspeed"]] = (
        scaler.fit_transform(
            data_csv[["datetime", "temp", "atemp", "humidity", "windspeed"]]
        )
    )
    return data_csv


features = [
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
]
labels = ["count"]
train_csv[features] = data_transformation(train_csv[features])
test_csv = data_transformation(test_csv)
X = torch.tensor(train_csv[features].to_numpy().astype("float32"))
Y = torch.tensor(train_csv[labels].to_numpy().astype("float32"))
eval_X = torch.tensor(test_csv.to_numpy().astype("float32"))
train_dataset = TensorDataset(X, Y)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
)
# print(Y)
# print(len(train_csv[labels]), torch.max(Y, dim=0))
# print(train_csv.head(10))
# print(max(train_csv["datetime"]))


# --- 定义模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, X):
        return self.fc(X)


class ImprovedModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))  # 输出 log(lambda)
        self.net = nn.Sequential(*layers)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedModel(
    input_size=9,
    # hidden_size=256,
    # output_size=1,
).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.PoissonNLLLoss()

epochs = 100

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for datas, labels in train_dataloader:
        datas, labels = datas.to(device), labels.to(device)
        outputs = model(datas)
        loss = criterion(outputs, labels)
        train_loss = train_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[{epoch}/{epochs}] >>> {train_loss}")

model.eval()
eval_X = eval_X.to(device)
with torch.no_grad():
    outputs = model(eval_X)
    outputs = torch.exp(outputs)
    outputs = outputs.int()
    outputs = outputs.flatten()
    # print(outputs.int())
    # print(torch.max(outputs, dim=0))
# print(outputs.shape)
submission = pd.DataFrame({"datetime": val_label, "count": outputs.cpu().numpy()})
submission.to_csv("submission.csv", index=False)
