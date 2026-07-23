import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

train_csv = pd.read_csv("./data/competitions/spaceship-titanic/train.csv")
test_csv = pd.read_csv("./data/competitions/spaceship-titanic/test.csv")


features = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "Age",
    "VIP",
    "Spend",
]
targets = "Transported"
# --- 数据处理
train_csv = train_csv.fillna(train_csv.mode().iloc[0])
hp_set = sorted(set(train_csv["HomePlanet"]))
train_csv["HomePlanet"] = train_csv["HomePlanet"].replace(
    {strs: index for index, strs in enumerate(hp_set, start=1)}
)
train_csv["CryoSleep"] = train_csv["CryoSleep"].replace({True: 1, False: 0})
go_set = sorted(set(train_csv["Destination"]))
train_csv["Destination"] = train_csv["Destination"].replace(
    {strs: index for index, strs in enumerate(go_set, start=1)}
)
train_csv["VIP"] = train_csv["VIP"].replace({True: 1, False: 0})
train_csv["Spend"] = train_csv[
    ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
].mean(axis=1)
train_csv["Transported"] = train_csv["Transported"].replace({True: 1, False: 0})
# print(train_csv[features])

# --- 归一化操作
scaler_std = StandardScaler()
train_csv[["Age", "Destination", "Spend"]] = scaler_std.fit_transform(
    train_csv[["Age", "Destination", "Spend"]]
)

# --- 创建数据集
X = torch.tensor(train_csv[features].to_numpy().astype("float32"))
Y = torch.tensor(train_csv[targets].to_numpy().astype("int64"))
train_dataset = TensorDataset(X, Y)
print(train_dataset)
print(train_csv[features].head(5))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
)


# --- 模型
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        return self.fc(X)


# print(f"X.shape: {X.shape}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(
    input_dim=6,
    hidden_dim=64,
    output_dim=2,
).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("---" * 10)
print("Input dtype:", X.dtype)  # 检查输入
print("Model weight dtype:", next(model.parameters()).dtype)  # 检查第一个参数

epochs = 100
loss_list = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for datas, labels in train_loader:
        datas, labels = datas.to(device), labels.to(device)
        output = model(datas)
        loss = criterion(output, labels)
        now_loss = loss.item()
        train_loss += now_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"[loss] >>> {now_loss}/{train_loss}")
    print(train_loss)
    loss_list.append(train_loss)

torch.save(model.state_dict(), "./model.pth")


# --- 数据验证
test_csv = test_csv.fillna(test_csv.mode().iloc[0])
hp_set = sorted(set(test_csv["HomePlanet"]))
test_csv["HomePlanet"] = test_csv["HomePlanet"].replace(
    {strs: index for index, strs in enumerate(hp_set, start=1)}
)
test_csv["CryoSleep"] = test_csv["CryoSleep"].replace({True: 1, False: 0})
go_set = sorted(set(test_csv["Destination"]))
test_csv["Destination"] = test_csv["Destination"].replace(
    {strs: index for index, strs in enumerate(go_set, start=1)}
)
test_csv["VIP"] = test_csv["VIP"].replace({True: 1, False: 0})
test_csv["Spend"] = test_csv[
    ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
].mean(axis=1)
scaler_std = StandardScaler()
test_csv[["Age", "Destination", "Spend"]] = scaler_std.fit_transform(
    test_csv[["Age", "Destination", "Spend"]]
)
X = torch.tensor(test_csv[features].to_numpy().astype("float32")).to(device)
model.load_state_dict(torch.load("./model.pth"))
model.eval()
with torch.no_grad():
    outputs = model(X)
    # print(outputs)
    _, preds = torch.max(outputs, dim=1)

output = model(X)
print(output)
print(preds, type(preds))
submission = pd.DataFrame(
    {"PassengerId": test_csv["PassengerId"], "Transported": preds.cpu().numpy()}
)
submission["Transported"] = submission["Transported"].replace({1: True, 0: False})
print(submission)
submission.to_csv("submission.csv", index=False)


def __things():
    """
    PassengerId：（舍弃）
    每位乘客都有一个唯一的ID。ID格式为，gggg_pp其中gggg表示乘客所属的旅行团，pp表示乘客在旅行团中的编号。旅行团成员通常是家庭成员，但并非总是如此。

    HomePlanet：（字符串）
    乘客出发的星球，通常是他们的永久居住星球。

    CryoSleep：（False/True）
    表示乘客是否选择在航程期间处于低温休眠状态。处于低温休眠状态的乘客将被限制在自己的舱房内。

    Cabin：（x/y/z）
    乘客所住的舱位号。格式为deck/num/side，其中side可以是P左舷（Port）或S右舷（Starboard）。

    Destination：（目的地）
    乘客将要下船前往的星球。

    Age：（年龄）
    乘客的年龄。

    VIP：（True/False）
    乘客是否在航行期间支付了特殊 VIP 服务费用。

    RoomService, FoodCourt, ShoppingMall, Spa, VRDeck：（额外支出费用）
    乘客在泰坦尼克号飞船FoodCourt的众多豪华设施中每一项ShoppingMall所支付的金额。SpaVRDeck

    Name：（姓名）
    乘客的姓名（包括姓和名）。

    Transported：（pred）
    乘客是否被传送到了另一个维度。这是目标，也是你要预测的那一列。
    """


def show():
    plt.figure(figsize=(16, 10))
    plt.plot(
        [x + 1 for x in range(100)],
        loss_list,
        label="loss value",
    )

    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    __things()
    show()
