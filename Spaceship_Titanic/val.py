import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_csv = pd.read_csv("./data/competitions/spaceship-titanic/test.csv")


features = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "Age",
    "VIP",
    "Spend",
]
# targets = "Transported"

# --- 数据处理
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
# test_csv["Transported"] = test_csv["Transported"].replace({True: 1, False: 0})
# print(train_csv[features])

# --- 归一化操作
scaler_std = StandardScaler()
test_csv[["Age", "Destination", "Spend"]] = scaler_std.fit_transform(
    test_csv[["Age", "Destination", "Spend"]]
)

# --- 创建数据集
X = torch.tensor(test_csv[features].to_numpy().astype("float32"))
X = X.to(device)


model = torch.load("./model.pth")
model.eval()
with torch.no_grad():
    outputs = model(X)
    print(outputs)

output = model(X)
print(output)
