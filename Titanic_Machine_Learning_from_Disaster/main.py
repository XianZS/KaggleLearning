import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torchvision import transforms
import torchvision
from sklearn.preprocessing import StandardScaler

train_csv = pd.read_csv("./kaggle/input/competitions/titanic/train.csv")
test_csv = pd.read_csv("./kaggle/input/competitions/titanic/test.csv")
# print(train_csv.head(5))
# print(list(train_csv.columns))
# [
# 'PassengerId', 'Survived', 'Pclass',
# 'Name', 'Sex', 'Age',
# 'SibSp', 'Parch', 'Ticket',
# 'Fare', 'Cabin', 'Embarked'
# ]

res = train_csv.head(5)
# print(len(res))
# print(res.iloc[0])

# PassengerId                          1    用户id
# Survived                             0    是否生存
# Pclass                               3    车票等级 1、2、3
# Name           Braund, Mr. Owen Harris    姓名
# Sex                               male    性别
# Age                               22.0    年龄
# SibSp                                1    兄弟姐妹的数量
# Parch                                0    父母
# Ticket                       A/5 21171    票号
# Fare                              7.25    票价
# Cabin                              NaN    舱位号
# Embarked                             S    出发港


features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    # "Cabin",
    "Embarked",
]

target = "Survived"
# print(type(train_csv))
# print("-" * 10)

train_csv["Sex"] = train_csv["Sex"].replace({"male": 0, "female": 1})
train_csv["Embarked"] = train_csv["Embarked"].replace({"C": 0, "S": 1, "Q": 2})
train_csv["Age"] = train_csv["Age"].fillna(train_csv["Age"].median())
train_csv["Fare"] = train_csv["Fare"].fillna(train_csv["Fare"].median())
train_csv["Embarked"] = train_csv["Embarked"].fillna(train_csv["Embarked"].median())

num_features = ["Age", "Fare", "SibSp", "Parch"]  # Pclass 是等级，可以不标准化
scaler = StandardScaler()
train_csv[num_features] = scaler.fit_transform(train_csv[num_features])

# print(train_csv.head(5))
X = train_csv[features].values.astype("float32")
Y = train_csv[target].values.astype("int64")
# print(X)
# print(Y)
X = torch.tensor(X)
Y = torch.tensor(Y)

train_dataset = TensorDataset(X, Y)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

print(X.shape)


#
# class SoftmaxModel(nn.Module):
#     def __init__(self, input_dim, output_dim=2):
#         super().__init__()
#         self.Linear = nn.Linear(
#             input_dim,
#             output_dim,
#         )
#
#     def forward(self, X):
#         return self.Linear(X)
# ---------- 模型定义（添加隐藏层） ----------
class SoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)  # 返回 logits，不包含 Softmax


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoftmaxModel(input_dim=7, output_dim=2).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    train_private = 0
    train_total = 0
    for datas, labels in train_loader:
        datas, labels = datas.to(device), labels.to(device)
        output = model(datas)
        # print(output)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred_index = torch.max(output.data, dim=1)
        train_total += len(pred_index)
        print(loss.item())

if __name__ == "__main__":
    pass
