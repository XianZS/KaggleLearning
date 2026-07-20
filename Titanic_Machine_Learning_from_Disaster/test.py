import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from sklearn.preprocessing import StandardScaler
from random import randint

# ---------- 数据加载 ----------
train_csv = pd.read_csv("./kaggle/input/competitions/titanic/train.csv")
test_csv = pd.read_csv("./kaggle/input/competitions/titanic/test.csv")

# ---------- 预处理 ----------
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

train_csv["Age"] = train_csv["Age"].fillna(train_csv["Age"].median())
train_csv["Fare"] = train_csv["Fare"].fillna(train_csv["Fare"].median())
train_csv["Embarked"] = train_csv["Embarked"].fillna(randint(0, 2))

# 编码分类特征
train_csv["Sex"] = train_csv["Sex"].replace({"male": 0, "female": 1})
train_csv["Embarked"] = train_csv["Embarked"].replace({"C": 0, "S": 1, "Q": 2})

# 标准化数值特征
num_features = ["Age", "Fare", "SibSp", "Parch"]
scaler = StandardScaler()
train_csv[num_features] = scaler.fit_transform(train_csv[num_features])

X = train_csv[features].values.astype("float32")
Y = train_csv[target].values.astype("int64")


train_dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# ---------- 模型 ----------
class TitanicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, X):
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # return self.fc2(x)
        return self.fc(X)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TitanicModel(input_dim=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- 训练 ----------
epochs = 30
best_val_acc = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for datas, labels in train_loader:
        datas, labels = datas.to(device), labels.to(device)
        outputs = model(datas)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# ---------- 测试集预处理（同样安全写法） ----------
test_csv["Age"] = test_csv["Age"].fillna(test_csv["Age"].median())
test_csv["Fare"] = test_csv["Fare"].fillna(test_csv["Fare"].median())
test_csv["Embarked"] = test_csv["Embarked"].fillna(randint(0, 2))
test_csv["Sex"] = test_csv["Sex"].replace({"male": 0, "female": 1})
test_csv["Embarked"] = test_csv["Embarked"].replace({"C": 0, "S": 1, "Q": 2})
test_csv[num_features] = scaler.transform(test_csv[num_features])

X_test = test_csv[features].values.astype("float32")
X_test_tensor = torch.tensor(X_test).to(device)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, preds = torch.max(outputs, 1)
    predictions = preds.cpu().numpy()

submission = pd.DataFrame(
    {"PassengerId": test_csv["PassengerId"], "Survived": predictions}
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
