import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# -------------------- 数据读取 --------------------
train_csv = pd.read_csv("./data/competitions/bike-sharing-demand/train.csv")
test_csv = pd.read_csv("./data/competitions/bike-sharing-demand/test.csv")

# 保存测试集的 datetime 用于提交
val_datetime = test_csv["datetime"].copy()


# -------------------- 特征工程函数 --------------------
def data_transformation(data_csv, is_train=True):
    """
    对数据进行特征工程和归一化
    """
    data_csv = data_csv.copy()  # 避免修改原数据

    # 1. 补全缺失值（用众数）
    data_csv = data_csv.fillna(data_csv.mode().iloc[0])

    # 2. 处理 datetime
    data_csv["datetime"] = pd.to_datetime(data_csv["datetime"])

    # 3. 时间分解特征
    data_csv["hour"] = data_csv["datetime"].dt.hour
    data_csv["dayofweek"] = data_csv["datetime"].dt.dayofweek  # 0=周一
    data_csv["month"] = data_csv["datetime"].dt.month
    data_csv["dayofyear"] = data_csv["datetime"].dt.dayofyear
    data_csv["weekend"] = (data_csv["datetime"].dt.dayofweek >= 5).astype(int)

    # 4. 交互特征
    data_csv["temp_humid"] = data_csv["temp"] * data_csv["humidity"]
    data_csv["temp_wind"] = data_csv["temp"] * data_csv["windspeed"]

    # 5. 相对时间（秒数）—— 用训练集最小时间为基准
    # 注意：训练和测试应使用相同的基准时间，这里用整个数据（训练+测试）的最小值？但测试时没有训练数据。
    # 为简单，我们分别处理，测试时用测试集自己的最小时间（实际竞赛中时间范围可能不同）
    # 更好的做法：预先定义基准时间，或统一使用训练集的最小时间。
    # 这里我们使用数据自身的最小值（训练和测试独立），但最好提前固定。
    # 为了对齐，我们在训练时记录基准时间，在测试时使用相同的基准。
    # 由于我们在这里无法跨函数传递，我们改在外部统一处理，或者使用全局变量。
    # 这里我们简单计算，但为了保持一致性，建议在外部统一处理，不在这里做秒数特征。
    # 我们将秒数特征放在外部处理，这样训练和测试可以使用相同的起始点。

    return data_csv


# 为了统一时间基准，我们全局设定基准时间 = 训练集最早时间
base_time = pd.to_datetime(train_csv["datetime"]).min()


def add_time_features(data_csv, base_time):
    """将时间特征添加到DataFrame，并归一化数值特征"""
    data_csv = data_csv.copy()
    data_csv["datetime"] = pd.to_datetime(data_csv["datetime"])

    # 时间分解
    data_csv["hour"] = data_csv["datetime"].dt.hour
    data_csv["dayofweek"] = data_csv["datetime"].dt.dayofweek
    data_csv["month"] = data_csv["datetime"].dt.month
    data_csv["dayofyear"] = data_csv["datetime"].dt.dayofyear
    data_csv["weekend"] = (data_csv["datetime"].dt.dayofweek >= 5).astype(int)

    # 相对秒数（从基准时间开始）
    data_csv["seconds_since"] = (data_csv["datetime"] - base_time).dt.total_seconds()

    # 交互特征
    data_csv["temp_humid"] = data_csv["temp"] * data_csv["humidity"]
    data_csv["temp_wind"] = data_csv["temp"] * data_csv["windspeed"]

    # 删除原始 datetime（已用完）
    data_csv = data_csv.drop(columns=["datetime"])

    return data_csv


# 对训练和测试分别添加时间特征
train_csv = add_time_features(train_csv, base_time)
test_csv = add_time_features(test_csv, base_time)

# 定义需要归一化的数值列（不包括类别/序数特征）
numeric_cols = [
    "seconds_since",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "weekend",
    "temp_humid",
    "temp_wind",
]
# 类别/序数特征：season, holiday, workingday, weather（它们已经是小整数，不归一化）

# 使用训练集的统计量归一化（防止数据泄露）
scaler = MinMaxScaler()
train_csv[numeric_cols] = scaler.fit_transform(train_csv[numeric_cols])
test_csv[numeric_cols] = scaler.transform(test_csv[numeric_cols])

# 定义特征列表（注意顺序）
features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "seconds_since",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "weekend",
    "temp_humid",
    "temp_wind",
]
labels = ["count"]

# 准备训练数据
X = torch.tensor(train_csv[features].values.astype(np.float32))
Y = torch.tensor(train_csv[labels].values.astype(np.float32))  # 泊松损失要求浮点数

# 划分训练集和验证集（按时间顺序，防止未来信息泄露）
# 可以按时间排序后切分，这里简单随机切分（但时间序列最好按顺序，我们按时间排序）
train_csv_sorted = train_csv.sort_values("seconds_since")  # 按时间排序
X_sorted = torch.tensor(train_csv_sorted[features].values.astype(np.float32))
Y_sorted = torch.tensor(train_csv_sorted[labels].values.astype(np.float32))

# 取前80%为训练，后20%为验证（时间顺序）
split_idx = int(0.8 * len(X_sorted))
X_train, X_val = X_sorted[:split_idx], X_sorted[split_idx:]
Y_train, Y_val = Y_sorted[:split_idx], Y_sorted[split_idx:]

# 创建 DataLoader
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 测试数据
eval_X = torch.tensor(test_csv[features].values.astype(np.float32))


# -------------------- 定义改进的模型 --------------------
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


# -------------------- 训练设置 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedModel(
    input_size=len(features), hidden_sizes=[256, 128, 64], dropout_rate=0.2
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.PoissonNLLLoss(full=True)  # 加常数项，损失变为非负
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

epochs = 200
best_val_loss = float("inf")
patience = 15
no_improve = 0

# -------------------- 训练循环 --------------------
for epoch in range(1, epochs + 1):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        train_loss += loss.item() * batch_X.size(0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    avg_train_loss = train_loss / len(train_dataset)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    avg_val_loss = val_loss / len(val_dataset)

    # 学习率调度
    scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # 早停与模型保存
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break

# -------------------- 加载最佳模型并预测 --------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
eval_X = eval_X.to(device)

with torch.no_grad():
    log_lambda = model(eval_X)
    pred_count = torch.exp(log_lambda)  # 转换为计数
    pred_count = pred_count.round().int().cpu().numpy().flatten()

# 生成提交文件
submission = pd.DataFrame({"datetime": val_datetime, "count": pred_count})
submission.to_csv("submission.csv", index=False)
print("预测完成，提交文件已保存为 submission.csv")
