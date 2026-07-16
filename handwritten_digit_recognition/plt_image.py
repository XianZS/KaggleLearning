import matplotlib.pyplot as plt
import numpy as np

# 1. 三组数据：(训练集准确率, 测试集准确率)
data = {
    "Softmax": (0.9236483333333334, 0.92643),
    "MLP": (0.9814833333333334, 0.97346),
    "CNN": (0.991085, 0.9896799999999999),
}

# 提取标签、训练准确率、测试准确率
models = list(data.keys())
train_acc = [data[item][0] for item in models]
test_acc = [data[item][1] for item in models]

# 2. 绘图基础设置
x = np.arange(len(models))  # x轴位置
width = 0.35  # 柱子宽度

fig, ax = plt.subplots(figsize=(8, 5))
# 绘制两组柱状图
bar1 = ax.bar(
    x - width / 2, train_acc, width, label="train dataset accuracy", color="#3498db"
)
bar2 = ax.bar(
    x + width / 2, test_acc, width, label="test dataset accuracy", color="#e74c3c"
)


# 3. 在柱子顶部标注数值
def add_label(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 文字偏移
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


add_label(bar1)
add_label(bar2)

# 4. 图表美化配置
ax.set_title("Softmax / MLP / CNN", fontsize=14, pad=15)
ax.set_ylabel("accuracy rate", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.10)  # 正确率0~1区间
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)  # 横向网格线

plt.tight_layout()
plt.show()
