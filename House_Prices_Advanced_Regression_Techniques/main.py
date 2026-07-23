import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch._higher_order_ops import inline_asm_elementwise
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


train_csv = pd.read_csv(
    "./data/competitions/house-prices-advanced-regression-techniques/train.csv"
)
test_csv = pd.read_csv(
    "./data/competitions/house-prices-advanced-regression-techniques/test.csv"
)


def _show():
    _desc_file = "./data/competitions/house-prices-advanced-regression-techniques/data_description.txt"
    with open(_desc_file, mode="r", encoding="utf-8") as file:
        text = file.read()
        print(text)


def _choose_col_to_train_model(data):
    print(data.dtypes)
    col = data.columns
    print(data[col[0]][0], type(data[col[0]][0]))


if __name__ == "__main__":
    _choose_col_to_train_model(train_csv)
