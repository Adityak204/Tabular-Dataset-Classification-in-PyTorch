import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_utils.data_prep import DataPrep
from data_utils.custom_dataset import TabularDataset
from tabular_model_pack.model import TorchTabular

# Loading Classification Dataset (Kaggle: Categorical Feature Encoding Challenge II)
train_base_data = pd.read_csv("")
test_base_data = pd.read_csv("")

# Data Preparation
test_base_data.loc[:, "target"] = -1
total_data = pd.concat([train_base_data, test_base_data]).reset_index(drop=True)
data_treat = DataPrep(data=total_data, categorical_var_list=['category', 'class'])
clean_data = data_treat.run_preprocessing(treat_na=True, label_encode=True)
embedding_details = data_treat.cat_to_embed()

train = clean_data.loc[clean_data.target != -1].reset_index(drop=True)
test = clean_data.loc[clean_data.target == -1].reset_index(drop=True)

train_df, valid_df = train_test_split(train, test_size=0.3, stratify=train.target)


# Generating Dataset for PyTorch consumption
train_dataset = TabularDataset(data=train_df.iloc[:, :-1].values, targets=train_df.targets)
valid_dataset = TabularDataset(data=valid_df.iloc[:, :-1].values, targets=valid_df.targets)
test_dataset = TabularDataset(data=test.iloc[:, :-1].values, targets=test.targets)

