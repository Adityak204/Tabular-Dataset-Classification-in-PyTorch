import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_utils.data_prep import DataPrep
from data_utils.custom_dataset import TabularDataset
from tabular_model_pack.model import TorchTabular

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model = True
batch_size = 64
num_epochs = 10

# Loading Classification Dataset (Kaggle: Categorical Feature Encoding Challenge II)
train_base_data = pd.read_csv("")
test_base_data = pd.read_csv("")

# Data Preparation
test_base_data.loc[:, "target"] = -1
total_data = pd.concat([train_base_data, test_base_data]).reset_index(drop=True)
data_treat = DataPrep(data=total_data, categorical_var_list=total_data.columns[:-1])
clean_data = data_treat.run_preprocessing(treat_na=True, label_encode=True)
embedding_details = data_treat.cat_to_embed()

train = clean_data.loc[clean_data.target != -1].reset_index(drop=True)
test = clean_data.loc[clean_data.target == -1].reset_index(drop=True)

train_df, valid_df = train_test_split(train, test_size=0.3, stratify=train.target)

# Generating Dataset for DataLoader consumption
train_dataset = TabularDataset(data=train_df.iloc[:, :-1].values, targets=train_df.targets)
valid_dataset = TabularDataset(data=valid_df.iloc[:, :-1].values, targets=valid_df.targets)
# test_dataset = TabularDataset(data=test.iloc[:, :-1].values, targets=test.targets)

# Defining Iterator
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
# train_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

tabular_model = TorchTabular(embed_details=embedding_details,
                             dropouts=0.3,
                             linear_layer_sz=100,
                             output_layer_sz=1).to(device)

optimizer = optim.Adam(tabular_model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
criterion = torch.nn.BCELoss()
loss_tracker = []

for epoch in range(num_epochs):
    tabular_model.train()
    losses = []
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch_idx, batch in loop:
        train_dt = batch['x'].to(device)
        target = batch['y'].to(device)

        # Forward prop
        output = tabular_model(data=train_dt)

        optimizer.zero_grad()
        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clipping exploding gradients
        torch.nn.utils.clip_grad_norm(tabular_model.parameters(), max_norm=1)
        # Gradient descent step
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    train_mean_loss = sum(losses)/len(losses)
    scheduler.step(train_mean_loss)

    tabular_model.eval()
    val_losses = []
    val_auc = []
    with torch.no_grad():
        for val_batch_idx, val_batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            val_dt = val_batch['x'].to(device)
            val_target = val_batch['y'].to(device)
            val_output = tabular_model(data=val_dt)
            val_loss = criterion(val_output, val_target)
            # validation_auc =
            val_losses.append(val_loss)
        val_mean_loss = sum(val_losses)/len(val_losses)

    loss_tracker.append(val_mean_loss)

    if epoch % 1 == 0:
        if save_model and val_mean_loss == np.min(loss_tracker):
            checkpoint = {
                "state_dict": tabular_model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint)

    print(f"Epoch [{epoch + 1}/{num_epochs}]: train_loss = {train_mean_loss}; val_loss = {val_mean_loss}; val_auc = {val_mean_auc}")
