import torch
import pandas as pd
from data_prep import DataPrep


class TabularDataset:
    def __init__(self, data, targets, model_type='classification'):
        self.data = data
        self.targets = targets
        self.model_type = model_type

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        return {
            "x": torch.tensor(current_sample, dtype=torch.long),
            "y": torch.tensor(current_target, dtype=torch.float)
            if self.model_type == 'regression' else torch.tensor(current_target, dtype=torch.long)
        }


if __name__ == '__main__':
    dt = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'a', 'c', 'd', 'e', 'c'],
                       'class': ['I', 'IV', 'V', None, 'I', 'V', None, 'VII', 'V'],
                       'targets': [0, 0, 0, 1, 1, 0, 1, 0, 1]
                       })
    data_treat = DataPrep(data=dt, categorical_var_list=['category', 'class'])
    clean_data = data_treat.run_preprocessing(treat_na=True, label_encode=True)
    dataset = TabularDataset(data=clean_data[['category', 'class']].values, targets=clean_data['targets'])
    print(dataset[8])
