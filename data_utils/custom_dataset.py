import torch


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
            "x": torch.tensor(current_sample),
            "y": torch.tensor(current_target, dtype=torch.float)
            if self.model_type == 'regression' else torch.tensor(current_target, dtype=torch.long)
        }
