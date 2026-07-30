import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class BurguersDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        x = data["x"]
        t = data["t"]
        u = data["u"]

        T, X  = np.meshgrid(t, x, idexing="ij")
        inputs_data = np.column_stack((T.ravel(), X.ravel()))
        targets_data = u.ravel().reshape(-1, 1)

        self.inputs = torch.from_numpy(inputs_data).float()
        self.targets = torch.from_numpy(targets_data).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]