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

def get_dataloader(
        datapath: str | Path,
        batch_size: int = 1024,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
):
    full_dataset = BurguersDataset(datapath)

    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader