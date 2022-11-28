import torch
import numpy as np
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 sequence_length: int = 1):

        self.sequence_length = sequence_length

        self.X = torch.tensor(x).float()
        self.Y = torch.tensor(y).float()

    def __len__(self):
        return self.X.__len__() - (self.sequence_length-1)

    def __getitem__(self, index: int):
        return self.X[index:index + self.sequence_length], self.Y[index + self.sequence_length - 1]
