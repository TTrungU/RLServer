import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset


class TimeSeriesDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return (self.x[idx], self.y[idx])

class AlignCollate(object):
    """ Transform data to the same format """
    def __init__(self):
      pass
    def __call__(self, batch):
        X, y = zip(*batch)

        X = torch.stack(X)
        y = torch.stack(y)
        return X.type(torch.float32), y.type(torch.float32)
