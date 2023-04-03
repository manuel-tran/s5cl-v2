import random
import torch
import numpy as np
from torch.utils.data import Dataset, Subset

#----------------------------------------------------------------------------

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

    return labeled_targets, labeled_dataset, unlabeled_dataset
