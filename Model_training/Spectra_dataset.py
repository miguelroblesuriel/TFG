import os
import pandas as pd
from sympy.printing.pytorch import torch
from torchvision.io import decode_image
import numpy as np
import math
from torch.utils.data import Dataset
import torch

class Spectra_dataset(Dataset):
    def __init__(self, dupla, tripletas, scores):
        self.dupla = dupla
        self.tripletas = tripletas
        self.scores = scores
        scores_squared = np.array([i ** 2 for i in self.scores])
        self.p_normalized = scores_squared / scores_squared.sum()
        self.indices = np.arange(len(self.tripletas))

    def __len__(self):
        return len(self.tripletas)

    def __getitem__(self, item):
        anchor = self.dupla[0]
        positive = self.dupla[1]
        chosen_index = np.random.choice(self.indices, p=self.p_normalized)
        negative = self.tripletas[chosen_index]
        anchor = torch.tensor(anchor, dtype=torch.float32)
        positive = torch.tensor(positive, dtype=torch.float32)
        negative = torch.tensor(negative, dtype=torch.float32)
        return anchor, positive, negative