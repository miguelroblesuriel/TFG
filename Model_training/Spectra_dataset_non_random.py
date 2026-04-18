import os
import pandas as pd
from torchvision.io import decode_image
import numpy as np
import math
from torch.utils.data import Dataset

class Spectra_dataset_non_random(Dataset):
    def __init__(self, dupla, tripletas, scores):
        self.dupla = dupla
        self.tripletas = tripletas
        self.scores = scores

    def __len__(self):
        return len(self.tripletas)

    def __getitem__(self, idx):
        anchor = self.dupla[0]
        positive = self.dupla[1]
        negative = self.tripletas[idx]
        return anchor, positive, negative