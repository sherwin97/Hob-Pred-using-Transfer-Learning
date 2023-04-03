import numpy as np


import pandas as pd
import torch
from torch_geometric.data import Dataset

import os
from tqdm import tqdm
import deepchem as dc
import rdkit
from rdkit import Chem


def seed_everything(seed):
    """Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoadSolDataset(Dataset):
    def __init__(self, root, raw_filename, transform=None, pre_transform=None):
        """
        root: directory of where raw file is at. Split into two path, processed and raw.
        filename: name of the raw file
        won't be using transform or pre-transform in this project
        """
        self.raw_filename = raw_filename
        super(LoadSolDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.raw_filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f"molecule_{i}.pt" for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["SMILES"])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = self._get_label(row["logS"])
            data.smiles = row["SMILES"]
            torch.save(data, os.path.join(self.processed_dir, f"molecule_{idx}.pt"))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"molecule_{idx}.pt"))


class LoadHOBDataset(Dataset):
    def __init__(self, root, raw_filename, transform=None, pre_transform=None):
        """
        root: directory of where raw file is at. Split into two path, processed and raw.
        filename: name of the raw file
        won't be using transform or pre-transform in this project
        """
        self.raw_filename = raw_filename
        super(LoadHOBDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.raw_filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f"molecule_{i}.pt" for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smile"])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = self._get_label(row["label_cutoff_50%"])
            data.smiles = row["smile"]
            torch.save(data, os.path.join(self.processed_dir, f"molecule_{idx}.pt"))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"molecule_{idx}.pt"))
