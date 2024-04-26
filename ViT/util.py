import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset

def set_seed(SEED=47):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)  # For CUDA devices
        

def normalize_per_voxel(voxels, means, std_devs):
    return (voxels - means) / std_devs

def mean_sd_map(voxels, device = None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    voxels.to(device)

    means = torch.mean(voxels, dim=0)
    std_devs = torch.std(voxels, dim=0)
    
    return torch.stack((means, std_devs))


class FilteredDataset(Dataset):

    def __init__(self, dataset, indices, max_count = None):
        self.dataset = dataset
        self.indices = indices
        self.max_count = max_count

        if self.max_count is not None:
            if len(self.indices) > self.max_count:
                self.indices = self.indices[:self.max_count]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.dataset[idx]




def find_mean_sd(dataset):
    target_size = dataset[0][1].shape[0]
    dataset_size = len(dataset)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=64, num_workers=4)

    mean = torch.tensor(0.0, device='cuda', dtype=torch.float64)
    count = torch.tensor(target_size * dataset_size, device='cuda', dtype=torch.float64)

    for _, targs in tqdm(dataloader, desc="Finding mean"):
        targs = targs.to('cuda', dtype=torch.float64)
        mean += torch.sum(targs) / count

    print(f"Mean: {mean.cpu().item()}")

    var = torch.tensor(0.0, device='cuda', dtype=torch.float64)
    for _, targs in tqdm(dataloader, desc="Finding std dev"):
        targs = targs.to('cuda', dtype=torch.float64)
        var += torch.sum((targs - mean)**2) / count


    std = torch.sqrt(var)
    print(f"Std dev: {std.cpu().item()}")

    mean = mean.cpu().item()
    std = std.cpu().item()

    return mean, std