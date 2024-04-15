import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

def set_seed(SEED=47):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)  # For CUDA devices


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