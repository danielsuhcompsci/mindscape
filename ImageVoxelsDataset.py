import torch
import pandas as pd
import h5py
import os.path
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset

class ImageVoxelsDataset(Dataset):
    def __init__(self, nsd_dir, subject, transform=None, target_transform=None, preload_imgs = False):
        self.transform = transform
        self.target_transform = target_transform
        self.responses_frame = pd.read_csv(os.path.join(
            nsd_dir, f'nsddata/ppdata/subj{subject:02n}/behav/responses.tsv'), delimiter='\t', usecols=[1, 4])
        self.masked_betas = np.load(os.path.join(nsd_dir, 
            f'nsddata_betas/masked/subj{subject:02n}.npy'))
        self.images = h5py.File(os.path.join(
            nsd_dir, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'), 'r')

        self.preload_imgs = preload_imgs

        if self.preload_imgs:
            self.loaded_imgs = []
            for brick in tqdm(self.images.get('imgBrick'), desc="Loading stimuli images..."):
                self.loaded_imgs.append(brick)
            self.image = None
        else:
            self.loaded_imgs = None

    def __len__(self):
        return len(self.responses_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        session = self.responses_frame.iloc[idx, 0]
        voxels = torch.tensor(self.masked_betas[idx])
        image_id = self.responses_frame.iloc[idx, 1]

        
        img_index = image_id = 1
        if self.preload_imgs:
            image = self.loaded_imgs[img_index]
        else:
            image = self.images.get('imgBrick')[img_index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            voxels = self.target_transform(voxels)

        return image, voxels
