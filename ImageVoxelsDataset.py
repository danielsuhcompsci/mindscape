import torch
import numpy as np
import pandas as pd
import h5py
import os.path
import nilearn.image
from torch.utils.data import Dataset, DataLoader

class ImageVoxelsDataset(Dataset):
    def __init__(self, nsd_dir, subject, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.responses_frame = pd.read_csv(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/behav/responses.tsv'.format(subject)), 
            delimiter='\t', usecols=[1, 5])
        self.betas_dir = os.path.join(nsd_dir, 
            'nsddata_betas/ppdata/subj{:02n}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(subject))
        self.images = h5py.File(os.path.join(
            nsd_dir, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'), 'r')
        self.atlas = nilearn.image.get_data(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/func1pt8mm/roi/streams.nii.gz'.format(subject)))

    def __len__(self):
        return len(self.responses_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        session = self.responses_frame.iloc[idx, 0]
        voxels = nilearn.image.get_data(nilearn.image.index_img(os.path.join(
            self.betas_dir, 'betas_session{:02n}.nii.gz').format(session), (idx - 750 * (session - 1))))
        voxels = np.concatenate((voxels[self.atlas == 1], voxels[self.atlas == 2], voxels[self.atlas == 5]))
        image = self.images.get('imgBrick')[self.responses_frame.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            betas_roi = self.target_transform(voxels)

        return image, voxels

