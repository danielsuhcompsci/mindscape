import torch
import pandas as pd
import h5py
import os.path
import nilearn.image
from torch.utils.data import Dataset, DataLoader

class ImageVoxelsDataset(Dataset):
    """Image Voxels dataset."""
    def __init__(self, subject):

        self.responses_frame = pd.read_csv('/teamspace/studios/this_studio/nsd/nsddata/ppdata/subj{:02n}/behav/responses.tsv'.format(subject), delimiter='\t')
        self.betas_dir = '/teamspace/studios/this_studio/nsd/nsddata_betas/ppdata/subj{:02n}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(subject)
        self.images = h5py.File('nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        betas = os.path.join(self.betas_dir, 'betas_session{:02n}.nii.gz'.format(self.responses_frame.iloc[idx, 1]))            
        betas = nilearn.image.index_img(betas, (idx - (self.responses_frame.iloc[idx, 1] - 1) * 750))
        image = self.images.get('imgBrick')[self.responses_frame.iloc[idx, 5]]
        return {'betas': betas, 'image': image}

