import torch
import pandas as pd
import h5py
import os.path
import nilearn.image
import gc
import nibabel as nib
from torch.utils.data import Dataset

class ImageVoxelsDataset(Dataset):
    def __init__(self, nsd_dir, subject, transform=None, target_transform=None, cache_size=0):
        self.transform = transform
        self.target_transform = target_transform
        self.responses_frame = pd.read_csv(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/behav/responses.tsv'.format(subject)), 
            delimiter='\t', usecols=[1, 4])
        self.betas_dir = os.path.join(nsd_dir, 
            'nsddata_betas/ppdata/subj{:02n}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(subject))
        self.images = h5py.File(os.path.join(
            nsd_dir, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'), 'r')
        self.prf_atlas = nilearn.image.get_data(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/func1pt8mm/roi/prf-visualrois.nii.gz'.format(subject)))
        self.floc_atlas = nilearn.image.get_data(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/func1pt8mm/roi/floc-faces.nii.gz'.format(subject)))


        #Basic LRU for session imgs
        self.cache = {}   
        self.cache_order = [] 
        self.cache_size = cache_size  


    def _load_voxel_data(self, session, idx):
        cache_key = session
        
        if cache_key in self.cache:
            # Retrieve from cache if present
            self.cache_order.remove(cache_key) #update cache order
            self.cache_order.append(cache_key)
            session_image = self.cache[cache_key] #retrieve img from cache
        else:
            voxels_path = os.path.join(self.betas_dir, f'betas_session{session:02d}.nii.gz')
             
            session_image = nib.load(voxels_path)  # load img from disk
            self.cache[cache_key] = session_image #update cache
            self.cache_order.append(cache_key)

            #enforce cache size
            if len(self.cache_order) > self.cache_size:
                lru_key = self.cache_order.pop(0)
                del self.cache[lru_key]
                gc.collect()
        

        specific_idx = idx - 750 * (session - 1)
        voxels = nilearn.image.get_data(nilearn.image.index_img(session_image, specific_idx))
        voxels = torch.tensor(voxels[((self.prf_atlas > 0) & (self.prf_atlas != 7)) | (self.floc_atlas > 0)])
        return voxels


    def __len__(self):
        return len(self.responses_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        session = self.responses_frame.iloc[idx, 0]
        voxels = self._load_voxel_data(session, idx)
        image_id = self.responses_frame.iloc[idx, 1]
        image = self.images.get('imgBrick')[image_id]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            voxels = self.target_transform(voxels)

        return image, voxels
