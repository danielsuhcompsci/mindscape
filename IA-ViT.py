import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import h5py
import os.path
import nilearn.image
import gc
import nibabel as nib
from transformers import ViTConfig, ViTModel
import numpy as np

class MatrixViTModel(ViTModel):
    def __init__(self, output_dimensions, num_rois, config=None):
        if config is None:
            config = ViTConfig()
        super().__init__(config)

        self.num_rois = num_rois
        
        # Initialize learnable parameters for ROI tokens
        self.roi_tokens = nn.Parameter(torch.randn(1, num_rois, config.hidden_size))

        # Initialize linear layers for each ROI token
        self.roi_linear_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, output_dim) for output_dim in output_dimensions
        ])

        # Attention mechanism parameters
        self.num_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // self.num_heads

        # Query, key, and value linear transformations for attention mechanism
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        # Output dimension for each ROI token
        self.output_dimensions = output_dimensions

    def forward(self, x, **kwargs):
        outputs = super().forward(x, **kwargs)  # Obtain outputs from the transformer encoder

        # Extract class token from the output of the transformer encoder
        class_token = outputs.last_hidden_state[:, 0, :]  # class token is first token

        # Compute attention scores
        query = self.query(self.roi_tokens).repeat(outputs.last_hidden_state.size(0), 1, 1)
        key = self.key(outputs.last_hidden_state)
        value = self.value(outputs.last_hidden_state)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.attention_head_size)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to obtain weighted sum of output embeddings
        weighted_sum = torch.matmul(attention_weights, value)

        # Compute voxel activations for each ROI token
        roi_outputs = []
        for i in range(self.num_rois):
            # Apply linear transformation for final output
            roi_output = self.roi_linear_layers[i](weighted_sum[:, i, :])
            roi_outputs.append(roi_output)

        # Concatenate outputs of all ROI linear layers
        flattened_outputs = torch.cat(roi_outputs, dim=1)

        return flattened_outputs

# Define a custom dataset to load image data and corresponding voxel activations
class ImageVoxelsDataset(Dataset):
    def __init__(self, nsd_dir, subject, transform=None, target_transform=None, cache_size=0):
        self.transform = transform
        self.target_transform = target_transform

        # Load responses frame containing image IDs and session information
        self.responses_frame = pd.read_csv(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/behav/responses.tsv'.format(subject)),
            delimiter='\t', usecols=[1, 5])
        
        # Directory containing beta images
        self.betas_dir = os.path.join(nsd_dir,
            'nsddata_betas/ppdata/subj{:02n}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(subject))

        # Load images file
        self.images = h5py.File(os.path.join(
            nsd_dir, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'), 'r')

        # Load the atlas files for PRF and FLOC
        self.prf_atlas = nilearn.image.get_data(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/func1pt8mm/roi/prf-visualrois.nii.gz'.format(subject)))
        self.floc_atlas = nilearn.image.get_data(os.path.join(
            nsd_dir, 'nsddata/ppdata/subj{:02n}/func1pt8mm/roi/floc-faces.nii.gz'.format(subject)))

        # Basic LRU for session images
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

    # Load voxel data for a session and specific index
    def _load_voxel_data(self, session, idx):
        cache_key = session
        
        if cache_key in self.cache:
            # Retrieve from cache if present
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            session_image = self.cache[cache_key]
        else:
            voxels_path = os.path.join(self.betas_dir, f'betas_session{session:02d}.nii.gz')
            session_image = nib.load(voxels_path)
            self.cache[cache_key] = session_image
            self.cache_order.append(cache_key)

            # Enforce cache size
            if len(self.cache_order) > self.cache_size:
                lru_key = self.cache_order.pop(0)
                del self.cache[lru_key]
                gc.collect()

        specific_idx = idx - 750 * (session - 1)
        voxels = nilearn.image.get_data(nilearn.image.index_img(session_image, specific_idx))
        voxels = torch.tensor(voxels[((self.prf_atlas > 0) & (self.prf_atlas != 7)) | (self.floc_atlas > 0)])
        return voxels

    # Get length of the dataset
    def __len__(self):
        return len(self.responses_frame)

    # Get an item from the dataset
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get session and voxel data
        session = self.responses_frame.iloc[idx, 0]
        voxels = self._load_voxel_data(session, idx)
        
        # Get image ID and load image
        image_id = self.responses_frame.iloc[idx, 1]
        image = self.images.get('imgBrick')[image_id]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            voxels = self.target_transform(voxels)

        return image, voxels

# Define the dataset and create DataLoader
output_dimensions = [128, 256, 512]  # Output dimensions for each ROI linear layer
num_rois = 10  # Number of ROIs
some_dataset = ImageVoxelsDataset(nsd_dir, subject, transform=None, target_transform=None, cache_size=0)
dataloader = DataLoader(some_dataset, batch_size=4, shuffle=True, num_workers=0)
