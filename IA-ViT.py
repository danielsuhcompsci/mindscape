import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class IAViT(nn.Module):
    def __init__(self, image_size, patch_size, num_rois, num_classes, num_layers, d_model, num_heads, mlp_dim, channels=3):
        super(IAViT, self).__init__()
        self.patch_dim = channels * patch_size ** 2
        self.num_patches = (image_size // patch_size) ** 2
        self.roi_embeddings = nn.Embedding(num_rois, d_model)
        self.patch_embeddings = nn.Embedding(self.num_patches, d_model)
        self.position_embeddings = nn.Embedding(self.num_patches, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, num_heads, mlp_dim) for _ in range(num_layers)])
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        
        # MLP for each ROI
        self.mlp_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_rois)])
        
        # Classifier head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, images):
        batch_size = images.size(0)
        
        # Patch embeddings
        patches = F.unfold(images, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        patches = patches.permute(0, 2, 1).reshape(batch_size, -1, self.patch_dim)
        patch_embeddings = self.patch_embeddings(patches)
        
        # Positional embeddings
        positions = torch.arange(self.num_patches).unsqueeze(0).repeat(batch_size, 1).to(images.device)
        position_embeddings = self.position_embeddings(positions)
        
        # CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        
        # Add positional embeddings to patch embeddings
        embeddings = patch_embeddings + position_embeddings
        
        # Transformer Encoder
        transformer_output = self.transformer_encoder(embeddings)
        
        # Attention computation for each ROI
        roi_activations = []
        for i in range(self.num_rois):
            # Apply attention computation for each ROI
            roi_embedding = self.roi_embeddings(torch.tensor(i).to(images.device))
            attention_output = self.mlp_layers[i](transformer_output + roi_embedding.unsqueeze(1))
            roi_activations.append(attention_output)
        
        # Concatenate and apply MLP for each ROI
        voxel_activations = [self.fc(roi_activation) for roi_activation in roi_activations]
        
        return voxel_activations

class FMRIImageDataset(Dataset):
    def __init__(self, images, voxel_activations):
        self.images = images
        self.voxel_activations = voxel_activations
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        voxel_activation = self.voxel_activations[idx]
        return image, voxel_activation

# Define your dataset and create DataLoader
some_dataset = FMRIImageDataset(images, voxel_activations)
dataloader = DataLoader(some_dataset, batch_size=4, shuffle=True, num_workers=0)
