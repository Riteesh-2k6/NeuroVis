import torch
import monai
import pandas as pd
import config
from dataset import BrainRSNADataset

def verify():
    print("Verifying model initialization...")
    if config.MODEL_NAME == "resnet18":
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
    else:
        model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
    
    print(f"Model {config.MODEL_NAME} initialized successfully.")
    
    print("\nVerifying dataset and augmentation...")
    sample_df = pd.DataFrame({'BraTS21ID': ['00000'], 'MGMT_value': [1]})
    ds = BrainRSNADataset(data=sample_df, mri_type="T1wCE", is_train=True)
    
    item = ds[0]
    img = item['image']
    print(f"Image shape: {img.shape}")
    print(f"Image mean: {img.mean():.4f}, std: {img.std():.4f}")
    
    if img.shape == (1, 256, 256, 64):
        print("Data shape is correct.")
    else:
        print(f"Warning: Unexpected shape {img.shape}")

if __name__ == "__main__":
    verify()
