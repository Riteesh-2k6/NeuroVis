import os
import zipfile
import tempfile
import shutil
import numpy as np
import torch
import monai
from monai.visualize import GradCAM
import pandas as pd
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import config
import utils
import config
import time

class InferenceEngine:
    def __init__(self, model_name="resnet10", fold=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
        self.model_name = model_name
        self.fold = fold
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()
        self.cam_extractor = GradCAM(nn_module=self.model, target_layers="layer4")

    def _load_weights(self):
        all_weights = os.listdir("../weights/")
        fold_files = [f for f in all_weights if self.model_name in f and f"fold{self.fold}" in f]
        if not fold_files:
            fold_files = [f for f in all_weights if self.model_name in f]
        
        if not fold_files:
            raise FileNotFoundError(f"No weights found for {self.model_name}")
            
        weights_path = os.path.join("../weights/", fold_files[0])
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def process_zip(self, zip_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find all DICOM files recursively
            dicom_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dicom_files.append(os.path.join(root, file))
            
            if not dicom_files:
                raise ValueError("No DICOM files found in the zip archives.")

            # Identify available MRI types
            start_time = time.time()
            mri_types = {}
            for f in dicom_files:
                fname = os.path.basename(f)
                # Heuristic to detect type from filename or path
                lower_f = f.lower()
                if "t1wce" in lower_f:
                    mri_types.setdefault("T1wCE", []).append(f)
                elif "flair" in lower_f:
                    mri_types.setdefault("FLAIR", []).append(f)
                elif "t2w" in lower_f:
                    mri_types.setdefault("T2w", []).append(f)
                elif "t1w" in lower_f: # Check T1w last to avoid confusion with T1wCE
                    mri_types.setdefault("T1w", []).append(f)
            
            # Selection Priority: T1wCE > FLAIR > T2w > T1w
            selected_type = None
            target_files = []
            
            if "T1wCE" in mri_types and mri_types["T1wCE"]:
                selected_type = "T1wCE"
                target_files = mri_types["T1wCE"]
            elif "FLAIR" in mri_types and mri_types["FLAIR"]:
                selected_type = "FLAIR"
                target_files = mri_types["FLAIR"]
            elif "T2w" in mri_types and mri_types["T2w"]:
                selected_type = "T2w"
                target_files = mri_types["T2w"]
            elif "T1w" in mri_types and mri_types["T1w"]:
                selected_type = "T1w"
                target_files = mri_types["T1w"]
            else:
                # Fallback if no type detected
                selected_type = "Unknown"
                target_files = dicom_files
                
            print(f"Selected MRI Type: {selected_type} with {len(target_files)} slices.")

            # 1. Identify central largest from target files
            valid_files = []
            resolutions = []
            
            for f in target_files:
                try:
                    res = utils.extract_cropped_image_size(f)
                    if res > 0: # Not blank
                        valid_files.append(f)
                        resolutions.append(res)
                except:
                    continue
            
            if not valid_files:
                raise ValueError("No valid non-blank scans found.")
                
            # 2. Identify the largest image
            resolutions = np.array(resolutions)
            largest_idx = resolutions.argmax()
            largest_file = valid_files[largest_idx]
            print(f"Largest Slice identified: {os.path.basename(largest_file)}")

            # 3. Load only the largest image and replicate for 3D
            largest_img = utils.load_dicom_image(largest_file, voi_lut=True) # [256, 256]
            
            # Replicate slice to match expected 3D depth (e.g., config.NUM_IMAGES_3D)
            target_depth = config.NUM_IMAGES_3D
            img3d = np.repeat(largest_img[:, :, np.newaxis], target_depth, axis=2) # [256, 256, 64]
                
            volume_tensor = torch.tensor(img3d).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 4. Classification and Heatmap for this single-slice volume
            with torch.set_grad_enabled(True):
                output = self.model(volume_tensor)
                prob = torch.sigmoid(output).item()
                
                # Grad-CAM result
                heatmap = self.cam_extractor(x=volume_tensor)
                heatmap = heatmap[0, 0].cpu().detach().numpy() # [256, 256, 64]
            
            # 5. Extract the heatmap for the specific largest slice (now replicated)
            # Since all slices are the same, we can just take the middle one or average
            final_heatmap = heatmap[:, :, target_depth // 2]
            
            # Apply background mask to suppress activations outside the brain
            mask = (largest_img > (largest_img.min() + 0.05 * (largest_img.max() - largest_img.min())))
            final_heatmap = final_heatmap * mask
            
            return prob, final_heatmap, largest_img, selected_type

def generate_aggregate_viz(avg_heatmap, background_slice, output_path):
    # Normalize
    heatmap_norm = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min() + 1e-8)
    
    # Colorize
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Background
    bg_norm = (background_slice - background_slice.min()) / (background_slice.max() - background_slice.min() + 1e-8)
    bg_color = np.stack([bg_norm]*3, axis=-1)
    
    # Overlay
    overlay = cv2.addWeighted(np.uint8(255 * bg_color), 0.6, heatmap_color, 0.4, 0)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title("Aggregated Attention Map (Commonly Identified Areas)")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()
    return output_path
