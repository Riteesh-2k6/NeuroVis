import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import monai
from monai.visualize import GradCAM
from dataset import BrainRSNADataset
import pandas as pd
import config
import cv2

def get_prediction(model, image, device):
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
    return prob

def visualize_gradcam(case_id, mri_type="FLAIR", fold=3):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model (ResNet-10)
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
    
    # Weight loading logic
    all_weights = os.listdir("../weights/")
    # Try to find weights for the specific mri_type first
    fold_files = [f for f in all_weights if f"resnet10_{mri_type}_fold{fold}" in f]
    
    if not fold_files:
        print(f"No specific weights found for {mri_type}. Checking for any available ResNet-10 weights...")
        fold_files = [f for f in all_weights if f"resnet10" in f and f"fold{fold}" in f]
    
    if not fold_files:
        print(f"No weights found for fold {fold}. Checking for any available ResNet-10 weights...")
        fold_files = [f for f in all_weights if "resnet10" in f]

    if not fold_files:
        raise FileNotFoundError("Could not find any ResNet-10 weights in ../weights/")

    weights_path = os.path.join("../weights/", fold_files[0])
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare data - Force 1 slice replicated to 3D
    df = pd.DataFrame({"BraTS21ID": [case_id], "MGMT_value": [0]})
    # We still use the dataset class but we will manually override the volume
    ds = BrainRSNADataset(data=df, mri_type=mri_type, is_train=True, ds_type=f"vis_{mri_type}", do_load=False)
    
    # We need to find the largest slice first
    import glob, re, utils
    case_id_str = str(case_id).zfill(5)
    path = f"../input/train/{case_id_str}/{mri_type}/*.dcm"
    files = sorted(glob.glob(path), key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    
    if not files:
        raise ValueError(f"No {mri_type} scans found for case {case_id}")
        
    resolutions = [utils.extract_cropped_image_size(f) for f in files]
    largest_idx = np.array(resolutions).argmax()
    largest_file = files[largest_idx]
    largest_img = utils.load_dicom_image(largest_file, voi_lut=True)
    
    # Replicate to 3D
    target_depth = config.NUM_IMAGES_3D
    img3d = np.repeat(largest_img[:, :, np.newaxis], target_depth, axis=2)
    image = torch.tensor(img3d).float().unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 256, 256, 64]
    
    # Get MGMT Prediction (Single Slice Replicated)
    mgmt_prob = get_prediction(model, image, device)
    
    # Initialize Grad-CAM
    cam = GradCAM(nn_module=model, target_layers="layer4")
    
    # Generate Heatmap
    result = cam(x=image)
    heatmap = result[0, 0].cpu().numpy() # [256, 256, 64]
    
    # Heatmap for the largest slice
    heatmap_slice = heatmap[:, :, target_depth // 2]
    
    # Apply background mask to suppress non-brain activations
    # Create mask where image signal is significant
    mask = (largest_img > (largest_img.min() + 0.05 * (largest_img.max() - largest_img.min())))
    heatmap_slice = heatmap_slice * mask
    
    # Normalize heatmap
    heatmap_slice = (heatmap_slice - heatmap_slice.min()) / (heatmap_slice.max() - heatmap_slice.min() + 1e-8)
    
    # Create overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_slice), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    img_colored = np.stack([largest_img]*3, axis=-1)
    img_colored = (img_colored - img_colored.min()) / (img_colored.max() - img_colored.min() + 1e-8)
    
    overlay = cv2.addWeighted(np.uint8(255 * img_colored), 0.6, heatmap_color, 0.4, 0)
    
    # Save results
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].set_title(f"Original Largest Slice ({mri_type})")
    axes[0].imshow(largest_img, cmap="gray")
    axes[0].axis("off")
    
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].imshow(heatmap_slice, cmap="jet")
    axes[1].axis("off")
    
    axes[2].set_title("Overlay (Single Slice Inference)")
    axes[2].imshow(overlay)
    axes[2].axis("off")
    
    save_path = f"{output_dir}/case_{case_id}_{mri_type}_heatmap.png"
    plt.savefig(save_path)
    plt.close()
    
    return save_path, mgmt_prob, largest_img, heatmap_slice

if __name__ == "__main__":
    # Example usage: Find an existing case ID from the directory
    train_dir = "../input/train"
    cases = os.listdir(train_dir)
    if cases:
        sample_case = int(cases[0])
        print(f"Generating visualization for sample case: {sample_case}")
        visualize_gradcam(sample_case)
    else:
        print("No cases found in input directory.")
