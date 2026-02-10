import argparse
import os

import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm

from dataset import BrainRSNADataset
import config
import monai.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="FLAIR", type=str)
parser.add_argument("--model_name", default="b0", type=str)
args = parser.parse_args()

data = pd.read_csv("../input/sample_submission.csv")


# model
if config.MODEL_NAME == "resnet18":
    model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
elif config.MODEL_NAME == "resnet10":
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
else:
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
device = torch.device("cpu")
model.to(device)
all_weights = os.listdir("../weights/")
fold_files = sorted([f for f in all_weights if (args.type in f) and (config.MODEL_NAME in f)])
criterion = nn.BCEWithLogitsLoss()


test_dataset = BrainRSNADataset(data=data, mri_type=args.type, is_train=False)
test_dl = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=0
)

preds_f = np.zeros(len(data))
for fold in range(5):
    image_ids = []
    model.load_state_dict(torch.load(f"../weights/{fold_files[fold]}", map_location=device))
    preds = []
    epoch_iterator_test = tqdm(test_dl)
    # TTA Transforms
    tta_transforms = [
        None, # Original
        T.RandFlip(prob=1.0, spatial_axis=0),
        T.RandFlip(prob=1.0, spatial_axis=1),
        T.RandFlip(prob=1.0, spatial_axis=2),
    ]

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_test):
            model.eval()
            images = batch["image"].to(device)
            
            # TTA logic
            batch_preds = []
            for tta_idx in range(min(config.TTA_STEPS, len(tta_transforms))):
                aug = tta_transforms[tta_idx]
                if aug is not None:
                    aug_images = aug(images)
                else:
                    aug_images = images
                
                outputs = model(aug_images)
                batch_preds.append(outputs.sigmoid().detach().cpu().numpy())
            
            # Average across TTA steps for this batch
            avg_batch_pred = np.mean(batch_preds, axis=0)
            preds.append(avg_batch_pred)
            image_ids.append(batch["case_id"].detach().cpu().numpy())

    preds_f += np.vstack(preds).T[0] / 5

    ids_f = np.hstack(image_ids)

data["BraTS21ID"] = ids_f
data["MGMT_value"] = preds_f

data = data.sort_values(by="BraTS21ID").reset_index(drop=True)
data.to_csv("c:/KJU/prediction_output.csv", index=False)
