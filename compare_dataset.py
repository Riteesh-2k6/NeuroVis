import os
import pandas as pd

def compare():
    partial_train_dir = r"c:\KJU\partial_train"
    labels_path = r"c:\KJU\train_labels.csv"
    
    # List folders in partial_train
    print(f"Listing folders in {partial_train_dir}...")
    dataset_folders = set(os.listdir(partial_train_dir))
    print(f"Total folders in partial_train: {len(dataset_folders)}")
    
    # Read train_labels.csv
    print(f"Reading {labels_path}...")
    df = pd.read_csv(labels_path)
    df['BraTS21ID'] = df['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    label_ids = set(df['BraTS21ID'].unique())
    print(f"Total IDs in train_labels.csv: {len(label_ids)}")
    
    # Compare
    intersection = dataset_folders.intersection(label_ids)
    only_in_dataset = dataset_folders - label_ids
    only_in_labels = label_ids - dataset_folders
    
    print("-" * 30)
    print(f"Matches (Found in both): {len(intersection)}")
    print(f"Folders in dataset but NOT in labels: {len(only_in_dataset)}")
    print(f"IDs in labels but NOT in dataset: {len(only_in_labels)}")
    print("-" * 30)
    
    if only_in_dataset:
        print("\nFirst 10 folders in dataset NOT in labels:")
        print(sorted(list(only_in_dataset))[:10])
        
    if only_in_labels:
        print("\nFirst 10 IDs in labels NOT in dataset:")
        print(sorted(list(only_in_labels))[:10])

if __name__ == "__main__":
    compare()
