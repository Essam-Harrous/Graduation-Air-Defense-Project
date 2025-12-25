#!/usr/bin/env python3
"""
Update dataset with CVAT annotations.
Replaces auto-generated labels with manually annotated ones.
"""

import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
CVAT_PEPSI = BASE_DIR / "Pepsi" / "obj_train_data"
DATASET_TRAIN_LABELS = BASE_DIR / "dataset" / "train" / "labels"
DATASET_TRAIN_IMAGES = BASE_DIR / "dataset" / "train" / "images"

def update_pepsi_labels():
    """Copy CVAT Pepsi annotations to dataset."""
    updated = 0
    
    for label_file in CVAT_PEPSI.glob("*.txt"):
        # CVAT label: "new_pepsi (23).jpg.txt" -> need to match "new_pepsi (23).jpg"
        # Our dataset label: "new_pepsi (23).txt"
        
        # Get the base name without .txt
        cvat_name = label_file.stem  # "new_pepsi (23).jpg"
        
        # Remove the extra .jpg if present
        if cvat_name.endswith('.jpg'):
            base_name = cvat_name[:-4]  # "new_pepsi (23)"
        else:
            base_name = cvat_name
        
        # Target label file in dataset
        target_label = DATASET_TRAIN_LABELS / f"{base_name}.txt"
        
        # Check if corresponding image exists
        target_image = DATASET_TRAIN_IMAGES / f"{base_name}.jpg"
        
        if target_image.exists():
            # Copy the CVAT label to dataset
            shutil.copy2(label_file, target_label)
            updated += 1
            print(f"✅ Updated: {base_name}.txt")
        else:
            print(f"⚠️ Image not found: {target_image.name}")
    
    return updated

def main():
    print("🔄 Updating dataset with CVAT annotations\n")
    
    # Update Pepsi labels
    print("📁 Processing Pepsi annotations...")
    pepsi_count = update_pepsi_labels()
    
    print(f"\n✅ Updated {pepsi_count} Pepsi labels with manual annotations")
    print("\n💡 To add Cola annotations:")
    print("   1. Annotate Cola images in CVAT")
    print("   2. Export as YOLO 1.1")
    print("   3. Place in 'Cola' folder")
    print("   4. Update this script to process Cola folder")
    print("\n🎯 Ready to retrain with improved labels!")

if __name__ == "__main__":
    main()
