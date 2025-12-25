#!/usr/bin/env python3
"""
Create a dataset with ONLY the 25 annotated Pepsi images.
Split: 20 train, 5 validation
"""

import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
CVAT_PEPSI = BASE_DIR / "Pepsi" / "obj_train_data"
TRAIN_IMAGES_SRC = BASE_DIR / "train_images"

# New dataset for annotated images only
DATASET_DIR = BASE_DIR / "dataset_annotated"

def setup_directories():
    """Create fresh dataset directory structure."""
    # Remove old if exists
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    
    dirs = [
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels",
        DATASET_DIR / "valid" / "images",
        DATASET_DIR / "valid" / "labels",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Created dataset_annotated directories")

def process_annotated_images():
    """Copy annotated images and labels, split into train/valid."""
    
    # Get all annotated label files
    label_files = sorted(CVAT_PEPSI.glob("*.txt"))
    
    # Split: first 20 for train, last 5 for valid
    train_labels = label_files[:20]
    valid_labels = label_files[20:]
    
    def copy_pair(label_file, dest_type):
        # CVAT label: "new_pepsi (23).jpg.txt"
        cvat_name = label_file.stem  # "new_pepsi (23).jpg"
        
        # Source image has double .jpg.jpg extension
        src_image = TRAIN_IMAGES_SRC / f"{cvat_name}.jpg"
        
        if not src_image.exists():
            print(f"⚠️ Image not found: {src_image}")
            return False
        
        # Clean name for destination (remove extra .jpg)
        if cvat_name.endswith('.jpg'):
            clean_name = cvat_name[:-4]  # "new_pepsi (23)"
        else:
            clean_name = cvat_name
        
        # Copy image
        dest_img = DATASET_DIR / dest_type / "images" / f"{clean_name}.jpg"
        shutil.copy2(src_image, dest_img)
        
        # Copy label
        dest_label = DATASET_DIR / dest_type / "labels" / f"{clean_name}.txt"
        shutil.copy2(label_file, dest_label)
        
        return True
    
    # Process train set
    train_count = 0
    for lf in train_labels:
        if copy_pair(lf, "train"):
            train_count += 1
    
    # Process valid set
    valid_count = 0
    for lf in valid_labels:
        if copy_pair(lf, "valid"):
            valid_count += 1
    
    return train_count, valid_count

def create_data_yaml():
    """Create data.yaml for single class (Pepsi only)."""
    yaml_content = f"""# YOLOv8 Dataset - Annotated Pepsi Images Only
path: {DATASET_DIR.absolute()}
train: train/images
val: valid/images

names:
  0: pepsi

nc: 1
"""
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"✅ Created {yaml_path}")

def main():
    print("🚀 Setting up dataset with annotated images only\n")
    
    setup_directories()
    
    print("\n📁 Processing annotated images...")
    train_count, valid_count = process_annotated_images()
    
    print(f"\n   Training: {train_count} images")
    print(f"   Validation: {valid_count} images")
    
    create_data_yaml()
    
    print("\n" + "="*50)
    print("✅ Dataset ready!")
    print(f"📂 Location: {DATASET_DIR}")
    print("\n🎯 Train with:")
    print("   yolo detect train data=dataset_annotated/data.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps")

if __name__ == "__main__":
    main()
