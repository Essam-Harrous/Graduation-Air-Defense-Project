#!/usr/bin/env python3
"""
Prepare YOLOv8 dataset for Pepsi/Cola logo detection.
This script:
1. Creates the proper folder structure
2. Copies images to train/valid folders
3. Generates YOLO format labels (assuming logo fills ~90% of image center)
4. Creates data.yaml config file
"""

import os
import shutil
from pathlib import Path
from PIL import Image

# Configuration
BASE_DIR = Path(__file__).parent
TRAIN_IMAGES_SRC = BASE_DIR / "train_images"
TEST_IMAGES_SRC = BASE_DIR / "test_images"
DATASET_DIR = BASE_DIR / "dataset"

# Class mapping: 0 = pepsi, 1 = cola
CLASSES = ["pepsi", "cola"]

def get_class_from_filename(filename):
    """Determine class from filename."""
    fname_lower = filename.lower()
    if "pepsi" in fname_lower:
        return 0  # pepsi
    elif "cola" in fname_lower:
        return 1  # cola
    return None

def create_label(image_path, class_id):
    """
    Create YOLO format label assuming logo fills most of the image.
    YOLO format: class_id x_center y_center width height (normalized 0-1)
    """
    # Assuming the logo is centered and fills ~90% of the image
    x_center = 0.5
    y_center = 0.5
    width = 0.9
    height = 0.9
    
    return f"{class_id} {x_center} {y_center} {width} {height}"

def setup_directories():
    """Create dataset directory structure."""
    dirs = [
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels",
        DATASET_DIR / "valid" / "images",
        DATASET_DIR / "valid" / "labels",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Created dataset directories")

def process_images(src_dir, dest_type):
    """Process images from source directory."""
    dest_images = DATASET_DIR / dest_type / "images"
    dest_labels = DATASET_DIR / dest_type / "labels"
    
    count = {"pepsi": 0, "cola": 0}
    
    for img_file in src_dir.iterdir():
        if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            continue
            
        class_id = get_class_from_filename(img_file.name)
        if class_id is None:
            print(f"⚠️ Skipping {img_file.name} - unknown class")
            continue
        
        # Clean up filename (remove double .jpg.jpg if present)
        clean_name = img_file.name
        if clean_name.endswith('.jpg.jpg'):
            clean_name = clean_name[:-4]  # Remove extra .jpg
        
        # Copy image
        dest_img = dest_images / clean_name
        shutil.copy2(img_file, dest_img)
        
        # Create label file
        label_name = Path(clean_name).stem + ".txt"
        label_content = create_label(img_file, class_id)
        (dest_labels / label_name).write_text(label_content)
        
        count[CLASSES[class_id]] += 1
    
    return count

def create_data_yaml():
    """Create data.yaml configuration file."""
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Pepsi and Cola Logo Detection

path: {DATASET_DIR.absolute()}
train: train/images
val: valid/images

# Classes
names:
  0: pepsi
  1: cola

# Number of classes
nc: 2
"""
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"✅ Created {yaml_path}")

def main():
    print("🚀 Preparing YOLOv8 dataset for Pepsi/Cola detection\n")
    
    # Setup directories
    setup_directories()
    
    # Process training images
    print("\n📁 Processing training images...")
    train_count = process_images(TRAIN_IMAGES_SRC, "train")
    print(f"   Pepsi: {train_count['pepsi']}, Cola: {train_count['cola']}")
    
    # Process test images -> validation set
    print("\n📁 Processing validation images...")
    valid_count = process_images(TEST_IMAGES_SRC, "valid")
    print(f"   Pepsi: {valid_count['pepsi']}, Cola: {valid_count['cola']}")
    
    # Create data.yaml
    print("\n📝 Creating data.yaml...")
    create_data_yaml()
    
    # Summary
    print("\n" + "="*50)
    print("✅ Dataset preparation complete!")
    print(f"   Training: {train_count['pepsi'] + train_count['cola']} images")
    print(f"   Validation: {valid_count['pepsi'] + valid_count['cola']} images")
    print(f"\n📂 Dataset location: {DATASET_DIR}")
    print("\n⚠️  NOTE: Labels assume logo fills 90% of image center.")
    print("   For better accuracy, use a labeling tool like LabelImg or Roboflow.")
    print("\n🎯 Next step: Run the training notebook or use this command:")
    print("   yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640")

if __name__ == "__main__":
    main()
