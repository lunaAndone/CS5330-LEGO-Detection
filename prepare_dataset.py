import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml

# Set paths - update these to match your setup
DATA_PATH = "./data"  # Path to your downloaded dataset
OUTPUT_PATH = "./processed_dataset"  # Where to save the processed dataset
MAX_IMAGES = 10000  # Reduce dataset to 10,000 images

def verify_annotation(image_file, xml_file):
    """Verify if annotation is valid."""
    try:
        # Check if image exists and can be opened
        img = cv2.imread(image_file)
        if img is None:
            return False
        
        height, width, _ = img.shape
        
        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Check if there are any objects
        objects = root.findall('object')
        if len(objects) == 0:
            return False
        
        # Check if bounding boxes are within image boundaries
        for obj in objects:
            bbox = obj.find('bndbox')
            if bbox is None:
                return False
                
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Check if coordinates are valid
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmin >= xmax or ymin >= ymax:
                return False
                
        return True
    except Exception as e:
        print(f"Error verifying {image_file}: {e}")
        return False

def convert_to_yolo_format(image_path, xml_path, output_path):
    """Convert PASCAL VOC annotation to YOLO format."""
    # Read image dimensions
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create YOLO txt file
    txt_path = os.path.join(output_path, os.path.basename(image_path).rsplit('.', 1)[0] + '.txt')
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            # Convert all object types to single class 'lego' (class_id = 0)
            class_id = 0
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            
            # Write to file
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def main():
    # Create output directory structure
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Paths to images and annotations
    images_path = os.path.join(DATA_PATH, "images")
    annotations_path = os.path.join(DATA_PATH, "annotations")
    
    # Get list of image files
    all_image_files = [f for f in os.listdir(images_path) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Limit number of images if specified
    if MAX_IMAGES and MAX_IMAGES < len(all_image_files):
        all_image_files = all_image_files[:MAX_IMAGES]
    
    # 1. Filter mislabeled images
    print("Step 1: Verifying annotations...")
    valid_images = []
    
    for img_file in tqdm(all_image_files):
        img_path = os.path.join(images_path, img_file)
        xml_file = os.path.join(annotations_path, img_file.rsplit('.', 1)[0] + '.xml')
        
        # Check if annotation exists and is valid
        if os.path.exists(xml_file) and verify_annotation(img_path, xml_file):
            valid_images.append(img_file)
    
    print(f"Found {len(valid_images)} valid images out of {len(all_image_files)}")
    
    # 2. Split dataset into train/val/test (70/15/15)
    print("Step 2: Splitting dataset...")
    train_files, temp_files = train_test_split(valid_images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    print(f"Split: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} testing")
    
    # 3. Create directories for each split
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(OUTPUT_PATH, split, subdir), exist_ok=True)
    
    # 4. Process each split
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"Processing {split_name} set...")
        
        for img_file in tqdm(files):
            # Source paths
            src_img = os.path.join(images_path, img_file)
            src_xml = os.path.join(annotations_path, img_file.rsplit('.', 1)[0] + '.xml')
            
            # Destination paths
            dst_img_dir = os.path.join(OUTPUT_PATH, split_name, 'images')
            dst_label_dir = os.path.join(OUTPUT_PATH, split_name, 'labels')
            
            # Copy image
            shutil.copy(src_img, os.path.join(dst_img_dir, img_file))
            
            # Convert annotation to YOLO format
            convert_to_yolo_format(src_img, src_xml, dst_label_dir)
    
    # 5. Create YAML config file for YOLOv8
    yaml_content = {
        'path': os.path.abspath(OUTPUT_PATH),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': os.path.join('test', 'images'),
        'nc': 1,  # Number of classes
        'names': ['lego']  # Class names
    }
    
    with open(os.path.join(OUTPUT_PATH, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)
    
    print("Dataset preparation complete!")
    print(f"YAML config saved to {os.path.join(OUTPUT_PATH, 'data.yaml')}")

if __name__ == "__main__":
    main()