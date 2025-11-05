"""
Helper Script to Create a Dummy Dataset for Testing

This script creates a dummy dataset with synthetic images and annotations
so you can test the training pipeline without real SAR data.

Usage:
    python create_dummy_dataset.py --output_dir ./dummy_dataset --num_images 50
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import random


def create_dummy_image(output_path, width=800, height=800):
    """Create a dummy SAR-like image (grayscale noise pattern)"""
    # Create a grayscale image with some structure (simulating SAR texture)
    img_array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Add some "structure" to make it more realistic
    # Add some bright spots (simulating ships)
    num_spots = random.randint(3, 8)
    for _ in range(num_spots):
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        w = random.randint(20, 60)
        h = random.randint(20, 60)
        brightness = random.randint(200, 255)
        img_array[y:y+h, x:x+w] = brightness
    
    # Convert to PIL Image and save as RGB (required by model)
    img = Image.fromarray(img_array, mode='L')
    img_rgb = img.convert('RGB')
    img_rgb.save(output_path)
    return img_rgb.size


def create_dummy_annotations(image_size, num_objects=2):
    """Create dummy bounding box annotations"""
    width, height = image_size
    boxes = []
    
    # Generate random bounding boxes
    for _ in range(num_objects):
        # Random box size
        box_w = random.randint(30, 80)
        box_h = random.randint(30, 80)
        
        # Random position (within image bounds)
        x1 = random.randint(10, width - box_w - 10)
        y1 = random.randint(10, height - box_h - 10)
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        boxes.append([x1, y1, x2, y2])
    
    return boxes


def create_coco_format_dataset(output_dir, num_images=50):
    """Create a COCO format dataset"""
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "ship", "supercategory": "vessel"}
        ]
    }
    
    annotation_id = 1
    
    print(f"Creating {num_images} dummy images...")
    for img_id in range(1, num_images + 1):
        # Create image
        img_filename = f"img_{img_id:04d}.jpg"
        img_path = images_dir / img_filename
        img_size = create_dummy_image(img_path)
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": f"images/{img_filename}",
            "width": img_size[0],
            "height": img_size[1]
        })
        
        # Create annotations
        num_ships = random.randint(1, 4)
        boxes = create_dummy_annotations(img_size, num_ships)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            # COCO format: [x, y, width, height]
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1
        
        if (img_id) % 10 == 0:
            print(f"  Created {img_id}/{num_images} images...")
    
    # Save annotation file
    annotation_file = output_dir / 'annotations.json'
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✅ COCO format dataset created!")
    print(f"   Location: {output_dir}")
    print(f"   Images: {images_dir}")
    print(f"   Annotations: {annotation_file}")
    print(f"\nTo train with this dataset:")
    print(f"  python train_pfarn_sarfish.py --dataset_path {output_dir} --format coco --annotation_file {annotation_file}")


def create_custom_format_dataset(output_dir, num_images=50):
    """Create a custom format dataset"""
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    annotations_dir = output_dir / 'annotations'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} dummy images...")
    for img_id in range(1, num_images + 1):
        # Create image
        img_filename = f"img_{img_id:04d}.jpg"
        img_path = images_dir / img_filename
        img_size = create_dummy_image(img_path)
        
        # Create annotations
        num_ships = random.randint(1, 4)
        boxes = create_dummy_annotations(img_size, num_ships)
        labels = [1] * len(boxes)  # All are ships (class 1)
        
        # Save annotation JSON
        ann_filename = f"img_{img_id:04d}.json"
        ann_path = annotations_dir / ann_filename
        
        ann_data = {
            "boxes": boxes,
            "labels": labels
        }
        
        with open(ann_path, 'w') as f:
            json.dump(ann_data, f, indent=2)
        
        if (img_id) % 10 == 0:
            print(f"  Created {img_id}/{num_images} images...")
    
    print(f"\n✅ Custom format dataset created!")
    print(f"   Location: {output_dir}")
    print(f"   Images: {images_dir}")
    print(f"   Annotations: {annotations_dir}")
    print(f"\nTo train with this dataset:")
    print(f"  python train_pfarn_sarfish.py --dataset_path {output_dir} --format custom")


def main():
    parser = argparse.ArgumentParser(description='Create dummy dataset for testing')
    parser.add_argument('--output_dir', type=str, default='./dummy_dataset',
                        help='Output directory for dataset')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of dummy images to create')
    parser.add_argument('--format', type=str, default='coco', choices=['coco', 'custom'],
                        help='Dataset format to create')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Creating Dummy Dataset for Testing")
    print("=" * 60)
    print(f"Format: {args.format}")
    print(f"Number of images: {args.num_images}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print()
    
    if args.format == 'coco':
        create_coco_format_dataset(args.output_dir, args.num_images)
    else:
        create_custom_format_dataset(args.output_dir, args.num_images)
    
    print("\n" + "=" * 60)
    print("⚠️  NOTE: This is a DUMMY dataset for testing only!")
    print("   For real training, you need actual SAR ship detection data.")
    print("   See TRAINING_GUIDE.md for dataset recommendations.")
    print("=" * 60)


if __name__ == '__main__':
    main()

