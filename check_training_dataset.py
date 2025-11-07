"""Check training dataset statistics"""
import json
from pathlib import Path

annotations_file = Path('training_dataset/annotations.json')
images_dir = Path('training_dataset/images')

if not annotations_file.exists():
    print("ERROR: annotations.json not found!")
    exit(1)

with open(annotations_file, 'r') as f:
    data = json.load(f)

print("=" * 60)
print("Training Dataset Statistics")
print("=" * 60)
print(f"Images: {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")
print(f"Categories: {len(data['categories'])}")

if len(data['images']) > 0:
    print(f"\nAnnotations per image:")
    annotations_per_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        annotations_per_image[img_id] = annotations_per_image.get(img_id, 0) + 1
    
    for img_id, count in sorted(annotations_per_image.items()):
        img_info = next((img for img in data['images'] if img['id'] == img_id), None)
        img_name = img_info['file_name'] if img_info else f"Image {img_id}"
        print(f"  {img_name}: {count} annotations")

if images_dir.exists():
    image_files = list(images_dir.glob('*.tif*'))
    print(f"\nImage files in directory: {len(image_files)}")
    if len(image_files) != len(data['images']):
        print(f"  WARNING: Mismatch! Expected {len(data['images'])} images, found {len(image_files)} files")

print("=" * 60)

