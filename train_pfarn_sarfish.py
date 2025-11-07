"""
Training Script for PFARN-SARfish Model

This script trains the PFARN-SARfish model for SAR ship detection.
It supports both COCO format datasets and custom datasets.

Usage:
    python train_pfarn_sarfish.py --dataset_path <path_to_dataset> --format coco --epochs 50
    python train_pfarn_sarfish.py --dataset_path <path_to_dataset> --format custom --epochs 50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import os
import sys
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb limit for large SAR images
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Import PFARN modules
from pfarn_modules import SSConv, PFA, CACHead
from SARfish import get_pfarn_sarfish_model


class SARShipDataset(Dataset):
    """
    Custom Dataset for SAR Ship Detection
    Supports both COCO format and simple custom format
    """
    def __init__(self, root_dir, annotation_file=None, transforms=None, format='coco'):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.format = format
        
        if format == 'coco':
            # Load COCO format annotations
            with open(annotation_file, 'r') as f:
                self.coco_data = json.load(f)
            
            # Create image to annotations mapping
            self.images = {img['id']: img for img in self.coco_data['images']}
            self.annotations = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append(ann)
            
            self.image_ids = list(self.images.keys())
            
        elif format == 'custom':
            # Custom format: simple directory structure
            # Expected structure:
            # root_dir/
            #   images/
            #     img1.jpg, img2.jpg, ...
            #   annotations/
            #     img1.json, img2.json, ...
            # Each annotation JSON: {"boxes": [[x1,y1,x2,y2], ...], "labels": [1,1,...]}
            self.images_dir = self.root_dir / 'images'
            self.annotations_dir = self.root_dir / 'annotations'
            
            if not self.images_dir.exists():
                raise ValueError(f"Images directory not found: {self.images_dir}")
            
            self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                    list(self.images_dir.glob('*.png')) +
                                    list(self.images_dir.glob('*.tif')))
            
            if len(self.image_files) == 0:
                raise ValueError(f"No images found in {self.images_dir}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def __len__(self):
        if self.format == 'coco':
            return len(self.image_ids)
        else:
            return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.format == 'coco':
            return self._get_coco_item(idx)
        else:
            return self._get_custom_item(idx)
    
    def _get_coco_item(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.root_dir / img_info['file_name']
        
        # Use rasterio for large TIFF files, PIL for others
        if img_path.suffix.lower() in ['.tif', '.tiff'] and HAS_RASTERIO:
            try:
                with rasterio.open(img_path) as src:
                    # Read all bands and convert to RGB
                    if src.count >= 3:
                        img = np.dstack([src.read(i+1) for i in range(3)])
                    else:
                        # Single band, replicate for RGB
                        band = src.read(1)
                        img = np.dstack([band, band, band])
                    # Normalize to 0-255 range if needed
                    if img.dtype != np.uint8:
                        img = (img / img.max() * 255).astype(np.uint8)
            except:
                # Fallback to PIL
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
        else:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
        
        # Resize large images to manageable size (max 2048x2048)
        # This is necessary for training as full-size SAR images are too large
        original_height, original_width = img.shape[:2]
        max_size = 2048
        if original_width > max_size or original_height > max_size:
            scale = min(max_size / original_width, max_size / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            img = Image.fromarray(img).resize((new_width, new_height), Image.Resampling.LANCZOS)
            img = np.array(img)
            # Update image info for annotation scaling
            width_scale = new_width / original_width
            height_scale = new_height / original_height
        else:
            width_scale = 1.0
            height_scale = 1.0
        
        # Get annotations
        boxes = []
        labels = []
        
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                # COCO format: [x, y, width, height] -> [x1, y1, x2, y2]
                bbox = ann['bbox']
                x, y, w, h = bbox
                # Scale bounding boxes if image was resized
                x = x * width_scale
                y = y * height_scale
                w = w * width_scale
                h = h * height_scale
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            # No annotations - create dummy box (will be ignored during training)
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        # Convert to tensor
        img = transforms.ToTensor()(img)
        
        # Apply normalization (same as SARfish.py)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        img = normalize(img)
        
        return img, target
    
    def _get_custom_item(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Load annotations
        ann_path = self.annotations_dir / (img_path.stem + '.json')
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            boxes = ann_data.get('boxes', [])
            labels = ann_data.get('labels', [1] * len(boxes))
        else:
            # No annotations
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        # Convert to tensor
        img = transforms.ToTensor()(img)
        
        # Apply normalization (same as SARfish.py)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
        img = normalize(img)
        
        return img, target


def collate_fn(batch):
    """Custom collate function for batching"""
    # Faster R-CNN expects a list of images, not a stacked tensor
    images, targets = tuple(zip(*batch))
    return list(images), list(targets)


def get_transforms(train=True):
    """Get data augmentation transforms"""
    # For Faster R-CNN, transforms are applied in the dataset
    # We'll handle normalization separately if needed
    # For now, return None - transforms will be applied in dataset
    return None


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """Train for one epoch"""
    model.train()
    
    loss_accumulator = {
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0,
        'total': 0.0
    }
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in loss_dict.keys():
            loss_accumulator[key] += loss_dict[key].item()
        loss_accumulator['total'] += losses.item()
        
        # Update progress bar
        if (batch_idx + 1) % print_freq == 0:
            pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'cls': f'{loss_dict["loss_classifier"].item():.4f}',
                'box': f'{loss_dict["loss_box_reg"].item():.4f}'
            })
    
    # Average losses
    num_batches = len(data_loader)
    avg_losses = {k: v / num_batches for k, v in loss_accumulator.items()}
    
    return avg_losses


def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train PFARN-SARfish Model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--format', type=str, default='coco', choices=['coco', 'custom'],
                        help='Dataset format: coco or custom')
    parser.add_argument('--annotation_file', type=str, default=None,
                        help='Path to COCO annotation file (required for coco format)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (background + ships)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Validate dataset format
    if args.format == 'coco' and args.annotation_file is None:
        raise ValueError("--annotation_file is required for coco format")
    
    # Create dataset
    print(f"Loading dataset from {args.dataset_path} (format: {args.format})")
    train_transforms = get_transforms(train=True)
    val_transforms = get_transforms(train=False)
    
    # For simplicity, using same dataset for train/val split
    # You can modify this to use separate train/val directories
    full_dataset = SARShipDataset(
        root_dir=args.dataset_path,
        annotation_file=args.annotation_file,
        transforms=train_transforms,
        format=args.format
    )
    
    # Split dataset (80% train, 20% val)
    # For small datasets, use all data for training and validation
    if len(full_dataset) <= 2:
        # If 1-2 images, use all for training, create empty val set
        train_size = len(full_dataset)
        val_size = 0
        train_dataset = full_dataset
        val_dataset = torch.utils.data.Subset(full_dataset, [])  # Empty subset
    else:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Only create val_loader if we have validation data
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("Creating PFARN-SARfish model...")
    model = get_pfarn_sarfish_model(num_classes=args.num_classes)
    model.to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch+1)
        print(f"Train Loss: {train_losses['total']:.4f}")
        print(f"  - Classifier: {train_losses['loss_classifier']:.4f}")
        print(f"  - Box Reg: {train_losses['loss_box_reg']:.4f}")
        print(f"  - Objectness: {train_losses['loss_objectness']:.4f}")
        print(f"  - RPN Box Reg: {train_losses['loss_rpn_box_reg']:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
        else:
            val_loss = float('inf')  # No validation data, use train loss for best model
            print(f"Val Loss: N/A (no validation data)")
        
        # Update learning rate
        lr_scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model (use train loss if no validation)
        if val_loader is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(args.output_dir, 'model.bin')
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model: {model_path}")
        else:
            # No validation, save model based on train loss
            if train_losses['total'] < best_val_loss:
                best_val_loss = train_losses['total']
                model_path = os.path.join(args.output_dir, 'model.bin')
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model (based on train loss): {model_path}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {os.path.join(args.output_dir, 'model.bin')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

