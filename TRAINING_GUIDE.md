# Training Guide for PFARN-SARfish Model

This guide explains how to train your own `model.bin` file for the PFARN-SARfish ship detection system.

## Quick Start

### 1. Prepare Your Dataset

You have two options for dataset format:

#### Option A: COCO Format (Recommended)

**Directory Structure:**
```
dataset/
  images/
    img1.jpg
    img2.jpg
    ...
  annotations.json  (COCO format)
```

**COCO Annotation Format:**
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img1.jpg",
      "width": 800,
      "height": 800
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height
    },
    ...
  ],
  "categories": [
    {"id": 1, "name": "ship"}
  ]
}
```

#### Option B: Custom Format

**Directory Structure:**
```
dataset/
  images/
    img1.jpg
    img2.jpg
    ...
  annotations/
    img1.json
    img2.json
    ...
```

**Custom Annotation Format:**
Each JSON file should contain:
```json
{
  "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
  "labels": [1, 1, ...]
}
```

### 2. Run Training

#### Basic Training (COCO Format)
```bash
python train_pfarn_sarfish.py \
  --dataset_path /path/to/dataset \
  --format coco \
  --annotation_file /path/to/annotations.json \
  --epochs 50 \
  --batch_size 4
```

#### Basic Training (Custom Format)
```bash
python train_pfarn_sarfish.py \
  --dataset_path /path/to/dataset \
  --format custom \
  --epochs 50 \
  --batch_size 4
```

### 3. Use Trained Model

After training completes, `model.bin` will be saved in the output directory. Copy it to the SARfish directory:

```bash
cp model.bin D:\PFARN\SARfish\model.bin
```

Then use it with `SARfish.py`:
```bash
python SARfish.py sample_image.tif detections.geojson 0.5
```

## Training Parameters

### Required Arguments

- `--dataset_path`: Path to your dataset directory
- `--format`: Dataset format (`coco` or `custom`)

### Optional Arguments

- `--annotation_file`: Path to COCO annotation file (required for coco format)
- `--num_classes`: Number of classes (default: 2, background + ships)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 4, adjust based on GPU memory)
- `--lr`: Learning rate (default: 0.001)
- `--output_dir`: Directory to save model (default: current directory)
- `--resume`: Path to checkpoint to resume training
- `--device`: Device to use (`cuda`, `cpu`, or `auto` - default: auto)
- `--save_freq`: Save checkpoint every N epochs (default: 5)

### Example with All Options

```bash
python train_pfarn_sarfish.py \
  --dataset_path ./sar_dataset \
  --format coco \
  --annotation_file ./sar_dataset/annotations.json \
  --epochs 100 \
  --batch_size 8 \
  --lr 0.0005 \
  --output_dir ./checkpoints \
  --device cuda \
  --save_freq 10
```

## Dataset Recommendations

### Minimum Dataset Size

- **Minimum**: 100-200 images (for fine-tuning)
- **Recommended**: 500+ images (for better results)
- **Optimal**: 1000+ images

### Data Quality Tips

1. **Image Quality**: Use high-quality SAR images (Sentinel-1 VH polarization recommended)
2. **Annotation Quality**: Ensure accurate bounding boxes around ships
3. **Diversity**: Include images from different:
   - Geographic locations
   - Weather conditions
   - Ship sizes and types
   - Image resolutions

### Recommended Datasets

1. **LS-SSDD-v1.0** (Large-Scale SAR Ship Detection Dataset)
   - Download from academic repositories
   - Contains ~15,000 annotated SAR ship images

2. **SSDD Dataset** (SAR Ship Detection Dataset)
   - Smaller dataset, good for testing

3. **Your Own Dataset**
   - Collect SAR images from Copernicus Open Access Hub
   - Annotate using tools like LabelImg, CVAT, or Roboflow

## Training Tips

### GPU Memory

- **Batch Size**: Start with 4, increase if you have more GPU memory
- **Image Size**: Model expects 800x800 images (automatically resized)
- **Mixed Precision**: Can be added for faster training (future enhancement)

### Learning Rate

- **Initial**: 0.001 (default)
- **Fine-tuning**: 0.0001 (if resuming from pretrained)
- **Learning Rate Schedule**: Automatically reduces by 10x every 10 epochs

### Training Time

- **100 images**: ~1-2 hours (on GPU)
- **500 images**: ~4-6 hours (on GPU)
- **1000+ images**: 8+ hours (on GPU)
- **CPU**: Much slower (10-20x slower)

### Monitoring Training

The script prints:
- **Loss values** for each component (classifier, box regression, etc.)
- **Validation loss** after each epoch
- **Learning rate** at each step
- **Checkpoint saves** when best model improves

### Resuming Training

If training is interrupted:
```bash
python train_pfarn_sarfish.py \
  --dataset_path /path/to/dataset \
  --format coco \
  --annotation_file /path/to/annotations.json \
  --resume ./checkpoints/checkpoint_epoch_20.pth \
  --epochs 50
```

## Expected Output

### Training Progress

```
Epoch 1/50
Train Loss: 2.3456
  - Classifier: 0.8234
  - Box Reg: 0.4567
  - Objectness: 0.5678
  - RPN Box Reg: 0.4977
Val Loss: 2.1234
Learning rate: 0.001000
Saved best model: model.bin (val_loss: 2.1234)
```

### Output Files

- `model.bin`: Best model weights (used by SARfish.py)
- `checkpoint_epoch_N.pth`: Periodic checkpoints (every N epochs)

## Troubleshooting

### Error: "CUDA out of memory"
- **Solution**: Reduce `--batch_size` (try 2 or 1)
- **Alternative**: Use `--device cpu` (slower but works)

### Error: "No images found"
- **Solution**: Check `--dataset_path` points to correct directory
- **Verify**: Images are in `images/` subdirectory for custom format

### Error: "Annotation file not found"
- **Solution**: Provide correct path with `--annotation_file`
- **Verify**: File exists and is valid JSON

### Training Loss Not Decreasing
- **Check**: Learning rate might be too high/low
- **Try**: Adjust `--lr` (try 0.0001 or 0.01)
- **Verify**: Dataset annotations are correct
- **Check**: Model architecture is correct

### Model Not Saving
- **Check**: `--output_dir` has write permissions
- **Verify**: Disk space available

## Next Steps

After training:

1. **Test the model**: Run `SARfish.py` with your trained `model.bin`
2. **Evaluate performance**: Check detection accuracy on test images
3. **Fine-tune**: Adjust detection threshold in `SARfish.py`
4. **Iterate**: Add more training data if needed

## Advanced Usage

### Transfer Learning from COCO

The model starts with ImageNet-pretrained ResNet-50 backbone. For faster convergence, you can:

1. Start with COCO-pretrained Faster R-CNN weights
2. Fine-tune on your SAR dataset
3. This typically requires 2-3x fewer epochs

### Custom Architecture Modifications

To modify the PFARN architecture:
1. Edit `pfarn_modules.py` (SSConv, PFA, CACHead)
2. Edit `SARfish.py` (get_pfarn_sarfish_model function)
3. Retrain with modified architecture

## Resources

- **LS-SSDD Dataset**: Search for "Large-Scale SAR Ship Detection Dataset"
- **SAR Ship Detection Papers**: Many provide datasets and model weights
- **Copernicus Open Access Hub**: https://scihub.copernicus.eu/ (for SAR images)

---

**Need Help?** Check the training script code comments or open an issue with your error message.

