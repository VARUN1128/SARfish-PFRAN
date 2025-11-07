# Next Steps After Conversion

## Step 1: Verify Conversion Results

After running the conversion, check the output:

```powershell
# Check if training_dataset was created
dir training_dataset

# Check annotations.json
python -c "import json; d=json.load(open('training_dataset/annotations.json')); print(f'Images: {len(d[\"images\"])}'); print(f'Annotations: {len(d[\"annotations\"])}')"
```

**Expected output:**
- `training_dataset/images/` folder with TIFF files
- `training_dataset/annotations.json` file
- Should show number of images and total annotations

## Step 2: Train the Model

Once conversion is successful, train your model:

```powershell
python train_pfarn_sarfish.py `
  --dataset_path training_dataset `
  --format coco `
  --annotation_file training_dataset/annotations.json `
  --epochs 50 `
  --batch_size 2 `
  --lr 0.001
```

**Or on a single line:**
```powershell
python train_pfarn_sarfish.py --dataset_path training_dataset --format coco --annotation_file training_dataset/annotations.json --epochs 50 --batch_size 2 --lr 0.001
```

**Parameters explained:**
- `--epochs 50`: Train for 50 epochs (adjust based on your needs)
- `--batch_size 2`: Process 2 images at a time (reduce to 1 if out of memory)
- `--lr 0.001`: Learning rate (default is usually fine)

## Step 3: Monitor Training

During training, you'll see:
- Training loss decreasing
- Validation loss (if you have validation data)
- Checkpoints saved every 5 epochs (default)
- Final model saved as `model.bin`

## Step 4: Use Your Trained Model

After training completes:

```powershell
python SARfish.py input_image.tif detections.geojson 0.5
```

The script will automatically use your new `model.bin`!

## Troubleshooting

**If conversion shows warnings:**
- Check that TIFF files have georeferencing
- Verify shapefile names match TIFF file dates
- Ensure all 4 shapefile components are present

**If training fails:**
- Reduce `--batch_size` to 1
- Check that annotations.json has valid data
- Ensure images are in `training_dataset/images/` folder

