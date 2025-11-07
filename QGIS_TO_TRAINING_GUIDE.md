# QGIS to Training Format - Complete Guide

## Your Workflow

1. **Download from Copernicus** → Extract ZIP
2. **Open in QGIS** → Load `.001.tif` files from `measurement/` folder
3. **Manually annotate ships** → Save as Shapefile (`.shp`)
4. **Convert to training format** → Use this script
5. **Train model** → Use `train_pfarn_sarfish.py`

## Quick Start

### Step 1: Organize Your Data

Your directory structure should look like:
```
your_data/
  ├── shapefiles/          # Your QGIS annotations
  │   ├── annotation1.shp
  │   ├── annotation1.shx
  │   ├── annotation1.dbf
  │   ├── annotation2.shp
  │   └── ...
  └── tiff_images/          # Copernicus TIFF files
      └── measurement/
          ├── image1.001.tif
          ├── image2.001.tif
          └── ...
```

### Step 2: Run Conversion

```bash
python convert_qgis_to_training.py \
  --shp_dir ./shapefiles \
  --tif_dir ./tiff_images/measurement \
  --output_dir ./training_dataset
```

### Step 3: Train Model

```bash
python train_pfarn_sarfish.py \
  --dataset_path ./training_dataset \
  --format coco \
  --annotation_file ./training_dataset/annotations.json \
  --epochs 50
```

## Detailed Usage

### Basic Command

```bash
python convert_qgis_to_training.py \
  --shp_dir /path/to/shapefiles \
  --tif_dir /path/to/tiff/images \
  --output_dir ./training_dataset
```

### Arguments

- `--shp_dir`: Directory containing your QGIS Shapefiles (`.shp`, `.shx`, `.dbf` files)
- `--tif_dir`: Directory containing TIFF images (looks for `.001.tif` files)
- `--output_dir`: Where to save the training dataset (default: `./training_dataset`)
- `--annotation_file`: Name of annotation file (default: `annotations.json`)

### Output Structure

After conversion, you'll have:
```
training_dataset/
  ├── images/
  │   ├── img_000001.tif
  │   ├── img_000002.tif
  │   └── ...
  └── annotations.json      # COCO format
```

## How It Works

1. **Finds Shapefiles**: Scans directory for all `.shp` files
2. **Finds TIFF Images**: Looks for `.001.tif` or `.tif` files
3. **Matches Files**: Tries to match shapefiles with corresponding TIFF images
4. **Converts Coordinates**: 
   - Reads lat/lon from Shapefile geometry
   - Converts to pixel coordinates using GeoTIFF transform
   - Creates bounding boxes
5. **Organizes Data**: 
   - Copies TIFF images to `images/` folder
   - Creates COCO format annotations
   - Saves everything in training-ready format

## Matching Shapefiles to TIFF Files

The script tries to match shapefiles with TIFF files by:
- Comparing filenames (removing suffixes/prefixes)
- If multiple matches, uses first available
- If no match found, skips that shapefile (with warning)

**Tip**: Name your shapefiles similar to TIFF files for better matching:
- TIFF: `S1A_IW_20240101_001.tif`
- Shapefile: `S1A_IW_20240101_annotations.shp`

## Coordinate Conversion

The script automatically:
- Reads coordinate system from GeoTIFF
- Converts lat/lon from Shapefile to pixel coordinates
- Handles different projections (EPSG:4326, UTM, etc.)

## Validation

After conversion, check:
1. Number of images matches expected
2. Number of annotations per image
3. Open `annotations.json` to verify format
4. Check a few images to ensure annotations match

## Troubleshooting

### Error: "geopandas not installed"
```bash
pip install geopandas
# or
conda install geopandas
```

### Error: "rasterio not installed"
```bash
pip install rasterio
# or
conda install rasterio
```

### Warning: "Could not match shapefile with TIFF"
- Check filenames match (or are similar)
- Manually rename shapefiles to match TIFF names
- Or organize in matching subdirectories

### Error: "Could not transform coordinates"
- Check GeoTIFF has proper georeferencing
- Verify Shapefile has valid geometry
- Check coordinate systems match

### No annotations created
- Check Shapefile actually contains features
- Verify geometry type (Polygon, Point work best)
- Check if bounding boxes are valid (not too small)

## Batch Processing

To process multiple datasets:

```bash
# Process dataset 1
python convert_qgis_to_training.py \
  --shp_dir ./dataset1/shapefiles \
  --tif_dir ./dataset1/tiff \
  --output_dir ./dataset1/training

# Process dataset 2
python convert_qgis_to_training.py \
  --shp_dir ./dataset2/shapefiles \
  --tif_dir ./dataset2/tiff \
  --output_dir ./dataset2/training

# Then combine or train separately
```

## Tips

1. **Organize First**: Keep shapefiles and TIFFs in organized folders
2. **Consistent Naming**: Use similar names for matching
3. **Validate**: Check annotations.json before training
4. **Backup**: Keep original shapefiles safe
5. **Batch**: Process all data at once for consistency

## Example Workflow

```bash
# 1. You annotated in QGIS, saved as:
#    ./my_annotations/ships_2024.shp

# 2. Your TIFF files are in:
#    ./copernicus_data/measurement/S1A_20240101_001.tif

# 3. Convert:
python convert_qgis_to_training.py \
  --shp_dir ./my_annotations \
  --tif_dir ./copernicus_data/measurement \
  --output_dir ./my_training_data

# 4. Verify output:
#    Check: ./my_training_data/images/ has TIFF files
#    Check: ./my_training_data/annotations.json exists

# 5. Train:
python train_pfarn_sarfish.py \
  --dataset_path ./my_training_data \
  --format coco \
  --annotation_file ./my_training_data/annotations.json \
  --epochs 50 \
  --batch_size 4

# 6. Use trained model:
#    model.bin is created, use with SARfish.py
```

---

**This is a one-time conversion step!** Once converted, you can train multiple times with the same dataset.







