# Preparing Training Data - Quick Guide

## Your Setup

You have:
- **4 TIFF files** → Place in `copernicus_data/measurement/`
- **16 shapefiles** → Place in `shapefiles/`

## File Organization

### Step 1: Organize TIFF Files

```
SARfish/
  └── copernicus_data/
      └── measurement/
          ├── image1.tif
          ├── image2.tif
          ├── image3.tif
          └── image4.tif
```

**Important:**
- Use original TIFF files from SAFE archive (with georeferencing)
- Not COG versions (they lack georeferencing)
- Files should have `.tif` or `.tiff` extension

### Step 2: Organize Shapefiles

```
SARfish/
  └── shapefiles/
      ├── annotation1.shp    ← Each annotation needs 4 files:
      ├── annotation1.shx    ← .shp (geometry)
      ├── annotation1.dbf    ← .shx (index)
      ├── annotation1.prj    ← .dbf (attributes)
      │                      ← .prj (projection)
      ├── annotation2.shp
      ├── annotation2.shx
      ├── annotation2.dbf
      ├── annotation2.prj
      └── ... (16 total shapefiles)
```

**Important:**
- Each shapefile needs all 4 files (.shp, .shx, .dbf, .prj)
- Name shapefiles to match TIFF files (by date or similar naming)
- Example: If TIFF is `s1a-20240121-001.tif`, name shapefile `s1a-20240121-annotations.shp`

## Matching Shapefiles to TIFF Files

The conversion script matches files by:
1. **Date matching** (extracts YYYYMMDD from filenames)
2. **Filename similarity** (removes underscores/dashes and compares)

**Best Practice:**
- Name shapefiles similar to TIFF files
- Example:
  - TIFF: `s1a-iw-grd-vv-20240121t173336-001.tif`
  - Shapefile: `s1a-iw-grd-vv-20240121t173336-annotations.shp`

## Running Conversion

Once files are organized:

```bash
python convert_qgis_to_training.py \
  --shp_dir shapefiles \
  --tif_dir copernicus_data/measurement \
  --output_dir training_dataset
```

**What happens:**
- Script finds all 16 shapefiles
- Script finds all 4 TIFF files
- Matches shapefiles to TIFF files
- Converts annotations to COCO format
- Creates `training_dataset/` with images and annotations.json

## Expected Output

```
training_dataset/
  ├── images/
  │   ├── img_000001.tif    ← From your 4 TIFF files
  │   ├── img_000002.tif
  │   ├── img_000003.tif
  │   └── img_000004.tif
  └── annotations.json      ← All 16 shapefiles converted
```

**Note:** If you have 16 shapefiles but only 4 TIFF files:
- Some shapefiles might not match (script will warn you)
- Only matched pairs will be used for training
- You can have multiple shapefiles per TIFF (different annotations)

## Training

After conversion:

```bash
python train_pfarn_sarfish.py \
  --dataset_path training_dataset \
  --format coco \
  --annotation_file training_dataset/annotations.json \
  --epochs 50 \
  --batch_size 2
```

## Tips

1. **More data = Better model**: 4 images with 16 annotations is a good start, but more images help
2. **Quality over quantity**: Well-annotated images are better than many poor annotations
3. **Georeferencing matters**: Ensure TIFF files have proper CRS/transform
4. **Check matches**: After conversion, check `annotations.json` to see how many annotations were created

