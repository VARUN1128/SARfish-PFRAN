# Quick Start Guide - PFARN-SARfish

## ✅ Current Status

All import errors are fixed! The script is ready to run.

## 🚀 Running the Script

### Step 1: Get a SAR Image File

You need a **Sentinel-1 SAR GeoTIFF image** (`.tif` file). Options:

**Option A: Download from Copernicus**
- Visit: https://scihub.copernicus.eu/
- Register (free)
- Search for Sentinel-1 SAR images
- Download a GeoTIFF file

**Option B: Use Existing Data**
- If you have SAR datasets, use those `.tif` files
- Place the file in `D:\PFARN\SARfish\` or provide full path

### Step 2: Run the Script

```powershell
cd D:\PFARN\SARfish
python SARfish.py your_image.tif output.geojson 0.5
```

Replace `your_image.tif` with your actual file path.

### Step 3: Check Output

The script will create `output.geojson` with ship detections.

## 📋 What You'll See

### If model.bin exists:
```
Loading pretrained model from D:\PFARN\SARfish/model.bin
Splitting image into shards
Finding ships
...
```

### If model.bin is missing:
```
Warning: model.bin not found at D:\PFARN\SARfish/model.bin
Running with untrained model weights (detections may be inaccurate)
Splitting image into shards
Finding ships
...
```

**Note**: Without trained weights, detections will be random/meaningless, but you can verify the pipeline works.

## 🔧 Troubleshooting

### Error: "Input file not found"
- Make sure the file path is correct
- Use full path if file is elsewhere: `python SARfish.py C:\path\to\file.tif output.geojson 0.5`

### Error: "No module named 'osgeo'"
- ✅ **FIXED** - Script automatically uses rasterio as fallback

### Error: "model.bin not found"
- ✅ **FIXED** - Script continues with untrained weights (warning shown)

## 📝 Next Steps

1. **Get a trained model**: Place `model.bin` in the SARfish directory for accurate detections
2. **Train your own model**: See Phase 5 in the documentation
3. **Test the pipeline**: Run with any GeoTIFF to verify the workflow

---
**Status**: ✅ All import errors resolved - Ready to run with valid SAR image file!

