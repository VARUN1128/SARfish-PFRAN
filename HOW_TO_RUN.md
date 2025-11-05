# How to Run PFARN-SARfish

## Quick Start

### Basic Command Format
```bash
cd D:\PFARN\SARfish
python SARfish.py <input_tiff_file> <output_geojson> <threshold>
```

### Example
```bash
python SARfish.py sample_image.tif detections.geojson 0.5
```

### ⚠️ Getting a Test SAR Image

You need a **Sentinel-1 SAR GeoTIFF image** to run the script. Here are options:

#### Option 1: Download from Copernicus Open Access Hub
1. Go to: https://scihub.copernicus.eu/
2. Register for a free account
3. Search for Sentinel-1 SAR images (VH polarization)
4. Download a GeoTIFF file

#### Option 2: Use Existing SAR Dataset
- If you have access to SAR datasets (LS-SSDD-v1.0, etc.), use those
- Ensure the file is in GeoTIFF format (.tif)

#### Option 3: Test with Sample Data
- Check if you have any `.tif` files in your directories
- The script works with any GeoTIFF file (though results are best with SAR imagery)

## Command Line Arguments

1. **`<input_tiff_file>`**: Path to your Sentinel-1 SAR GeoTIFF image file
   - Example: `sample.tif`, `sar_image.tif`
   - Must be a valid GeoTIFF file

2. **`<output_geojson>`**: Name for the output GeoJSON file with detections
   - Example: `detections.geojson`, `ship_detections.geojson`
   - Will be created in the same directory as the script

3. **`<threshold>`**: Detection confidence threshold (0.0 to 1.0)
   - Example: `0.5` (50% confidence)
   - Lower values = more detections (including false positives)
   - Higher values = fewer detections (higher precision)

## Prerequisites

### Required Files
- ✅ `world_land_areas.geojson` - Should already exist in the directory
- ⚠️ `model.bin` - **You need a trained model file**
  - If you don't have this, you'll get an error when loading the model
  - Options:
    1. Use an existing trained model
    2. Train a new model first
    3. For testing, you can temporarily comment out the `load_state_dict` line

### Required Python Packages
Make sure you have installed:
- torch, torchvision
- numpy, pandas
- PIL (Pillow)
- rasterio
- geopandas
- shapely
- osgeo/gdal
- tqdm

## Step-by-Step Example

### 1. Navigate to the directory
```powershell
cd D:\PFARN\SARfish
```

### 2. Prepare your input file
- Ensure you have a Sentinel-1 SAR GeoTIFF file
- Place it in the `D:\PFARN\SARfish\` directory or provide full path

### 3. Run the detection
```powershell
python SARfish.py my_sar_image.tif results.geojson 0.5
```

### 4. Check the output
- The script will create `results.geojson` in the same directory
- The file contains detected ship locations with:
  - Coordinates (lat/lon)
  - Detection confidence scores
  - Onshore/offshore flags

## What the Script Does

1. **Splits** the input SAR image into 800x800 pixel shards
2. **Processes** each shard through the PFARN detection model
3. **Converts** pixel coordinates to geographic coordinates (lat/lon)
4. **Filters** detections on land using the world land areas map
5. **Outputs** a GeoJSON file with all detections

## Troubleshooting

### Error: "No module named 'osgeo'"
```bash
# Install GDAL using conda (recommended)
conda install -c conda-forge gdal

# Or use pip
pip install gdal
```

### Error: "FileNotFoundError: model.bin"
- You need a trained model file
- For testing architecture, you can temporarily modify the script to skip model loading:
  ```python
  # Comment out this line:
  # model_ft.load_state_dict(torch.load(rootdir+"/"+"model.bin", map_location=torch.device('cpu')))
  ```

### Error: "No such file or directory: <input_file>"
- Make sure the input file path is correct
- Use full path if file is in different directory:
  ```bash
  python SARfish.py "C:\path\to\your\image.tif" output.geojson 0.5
  ```

## Testing Without a Model File

If you want to test the architecture without a trained model, you can temporarily modify the script:

```python
# In SARfish.py, around line 392, comment out:
# model_ft.load_state_dict(torch.load(rootdir+"/"+"model.bin", map_location=torch.device('cpu')))
```

Note: Detections will be random/meaningless without a trained model, but you can verify the pipeline works.

## Expected Output

The output GeoJSON file will contain:
- **geometry**: Point coordinates (lat/lon) for each detection
- **onshore_detection**: Boolean (True if on land, False if at sea)
- **detection_confidence**: Confidence score (0.0 to 1.0)

Example output structure:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [lon, lat]
      },
      "properties": {
        "onshore_detection": false,
        "detection_confidence": 0.87
      }
    }
  ]
}
```

---
For more information, see the README.md file.

