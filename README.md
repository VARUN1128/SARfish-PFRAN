# SARfish - Ship Detection in Sentinel-1 SAR Imagery

![SARfish](https://github.com/MJCruickshank/SARfish/blob/main/title_image.jpg)

## Description

SARfish is a deep learning-based ship detection system designed for Sentinel-1 Synthetic Aperture Radar (SAR) imagery. The system uses a PFARN (Pyramid Feature Aggregation with ResNet) architecture based on Faster R-CNN to automatically detect ships in SAR images.

**Use Cases:**
- Open Source Intelligence (OSINT) research
- Maritime traffic monitoring
- Detection of vessels with AIS transponders disabled
- Military and security applications

**Note:** This program is actively being developed. Outputs should be validated before use in critical applications.

## Features

- **Automatic Ship Detection**: Detects ships in Sentinel-1 VH polarization SAR images
- **Georeferenced Output**: Produces GeoJSON files with lat/lon coordinates
- **Land Filtering**: Automatically filters out onshore detections
- **Confidence Scores**: Provides detection confidence for each detection
- **Custom Training**: Train your own model with custom annotated data
- **QGIS Integration**: Works seamlessly with QGIS for annotation and visualization

## Architecture

The system uses:
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Detector**: Faster R-CNN
- **PFARN Modules**: Shape-Scale Convolution (SSConv), Pyramid Feature Aggregation (PFA), Center-Aware Classification Head (CACHead)
- **Processing**: Images are split into 800x800 pixel shards for efficient processing

---

## Installation

### Requirements

- **Python 3.9+**
- **Conda** (recommended for geospatial libraries)

### Step 1: Create Conda Environment

```bash
conda create -n sarfish -c conda-forge python gdal numpy pandas shapely matplotlib pytorch torchvision rasterio ipython tqdm geopandas
```

### Step 2: Activate Environment

```bash
conda activate sarfish
```

### Step 3: Install Additional Dependencies

```bash
pip install pillow
```

### Step 4: Verify Installation

```bash
python -c "import torch; import rasterio; import geopandas; print('All dependencies installed!')"
```

---

## Quick Start (Using Pre-trained Model)

### Step 1: Download Pre-trained Model

Download model weights from: [Google Drive](https://drive.google.com/file/d/1f4hJH9YBeTlNkbWUrbCP-C8ELh0eWJtT/view)

Save the `model.bin` file to the SARfish directory.

### Step 2: Get SAR Image

Download a Sentinel-1 SAR VH polarization image from:
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- [SentinelHub EO Browser](https://apps.sentinel-hub.com/eo-browser/)

**Important**: The image should be:
- GeoTIFF format (`.tif` or `.tiff`)
- VH polarization
- Properly georeferenced (with CRS/transform information)
- 8-bit integer data type

### Step 3: Run Detection

```bash
python SARfish.py input_image.tif output_detections.geojson 0.5
```

**Parameters:**
- `input_image.tif` - Path to your SAR image
- `output_detections.geojson` - Output filename for detections
- `0.5` - Confidence threshold (0.0 to 1.0)

**Example:**
```bash
python SARfish.py sample_image.tiff detections.geojson 0.5
```

### Step 4: View Results

Open the output GeoJSON file in:
- **QGIS**: Layer → Add Vector Layer → Select `detections.geojson`
- **Online**: Upload to [geojson.io](https://geojson.io/)
- **Python**: Use the provided visualization scripts

---

## Complete Workflow: Training Your Own Model

### Phase 1: Data Preparation

#### Step 1.1: Download SAR Images

1. Go to [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
2. Search for Sentinel-1 SAR images (VH polarization)
3. Download ZIP files
4. Extract ZIP files to get `.SAFE` folders
5. Locate `.tif` files in `SAFE/measurement/` folder

**Directory Structure:**
```
SARfish/
  └── copernicus_data/
      └── measurement/
          ├── s1a-iw-grd-vv-20240121-001.tif
          └── ...
```

**Important**: Use the original TIFF files from the SAFE archive (not COG versions) as they have proper georeferencing.

#### Step 1.2: Annotate Ships in QGIS

1. Open QGIS
2. Load the `.tif` file from `measurement/` folder
3. Create a new Shapefile layer:
   - Layer → Create Layer → New Shapefile Layer
   - Choose "Polygon" or "Point" geometry type
4. Manually draw polygons/points around ships
5. Save the Shapefile

**Directory Structure:**
```
SARfish/
  └── shapefiles/
      ├── annotation1.shp    ← Geometry
      ├── annotation1.shx    ← Index
      ├── annotation1.dbf    ← Attributes
      ├── annotation1.prj    ← Projection (important!)
      └── ...
```

**Note**: All 4 files (`.shp`, `.shx`, `.dbf`, `.prj`) are required for proper conversion.

### Phase 2: Convert Annotations to Training Format

#### Step 2.1: Organize Files

Ensure your files are organized:
```
SARfish/
  ├── shapefiles/              # QGIS annotations
  │   ├── annotation1.shp
  │   ├── annotation1.shx
  │   ├── annotation1.dbf
  │   └── annotation1.prj
  └── copernicus_data/
      └── measurement/         # Original TIFF files
          ├── image1.tif
          └── ...
```

#### Step 2.2: Run Conversion Script

```bash
cd D:\PFARN\SARfish; python convert_qgis_to_training.py --shp_dir shapefiles --tif_dir copernicus_data/measurement --output_dir training_dataset
```

**What it does:**
- Reads all `.shp` files from `shapefiles/` folder
- Matches them with corresponding TIFF files by date/filename
- Converts lat/lon coordinates to pixel coordinates
- Creates COCO format annotations
- Copies images to organized structure

**Output:**
```
training_dataset/
  ├── images/
  │   ├── img_000001.tif
  │   └── ...
  └── annotations.json    # COCO format
```

**Troubleshooting:**
- If TIFF files aren't found: Ensure filenames match or use original SAFE archive files
- If coordinates are wrong: Check that TIFF files have proper georeferencing (CRS/transform)
- If shapefile can't be read: Ensure all 4 files (.shp, .shx, .dbf, .prj) are present

### Phase 3: Train the Model

#### Step 3.1: Run Training

```bash
python train_pfarn_sarfish.py \
  --dataset_path training_dataset \
  --format coco \
  --annotation_file training_dataset/annotations.json \
  --epochs 50 \
  --batch_size 2 \
  --lr 0.001
```

**Parameters:**
- `--dataset_path`: Path to training dataset directory
- `--format`: Dataset format (`coco` or `custom`)
- `--annotation_file`: Path to COCO annotations JSON file
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 2-4, adjust based on GPU memory)
- `--lr`: Learning rate (default: 0.001)
- `--output_dir`: Directory to save model (default: current directory)
- `--device`: Device to use (`cuda`, `cpu`, or `auto`)

**Training Output:**
```
SARfish/
  ├── model.bin              ← Final trained model
  ├── checkpoint_epoch_5.pth ← Periodic checkpoints
  ├── checkpoint_epoch_10.pth
  └── ...
```

**Note**: Large SAR images are automatically resized to 2048x2048 pixels during training for memory efficiency.

### Phase 4: Use Your Trained Model

Once training completes, use your `model.bin` for detection:

```bash
python SARfish.py input_image.tif detections.geojson 0.5
```

The script automatically loads `model.bin` from the current directory.

---

## Output Format

The detection script produces a GeoJSON file with the following structure:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [longitude, latitude]
      },
      "properties": {
        "detection_confidence": 0.85,
        "onshore_detection": false
      }
    }
  ]
}
```

**Fields:**
- `coordinates`: [longitude, latitude] in WGS84 (EPSG:4326)
- `detection_confidence`: Model confidence score (0.0 to 1.0)
- `onshore_detection`: `true` if detection is on land, `false` if offshore

---

## File Structure

```
SARfish/
├── SARfish.py                    # Main detection script
├── train_pfarn_sarfish.py        # Training script
├── convert_qgis_to_training.py   # QGIS to COCO converter
├── pfarn_modules.py              # PFARN model architecture
├── model.bin                     # Trained model (after training)
├── world_land_areas.geojson     # Land mask for filtering
│
├── shapefiles/                   # QGIS annotations
│   ├── *.shp, *.shx, *.dbf, *.prj
│
├── copernicus_data/              # SAR images
│   └── measurement/
│       └── *.tif
│
├── training_dataset/             # Training data (after conversion)
│   ├── images/
│   └── annotations.json
│
└── Documentation/
    ├── README.md                 # This file
    ├── COMPLETE_WORKFLOW.md      # Detailed workflow
    ├── TRAINING_GUIDE.md         # Training instructions
    └── ...
```

---

## Troubleshooting

### Common Issues

#### 1. "model.bin not found"
- **Solution**: Either download the pre-trained model or train your own model first
- The script will run with untrained weights (inaccurate detections)

#### 2. "TIFF directory not found"
- **Solution**: Check the path to your TIFF files
- Ensure the directory exists and contains `.tif` or `.tiff` files

#### 3. "No .tif files found"
- **Solution**: Check file extensions (`.tif`, `.tiff`, `.TIF`, `.TIFF`)
- The script searches for all common TIFF extensions

#### 4. "Shapefile missing .shx or .dbf"
- **Solution**: Ensure all 4 shapefile components are present:
  - `.shp` (geometry)
  - `.shx` (index)
  - `.dbf` (attributes)
  - `.prj` (projection)

#### 5. "Coordinates are incorrect"
- **Solution**: Ensure TIFF files have proper georeferencing:
  - Use original files from SAFE archive (not COG versions)
  - Check that CRS and transform information are present
  - Verify with: `python -c "import rasterio; src=rasterio.open('file.tif'); print(src.crs, src.transform)"`

#### 6. "Out of memory during training"
- **Solution**: 
  - Reduce `--batch_size` (try 1 or 2)
  - Use smaller images
  - Images are automatically resized to 2048x2048 during training

#### 7. "DecompressionBombError"
- **Solution**: Already handled in the code - large images are automatically resized

#### 8. "Missing shard files"
- **Solution**: The script now handles missing shards gracefully
- If issues persist, ensure sufficient disk space

---

## Data Requirements

### Input SAR Images

- **Format**: GeoTIFF (`.tif` or `.tiff`)
- **Polarization**: VH (Vertical-Horizontal)
- **Coordinate System**: EPSG:4326 (WGS84) recommended
- **Data Type**: 8-bit integer
- **Georeferencing**: Must have CRS and transform information

### Annotations

- **Format**: Shapefile (`.shp` with `.shx`, `.dbf`, `.prj`)
- **Geometry**: Polygon or Point
- **Coordinate System**: Should match the SAR image CRS

---

## Performance Tips

1. **Confidence Threshold**: 
   - Lower (0.3-0.4): More detections, more false positives
   - Higher (0.6-0.7): Fewer detections, higher precision
   - Default (0.5): Balanced

2. **Training**:
   - More epochs = better accuracy (but diminishing returns)
   - More training images = better generalization
   - Use data augmentation for small datasets

3. **Detection**:
   - Processing time depends on image size
   - Large images are split into 800x800 shards
   - Each shard takes ~1-3 seconds on CPU

---

## Known Limitations

1. **False Positives**: The model may detect:
   - Oil platforms
   - Rocks and small islands
   - Other bright objects in SAR imagery

2. **Edge Effects**: Areas on image edges may not be fully scanned if the image isn't divisible by 800x800

3. **Coordinate Accuracy**: Depends on input image georeferencing quality

4. **Model Accuracy**: Depends on training data quality and quantity

---

## Citation

If you use SARfish in your research, please cite:

```
SARfish: Ship Detection in Sentinel-1 SAR Imagery
Based on PFARN (Pyramid Feature Aggregation with ResNet)
```

---

## License

[Add your license information here]

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## Support

For issues, questions, or contributions, please:
1. Check the documentation files in the repository
2. Review the troubleshooting section above
3. Open an issue on GitHub

---

## Acknowledgments

- Based on Faster R-CNN architecture
- Trained on LS-SSDD-v1.0 dataset
- Uses Sentinel-1 data from Copernicus Open Access Hub

---

**Last Updated**: 2025
**Version**: 1.0
