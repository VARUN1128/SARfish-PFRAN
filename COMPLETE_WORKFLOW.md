# Complete Workflow: From QGIS Annotations to Ship Detection Output

## Overview Diagram

```
QGIS Annotations (.shp)
    ↓
[Convert to Training Format]
    ↓
Training Dataset (COCO format)
    ↓
[Train PFARN Model]
    ↓
Trained Model (model.bin)
    ↓
[SARfish Detection]
    ↓
Ship Detections (GeoJSON)
```

---

## Step-by-Step Workflow

### **PHASE 1: Data Preparation (QGIS)**

#### Step 1.1: Download SAR Images
```
1. Go to Copernicus Open Access Hub (https://scihub.copernicus.eu/)
2. Download Sentinel-1 SAR images (VH polarization)
3. Extract ZIP files
4. Locate .001.tif files in measurement/ folder
```

**File Location:**
```
D:\PFARN\SARfish\copernicus_data\
  └── measurement\
      ├── S1A_IW_20240101_001.tif
      ├── S1A_IW_20240102_001.tif
      └── ...
```

#### Step 1.2: Annotate in QGIS
```
1. Open QGIS
2. Load .001.tif file from measurement/ folder
3. Create new Shapefile layer
4. Manually draw polygons/points around ships
5. Save as Shapefile (.shp)
```

**File Location:**
```
D:\PFARN\SARfish\shapefiles\
  ├── annotation1.shp       ← Geometry
  ├── annotation1.shx       ← Index
  ├── annotation1.dbf       ← Attributes
  ├── annotation2.shp
  └── ...
```

**What you have now:**
- ✅ SAR images (.tif files)
- ✅ Ship annotations (.shp files with lat/lon coordinates)

---

### **PHASE 2: Format Conversion (One-Time)**

#### Step 2.1: Run Conversion Script

**Purpose:** Convert QGIS Shapefiles to ML training format

**Command:**
```bash
python convert_qgis_to_training.py \
  --shp_dir D:\PFARN\SARfish\shapefiles \
  --tif_dir D:\PFARN\SARfish\copernicus_data\measurement \
  --output_dir D:\PFARN\SARfish\training_dataset
```

**What the script does:**
1. Reads all .shp files from `shapefiles/` folder
2. Finds matching .tif files from `measurement/` folder
3. Converts lat/lon coordinates → pixel coordinates
4. Extracts bounding boxes from polygons/points
5. Creates COCO format annotations
6. Copies images to organized structure

**Output Structure:**
```
D:\PFARN\SARfish\training_dataset\
  ├── images\
  │   ├── img_000001.tif     ← Copied from measurement/
  │   ├── img_000002.tif
  │   └── ...
  └── annotations.json       ← COCO format (pixel coordinates)
```

**What you have now:**
- ✅ Training-ready images in `images/` folder
- ✅ COCO format annotations (JSON file)
- ✅ Pixel coordinates (not lat/lon anymore)

---

### **PHASE 3: Model Training (PFARN)**

#### Step 3.1: Train PFARN Model

**Purpose:** Train the PFARN-SARfish model on your data

**Command:**
```bash
python train_pfarn_sarfish.py \
  --dataset_path D:\PFARN\SARfish\training_dataset \
  --format coco \
  --annotation_file D:\PFARN\SARfish\training_dataset\annotations.json \
  --epochs 50 \
  --batch_size 4
```

**What happens during training:**
1. **Loads dataset:**
   - Reads images from `training_dataset/images/`
   - Reads annotations from `annotations.json`
   - Creates data loader with batches

2. **PFARN Architecture:**
   - ResNet-50 backbone (feature extraction)
   - SSConv module (Shape-Scale Convolution)
   - PFA module (Pyramid Feature Aggregation)
   - CACHead (Center-Aware Classification Head)
   - Faster R-CNN detector

3. **Training loop:**
   - Forward pass: Model predicts ship locations
   - Loss calculation: Compare predictions with annotations
   - Backward pass: Update model weights
   - Validation: Check performance on validation set
   - Repeat for specified epochs

4. **Saves checkpoints:**
   - Best model → `model.bin`
   - Periodic checkpoints → `checkpoint_epoch_N.pth`

**Training Output:**
```
D:\PFARN\SARfish\
  ├── model.bin              ← YOUR TRAINED MODEL! (Main file)
  ├── checkpoint_epoch_5.pth  ← Periodic saves
  ├── checkpoint_epoch_10.pth
  └── ...
```

**What you have now:**
- ✅ `model.bin` - Trained PFARN model with learned weights
- ✅ Model knows how to detect ships in SAR images

---

### **PHASE 4: Ship Detection (SARfish)**

#### Step 4.1: Run SARfish Detection

**Purpose:** Use trained model to detect ships in new SAR images

**Command:**
```bash
python SARfish.py \
  new_sar_image.tif \
  ship_detections.geojson \
  0.5
```

**Parameters:**
- `new_sar_image.tif` - Input SAR image to analyze
- `ship_detections.geojson` - Output file name
- `0.5` - Confidence threshold (0.0 to 1.0)

**What SARfish.py does:**

1. **Load Model:**
   ```python
   # Automatically loads model.bin from current directory
   model = get_pfarn_sarfish_model(num_classes=2)
   model.load_state_dict(torch.load('model.bin'))
   ```

2. **Process Image:**
   - Splits large SAR image into 800x800 pixel shards
   - Converts each shard to RGB format
   - Normalizes pixel values

3. **Run Detection:**
   - Passes each shard through PFARN model
   - Model predicts ship locations with confidence scores
   - Filters detections by confidence threshold (0.5)

4. **Coordinate Conversion:**
   - Converts pixel coordinates → lat/lon coordinates
   - Uses GeoTIFF transform information

5. **Land Filtering:**
   - Checks if detections are on land (using world_land_areas.geojson)
   - Adds `onshore_detection` flag (True/False)

6. **Output GeoJSON:**
   - Creates GeoJSON file with ship detections
   - Includes: coordinates, confidence, onshore flag

**Output File:**
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
        "detection_confidence": 0.85,
        "onshore_detection": false
      }
    },
    ...
  ]
}
```

**What you have now:**
- ✅ `ship_detections.geojson` - Ship locations with metadata
- ✅ Ready to visualize in QGIS or other GIS software

---

## Complete Workflow Summary

### Input Files:
```
1. SAR Images:     copernicus_data/measurement/*.001.tif
2. Annotations:    shapefiles/*.shp (from QGIS)
```

### Processing:
```
1. Conversion:     convert_qgis_to_training.py
   Input:  .shp files + .tif files
   Output: training_dataset/ (images + annotations.json)

2. Training:       train_pfarn_sarfish.py
   Input:  training_dataset/
   Output: model.bin (trained model)

3. Detection:      SARfish.py
   Input:  new_sar_image.tif + model.bin
   Output: ship_detections.geojson
```

### Output Files:
```
1. model.bin              - Trained PFARN model
2. ship_detections.geojson - Ship detections with coordinates
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: QGIS Annotation                                │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Copernicus SAR Images          QGIS Annotations         │
│  (.001.tif files)              (manual drawing)         │
│         │                              │                 │
│         └──────────┬───────────────────┘                 │
│                    ↓                                     │
│              shapefiles/*.shp                            │
│              (lat/lon coordinates)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 2: Format Conversion                               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  convert_qgis_to_training.py                            │
│         │                                                 │
│         ├─ Reads .shp files                               │
│         ├─ Matches with .tif files                        │
│         ├─ Converts lat/lon → pixels                      │
│         ├─ Creates bounding boxes                         │
│         └─ Outputs COCO format                           │
│                    ↓                                     │
│        training_dataset/                                 │
│        ├── images/*.tif                                 │
│        └── annotations.json                              │
│        (pixel coordinates, COCO format)                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 3: PFARN Model Training                            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  train_pfarn_sarfish.py                                  │
│         │                                                 │
│         ├─ Loads training_dataset/                       │
│         ├─ PFARN Architecture:                            │
│         │   • ResNet-50 backbone                         │
│         │   • SSConv (Shape-Scale Conv)                  │
│         │   • PFA (Pyramid Feature Aggregation)          │
│         │   • CACHead (Center-Aware Head)                │
│         │   • Faster R-CNN detector                       │
│         ├─ Training loop (50 epochs)                     │
│         ├─ Loss optimization                             │
│         └─ Saves best model                              │
│                    ↓                                     │
│              model.bin                                   │
│        (trained weights, learned patterns)               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 4: Ship Detection (SARfish)                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  SARfish.py + model.bin                                  │
│         │                                                 │
│         ├─ Loads model.bin                               │
│         ├─ Splits image into 800x800 shards              │
│         ├─ Runs PFARN detection on each shard            │
│         ├─ Gets ship predictions with confidence         │
│         ├─ Converts pixels → lat/lon                    │
│         ├─ Filters land detections                       │
│         └─ Outputs GeoJSON                              │
│                    ↓                                     │
│        ship_detections.geojson                           │
│        (ship locations, confidence, onshore flags)       │
└─────────────────────────────────────────────────────────┘
```

---

## Key Points

### One-Time Steps:
- ✅ **Conversion** - Only needed once per dataset
- ✅ **Training** - Only needed once (or when you add more data)

### Repeated Steps:
- 🔄 **Detection** - Run every time you have a new SAR image

### File Formats:
- **Input:** `.shp` (QGIS) → `.json` (COCO) → `model.bin` (PyTorch)
- **Output:** `.geojson` (GIS visualization)

### Coordinate Systems:
- **QGIS:** Lat/Lon (EPSG:4326)
- **Training:** Pixel coordinates (0 to image_width/height)
- **Output:** Lat/Lon (EPSG:4326)

---

This is your complete end-to-end workflow from QGIS annotations to ship detection results!

