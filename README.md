# SARfish
Ship detection in Sentinel 1 Synthetic Aperture Radar (SAR) imagery

!["SARfish"](https://github.com/MJCruickshank/SARfish/blob/main/title_image.jpg)

## Description

*Note: This program is very much a work in progress, and its outputs should not be relied upon for important tasks.*

SARfish is a program designed to help Open Source Intelligence (OSINT) researchers investigate maritime traffic. While the automatic identification system (AIS) tracks most commercial vessels, a small percentage of vessels sail with their AIS transponders off. These vessels are often military vessels, or shipments of illicit/clandestine cargoes, making them of particular interest to researchers. 

The program runs on a Faster R-CNN model with a ResNet-50-FPN backbone retrained on the Large-Scale SAR Ship Detection Dataset-v1.0 (LS-SSDD-v1.0). It takes Sentinel-1 VH polarisation images as an input and outputs a geojson file with points where a ship has possibly been detected. 

Specifically, SARfish breaks down the input SAR geotiff file into 800x800 shards. Each of these shards is converted to a .jpg image and the model searches it for detections. The x,y coordinates of the detections are then converted into lat/lon and added to a list, before the program moves onto the next shard. Once all detections have been performed, the coordinates of potential ship detections are then checked for intersection with a buffered map of global land areas, and given a True/False value based on this, allowing for onshore detections to be filtered out.  

## Getting Started

### Requirements

- **Python 3.9**+
- **conda**: Due to the geo-spatial libraries required it is easiest to install the dependencies with conda

### Installing Package Dependencies

1. Create the conda environment. This will install all necessary package dependencies too.

```shell
conda create -n sarfish -c conda-forge python gdal numpy pandas shapely matplotlib pytorch torchvision rasterio ipython tqdm geopandas
```

2. Activate the conda environment created.

```shell
conda activate SARfish
```
## Complete Workflow

### Option 1: Using Pre-trained Model (Quick Start)

1) Download a Sentinel 1 SAR VH polarisation image, for more details check the [Data Specifics](#data-specifics) section below
2) Convert raw .tiff image to .tif (Can be performed in QGIS)
3) Download model weights here (https://drive.google.com/file/d/1f4hJH9YBeTlNkbWUrbCP-C8ELh0eWJtT/view) and save the `model.bin` file to the SARfish directory.
4) Run: 
```shell
python SARfish.py input_tif_image_name output_geojson_filename prediction_confidence_threshold 
```
   Example: 
```shell
python SARfish.py VH_test_image.tif detections.geojson 0.5
```
5) Plot detections / imagery in GIS software. Use the "onshore_detection" field in the output geojson file to filter out erronous detections on land. Alternatively, use the "detection_confidence" field to visualise the model's confidence that a given detection is a ship.

### Option 2: Training Your Own Model

#### Step 1: Prepare Your Dataset

1. Download Sentinel-1 SAR images from [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
2. Extract the ZIP files and locate `.001.tif` files in the `measurement` folder
3. Open images in QGIS and manually annotate ships
4. Save annotations as Shapefiles (`.shp` format)

#### Step 2: Convert Annotations to Training Format

Organize your files:
```
D:\PFARN\SARfish\
  ├── shapefiles\              # Your QGIS annotations (.shp files)
  │   ├── annotation1.shp
  │   ├── annotation1.shx
  │   ├── annotation1.dbf
  │   └── ...
  └── copernicus_data\         # Your TIFF images
      └── measurement\
          ├── S1A_20240101_001.tif
          └── ...
```

Convert Shapefiles to COCO training format:
```shell
python convert_qgis_to_training.py \
  --shp_dir D:\PFARN\SARfish\shapefiles \
  --tif_dir D:\PFARN\SARfish\copernicus_data\measurement \
  --output_dir D:\PFARN\SARfish\training_dataset
```

This creates:
```
training_dataset/
  ├── images/              # TIFF images for training
  │   ├── img_000001.tif
  │   └── ...
  └── annotations.json     # COCO format annotations
```

#### Step 3: Train the Model

```shell
python train_pfarn_sarfish.py \
  --dataset_path D:\PFARN\SARfish\training_dataset \
  --format coco \
  --annotation_file D:\PFARN\SARfish\training_dataset\annotations.json \
  --epochs 50
```

#### Step 4: Training Output

After training completes, you'll have:

```
D:\PFARN\SARfish\
  ├── model.bin              ← YOUR TRAINED MODEL! (Main output)
  ├── checkpoint_epoch_5.pth  ← Periodic checkpoints
  ├── checkpoint_epoch_10.pth
  └── ...
```

#### Step 5: Use Your Trained Model

Now use your trained `model.bin` for detection:

```shell
python SARfish.py \
  sample_image.tif \
  ship_detections.geojson \
  0.5
```

This will:
- Load `model.bin` automatically
- Detect ships in `sample_image.tif`
- Save results to `ship_detections.geojson`
- `0.5` is the confidence threshold

## To Run

**Quick Detection (with pre-trained or trained model):**
```shell
python SARfish.py input_tif_image_name output_geojson_filename prediction_confidence_threshold 
```

**Example:**
```shell
python SARfish.py VH_test_image.tif detections.geojson 0.5
```

**Note:** The script automatically loads `model.bin` if it exists in the SARfish directory. If not found, it will run with untrained weights (detections will be inaccurate). 

### Data Specifics
You can download Sentinel 1 products from [Copernicus Open Access Hub](https://scihub.copernicus.eu/) or 
[SentinelHub EO Browser](https://apps.sentinel-hub.com/eo-browser/). The pipeline currently expects the Sentinel tile 
to be in EPSG:4326, so either you download the tile in that coordinate system or you need to reproject it. 
The datatype of the tile should be an `8-bit` integer.

## Known Issues

Currently the model's detection threshold is set quite low. This can result in false positives where objects like stationary oil platforms, rocks, or small islands can be detected as ships. 

Areas on the edge of the input raster may not be properly scanned, due to the image not being perfectly divisible by the 800x800 detection window. 
