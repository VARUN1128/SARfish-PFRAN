"""
Convert QGIS Shapefile Annotations to Training Format

This script converts your QGIS-annotated Shapefiles (.shp) to COCO format
for training the PFARN-SARfish model.

Workflow:
1. You annotate in QGIS → saves as .shp files
2. This script converts .shp → COCO format
3. Train with train_pfarn_sarfish.py

Usage:
    python convert_qgis_to_training.py \
        --shp_dir /path/to/shapefiles \
        --tif_dir /path/to/tiff/images \
        --output_dir ./training_dataset
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import geopandas as gpd
    from shapely.geometry import box, Polygon, Point
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("Warning: geopandas not installed. Install with: pip install geopandas")

try:
    import rasterio
    from rasterio.transform import xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. Install with: pip install rasterio")

from PIL import Image
import numpy as np


def find_tif_files(tif_dir: Path) -> Dict[str, Path]:
    """
    Find all .tif files, especially those ending in .001
    Returns dict mapping filename (without extension) to full path
    """
    tif_files = {}
    
    # Look for .001.tif files first (as mentioned in workflow)
    patterns = ['*.001.tif', '*.001.TIF', '*.tif', '*.TIF']
    
    for pattern in patterns:
        for tif_file in tif_dir.rglob(pattern):
            # Use stem (filename without extension) as key
            key = tif_file.stem
            # Remove .001 if present
            if key.endswith('.001'):
                key = key[:-4]
            tif_files[key] = tif_file
    
    return tif_files


def find_shp_files(shp_dir: Path) -> List[Path]:
    """Find all .shp files in directory"""
    shp_files = []
    for shp_file in shp_dir.rglob('*.shp'):
        shp_files.append(shp_file)
    return shp_files


def get_image_size(tif_path: Path) -> Tuple[int, int]:
    """Get image dimensions from TIFF file"""
    try:
        with rasterio.open(tif_path) as src:
            return (src.width, src.height)
    except:
        # Fallback: use PIL
        img = Image.open(tif_path)
        return img.size


def geometry_to_bbox(geometry, image_path: Path, image_size: Tuple[int, int]) -> List[float]:
    """
    Convert shapefile geometry to bounding box in pixel coordinates
    
    Args:
        geometry: Shapely geometry object (Polygon, Point, etc.)
        image_path: Path to the TIFF image
        image_size: (width, height) of image in pixels
    
    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    width, height = image_size
    
    try:
        # Get bounding box from geometry
        if isinstance(geometry, Point):
            # Point: create small box around it
            x, y = geometry.x, geometry.y
            # Convert lat/lon to pixel coordinates
            pixel_coords = latlon_to_pixel(x, y, image_path)
            # Create small box (20x20 pixels)
            x1 = max(0, pixel_coords[0] - 10)
            y1 = max(0, pixel_coords[1] - 10)
            x2 = min(width, pixel_coords[0] + 10)
            y2 = min(height, pixel_coords[1] + 10)
            return [float(x1), float(y1), float(x2), float(y2)]
        
        elif isinstance(geometry, (Polygon, box)):
            # Polygon: get bounding box
            bounds = geometry.bounds  # (minx, miny, maxx, maxy) in lat/lon
            
            # Convert corners to pixel coordinates
            minx, miny, maxx, maxy = bounds
            
            # Convert to pixel coordinates
            px_min, py_min = latlon_to_pixel(minx, miny, image_path)
            px_max, py_max = latlon_to_pixel(maxx, maxy, image_path)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(px_min, px_max))
            y1 = max(0, min(py_min, py_max))
            x2 = min(width, max(px_min, px_max))
            y2 = min(height, max(py_min, py_max))
            
            # Ensure valid box
            if x2 <= x1:
                x2 = x1 + 10
            if y2 <= y1:
                y2 = y1 + 10
            
            return [float(x1), float(y1), float(x2), float(y2)]
        
        else:
            # Other geometry types: get bounds
            bounds = geometry.bounds
            minx, miny, maxx, maxy = bounds
            px_min, py_min = latlon_to_pixel(minx, miny, image_path)
            px_max, py_max = latlon_to_pixel(maxx, maxy, image_path)
            
            x1 = max(0, min(px_min, px_max))
            y1 = max(0, min(py_min, py_max))
            x2 = min(width, max(px_min, px_max))
            y2 = min(height, max(py_min, py_max))
            
            if x2 <= x1:
                x2 = x1 + 10
            if y2 <= y1:
                y2 = y1 + 10
            
            return [float(x1), float(y1), float(x2), float(y2)]
    
    except Exception as e:
        print(f"Warning: Error converting geometry: {e}")
        return None


def latlon_to_pixel(lon: float, lat: float, image_path: Path) -> Tuple[int, int]:
    """
    Convert lat/lon coordinates to pixel coordinates
    
    Args:
        lon: Longitude
        lat: Latitude
        image_path: Path to GeoTIFF file
    
    Returns:
        (x, y) pixel coordinates (column, row)
    """
    try:
        with rasterio.open(image_path) as src:
            # Rasterio rowcol: (row, col) = rowcol(transform, x, y)
            # x = longitude, y = latitude
            row, col = rasterio.transform.rowcol(src.transform, lon, lat)
            # Return as (x, y) = (col, row)
            return (int(col), int(row))
    except Exception as e:
        # Fallback: if transformation fails
        print(f"Warning: Could not transform coordinates ({lon}, {lat}): {e}")
        return (0, 0)


def convert_shp_to_coco(
    shp_files: List[Path],
    tif_files: Dict[str, Path],
    output_dir: Path,
    images_dir: Path
) -> Dict:
    """
    Convert Shapefiles to COCO format
    
    Returns:
        COCO format dictionary
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "ship", "supercategory": "vessel"}
        ]
    }
    
    image_id = 1
    annotation_id = 1
    
    print(f"\nProcessing {len(shp_files)} shapefile(s)...")
    
    for shp_idx, shp_path in enumerate(shp_files, 1):
        print(f"\n[{shp_idx}/{len(shp_files)}] Processing: {shp_path.name}")
        
        try:
            # Read shapefile
            gdf = gpd.read_file(shp_path)
            print(f"  Found {len(gdf)} features")
            
            # Try to match with TIFF file
            # Strategy: try to match by filename
            shp_stem = shp_path.stem
            
            # Try different matching strategies
            matched_tif = None
            for key, tif_path in tif_files.items():
                # Remove common suffixes/prefixes
                key_clean = key.replace('_', '').replace('-', '').lower()
                shp_clean = shp_stem.replace('_', '').replace('-', '').lower()
                
                if key_clean in shp_clean or shp_clean in key_clean:
                    matched_tif = tif_path
                    break
            
            # If no match, try first available or ask user
            if matched_tif is None:
                if len(tif_files) == 1:
                    matched_tif = list(tif_files.values())[0]
                    print(f"  Using single available TIFF: {matched_tif.name}")
                else:
                    print(f"  ⚠️  Warning: Could not match {shp_path.name} with TIFF file")
                    print(f"     Available TIFF files: {list(tif_files.keys())[:5]}...")
                    print(f"     Skipping this shapefile")
                    continue
            
            print(f"  Matched with: {matched_tif.name}")
            
            # Get image size
            img_size = get_image_size(matched_tif)
            width, height = img_size
            
            # Copy image to images directory
            img_filename = f"img_{image_id:06d}.tif"
            img_output_path = images_dir / img_filename
            shutil.copy2(matched_tif, img_output_path)
            
            # Add image to COCO data
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"images/{img_filename}",
                "width": width,
                "height": height
            })
            
            # Process each feature in the shapefile
            valid_annotations = 0
            for idx, row in gdf.iterrows():
                geometry = row.geometry
                
                # Convert geometry to bounding box
                bbox = geometry_to_bbox(geometry, matched_tif, img_size)
                
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # COCO format: [x, y, width, height]
                bbox_coco = [x1, y1, x2 - x1, y2 - y1]
                area = (x2 - x1) * (y2 - y1)
                
                # Skip invalid boxes
                if area <= 0 or x2 <= x1 or y2 <= y1:
                    continue
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # All ships for now
                    "bbox": bbox_coco,
                    "area": area,
                    "iscrowd": 0
                })
                
                annotation_id += 1
                valid_annotations += 1
            
            print(f"  ✓ Created {valid_annotations} annotations for image {image_id}")
            image_id += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {shp_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description='Convert QGIS Shapefile annotations to COCO training format'
    )
    parser.add_argument('--shp_dir', type=str, required=True,
                        help='Directory containing .shp files from QGIS')
    parser.add_argument('--tif_dir', type=str, required=True,
                        help='Directory containing .tif files (measurement folder)')
    parser.add_argument('--output_dir', type=str, default='./training_dataset',
                        help='Output directory for training dataset')
    parser.add_argument('--annotation_file', type=str, default='annotations.json',
                        help='Output annotation filename')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_GEO:
        print("\nERROR: geopandas is required. Install with:")
        print("  pip install geopandas")
        print("  or")
        print("  conda install geopandas")
        return
    
    if not HAS_RASTERIO:
        print("\nERROR: rasterio is required. Install with:")
        print("  pip install rasterio")
        print("  or")
        print("  conda install rasterio")
        return
    
    # Setup paths
    shp_dir = Path(args.shp_dir)
    tif_dir = Path(args.tif_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    
    # Validate inputs
    if not shp_dir.exists():
        print(f"\nERROR: Shapefile directory not found: {shp_dir}")
        return
    
    if not tif_dir.exists():
        print(f"\nERROR: TIFF directory not found: {tif_dir}")
        return
    
    # Create output directory
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("QGIS to Training Format Converter")
    print("=" * 60)
    print(f"Shapefile directory: {shp_dir}")
    print(f"TIFF directory: {tif_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Find files
    print("\nScanning for files...")
    shp_files = find_shp_files(shp_dir)
    tif_files = find_tif_files(tif_dir)
    
    print(f"Found {len(shp_files)} shapefile(s)")
    print(f"Found {len(tif_files)} TIFF file(s)")
    
    if len(shp_files) == 0:
        print("\nERROR: No .shp files found in shapefile directory")
        return
    
    if len(tif_files) == 0:
        print("\nERROR: No .tif files found in TIFF directory")
        return
    
    # Convert
    coco_data = convert_shp_to_coco(shp_files, tif_files, output_dir, images_dir)
    
    # Save COCO format file
    annotation_path = output_dir / args.annotation_file
    with open(annotation_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Images folder: {images_dir}")
    print(f"Annotation file: {annotation_path}")
    print("\nTo train with this dataset:")
    print(f"  python train_pfarn_sarfish.py \\")
    print(f"    --dataset_path {output_dir} \\")
    print(f"    --format coco \\")
    print(f"    --annotation_file {annotation_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

