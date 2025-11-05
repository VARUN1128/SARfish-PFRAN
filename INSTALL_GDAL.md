# Installing GDAL/osgeo on Windows

## ✅ Solution Applied

The script has been updated to **automatically use rasterio + pyproj** as a fallback when osgeo/GDAL is not available. You can now run the script without installing GDAL!

## Current Status

The script will:
- ✅ Try to use osgeo/GDAL if available (preferred method)
- ✅ Automatically fall back to rasterio + pyproj if osgeo is not found
- ✅ Work the same way for coordinate conversion

## If You Still Want to Install GDAL (Optional)

### Option 1: Using Conda (Recommended - Easiest)
```bash
# Install Anaconda/Miniconda first, then:
conda install -c conda-forge gdal
```

### Option 2: Using Pre-built Wheels
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
# Then install:
pip install GDAL-3.x.x-cp311-cp311-win_amd64.whl
```

### Option 3: OSGeo4W Installer
1. Download OSGeo4W from: https://trac.osgeo.org/osgeo4w/
2. Install it
3. Add to PATH: `C:\OSGeo4W64\bin`
4. Then: `pip install gdal`

## Testing

You can now run:
```powershell
python SARfish.py sample_image.tif detections.geojson 0.5
```

The script will automatically use rasterio if GDAL is not available.

---

**Note**: The rasterio fallback method provides the same functionality for coordinate conversion. Both methods work identically for this use case.

