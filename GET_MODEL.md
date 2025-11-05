# How to Get model.bin

## Option 1: Download from Original SARfish Repository (Currently Unavailable)

⚠️ **The original Google Drive link is no longer working** - The file appears to have been removed or the link is broken.

### Alternative Sources to Try:

1. **Check the Original Repository:**
   - Visit: https://github.com/MJCruickshank/SARfish
   - Check if there are updated links in the README or issues
   - Look for releases or assets section

2. **Contact the Repository Maintainer:**
   - Open an issue on GitHub asking for the model weights
   - Check if there's a new download link

3. **Search for Alternative Sources:**
   - Some users may have shared the model elsewhere
   - Check research paper repositories or datasets
   - Look for LS-SSDD-v1.0 related resources

### Important Notes:
- ⚠️ **Any original model was trained on Faster R-CNN + FPN architecture**
- ⚠️ **It may not be fully compatible with PFARN architecture** (different model structure)
- The weights might load but may not work perfectly since we're using PFARN components

---

## Option 2: Train Your Own PFARN Model (Recommended for PFARN)

Since you're using PFARN architecture, the best approach is to train your own model:

### What You'll Need:
1. **Training Dataset:**
   - Large-Scale SAR Ship Detection Dataset-v1.0 (LS-SSDD-v1.0)
   - Or your own annotated SAR ship detection dataset

2. **Training Script:**
   - Create `train_pfarn_sarfish.py`
   - Use PyTorch training loop
   - Save model weights as `model.bin`

3. **Training Process:**
   - Fine-tune the PFARN model on your dataset
   - Use appropriate loss functions
   - Save the trained weights

---

## Option 3: Use Original Model (Temporary Testing)

If you want to test with the original model weights:

1. **Download from Google Drive** (link above)
2. **Place in SARfish directory**
3. **Note:** You may get shape mismatch errors if the model architecture differs significantly
4. **If errors occur:** You'll need to either:
   - Use the original SARfish code (without PFARN)
   - Train a new PFARN model

---

## Compatibility Check

### Current PFARN Architecture:
- Uses `resnet_fpn_backbone` (standard FPN for now)
- Has `CACHead` wrapper (but currently just passes through)
- Model structure: Faster R-CNN with ResNet-50-FPN

### Original SARfish Model:
- Trained on Faster R-CNN with ResNet-50-FPN
- Standard detection head (not CACHead)

### Compatibility:
- ✅ **Should work** - The backbone structure is the same
- ⚠️ **May have minor issues** - The CACHead wrapper might cause shape mismatches
- **If errors occur:** You can temporarily bypass CACHead for testing

---

## Quick Download Instructions

1. **Open this link in your browser:**
   ```
   https://drive.google.com/file/d/1f4hJH9YBeTlNkbWUrbCP-C8ELh0eWJtT/view
   ```

2. **Download the file** (it may be a `.bin` or `.pth` file)

3. **Move/Rename to:**
   ```
   D:\PFARN\SARfish\model.bin
   ```

4. **Run the script again:**
   ```powershell
   python SARfish.py sample_image.tiff detections.geojson 0.5
   ```

---

## Troubleshooting

### Error: "size mismatch" or "shape mismatch"
- The original model weights may not match PFARN architecture exactly
- Solution: Train your own PFARN model or use original SARfish code

### Error: "Unexpected key(s) in state_dict"
- Some layers have different names (e.g., CACHead wrapper)
- Solution: Modify loading code to handle key mismatches

### File not found after download
- Make sure the file is named exactly `model.bin`
- Check it's in `D:\PFARN\SARfish\` directory

---

**Recommendation:** For best results with PFARN, train your own model on the PFARN architecture.

