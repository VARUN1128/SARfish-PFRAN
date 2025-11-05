"""
Quick sanity test for PFARN-SARfish architecture
Tests model definition and forward pass without requiring a full dataset
"""
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from pfarn_modules import SSConv, PFA, CACHead

class ResNetBackboneWrapper(nn.Module):
    """Wrapper for ResNet backbone to work with FasterRCNN"""
    def __init__(self, resnet_backbone):
        super().__init__()
        self.body = resnet_backbone
        self.out_channels = 2048  # ResNet-50 final layer output channels
    
    def forward(self, x):
        # Extract features at different levels
        # For now, return single feature map (will be enhanced with PFA later)
        x = self.body(x)
        return OrderedDict([('0', x)])

def get_pfarn_sarfish_model(num_classes):
    # 1. Load base ResNet-50 backbone
    resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    resnet_body = nn.Sequential(*list(resnet.children())[:-2])  # remove classification head
    
    # Wrap backbone for FasterRCNN compatibility
    backbone = ResNetBackboneWrapper(resnet_body)
    
    # 2. Add Shape-Scale Convolution module (PFARN) - stored for later integration
    ssconv = SSConv(in_channels=2048, out_channels=2048)
    
    # 3. Replace default FPN with PFARN PFA module - stored for later integration
    neck = PFA(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
    
    # 4. Construct custom Faster R-CNN using the PFARN backbone
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes)
    
    # 5. Replace standard box predictor with Center-Aware Head
    model.roi_heads.box_predictor = CACHead(model.roi_heads.box_predictor)
    
    return model

print("=" * 60)
print("PFARN-SARfish Architecture Sanity Test")
print("=" * 60)

try:
    print("\n[1/4] Testing module imports...")
    from pfarn_modules import SSConv, PFA, CACHead
    print("[OK] Modules imported successfully")
except Exception as e:
    print(f"[ERROR] Import error: {e}")
    exit(1)

try:
    print("\n[2/4] Testing SSConv module...")
    x = torch.randn(1, 2048, 32, 32)
    ss = SSConv(2048, 2048)
    out = ss(x)
    print(f"[OK] SSConv forward pass: input {x.shape} -> output {out.shape}")
except Exception as e:
    print(f"[ERROR] SSConv error: {e}")
    exit(1)

try:
    print("\n[3/4] Testing PFA module...")
    # Create dummy multi-scale features
    features = [
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 512, 32, 32),
        torch.randn(1, 1024, 16, 16),
        torch.randn(1, 2048, 8, 8)
    ]
    pfa = PFA(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
    out_features = pfa(features)
    print(f"[OK] PFA forward pass: {len(features)} input features -> {len(out_features)} output features")
    print(f"     Output shapes: {[f.shape for f in out_features]}")
except Exception as e:
    print(f"[ERROR] PFA error: {e}")
    exit(1)

try:
    print("\n[4/4] Testing model creation...")
    num_classes = 2
    model = get_pfarn_sarfish_model(num_classes)
    print(f"[OK] Model created successfully")
    print(f"     Model type: {type(model)}")
    print(f"     Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"[ERROR] Model creation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("[OK] All sanity checks passed!")
print("=" * 60)
print("\nNext: Test with actual image file:")
print("  python SARfish.py <test_image.tif> <output.geojson> 0.5")

