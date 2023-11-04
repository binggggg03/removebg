import os
import numpy as np
from PIL import Image
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import tkinter as tk
# 1. Load the pre-trained model:

model_path = 'C:/Users/yanbi/Desktop/DIP Project/MODELS/best_model_3_nov_new.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# Neural Network Architecture (U^2Net)
# ==============================================================================

# Submodules used in the U2NET architecture
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, dropout_rate=0.5):
        super(REBNCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class RSU7(nn.Module):
    def __init__(self, in_ch=128, mid_ch=64, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconv1 = REBNCONV(in_ch, out_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(out_ch * 2, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch * 2, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch * 2, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch * 2, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch * 2, mid_ch)
        self.pool6 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv7 = REBNCONV(mid_ch * 2, mid_ch)
        self.pool7 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv8 = REBNCONV(mid_ch * 2, mid_ch)
        self.pool8 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv9 = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv10 = REBNCONV(mid_ch * 2, mid_ch)
        
        #Upsampling
        self.rebnconv9d = REBNCONV(mid_ch * 4, mid_ch)
        
        self.rebnconv8d = REBNCONV(mid_ch * 4, mid_ch)
        
        self.rebnconv7d = REBNCONV(mid_ch * 4, mid_ch) 
        
        self.rebnconv6d = REBNCONV(mid_ch * 4, mid_ch)
        
        self.rebnconv5d = REBNCONV(mid_ch * 4, mid_ch)
        
        self.rebnconv4d = REBNCONV(mid_ch * 4, mid_ch)
        
        self.rebnconv3d = REBNCONV(mid_ch * 4, mid_ch)
        
        self.rebnconv2d = REBNCONV(mid_ch * 4, out_ch)

    def forward(self, x):
        hx = x

        hx1 = self.rebnconv1(hx)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)   
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx = self.pool6(hx6)
        
        hx7 = self.rebnconv7(hx)
        hx = self.pool7(hx7)
        
        hx8 = self.rebnconv8(hx)
        hx = self.pool8(hx8)

        hx9 = self.rebnconv9(hx8)
        
        hx10 = self.rebnconv10(hx9)
        
        hx9d = self.rebnconv9d(torch.cat((hx10, hx9), 1))
        
        hx8 = F.interpolate(hx8, size=(hx9d.size(2), hx9d.size(3)), mode='bilinear', align_corners=True)
        hx8d =  self.rebnconv8d(torch.cat((hx9d, hx8), 1))
        
        hx7 = F.interpolate(hx7, size=(hx8d.size(2), hx8d.size(3)), mode='bilinear', align_corners=True)
        hx7d =  self.rebnconv7d(torch.cat((hx8d, hx7), 1))
        
        hx6 = F.interpolate(hx6, size=(hx7d.size(2), hx7d.size(3)), mode='bilinear', align_corners=True)
        hx6d =  self.rebnconv6d(torch.cat((hx7d, hx6), 1))
        
        hx5 = F.interpolate(hx5, size=(hx6d.size(2), hx6d.size(3)), mode='bilinear', align_corners=True)
        hx5d =  self.rebnconv5d(torch.cat((hx6d, hx5), 1))
        
        hx4 = F.interpolate(hx4, size=(hx5d.size(2), hx5d.size(3)), mode='bilinear', align_corners=True)
        hx4d =  self.rebnconv4d(torch.cat((hx5d, hx4), 1))
        
        hx3 = F.interpolate(hx3, size=(hx4d.size(2), hx4d.size(3)), mode='bilinear', align_corners=True)
        hx3d =  self.rebnconv3d(torch.cat((hx4d, hx3), 1))
        
        hx2 = F.interpolate(hx2, size=(hx3d.size(2), hx3d.size(3)), mode='bilinear', align_corners=True)
        hx2d =  self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        
        hx2d = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=True)
        
        return hx1 + hx2d

#U^2Net Architecture Definition
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 32)
        
        self.stage2 = RSU7(64, 32, 64)
        
        self.stage3 = RSU7(128, 64, 128)

        self.stage2d = REBNCONV(256, 128)
        
        self.stage1d = REBNCONV(128, 64)

        self.side1 = nn.Conv2d(128, out_ch, kernel_size=3, padding=1)

        self.channel_reducer1 = nn.Conv2d(256, 128, 1)
        
        self.channel_reducer2 = nn.Conv2d(256, 64, 1)
        
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hx = x

        # Stage 1: Input -> 64 channels
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # Stage 2: 64 -> 128 channels
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # Stage 3: 128 -> 256 channels
        hx3 = self.stage3(hx)

        # Reduce channels: 256 -> 128
        hx3_reduced = self.channel_reducer1(hx3)
        
        # Upsampling and Stage 2d: 128 channels from reduced hx3 and 128 channels from hx2 -> 128 channels
        hx2d = self.stage2d(torch.cat((F.interpolate(hx3_reduced, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=True), hx2), dim=1))

        # Reduce channels: 128 -> 64
        hx2d_reduced = self.channel_reducer2(hx2d)

        # Upsampling and Stage 1d: 64 channels from reduced hx2d and 64 channels from hx1 -> 64 channels
        hx1d = self.stage1d(torch.cat((F.interpolate(hx2d_reduced, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=True), hx1), dim=1))

        # Side output: 64 -> 1 channel
        d1 = self.side1(hx1d)

        # Resizing `d1` to have the same width and height as `x`
        d1 = F.interpolate(d1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        out = torch.sigmoid(d1)

        return out
    
# Assuming you have a U-Net architecture model defined as 'UNet', otherwise replace the following line:
model = U2NET().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 2. Preprocess the input image:

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Or whatever size you've used during training
    transforms.ToTensor(),
    # Add any normalization if used during training e.g.,
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    return transform(img)

# 3. Use the model to get the mask:

def get_mask(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred = model(image_tensor)
    mask_pred = torch.sigmoid(mask_pred)
    mask_pred = (mask_pred > 0.5).float()
    return mask_pred.squeeze().cpu().numpy()

# 4. Remove the background:

def remove_background(image_path, mask):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    # Resize the mask to match the original image size
    mask_resized = np.array(Image.fromarray((mask*255).astype(np.uint8)).resize(img.size, Image.BILINEAR))
    mask_resized = mask_resized/255  # Normalize back to range [0,1]

    mask_3channel = np.stack([mask_resized]*3, axis=-1)
    foreground = img_np * mask_3channel
    return Image.fromarray(foreground.astype(np.uint8))

# 5. Process an image:

image_path = 'C:/Users/yanbi/Desktop/DIP Project/IMAGES/bing.jpg'



# Preprocess the image
image_tensor = preprocess_image(image_path, transform)

# Get the mask
mask = get_mask(model, image_tensor, device)

# Remove the background
result = remove_background(image_path, mask)

# Save or display the result
result.save('C:/Users/yanbi/Desktop/DIP Project/IMAGES/bing_rembg.jpg')
result.show()
