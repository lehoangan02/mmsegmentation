import os
import torch
import mmcv
from mmseg.apis import init_model

config_file = 'deeplabv3_r18b-d8_4xb2-80k_cityscapes-769x769.py'
checkpoint_file = 'deeplabv3_r18b-d8_769x769_80k_cityscapes_20201225_094144-fdc985d9.pth'
model = init_model(config_file, checkpoint_file, device='mps')

input_folder = './../BBAVectors-Oriented-Object-Detection/datasets/MiniTrainV1.1/images'

batch_size = 8  # Set batch size to 8
batch_imgs = []  # List to accumulate images

for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, img_name)
        img = mmcv.imread(img_path)
        
        # Resize image to the model's expected input size (769x769)
        img = mmcv.imresize(img, (608, 608))  # Critical step for batching
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to('mps')
        batch_imgs.append(img_tensor)  # Add to batch list
        
        # When batch is full, process it
        if len(batch_imgs) == batch_size:
            batch = torch.cat(batch_imgs, dim=0)  # Shape [8, C, H, W]
            raw_output = model.forward(batch, mode='tensor')
            print(f'Batch output shape: {raw_output.shape}')
            batch_imgs = []  # Reset batch list

# Process remaining images (if total not divisible by 8)
if len(batch_imgs) > 0:
    batch = torch.cat(batch_imgs, dim=0)
    raw_output = model.forward(batch, mode='tensor')
    print(f'Last batch (size {len(batch_imgs)}) output shape: {raw_output.shape}')