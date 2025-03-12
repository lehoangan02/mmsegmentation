import os
import torch
import mmcv
from mmseg.apis import init_model

# Specify your model configuration and checkpoint paths.
# config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
config_file = 'deeplabv3_r18b-d8_4xb2-80k_cityscapes-769x769.py'
checkpoint_file = 'deeplabv3_r18b-d8_769x769_80k_cityscapes_20201225_094144-fdc985d9.pth'

# Initialize the model.
model = init_model(config_file, checkpoint_file, device='mps')

# Folder containing the input images.
input_folder = './../BBAVectors-Oriented-Object-Detection/datasets/MiniTrainV1.1/images'
# input_folder = 'input_images'

# Process each image in the folder.
for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, img_name)
        # Read the image using mmcv
        img = mmcv.imread(img_path)
        # Convert the image from HWC to CHW and add a batch dimension.
        # (Assumes the image is in RGB format; adjust if needed.)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to('mps')
        
        # Forward pass in "tensor" mode to get the raw output.
        raw_output = model.forward(img_tensor, mode='tensor')
        
        # Print the raw tensor output.
        print(f'Raw tensor output for {img_name}:')
        print(raw_output.shape)
