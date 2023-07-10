#!/usr/bin/env python3  

import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

def encoder(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
        nn.MaxPool2d(kernel_size=2, stride=2)
        # Add more convolutional layers as needed
    )


def disparity_estimation(input_channels, output_channels):
    return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)

def compute_cost_volume(encoded_IL, encoded_IR):
    batch_size, num_channels, height_IL, width_IL = encoded_IL.size()
    batch_size, _, height_IR, width_IR = encoded_IR.size()
    
    # Resize the tensors to have the same spatial dimensions
    if height_IL != height_IR or width_IL != width_IR:
        encoded_IL = nn.functional.interpolate(encoded_IL, size=(height_IR, width_IR), mode='bilinear', align_corners=False)
    
    encoded_IL_expanded = encoded_IL.unsqueeze(2).expand(-1, -1, width_IR, -1, -1)
    encoded_IR_expanded = encoded_IR.unsqueeze(3).expand(-1, -1, -1, height_IR, -1)
    
    cost_volume = torch.cat((encoded_IL_expanded, encoded_IR_expanded), dim=1)
    cost_volume = cost_volume.view(batch_size, -1, height_IR, width_IR)
    
    return cost_volume


def disparity_net(IL, IR):
    encoded_IL = encoder(IL.size(1), 64)(IL)
    encoded_IR = encoder(IR.size(1), 64)(IR)
    
    cost_volume = compute_cost_volume(encoded_IL, encoded_IR)
    
    disparity_map = disparity_estimation(cost_volume.size(1), 1)(cost_volume)
    
    return disparity_map

def depth_estimation(disparity_map, baseline, focal_length):
    # Calculate depth using the formula: depth = (baseline * focal_length) / disparity
    depth_map = (baseline * focal_length) / disparity_map
    
    return depth_map

def save_disparity(disparity_map, output_dir, file_name):
    # Convert the disparity map tensor to a numpy array
    disparity_map_np = disparity_map.squeeze().cpu().detach().numpy()
    
    # Normalize the disparity map between 0 and 255
    disparity_map_np = (disparity_map_np - disparity_map_np.min()) / (disparity_map_np.max() - disparity_map_np.min()) * 255
    disparity_map_np = disparity_map_np.astype(np.uint8)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the disparity map as an image
    output_path = os.path.join(output_dir, file_name)
    Image.fromarray(disparity_map_np, mode='L').save(output_path)


def main():
    # Directory paths for left and right images
    left_dir = "left_image"
    right_dir = "right_image"
    output_dir = "output"
    
    # List left and right image file names
    left_images = sorted(os.listdir(left_dir))
    right_images = sorted(os.listdir(right_dir))
    
    # Check if the number of left and right images match
    if len(left_images) != len(right_images):
        print("Error: Number of left and right images does not match.")
        return
    
    # Iterate through each pair of left and right images
    for i in range(len(left_images)):
        # Read left and right images
        left_image_path = os.path.join(left_dir, left_images[i])
        right_image_path = os.path.join(right_dir, right_images[i])
        IL = Image.open(left_image_path)
        IR = Image.open(right_image_path)
        
        # Convert images to tensors
        IL_tensor = torch.unsqueeze(torch.from_numpy(np.array(IL)), dim=0).permute(0, 3, 1, 2).float() / 255.0
        IR_tensor = torch.unsqueeze(torch.from_numpy(np.array(IR)), dim=0).permute(0, 3, 1, 2).float() / 255.0
        
        # Forward pass through the disparity network
        disparity_map = disparity_net(IL_tensor, IR_tensor)
        
        # Define baseline distance and focal length
        baseline = 0.1  # Example baseline distance in meters
        focal_length = 0.5  # Example focal length in pixels
        
        # Estimate the depth map
        depth_map = depth_estimation(disparity_map, baseline, focal_length)
        
        # Print the shape of the predicted disparity map and depth map
        print("Disparity Map Shape:", disparity_map.shape)
        print("Depth Map Shape:", depth_map.shape)
        
        # Save the disparity map
        save_disparity(disparity_map, output_dir, f"disparity_{i}.png")

if __name__ == '__main__':
    main()
