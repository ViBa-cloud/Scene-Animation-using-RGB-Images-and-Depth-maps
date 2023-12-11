import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt


def custom_sobel_same_size(image, stride, dx=1, dy=0, ksize=3):
    rows, cols = image.shape[:2]
    print("rows, cols: ", rows, cols)
    sobelx = np.zeros_like(image, dtype=np.float32)
    sobely = np.zeros_like(image, dtype=np.float32)

    for i in range(0, rows - ksize + 1, stride):
        for j in range(0, cols - ksize + 1, stride):
            window = image[i:i+ksize, j:j+ksize]

            gradient_x = cv2.Sobel(window, cv2.CV_32F, dx, dy, ksize=ksize)
            gradient_y = cv2.Sobel(window, cv2.CV_32F, dy, dx, ksize=ksize)

            sobelx[i:i+ksize, j:j+ksize] = gradient_x
            sobely[i:i+ksize, j:j+ksize] = gradient_y

    return sobelx, sobely

def average_convolution(image, kernel_size):
    # Define the average filter kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Apply the convolution operation
    output_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    return output_image

# Define paths to directories containing depth and RGB images
depth_dir = "/Users/nitin/Desktop/Michigan Started/F23/EECS 504/animate_proj/for slides/depth_rw" ##Change the path to depth images
rgb_dir = "/Users/nitin/Desktop/Michigan Started/F23/EECS 504/animate_proj/for slides/rgb_rw" ##Change the path to rgb images
output_dir = "/Users/nitin/Desktop/Michigan Started/F23/EECS 504/animate_proj/for slides/output_rw" ##Change the path to output directory

# Get a list of file names in the directories
depth_files = os.listdir(depth_dir)
rgb_files = os.listdir(rgb_dir)

# Ensure the directories exist for output
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each depth file
for depth_file in depth_files:
    if depth_file.endswith(".png"):  # Check if it's a PNG file
        depth_path = os.path.join(depth_dir, depth_file)
        rgb_file = depth_file.replace("_depth.png", "_rgb.png")  # Derive RGB file name from depth file
        rgb_path = os.path.join(rgb_dir, rgb_file)

        if os.path.exists(rgb_path):  # Check if corresponding RGB file exists
            # Process depth image
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_array = np.array(depth_img)
            normalized_depth = cv2.normalize(depth_array, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            sobelx, sobely = custom_sobel_same_size(normalized_depth, 2)
            sobel = sobelx + sobely
            # absolute
            sobel = abs(sobel)
            # averaging
            sobel = average_convolution(sobel, 5)
            # normalise
            sobel = (sobel - np.min(sobel)) / (np.max(sobel) - np.min(sobel))
            
            plt.imshow(sobel)

            rgb_img = Image.open(rgb_path)
            rgb_image = rgb_img.convert('RGB')
            rgb_array = np.array(rgb_image)
            hsv_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

            sat_channel = hsv_image[:, :, 1]
            modified_sat_value = np.multiply(sat_channel, normalized_depth)
            modified_sat_value = np.clip(modified_sat_value, 0 , 255)

            modified_val_value = hsv_image[:, :, 2]
            modified_val_value[sobel > 0.15] = 255

            hsv_image[:, :, 1] = modified_sat_value
            hsv_image[:, :, 2] = modified_val_value

            # Convert the modified HSV image back to BGR
            updated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            Image.fromarray(updated_image.astype(np.uint8)).save(updated_image_path)
            print(f"Processed: {depth_file}")
        else:
            print(f"No RGB image found for {depth_file}")