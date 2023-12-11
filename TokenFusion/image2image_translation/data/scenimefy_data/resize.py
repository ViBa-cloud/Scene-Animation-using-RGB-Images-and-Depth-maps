from PIL import Image, ImageFile
import os
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

directory = '/home/suryasin/TokenFusion/image2image_translation/data/scenimefy_data/edge_occlusion'

# Get a list of files in the directory
files = [filename for filename in os.listdir(directory) if filename.endswith('.png')]

# Create a tqdm progress bar for the loop
for filename in tqdm(files, desc='Resizing images'):
    file_path = os.path.join(directory, filename)
    img = Image.open(file_path)
    
    # Resize the image to 512x512
    resized_img = img.resize((512, 512), Image.LANCZOS)

    resized_img.save(file_path)
