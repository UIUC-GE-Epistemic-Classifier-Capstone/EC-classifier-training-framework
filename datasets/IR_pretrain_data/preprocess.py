import os
import shutil
from PIL import Image

# Define your source and destination directories
src_dir = './cam3'
dest_dir = './'

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Image formats to consider
image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# Counter for image names
img_counter = 1

# Go through all folders and files recursively
for subdir, dirs, files in os.walk(src_dir):
    for file in files:
        # Check if the file is an image
        if file.endswith(image_formats):
            # Open and save the image in the new location
            img = Image.open(os.path.join(subdir, file))
            img.save(os.path.join(dest_dir, f"{img_counter:05d}.jpg")) # save image as .jpg
            img_counter += 1

