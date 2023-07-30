# import os
# import cv2
# import numpy as np
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import easyocr
# from scipy.spatial.distance import cdist

# # Define the boundaries of the regions
# max_temp_region_bounds = (560, 8, 617, 47)
# min_temp_region_bounds = (570, 360, 617, 390)
# scale_region_bounds = (610, 56, 628, 342)

# dirs = ['./day/thermal/', './night/thermal/']

# # Load the image
# image = cv2.imread('./day/thermal/0009.JPG')

# # Define the region that contains the temperature scale numbers
# max_temp_region = image[max_temp_region_bounds[1]:max_temp_region_bounds[3], 
#                         max_temp_region_bounds[0]:max_temp_region_bounds[2]]

# min_temp_region = image[min_temp_region_bounds[1]:min_temp_region_bounds[3], 
#                         min_temp_region_bounds[0]:min_temp_region_bounds[2]]

# # Define the region that contains the color scale
# scale_region = image[scale_region_bounds[1]:scale_region_bounds[3], 
#                      scale_region_bounds[0]:scale_region_bounds[2]]

# # Define a function to find the closest color in the color scale for a given color
# def find_closest_color(color, color_scale):
#     distances = cdist([color], color_scale, 'euclidean')
#     min_index = np.argmin(distances)
#     return color_scale[min_index]

# def trim_whitespace(image):
#     _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((5, 5), np.uint8)
#     dilated = cv2.dilate(binary, kernel, iterations=1)
#     eroded = cv2.erode(dilated, kernel, iterations=1)
#     contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     x, y, w, h = cv2.boundingRect(np.concatenate(contours))
#     cropped_image = image[y:y+h, x:x+w]
#     return cropped_image

# def preprocess_image(image):
#     scale_percent = 300 
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9, 9), 2)
#     ret, thresh_img = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
#     return thresh_img

# def extract_temp(region):
#     reader = easyocr.Reader(['en'])  
#     result = reader.readtext(region)
#     temp_values = []
#     for res in result:
#         try:
#             temp = float(res[1].replace('°', '').replace('-', ''))
#             if '-' in res[1]:
#                 temp *= -1
#             temp_values.append(temp)
#         except ValueError:
#             continue
#     return min(temp_values) if temp_values else None
# # Define a function to preprocess image and extract temperatures
# def preprocess_and_extract_temps(image):
#     # Define the region that contains the temperature scale numbers
#     max_temp_region = image[max_temp_region_bounds[1]:max_temp_region_bounds[3], 
#                             max_temp_region_bounds[0]:max_temp_region_bounds[2]]

#     min_temp_region = image[min_temp_region_bounds[1]:min_temp_region_bounds[3], 
#                             min_temp_region_bounds[0]:min_temp_region_bounds[2]]

#     max_temp_region = preprocess_image(max_temp_region)
#     min_temp_region = preprocess_image(min_temp_region)

#     # Extract the top and bottom temperature values
#     temp_max = extract_temp(max_temp_region)
#     temp_min = extract_temp(min_temp_region)
    
#     return temp_max, temp_min

# # Initialize the global min and max temperatures
# global_min_temp = float('inf')
# global_max_temp = float('-inf')

# # Load all images
# for dir_path in dirs:
#     for filename in os.listdir(dir_path):
#         filepath = os.path.join(dir_path, filename)
#         if os.path.exists(filepath):  # making sure the file exists
#             # Load the image
#             image = cv2.imread(filepath)
#             temp_max, temp_min = preprocess_and_extract_temps(image)
#             print(f"{filename} temp_max:", temp_max)
#             print(f"{filename} temp_min:", temp_min)
#             # Update global min and max
#             if temp_max is not None:
#                 global_max_temp = max(global_max_temp, temp_max)
#             if temp_min is not None:
#                 global_min_temp = min(global_min_temp, temp_min)

# # print the global max and min temperatures
# print("Global max temp:", global_max_temp)
# print("Global min temp:", global_min_temp)

# # Process all images
# for dir_path in dirs:
#     for filename in os.listdir(dir_path):
#         filepath = os.path.join(dir_path, filename)
#         if os.path.exists(filepath):  # making sure the file exists
#             # Load the image
#             image = cv2.imread(filepath)
        
#             # Extract the local min and max temperature values
#             temp_max, temp_min = preprocess_and_extract_temps(image)

#             # Define the region that contains the color scale
#             scale_region = image[scale_region_bounds[1]:scale_region_bounds[3], 
#                                  scale_region_bounds[0]:scale_region_bounds[2]]
            
#             flattened_colors = scale_region.reshape(-1, scale_region.shape[-1])
#             unique_colors = flattened_colors[::-1]

#             color_to_temp = {}
#             color_positions = np.linspace(temp_min, temp_max, len(unique_colors))

#             for i, color in enumerate(unique_colors):
#                 color_to_temp[tuple(color)] = color_positions[i]
                
#             temp_image = np.zeros_like(image, dtype=float)

#             for i in range(image.shape[0]):
#                 for j in range(image.shape[1]):
#                     color = tuple(image[i, j])
#                     if color not in color_to_temp:
#                         closest_color = find_closest_color(color, unique_colors)
#                         temp_image[i, j] = color_to_temp.get(tuple(closest_color), 0)
#                     else:
#                         temp_image[i, j] = color_to_temp.get(color, 0)

#             # Rescale local temperatures to the global range
#             local_range = [temp_min, temp_max]
#             global_range = [global_min_temp, global_max_temp]
#             scale_factor = (global_range[1] - global_range[0]) / (local_range[1] - local_range[0])
#             temp_image_rescaled = global_range[0] + (temp_image - local_range[0]) * scale_factor

#             # Normalize rescaled temperatures to 0-255 and convert to grayscale
#             min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
#             normalized_temp_image = min_max_scaler.fit_transform(temp_image_rescaled.reshape(-1, 1)).reshape(temp_image_rescaled.shape)
#             grayscale_image = normalized_temp_image.astype(np.uint8)

#             # Save the grayscale image
#             # Directory to save cropped images
#             output_dir = 'datasets/data/thermal_normalized_/'
#             os.makedirs(output_dir, exist_ok=True)
#             output_path = os.path.join(output_dir, filename)
#             cv2.imwrite(output_path, grayscale_image)

#             # Display   
#             image_with_boxes = image.copy()
#             plt.imshow(image_with_boxes)
#             plt.show()

#             plt.imshow(scale_region)
#             plt.show()



#             print("temp_max:", temp_max)

#             print("temp_min:", temp_min)

#             plt.imshow(grayscale_image, cmap='gray')
#             plt.show()


import os
import numpy as np
from PIL import Image

os.makedirs('./thermal', exist_ok=True)
os.makedirs('./visible', exist_ok=True)
# Reset the sequence number for IR RGB
sequence_num = 1
# Directories containing images to crop
input_dirs = ['./day/thermal/', './night/thermal/']

# Loop through each directory
for input_dir in input_dirs:

    # Get the list of image file names
    filenames = os.listdir(input_dir)


    # Loop through each file
    for filename in filenames:
        # Construct full file path
        filepath = os.path.join(input_dir, filename)
        
        # Open image
        img = Image.open(filepath).convert('L')

        # Crop the image
        width, height = img.size
        left = width * 0.01
        top = height * 0.095
        right = width * 0.9
        bottom = height * 0.83
        crop_img = img.crop((left, top, right, bottom))

        # Normalize the image
        img_array = np.array(crop_img)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
        img = Image.fromarray(img_array.astype('uint8'))

        # Construct output file path with the sequence number
        output_filename = f"{sequence_num:05d}.jpg"  # sequence numbers as 00001, 00002, etc.
        output_filepath = os.path.join('./thermal', output_filename)
        
        # Save the cropped image
        img.save(output_filepath)

        # Increment the sequence number
        sequence_num += 1



# Reset the sequence number for IR RGB
sequence_num = 1
# Directories containing images to crop
input_dirs = ['./day/visible/', './night/visible/']

# Loop through each directory
for input_dir in input_dirs:
    if input_dir =='./day/visible/':
        print("day")
    # Get the list of image file names
    filenames = os.listdir(input_dir)
    filenames.sort()  # Sort the filenames for consistent sequencing


    # Loop through each file
    for filename in filenames:
        # Construct full file path
        filepath = os.path.join(input_dir, filename)
        
        # Open image
        img = Image.open(filepath)
        # Construct output file path with the sequence number
        output_filename = f"{sequence_num:05d}.jpg"  # sequence numbers as 00001, 00002, etc.
        output_filepath = os.path.join('./visible', output_filename)
        
        # Save the cropped image
        img.save(output_filepath)

        # Increment the sequence number
        sequence_num += 1

