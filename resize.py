import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Set your input and output directories
INPUT_DIR = "data/"
OUTPUT_DIR = "resized_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define resizing transform
resize_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
])

# Traverse nested folders
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            input_path = os.path.join(root, file)
            
            # Get relative path (e.g., slide_001/tile_001.jpeg)
            relative_path = os.path.relpath(input_path, INPUT_DIR)
            output_path = os.path.join(OUTPUT_DIR, relative_path)
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Resize and save
            img = Image.open(input_path).convert("RGB")
            img_resized = resize_transform(img)
            img_resized.save(output_path)
