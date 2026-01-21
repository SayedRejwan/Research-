import os
import numpy as np
from PIL import Image

def create_dummy_images(base_path, num_images=20):
    classes = ["Aedes aegypti", "Culex quinquefasciatus"]
    for cls in classes:
        # Train
        path = os.path.join(base_path, "train", cls)
        os.makedirs(path, exist_ok=True)
        print(f"Generating {num_images} images in {path}...")
        for i in range(num_images):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(path, f"dummy_{i}.jpg"))
            
        # Val
        path = os.path.join(base_path, "val", cls)
        os.makedirs(path, exist_ok=True)
        print(f"Generating {5} images in {path}...")
        for i in range(5):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(path, f"dummy_{i}.jpg"))

if __name__ == "__main__":
    create_dummy_images("dataset")
