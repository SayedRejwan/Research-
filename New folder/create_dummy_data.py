import os
import numpy as np
from PIL import Image

DATA_ROOT = "dataset"
CLASSES = ["larva_positive", "larva_negative"]
SPLITS = ["train", "test"]
IMG_SIZE = (224, 224)
SAMPLES_PER_CLASS = {"train": 50, "test": 20} # Small number for quick demo

def create_dummy_dataset():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
    
    for split in SPLITS:
        for cls in CLASSES:
            dir_path = os.path.join(DATA_ROOT, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            num_samples = SAMPLES_PER_CLASS[split]
            print(f"Generating {num_samples} images for {split}/{cls}...")
            
            for i in range(num_samples):
                # Generate random noise image
                # Make positive class slightly different (higher mean) to allow some learning
                if cls == "larva_positive":
                    data = np.random.randint(50, 255, (IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
                else:
                    data = np.random.randint(0, 200, (IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
                
                img = Image.fromarray(data)
                img.save(os.path.join(dir_path, f"img_{i}.jpg"))

if __name__ == "__main__":
    create_dummy_dataset()
    print("Dummy dataset created successfully at ./dataset")
