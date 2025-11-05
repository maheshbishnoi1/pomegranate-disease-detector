import os
import shutil
import random

# ✅ Step 1: Define source folder (your dataset path)
SOURCE_DIR = r"C:\Users\MAHESH BISHNOI\OneDrive\Desktop\pythonproject\pomegranate-disease-detector\Pomegranate Diseases Dataset"

# ✅ Step 2: Define output folders for organized data
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Create train/test directories if not exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# ✅ Step 3: Split each disease category into train/test
for category in os.listdir(SOURCE_DIR):
    src_folder = os.path.join(SOURCE_DIR, category)

    if not os.path.isdir(src_folder):
        continue  # Skip non-folder files

    print(f"Processing category: {category}")

    # Collect all image files
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle for randomness
    random.shuffle(images)

    # Split 80% train / 20% test
    split_index = int(0.8 * len(images))
    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    # Create category folders
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

    # Copy images to train folder
    for img in train_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(TRAIN_DIR, category, img))

    # Copy images to test folder
    for img in test_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(TEST_DIR, category, img))

print("\n✅ Dataset organized successfully into:")
print(f"   {TRAIN_DIR} (80%)")
print(f"   {TEST_DIR} (20%)")
