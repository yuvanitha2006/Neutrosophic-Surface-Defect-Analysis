import cv2
import os
import pandas as pd
import numpy as np

# 🔹 Dataset path (DO NOT CHANGE unless folder name is different)
DATASET_PATH = "../Magnetic-Tile-Defect"


output = []

# Folder to label mapping
label_map = {
    "MT_Free": "normal",
    "MT_Blowhole": "defect",
    "MT_Break": "defect",
    "MT_Crack": "defect",
    "MT_Fray": "defect"
}

for folder_name, label in label_map.items():
    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue
    img_folder = os.path.join(folder_path, "Imgs")

    if not os.path.exists(img_folder):
        print(f"❌ Image folder not found: {img_folder}")
        continue

    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (128, 128))

        texture_score = np.std(img) / 255
        mean_intensity = np.mean(img) / 255

        Truth = texture_score * 100
        Falsity = mean_intensity * 100
        Indeterminacy = abs(texture_score - mean_intensity) * 100

        output.append([
            img_name,
            label,
            round(Truth, 2),
            round(Indeterminacy, 2),
            round(Falsity, 2)
        ])

    

# Create DataFrame
df = pd.DataFrame(
    output,
    columns=["Image_Name", "Label", "Truth", "Indeterminacy", "Falsity"]
)

# Save CSV
df.to_csv("image_features.csv", index=False)

print("✅ image_features.csv created successfully with neutrosophic values")
