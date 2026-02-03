import pandas as pd
import numpy as np
import cv2
import os

# Load FER-2013 CSV
df = pd.read_csv("fer2013.csv")

# Emotion mapping (only required emotions)
emotion_map = {
    0: "angry",
    3: "happy",
    4: "sad",
    5: "surprise"
}

# Create required folders
for split in ["train", "test"]:
    for emotion in emotion_map.values():
        os.makedirs(f"dataset/{split}/{emotion}", exist_ok=True)

img_count = 0

# Loop through dataset
for index, row in df.iterrows():

    emotion = row["emotion"]

    # Skip unwanted emotions
    if emotion not in emotion_map:
        continue

    # Convert pixel string to image
    pixels = row["pixels"].split()
    pixels = np.array(pixels, dtype="uint8")
    image = pixels.reshape(48, 48)

    # Decide train or test
    if row["Usage"] == "Training":
        split = "train"
    else:
        split = "test"

    emotion_name = emotion_map[emotion]

    # Save image
    file_path = f"dataset/{split}/{emotion_name}/{emotion_name}_{img_count}.jpg"
    cv2.imwrite(file_path, image)

    img_count += 1

print("✅ Dataset successfully created!")
print(f"📸 Total images saved: {img_count}")
