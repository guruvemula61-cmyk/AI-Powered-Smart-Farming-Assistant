import os
import shutil


SOURCE_DIR = "data/raw/PlantVillage"
TARGET_DIR = "data/plantvillage"


CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy",
]


def prepare_dataset():
    os.makedirs(TARGET_DIR, exist_ok=True)

    for class_name in CLASSES:
        source = os.path.join(SOURCE_DIR, class_name)
        target = os.path.join(TARGET_DIR, class_name)

        if not os.path.exists(source):
            print(f"Missing: {class_name}")
            continue

        os.makedirs(target, exist_ok=True)

        for filename in os.listdir(source):
            source_file = os.path.join(source, filename)
            target_file = os.path.join(target, filename)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_file)

        print(f"Prepared: {class_name}")

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    prepare_dataset()
