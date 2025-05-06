import json
from pathlib import Path
from PIL import Image

file_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.gif"]
root_dir = Path("dataset/SMOT4SB/private_test")
output_dir = Path("dataset/SMOT4SB/private_test_anns")
output_dir.mkdir(parents=True, exist_ok=True)

for folder in sorted(root_dir.iterdir()):
    if not folder.is_dir():
        continue

    folder_name = folder.name
    images_info = []

    for extension in file_extensions:
        for filename in sorted(folder.glob(extension)):
            try:
                image = Image.open(filename)
                width, height = image.size
                images_info.append([filename.name, int(height), int(width)])
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    images = [
        {
            "file_name": image_info[0],
            "height": image_info[1],
            "width": image_info[2],
            "id": i,
        }
        for i, image_info in enumerate(images_info, start=1)
    ]

    ann_file = {
        "categories": [{"id": 1, "name": "bird"}],
        "annotations": [],
        "images": images,
    }

    json_path = output_dir / f"{folder_name}.json"
    with open(json_path, 'w') as fp:
        json.dump(ann_file, fp)

    print(f"Saved: {json_path}")
