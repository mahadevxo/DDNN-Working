import os
from pathlib import Path
from collections import defaultdict

def verify_places365_structure(dataset_root='places365'):
    required_splits = ['train', 'val']
    valid_exts = {'.jpg', '.jpeg', '.png'}

    for split in required_splits:
        split_path = Path(dataset_root) / split
        if not split_path.exists():
            print(f"❌ Missing split: {split_path}")
            continue

        print(f"✅ Found split: {split_path}")
        class_folders = [f for f in split_path.iterdir() if f.is_dir()]
        print(f"📁 {split} - Found {len(class_folders)} class folders.")

        if len(class_folders) != 365:
            print(f"⚠️ Expected 365 classes, found {len(class_folders)}.")

        empty_folders = []
        ext_errors = defaultdict(list)

        for class_folder in class_folders:
            images = list(class_folder.glob("*"))
            if not images:
                empty_folders.append(class_folder.name)
            for img in images:
                if img.suffix.lower() not in valid_exts:
                    ext_errors[class_folder.name].append(img.name)

        if empty_folders:
            print(f"🚫 Empty class folders in '{split}':", empty_folders)

        if ext_errors:
            print("⚠️ Non-image files or bad extensions found:")
            for cls, bad_files in ext_errors.items():
                print(f"  - {cls}: {len(bad_files)} file(s)")

        print()

    print("✅ Structure check complete.")

if __name__ == "__main__":
    verify_places365_structure("places365")