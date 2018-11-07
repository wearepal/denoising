from loader import HuaweiDataset
from pathlib import Path
import csv
from sys import argv
from random import seed
from torchvision import transforms

def main(patches, patch_size, new_path=None):
    rand_crop = transforms.RandomCrop(size=patch_size)
    transform = transforms.Compose([rand_crop])

    if new_path is not None:
        root_path = Path(root_dir).resolve()
    else:
        # Use the default path, assumes repo was cloned alongside a `data` folder
        root_path = Path(__file__).resolve().parent.parent.parent / "data"
    if not root_path.is_dir():
        raise ValueError("No valid top directory specified")
    transformed_path = root_path / "transformed"
    transformed_path.mkdir()
    data = HuaweiDataset()
    iso_data = [("folder_idx", "iso")]
    for image_no, pair in enumerate(data):
        iso_data.append((image_no, pair['iso']))
        clean_path = transformed_path / str(image_no) / "clean"
        clean_path.mkdir(parents=True)
        noisy_path = transformed_path / str(image_no) / "noisy"
        noisy_path.mkdir()
        seed(image_no)
        for i in range(patches):
            clean = transform(pair['clean'])
            clean.save(clean_path / f"{i}.png")
        seed(image_no)
        for i in range(patches):
            noisy = transform(pair['noisy'])
            noisy.save(noisy_path / f"{i}.png")
    with open(Path(transformed_path).resolve() / "info.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(iso_data)



if __name__ == "__main__":
    main(1000, 64)