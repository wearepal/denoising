"""Transform the Huawei dataset"""
from pathlib import Path
import csv
from random import seed
from torchvision import transforms
from tqdm import tqdm
from loader import HuaweiDataset


def main(patches, patch_size, old_path=None, new_path=None):
    rand_crop = transforms.RandomCrop(size=patch_size)
    transform = transforms.Compose([rand_crop])

    if old_path is not None:
        root_path = Path(old_path).resolve()
    else:
        # Use the default path, assumes repo was cloned alongside a `data` folder
        root_path = Path(__file__).resolve().parent.parent.parent / "data"
    if not root_path.is_dir():
        raise ValueError("No valid data directory specified")

    if new_path is not None:
        transformed_path = Path(new_path).resolve()
    else:
        # Adds transformed folder to existing data dir
        transformed_path = root_path / "transformed"

    transformed_path.mkdir()
    data = HuaweiDataset(root_dir=root_path)
    iso_data = [("folder_idx", "iso")]
    dataset_info_writer = _get_dataset_info_writer(transformed_path, patches)
    for image_no, pair in enumerate(tqdm(data)):
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
        dataset_info_writer(noisy_path, clean_path, pair['iso'])
    with open(Path(transformed_path).resolve() / "info.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(iso_data)


def _get_dataset_info_writer(transformed_path, patches):
    base_path = transformed_path.resolve()
    csv_path = base_path / "dataset.csv"
    with csv_path.open('w') as csv_file:  # reset CSV file by writing the heading
        csv.writer(csv_file).writerows([("noisy_path", "clean_path", "iso")])

    def _write_dataset_info(noisy_path, clean_path, iso):
        # relative paths
        noisy_rel = noisy_path.resolve().relative_to(base_path)
        clean_rel = clean_path.resolve().relative_to(base_path)
        dataset_info = [(noisy_rel / f"{i}.png", clean_rel / f"{i}.png", iso)
                        for i in range(patches)]
        with csv_path.open('a') as csv_file:  # append to the CSV file
            csv.writer(csv_file).writerows(dataset_info)

    return _write_dataset_info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate transformed images")
    parser.add_argument("crops", metavar="c", type=int, help="Number of random crops per image")
    parser.add_argument("size", metavar="s", type=int, help="Dimension of cropped image")
    parser.add_argument("--data-path", dest="old_path", default=None, help="Data folder path")
    parser.add_argument("--new-path", dest="new_path", default=None,
                        help="Folder to store transformed data, must not exist yet")
    args = parser.parse_args()
    main(args.crops, args.size, old_path=args.old_path, new_path=args.new_path)
