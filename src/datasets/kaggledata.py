import torch.utils.data.dataset
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

from src.utils import parse_xml


class KaggleDataSet(torch.utils.data.Dataset):
    """Kaggle dataset from here: https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection/data
    """

    def __init__(self, max_num_samples: int = None):
        super(KaggleDataSet, self).__init__()
        self.num_classes = 2
        # local path where data is saved
        rel_path = Path(f"data/kaggle_data/")
        self.image_paths = Path(f"{rel_path}/images").glob("*.png")
        self.image_paths = [path for path in self.image_paths]
        self.label_paths = Path(f"{rel_path}/annotations").glob("*.xml")
        self.label_paths = [path for path in self.label_paths]
        # download dataset if not already downloaded
        if not rel_path.exists():
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                dataset_url = 'andrewmvd/dog-and-cat-detection'
                destination = rel_path
                api.dataset_download_files(
                    dataset_url, path=destination, unzip=True)
            except:
                error_msg = f"Data not present at {rel_path} and the Automatic kaggle donwload failed, make sure kaggle package is installed and your API credentials exist."
                raise ValueError(error_msg)

        self.num_samples = len(
            self.image_paths) if max_num_samples is None else max_num_samples

        self.labels_map = {
            0: 'cat',
            1: 'dog',
        }
        self.inv_labels_map = {value: key for key,
                               value in self.labels_map.items()}

    def __getitem__(self, idx):
        # load image and label dictionary
        img = Image.open(self.image_paths[idx])
        # Check the number of channels
        if img.mode == 'RGBA':
            # Convert RGBA to RGB by discarding the alpha channel
            img = img.convert('RGB')
        elif img.mode == 'L':
            # If the image is grayscale, convert it to RGB
            img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        label_og = parse_xml(self.label_paths[idx])

        # convert into standard label form which can be used for different datasets
        label_obj = label_og['annotation']['object']
        if not isinstance(label_obj, list):
            label_obj = [label_obj]

        label = {'object': [{
                            'class': label_obj[idx]['name'],
                            'bndbox': label_obj[idx]['bndbox']
                            }
                            for idx in range(len(label_obj))
                            ]
                 }

        return img, label

    def __len__(self):
        return self.num_samples
