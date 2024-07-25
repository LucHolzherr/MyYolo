import torch.utils.data.dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image

from src.utils import parse_xml
from src.utils import copy_content_to_folder


class VOCDataSet(torch.utils.data.Dataset):
    """Dataset of VOCData 2007 and 2012
    """

    def __init__(self, max_num_samples: int = None):
        super(VOCDataSet, self).__init__()
        self.num_classes = 20
        # local path where data is saved
        rel_path = Path(f"data/VOCdevkit/{'AllData'}")
        self.image_paths = Path(f"{rel_path}/JPEGImages").glob("*.jpg")
        self.image_paths = [path for path in self.image_paths]
        self.label_paths = Path(f"{rel_path}/Annotations").glob("*.xml")
        self.label_paths = [path for path in self.label_paths]
        # download dataset if not already downloaded
        if not rel_path.exists():
            # download dataset from 2007 and 2012 if not on disk
            path_2007 = Path(f'data/VOCdevkit/VOC2007')
            if not path_2007.exists():
                datasets.VOCDetection(
                    root='data', year='2007', download=True, transform=transforms.ToTensor())
            path_2012 = Path(f'data/VOCdevkit/VOC2012')
            if not path_2012.exists():
                datasets.VOCDetection(
                    root='data', year='2012', download=True, transform=transforms.ToTensor())

            # put all the data into the same folder
            dir_2007 = 'data/VOCdevkit/VOC2007/'
            dir_2012 = 'data/VOCdevkit/VOC2012/'
            new_dir = 'data/VOCdevkit/AllData/'
            for dir in [dir_2007, dir_2012]:
                print(f"Copying Content of {dir} to Alldata")
                copy_content_to_folder(
                    path_dir1=f'{dir}/',
                    path_dir2=f'{new_dir}/',
                    white_list=['Annotations', 'JPEGImages']
                )

        self.num_samples = len(
            self.image_paths) if max_num_samples is None else max_num_samples

        self.labels_map = {
            0: 'aeroplane',
            1: 'bicycle',
            2: 'bird',
            3: 'boat',
            4: 'bottle',
            5: 'bus',
            6: 'car',
            7: 'cat',
            8: 'chair',
            9: 'cow',
            10: 'diningtable',
            11: 'dog',
            12: 'horse',
            13: 'motorbike',
            14: 'person',
            15: 'pottedplant',
            16: 'sheep',
            17: 'sofa',
            18: 'train',
            19: 'tvmonitor'
        }
        self.inv_labels_map = {value: key for key,
                               value in self.labels_map.items()}

    def __getitem__(self, idx):
        # load image and label dictionary
        img = Image.open(self.image_paths[idx])
        img = transforms.ToTensor()(img)
        label_og = parse_xml(self.label_paths[idx])
        if not isinstance(label_og['annotation']['object'], list):
            label_og['annotation']['object'] = [
                label_og['annotation']['object']]

        # convert into standard label form which can be used for different datasets
        label = {'object': []}
        for obj in label_og['annotation']['object']:
            label['object'].append(
                {'class': obj['name'], 'bndbox': obj['bndbox']})
        return img, label

    def __len__(self):
        return self.num_samples
