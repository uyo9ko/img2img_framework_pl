import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import albumentations as A
from PIL import Image
import numpy as np

# create transformation pipeline
transform = A.Compose([
    A.Resize(256, 256),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize()
],additional_targets={'image0': 'image'})


class MyCustomDataset(Dataset):
    def __init__(self, root_dir, is_train, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform

        # read data file paths
        if is_train:
            data_file = 'train.txt'
        else:
            data_file = 'test.txt'
        with open(os.path.join(root_dir, data_file), 'r') as file:
            self.data_list = file.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # read input image
        image_name = self.data_list[index].strip()
        input_image_path = os.path.join(self.root_dir,'raw-890',image_name)
        input_image = np.array(Image.open(input_image_path))

        # read ground truth image
        gt_image_path = os.path.join(self.root_dir, 'reference-890', image_name)
        gt_image = np.array(Image.open(gt_image_path))

        # apply transformation pipeline
        data = self.transform(image = input_image, image0 = gt_image)
        input_image, gt_image = data['image'], data['image0']
        input_image = input_image.transpose((2, 0, 1))
        gt_image = gt_image.transpose((2, 0, 1))


        return input_image, gt_image, image_name

class MyDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # define training transformation pipeline
        self.train_transform = A.Compose([
            A.Resize(256, 256),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize()
        ],additional_targets={'image0': 'image'})

        # define validation transformation pipeline
        self.val_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize()
        ],additional_targets={'image0': 'image'})

    def setup(self, stage=None):
        self.train_dataset = MyCustomDataset(self.root_dir, True, transform=self.train_transform)
        self.test_dataset = MyCustomDataset(self.root_dir, False, transform=self.val_transform)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


