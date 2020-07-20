import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils


class CamVid(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.


    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'trainannot'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'valannot'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'testannot'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.pil_loader,
                 preload=True):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            datadir = self.train_folder
            labeldir = self.train_lbl_folder
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            datadir = self.val_folder
            labeldir = self.val_lbl_folder
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            datadir = self.test_folder
            labeldir = self.test_lbl_folder
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
            
        self.data_files = utils.get_files(
            os.path.join(root_dir, datadir),
            extension_filter=self.img_extension)

        self.label_files = utils.get_files(
            os.path.join(root_dir, labeldir),
            extension_filter=self.img_extension)
        
        self.preload=preload
        if preload:
            self.data = [self.transform_pair(*self.loader(*path)) 
                for path in zip(self.data_files, self.label_files)]
        
    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.preload:
            return self.data[index]
        
        data_path, label_path = self.data_files[index], self.label_files[
            index]

        img, label = self.loader(data_path, label_path)
        return self.transform_pair(img, label)

    def transform_pair(self, img, label):
        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_files)
            
    