from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class FRDEEPN(data.Dataset):
    """`FRDEEP-N <https://github.com/HongmingTang060313/FR-DEEP/>`_Dataset

    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``htru1-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = 'NVSS_PNG_dataset'
    url = "http://www.jb.man.ac.uk/research/ascaife/NVSS_PNG_dataset.tar.gz"
    filename = "NVSS_PNG_dataset.tar.gz"
    tgz_md5 = '2584ed1e174ea71f581d0e0d6f32ef38'
    train_list = [
                  ['data_batch_1', '3a2a15d88756ba61c796378fc8574540'],
                  ['data_batch_2', '6a04e3985397e1f67f0ad42153dca64e'],
                  ['data_batch_3', 'd852c8200f3bbb63beacf31f3e954f9a'],
                  ['data_batch_4', 'a5739996ca44a1a1841f2d0e6b844dd6'],
                  ['data_batch_5', '8e2fdb3f60bf7541ca135fc8e2407f7a'],
                  ['data_batch_6', '9e5a82500bd9742f8fefe412ada95336'],
                  ['data_batch_7', 'f66af7795265fbe24376f669200412c4'],
                  ['data_batch_8', '75982afc09bf480ecc521acdb39cbe46'],
                  ['data_batch_9', '72aee306fef9acee21a0e5537bb681e4'],
                  ['data_batch_10', '7a039ce8062a533b23b401a612c5f9b7'],
                  ['data_batch_11', 'c0013314098c96ca4c7c20c0f17abcd3'],
                  ]

    test_list = [
                 ['test_batch', '39fd167b9a7df12cee1ef9a804f9fa86'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': '655493bdee948954f3939727b3f9e735',
                }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img,(150,150))
        img = Image.fromarray(img,mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class FRDEEPF(data.Dataset):
    """`FRDEEP-F <https://github.com/HongmingTang060313/FR-DEEP/>`_Dataset

    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``htru1-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = 'FIRST_PNG_dataset'
    url = "http://www.jb.man.ac.uk/research/ascaife/FIRST_PNG_dataset.tar.gz"
    filename = "FIRST_PNG_dataset.tar.gz"
    tgz_md5 = '2f39461e6c62fb45289559915106013a'
    train_list = [
                  ['data_batch_1', 'f34da44757c7fa3f6e6cd3d0839a4634'],
                  ['data_batch_2', 'f56cda0d9a99305fee2bad7de0560f95'],
                  ['data_batch_3', '93265dd849331af4e1b092f74b06450b'],
                  ['data_batch_4', '0de8f4c18b775251f4e553e2990cd446'],
                  ['data_batch_5', 'c6aa87400a1be6007da7cfcefd2c3e5c'],
                  ['data_batch_6', 'cebd3fdea93abbc048a3a4d5e58528e0'],
                  ['data_batch_7', '49497445e9380f157e78cf8d74fca1eb'],
                  ['data_batch_8', '88e298eed2d87bbdddad83fef1482723'],
                  ['data_batch_9', '8c40117dbf4d456e63a8a665b245aa63'],
                  ['data_batch_10', 'f24d110cc5811ba4651630b9ee9b2989'],
                  ['data_batch_11', 'b843dc3b7f48606235029f135d41c85e'],
                  ]

    test_list = [
                 ['test_batch', '4e06889b1e7713deb46e62887eb37727'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': '655493bdee948954f3939727b3f9e735',
                }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img,(150,150))
        img = Image.fromarray(img,mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
