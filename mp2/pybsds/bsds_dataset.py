import os
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.color import rgb2grey
from skimage.io import imread
from scipy.io import loadmat


class BSDSDataset (object):
    """
    BSDS dataset wrapper

    Given the path to the root of the BSDS dataset, this class provides
    methods for loading images, ground truths and evaluating predictions

    Attribtes:

    bsds_path - the root path of the dataset
    data_path - the path of the data directory within the root
    images_path - the path of the images directory within the data dir
    gt_path - the path of the groundTruth directory within the data dir
    train_sample_names - a list of names of training images
    val_sample_names - a list of names of validation images
    test_sample_names - a list of names of test images
    """
    def __init__(self, bsds_path):
        """
        Constructor

        :param bsds_path: the path to the root of the BSDS dataset
        """
        self.bsds_path = bsds_path
        self.data_path = os.path.join(bsds_path, 'BSDS500', 'data')
        self.images_path = os.path.join(self.data_path, 'images')
        self.gt_path = os.path.join(self.data_path, 'groundTruth')

        self.train_sample_names = self._sample_names(self.images_path, 'train')
        self.val_sample_names = self._sample_names(self.images_path, 'val')
        self.test_sample_names = self._sample_names(self.images_path, 'test')

    @staticmethod
    def _sample_names(dir, subset):
        names = []
        files = os.listdir(os.path.join(dir, subset))
        for fn in files:
            dir, filename = os.path.split(fn)
            name, ext = os.path.splitext(filename)
            if ext.lower() == '.jpg':
                names.append(os.path.join(subset, name))
        return names

    def read_image(self, name):
        """
        Load the image identified by the sample name (you can get the names
        from the `train_sample_names`, `val_sample_names` and
        `test_sample_names` attributes)
        :param name: the sample name
        :return: a (H,W,3) array containing the image, scaled to range [0,1]
        """
        path = os.path.join(self.images_path, name + '.jpg')
        return img_as_float(imread(path))

    def get_image_shape(self, name):
        """
        Get the shape of the image identified by the sample name (you can
        get the names from the `train_sample_names`, `val_sample_names` and
        `test_sample_names` attributes)
        :param name: the sample name
        :return: a tuple of the form `(height, width, channels)`
        """
        path = os.path.join(self.images_path, name + '.jpg')
        img = Image.open(path)
        return img.height, img.width, 3

    def ground_truth_mat(self, name):
        """
        Load the ground truth Matlab file identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: the `groundTruth` entry from the Matlab file
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_ground_truth_mat(path)

    def segmentations(self, name):
        """
        Load the ground truth segmentations identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_segmentations(path)

    def boundaries(self, name):
        """
        Load the ground truth boundaries identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_boundaries(path)

    @staticmethod
    def load_ground_truth_mat(path):
        """
        Load the ground truth Matlab file at the specified path
        and return the `groundTruth` entry.
        :param path: path
        :return: the 'groundTruth' entry from the Matlab file
        """
        gt = loadmat(path)
        return gt['groundTruth']

    @staticmethod
    def load_segmentations(path):
        """
        Load the ground truth segmentations from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        gt = BSDSDataset.load_ground_truth_mat(path)
        num_gts = gt.shape[1]
        return [gt[0,i]['Segmentation'][0,0].astype(np.int32) for i in range(num_gts)]

    @staticmethod
    def load_boundaries(path):
        """
        Load the ground truth boundaries from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        gt = BSDSDataset.load_ground_truth_mat(path)
        num_gts = gt.shape[1]
        return [gt[0,i]['Boundaries'][0,0] for i in range(num_gts)]
