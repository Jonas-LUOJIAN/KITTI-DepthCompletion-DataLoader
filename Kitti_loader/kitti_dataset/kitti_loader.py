import re
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
'''
attention:
    There is mistake in 2011_09_26_drive_0009_sync/proj_depth 4 files were
    left out 177-180 .png. Hence these files were also deleted in rgb
'''


class Kitti_dataset(object):
    def __init__(self, kitti_root):
        self.train_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.val_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.selected_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.test_files = {'img': [], 'lidar_in': []}
        #
        self.dataset_path = kitti_root
        self.left_side_selection = 'image_02'
        self.right_side_selection = 'image_03'
        self.depth_keyword = 'proj_depth'
        self.rgb_keyword = 'rgb'
        self.date_selection = '2011_09_26'

    def get_paths(self):
        # train and validation dirs
        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):
                if re.search(self.depth_keyword, root):
                    self.train_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files
                                                        if re.search('velodyne_raw', root)
                                                        and re.search('train', root)]))
                    self.val_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files
                                                              if re.search('velodyne_raw', root)
                                                              and re.search('val', root)]))
                    self.train_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                                          if re.search('groundtruth', root)
                                                          and re.search('train', root)]))
                    self.val_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                                        if re.search('groundtruth', root)
                                                        and re.search('val', root)]))
                if re.search(self.rgb_keyword, root):
                    self.train_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                               if re.search('train', root)
                                                               and (re.search('image_02', root) or re.search('image_03', root))
                                                               and re.search('data', root)])[5:-5])
                    self.val_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                            if re.search('val', root)
                                                            and (re.search('image_02', root) or re.search('image_03', root))
                                                            and re.search('data', root)])[5:-5])
    def get_selected_paths(self, selection):
        files = []
        for file in sorted(os.listdir(os.path.join(self.dataset_path, selection))):
            files.append(os.path.join(self.dataset_path, os.path.join(selection, file)))
        return files

    def read_kitti_from_local(self):
        path_to_val_sel = 'depth_selection/val_selection_cropped'
        path_to_test = 'depth_selection/test_depth_completion_anonymous'
        self.get_paths()
        self.selected_paths['lidar_in'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'velodyne_raw'))
        self.selected_paths['gt'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'groundtruth_depth'))
        self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))
        self.test_files['lidar_in'] = self.get_selected_paths(os.path.join(path_to_test, 'velodyne_raw'))
        self.test_files['img'] = self.get_selected_paths(os.path.join(path_to_test, 'image'))
        print(len(self.train_paths['lidar_in']))
        print(len(self.train_paths['img']))
        print(len(self.train_paths['gt']))
        print(len(self.val_paths['lidar_in']))
        print(len(self.val_paths['img']))
        print(len(self.val_paths['gt']))
        print(len(self.test_files['lidar_in']))
        print(len(self.test_files['img']))


# dataset_path = "D:/kitti"
# test = Kitti_dataset(dataset_path)
# print(test.train_paths)
# test.read_kitti_from_local()
# depth = read_depth(test.train_paths['lidar_in'][78])
# print(depth.shape)
# params = test.compute_mean_std()
# mu_std = params[0:2]
# max_lst = params[-1]
# print('Means and std equals {} and {}'.format(*mu_std))
# plt.hist(max_lst, bins='auto')
# plt.title('Histogram for max depth')
# plt.show()
# depth = read_depth(test.train_paths['lidar_in'][78])
# print(depth.shape)
# print(test.train_paths['img'][78])
# print(test.train_paths['lidar_in'][78])
# print(test.train_paths['gt'][78])
# print(len(test.train_paths['img']))
# print(len(test.train_paths['lidar_in']))
# print(len(test.train_paths['gt']))
# print(test.val_paths['img'][0])
# print(test.val_paths['lidar_in'][0])
# print(test.val_paths['gt'][0])
# print(len(test.val_paths['img']))
# print(len(test.val_paths['lidar_in']))
# print(len(test.val_paths['gt']))

class Kitti_Dataset(Dataset):

    """Dataset with labeled lanes"""

    def __init__(self, dataset_type, transform=False, num_samples=None):

        # Constants
        self.dataset_type = dataset_type
        self.is_transform = transform
        # Names
        self.img_name = 'img'
        self.lidar_name = 'lidar_in'
        self.gt_name = 'gt'

        # Define random sampler
        self.num_samples = num_samples
        self.ToTenser = transforms.ToTensor()

    def __len__(self):
        """
        Conventional len method
        """
        len_liar = len(self.dataset_type['lidar_in'])
        len_img = len(self.dataset_type['img'])
        len_gt = len(self.dataset_type['gt'])
        return len_liar

    def __getitem__(self, idx):
        """
        Args: idx (int): Index of images to make batch
        Returns (tuple): Sample of velodyne data and ground truth.
        """
        sparse_depth_name = self.dataset_type[self.lidar_name][idx]
        gt_name = self.dataset_type[self.gt_name][idx]
        img_name = self.dataset_type[self.img_name][idx]
        print(sparse_depth_name)
        print(gt_name)
        print(img_name)
        with open(sparse_depth_name, 'rb') as f:
            sparse_depth = Image.open(f)
            sparse_depth = F.crop(sparse_depth, 0, 0, 256, 1216)
            f.close()
        with open(gt_name, 'rb') as f:
            gt = Image.open(f)
            gt = F.crop(gt, 0, 0, 256, 1216)
            f.close()

        with open(img_name, 'rb') as f:
            img = (Image.open(f).convert('RGB'))
            img = F.crop(img, 0, 0, 256, 1216)
            f.close()
        if self.is_transform:
            return self.apply_transforms(sparse_depth, img, gt)
        else:
            return sparse_depth, img, gt

    def apply_transforms(self, sparse_depth, image, g_truth):
        # if do flip in depth,image,ground_truth
        do_flip = np.random.uniform(0.0, 1.0) > 0.5
        if do_flip:
            sparse_depth, image, g_truth = F.hflip(sparse_depth), F.hflip(image), F.hflip(g_truth)

        _sparse_depth_np, _image_np, _g_truth_np = self.ToTenser(sparse_depth).float(), \
                                        self.ToTenser(image).float(), \
                                        self.ToTenser(g_truth).float()

        return _sparse_depth_np, _image_np, _g_truth_np


# dataset_path = "D:/kitti"
# kitti = Kitti_dataset(dataset_path)
# print(kitti.train_paths)
# kitti.read_kitti_from_local()
#
# kitti_train = Kitti_Dataset(kitti.train_paths, transform=True)
# kitti_val = Kitti_Dataset(kitti.val_paths)
# kitti_selection = Kitti_Dataset(kitti.selected_paths)
# kitti_test = Kitti_Dataset(kitti.test_files)
#
# s, i, g = kitti_train.__getitem__(0)
# print(kitti_train.__len__())
# # print(i.size)
# print(i.shape)
# print(i.max(), i.min())
# # (1242, 375)
