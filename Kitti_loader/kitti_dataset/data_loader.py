from kitti_dataset.kitti_loader import Kitti_dataset, Kitti_Dataset
from torch.utils.data import DataLoader
import random

# dataset_path = "D:/kitti"
# kitti = Kitti_dataset(dataset_path)
# print(kitti.train_paths)
# kitti.read_kitti_from_local()
#
# kitti_train = Kitti_Dataset(kitti.train_paths, transform=True)
# kitti_val = Kitti_Dataset(kitti.val_paths)
# kitti_selection = Kitti_Dataset(kitti.selected_paths)
# kitti_test = Kitti_Dataset(kitti.test_files)


def get_loader(kitti_train, kitti_val, kitti_selection, batch_size=5):
    """
    Define the different dataloaders for training and validation
    """


    train_loader = DataLoader(
        kitti_train, batch_size=batch_size, sampler=None,
        shuffle=True, num_workers=8,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        kitti_val, batch_size=batch_size,  sampler=None,
        shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True)
    val_selection_loader = DataLoader(
        kitti_selection, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True)
    return train_loader, val_loader, val_selection_loader

# train_loader, val_loader, val_selection_loader = get_loader(kitti_train, kitti_val, kitti_selection)
