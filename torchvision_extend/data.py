import torchvision.transforms as transforms
import transforms as my_transforms
import torch.utils.data as Data
import numpy as np
import torch
import glob
import cv2
import os

class WaitTensorDataset(Data.Dataset):
    def __init__(self, real_img_root_dir, wait_img_root_dir, transform = None):
        if not (os.path.exists(real_img_root_dir) and os.path.exists(wait_img_root_dir)):
            raise Exception("root_dir not exist in WaitTensorDataset...")
        self.real_img_root_dir = real_img_root_dir
        self.wait_img_root_dir = wait_img_root_dir
        self.transform = transform
        self.real_img_list = glob.glob(os.path.join(self.real_img_root_dir, "*.jpg"))
        self.wait_img_list = glob.glob(os.path.join(self.wait_img_root_dir, "*.jpg"))
        self.fill()

    def fill(self):
        """
            Over-sampling to make the data balance
        """
        if len(self.real_img_list) == len(self.wait_img_list):
            return
        max_num, min_num = len(self.real_img_list), len(self.wait_img_list)
        (max_num, min_num) = (max_num, min_num) if max_num > min_num else (min_num, max_num)
        if min_num >= max_num:
            raise Exception("index error (or maybe there're no image in the folder?) ...")
        img_num_diff = max_num - min_num
        random_idx = np.random.randint(low=0, high=min_num, size=img_num_diff)
        for i in range(len(random_idx)):
            if len(self.real_img_list) > len(self.wait_img_list):
                self.wait_img_list.append(self.wait_img_list[random_idx[i]])
            else:
                self.real_img_list.append(self.real_img_list[random_idx[i]])
        self.img_num = max_num

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        real_img_name = self.real_img_list[idx]
        wait_img_name = self.wait_img_list[idx]
        sample = {
            'real': cv2.imread(real_img_name),
            'wait': cv2.imread(wait_img_name)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample['real'], sample['wait']

class WaitDataLoader(Data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(WaitDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.dataset = dataset
        self.iter_num = self.getIterNumber()

    def getIterNumber(self):
        import glob
        import os
        # _list = glob.glob(os.path.join(self.dataset.real_img_root_dir, '*.jpg'))
        # return round(len(_list) / self.batch_size)
        return round(len(self.dataset.data_tensor) / self.batch_size)

if __name__ == '__main__':
    dataset = WaitTensorDataset(
        real_img_root_dir = '../train2014/', 
        wait_img_root_dir = '../wait/', 
        transform = transforms.Compose([
            my_transforms.Rescale((160, 320)),
            my_transforms.ToTensor()
        ])
    )
    real_img, wait_img = dataset[1]
    # cv2.imshow('1', real_img)
    # cv2.imshow('2', wait_img)
    # cv2.waitKey()
