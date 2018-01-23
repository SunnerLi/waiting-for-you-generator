import torchvision.transforms as transforms
import transforms as my_transforms
import torch.utils.data as Data
import numpy as np
import torch
import glob
import cv2
import os

class ImageDataset(Data.Dataset):
    def __init__(self, root_list, use_cv = True, transform = None):
        import glob
        import os
        self.root_list = root_list
        self.folder_list = []
        self.use_cv = use_cv
        self.transform = transform
        if not isinstance(root_list, (list, int)) and not isinstance(root_list, (tuple, int)):
            raise Exception('The type of 1st parameter should be tuple or list')
        for root in root_list:
            if os.path.exists(root):
                img_list = glob.glob(os.path.join(root, '*'))
                channel_format_desc = "cv" if use_cv else "sklearn"
                print("[ ImageDataset ] path: %s \t image number: %d \t channel format: %s" 
                    % (root, len(img_list), channel_format_desc)
                )
                self.folder_list.append(img_list)
            else:
                raise Exception("root folder not found...")
        
    def __len__(self):
        import math
        import os
        if len(self.root_list) == 0:
            return 0
        else:
            num = math.inf
        for path in self.root_list:
            _list = glob.glob(os.path.join(path, '*'))
            num = len(_list) if len(_list) < num else num
        return num

    def __getitem__(self, idx):
        return_list = []
        if self.use_cv:
            import cv2
            for i in range(len(self.folder_list)):
                img = cv2.imread(self.folder_list[i][idx])
                if self.transform:
                    img = self.transform(img)
                return_list.append(img)
            return return_list

class ImageLoader(Data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers = 1):
        super(ImageLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)
        self.dataset = dataset
        self.iter_num = self.getIterNumber()

    def getIterNumber(self):       
        return round(len(self.dataset) / self.batch_size)
        # return round(len(self.dataset.data_tensor) / self.batch_size)

if __name__ == '__main__':
    dataset = ImageDataset(
        root_list = ['./train2014'],
        transform = transforms.Compose([
            my_transforms.Rescale((160, 320)),
            my_transforms.ToTensor(),
            my_transforms.Transpose(my_transforms.BHWC2BCHW),
            my_transforms.Normalize([127., 127., 127.], [127., 127., 127.])
        ]) 
    )
    loader = ImageLoader(dataset, batch_size=32)
    loader_iter = iter(loader)
    for i in range(loader.getIterNumber()):
        batch_img1 = torch.stack(loader_iter.next()[0], 0)
        print(batch_img1.size())

        batch_img1 = my_transforms.tensor2Numpy(batch_img1, transform = transforms.Compose([
            my_transforms.UnNormalize([127., 127., 127.], [127., 127., 127.]),
            my_transforms.Transpose(my_transforms.BCHW2BHWC),
        ]))
        
        
        batch_img = batch_img1[0].astype(np.uint8)
        cv2.imshow('0', batch_img)
        cv2.waitKey()
        break