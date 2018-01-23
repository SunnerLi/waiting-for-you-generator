from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import torch
import cv2

class ImageDataset(Data.Dataset):
    def __init__(self, root, use_cv = True, transform = None):
        import glob
        import os
        if os.path.exists(root):
            self.root = root
            self.img_list = glob.glob(
                os.path.join(self.root, '*')
            ) 
            self.use_cv = use_cv
            self.transform = transform
            channel_format_desc = "cv" if use_cv else "sklearn"
            print("[ ImageDataset ] path: %s \t image number: %d \t channel format: %s" 
                % (self.root, len(self.img_list), channel_format_desc)
            )
        else:
            raise Exception("root folder not found...")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.use_cv:
            import cv2
            img = cv2.imread(self.img_list[idx])
            if self.transform:
                img = self.transform(img)
            return img

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

        # -------------------------------------------------------
        # Reverse the order 
        # cv2       order: [width, height]
        # pytorch   order: [hdight, width]
        # -------------------------------------------------------
        if len(self.output_size) == 2:
            self.output_size = tuple(reversed(list(self.output_size)))

    def __call__(self, sample):
        return cv2.resize(sample, self.output_size)
            
class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)

class Transpose(object):
    def __call__(self, sample):
        return sample.transpose(1, 2).transpose(0, 1)

if __name__ == '__main__':
    dataset = ImageDataset(root='train2014', transform = transforms.Compose([
        Rescale((160, 320)),
        ToTensor(),
        Transpose()
    ]))
    loader = Data.DataLoader(dataset, batch_size=32)
    loader = iter(loader)
    for i in range(10):
        batch_img = loader.next()
        print(np.shape(batch_img))