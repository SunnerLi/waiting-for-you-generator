import _init_paths
from torch.utils import data as Data
from torch.autograd import Variable
from torchvision import transforms
from model import CustomCycleGAN
import torchvision_extend.transforms as my_transforms
import torchvision_extend.data as mData
import numpy as np
import torch

VERBOSE_PEROID = 2

if __name__ == '__main__':
    # Generate data
    dataset = mData.WaitTensorDataset(
        real_img_root_dir = './train2014/', 
        wait_img_root_dir = './wait/', 
        transform = transforms.Compose([
                my_transforms.Rescale((160, 320)),
                my_transforms.ToTensor(),
                my_transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
            ])
    )
    loader = mData.WaitDataLoader(dataset = dataset, batch_size=32, shuffle=True, num_workers = 2)

    # Train (Usual cycleGAN)
    model = CustomCycleGAN(adopt_custom = False, model_folder = './cycleGAN_model/', output_folder = './cycleGAN_output/')
    model.cuda()
    model.train(loader, verbose_period=VERBOSE_PEROID, epoch=3)
    model.storeCSV(csv_name = './cycleGAN_output.csv')
    model.plot(period_times = VERBOSE_PEROID, title = 'CycleGAN', fig_name = './cycleGAN_output.png')

    # Train (Mask cycleGAN)
    model = CustomCycleGAN(adopt_custom = True, model_folder = './mask_cycleGAN_model/', output_folder = './mask_cycleGAN_output/')
    model.cuda()
    model.train(loader, verbose_period=VERBOSE_PEROID, epoch=3)
    model.storeCSV(csv_name = './mask_cycleGAN_output.csv')
    model.plot(period_times = VERBOSE_PEROID, title = 'Mask-CycleGAN', fig_name = './mask_cycleGAN_output.png')    