import _init_paths
from torch.utils import data as Data
from torch.autograd import Variable
from torchvision import transforms
from model import CustomCycleGAN
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import numpy as np
import torch

VERBOSE_PEROID = 20

if __name__ == '__main__':
    # Generate data
    dataset = sunnerData.ImageDataset(
        root_list = ['./train2014', './wait'],
        sample_method = sunnerData.OVER_SAMPLING,
        transform = transforms.Compose([
            sunnertransforms.Rescale((160, 320)),
            sunnertransforms.ToTensor(),

            # BHWC -> BCHW
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize()
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=16, shuffle=True, num_workers = 2)

    # Train (Usual cycleGAN)
    model = CustomCycleGAN(adopt_mask = False, model_folder = './cycleGAN_model/', output_folder = './cycleGAN_output/')
    model.cuda()
    model.train(loader, verbose_period=VERBOSE_PEROID, epoch=1)
    model.storeCSV(csv_name = './cycleGAN_output.csv')
    model.plot(period_times = VERBOSE_PEROID, title = 'CycleGAN', fig_name = './cycleGAN_output.png')

    # Train (Mask cycleGAN)
    model = CustomCycleGAN(adopt_mask = True, model_folder = './mask_cycleGAN_model/', output_folder = './mask_cycleGAN_output/')
    model.cuda()
    model.train(loader, verbose_period=VERBOSE_PEROID, epoch=1)
    model.storeCSV(csv_name = './mask_cycleGAN_output.csv')
    model.plot(period_times = VERBOSE_PEROID, title = 'Mask-CycleGAN', fig_name = './mask_cycleGAN_output.png')