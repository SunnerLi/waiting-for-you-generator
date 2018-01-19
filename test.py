import _init_paths
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision_extend.transforms as my_transforms
import torchvision_extend.data as Data
import numpy as np
import torch
import cv2

dataset = Data.WaitTensorDataset(
    real_img_root_dir = './train2014/', 
    wait_img_root_dir = './wait/', 
    transform = transforms.Compose([
        my_transforms.Rescale((160, 320)),
        my_transforms.ToTensor(),
        my_transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
    ])
)
loader = Data.WaitDataLoader(dataset, batch_size=32, shuffle=True)

for batch_real_tensor, batch_wait_tensor in loader:
    # Reverse normalization
    unNorm = my_transforms.UnNormalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
    batch_real_tensor = unNorm(batch_real_tensor)
    batch_wait_tensor = unNorm(batch_wait_tensor)

    print(torch.max(batch_real_tensor))
    
    # Transfer into HWC format before we show
    real_img = batch_real_tensor.transpose(1, 2).transpose(2, 3).numpy()[0].astype(np.uint8)
    wait_img = batch_wait_tensor.transpose(1, 2).transpose(2, 3).numpy()[0].astype(np.uint8)

    cv2.imshow('real', real_img)
    cv2.imshow('wait', wait_img)
    cv2.waitKey()
    exit()