import _init_paths
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision_extend.transforms as my_transforms
import torchvision_extend.dataset as dataset
import cv2

dataset = dataset.WaitTensorDataset(
    real_img_root_dir = './train2014/', 
    wait_img_root_dir = './wait/', 
    transform = transforms.Compose([
        my_transforms.Rescale((160, 320)),
        my_transforms.ToTensor()
    ])
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch_real_tensor, batch_wait_tensor in loader:
    real_img = batch_real_tensor.numpy()[0]
    wait_img = batch_wait_tensor.numpy()[0]
    cv2.imshow('real', real_img)
    cv2.imshow('wait', wait_img)
    cv2.waitKey()
    exit()