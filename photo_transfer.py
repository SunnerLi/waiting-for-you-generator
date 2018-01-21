import _init_paths
from torch.autograd import Variable
from model import CustomCycleGAN
import torchvision.transforms.functional as F
import numpy as np
import argparse
import torch
import cv2
import os

"""
    This code will transfer the real-world image into waiting-for-you latend space
"""

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def Unnormalize(tensor, mean, std):
    """Inversed-normalize a tensor image with mean and standard deviation.

        This part is revised from official F.normalize implementation
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def transferImage(img, output_folder = './', result_img_name = 'result.png'):
    # Make as variable and normalized
    img = torch.from_numpy(img).float()
    img = img.transpose(1, 2).transpose(0, 1)
    mean_list = [127.5, 127.5, 127.5]
    std_list = [127.5, 127.5, 127.5]
    img = F.normalize(img, mean_list, std_list)
    img = img.numpy()[np.newaxis, :]
    img = Variable(torch.from_numpy(img).float()).cuda()

    # Transfer
    model = CustomCycleGAN(model_folder = './cycleGAN_model/')
    model.cuda()
    model.load()
    img = model(img)

    # Save
    img = img.data.cpu().numpy()[0]
    img = torch.from_numpy(img).float()
    img = Unnormalize(img, mean_list, std_list)
    img = img.transpose(0, 1).transpose(1, 2)
    img = img.numpy()
    img = (img).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, result_img_name), img)

if __name__ == '__main__':
    # Deal with parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='COCO_train2014_000000000009.jpg', dest='input', help='The name of input image')
    parser.add_argument('--output', type=str, default='result.png', dest='output', help='The name of output image')
    args = parser.parse_args()
    IMAGE_NAME = args.input
    OUTPUT_NAME = args.output

    # work
    img = cv2.imread(IMAGE_NAME)
    transferImage(img, result_img_name = OUTPUT_NAME)