import _init_paths
from torch.autograd import Variable
from cycle_gan import CycleGAN
from util import saveImg
from skimage import io
import torchvision.transforms.functional as F
import numpy as np
import argparse
import torch
import cv2
import os

"""
    This code will transfer the real-world image into waiting-for-you latend space
"""

def transferImage(img, model_path = './model/', output_folder = './', result_img_name = 'result.png'):
    # Resize as the same as the training size
    img = np.asarray(img)
    img = cv2.resize(img, (320, 160))

    # Make as variable and normalized
    img = torch.from_numpy(img).float()
    img = img.transpose(1, 2).transpose(0, 1)

    # Normalize
    img_tensor_list = []
    for t in img:
        img_tensor_list.append(torch.div(t, 255).mul_(2).add_(-1))
    img = torch.stack(img_tensor_list, 0)

    # Form the input tensor
    img = img.numpy()[np.newaxis, :]
    img = torch.from_numpy(img).float()

    # Transfer
    model = CycleGAN(model_path, \
        isTrain = False, \
        input_channel = 3, \
        output_channel = 3, \
        base_filter = 32, \
        batch_size = 1, \
        use_dropout = False, \
        use_gpu = True)
    data = {'A': img}
    model.set_input(data, 'AtoB')
    model.test()
    visual = model.get_current_visuals()

    # Save
    saveImg(visual, output_folder, result_img_name)

if __name__ == '__main__':
    # Deal with parameter 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='COCO_train2014_000000000009.jpg', dest='input', help='The name of input image')
    parser.add_argument('--model', type=str, default='./model/', dest='model', help='The path of model')
    parser.add_argument('--output', type=str, default='result.png', dest='output', help='The name of output image')
    args = parser.parse_args()
    IMAGE_NAME = args.input
    model_path = args.model
    OUTPUT_NAME = args.output

    # work
    img = io.imread(IMAGE_NAME)
    transferImage(img, model_path = model_path, result_img_name = OUTPUT_NAME)