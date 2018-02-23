from torch.autograd import Variable
from skimage import io
import numpy as np
import random
import torch
import os

def tensor2im(image_tensor, imtype=np.uint8):
    """
        Converts a Tensor into a Numpy array
        |imtype|: the desired type of the converted numpy array
    """
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def saveImg(im_arr, img_dir, img_name):
    """
        Save CycleGAN transformation result
        Arg:    im_arr      - The orderdict object of images
                img_name    - The name of result image
    """
    img_dir = os.path.join(img_dir, 'photo_output')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    result_img = None

    # Form the rows
    row_list = []
    if im_arr['real_A'] is not None:
        row = im_arr['real_A']
        row = np.concatenate((row, im_arr['fake_B']), axis=1)
        row = np.concatenate((row, im_arr['rec_A']), axis=1)
        row_list.append(row)

    # Form the row B
    if im_arr['real_B'] is not None:
        row = im_arr['real_B']
        row = np.concatenate((row, im_arr['fake_A']), axis=1)
        row = np.concatenate((row, im_arr['rec_B']), axis=1)
        row_list.append(row)

    if len(row_list):
        for i, row in enumerate(row_list):
            if i == 0:
                result_img = row
            else:
                result_img = np.concatenate((result_img, row), axis=0)
    io.imsave(os.path.join(img_dir, img_name), result_img)

class ImagePool():
    def __init__(self):
        self.pool_size = 50
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images