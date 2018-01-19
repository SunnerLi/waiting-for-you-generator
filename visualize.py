import torchvision_extend.transforms as my_transforms
import numpy as np
import torch
import cv2
import os

def saveTransformResult(output_dir, img_name, real_variable, wait_variable, wait_latent_img, real_latent_img):
    """
        Visualize for 4 image
    """
    # Reverse normalization
    unNorm = my_transforms.UnNormalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
    real_variable = unNorm(real_variable)
    wait_latent_img = unNorm(wait_latent_img)
    wait_variable = unNorm(wait_variable)
    real_latent_img = unNorm(real_latent_img)

    # BCHW -> BHWC
    real_variable = real_variable.transpose(1, 2).transpose(2, 3)
    wait_latent_img = wait_latent_img.transpose(1, 2).transpose(2, 3)
    wait_variable = wait_variable.transpose(1, 2).transpose(2, 3)
    real_latent_img = real_latent_img.transpose(1, 2).transpose(2, 3)

    # Check if output folder exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Merge
    result_img1 = None
    for idx in range(min(4, real_variable.size(0))):
        real_img = real_variable.data.cpu().numpy()[idx]
        wait_img = wait_latent_img.data.cpu().numpy()[idx]

        column = np.concatenate((real_img, wait_img), axis=0)
        if result_img1 is None:
            result_img1 = column
        else:
            result_img1 = np.concatenate((result_img1, column), axis=1)
    result_img2 = None
    for idx in range(min(4, real_variable.size(0))):
        wait_img = wait_variable.data.cpu().numpy()[idx]
        real_img = real_latent_img.data.cpu().numpy()[idx]

        column = np.concatenate((wait_img, real_img), axis=0)
        if result_img2 is None:
            result_img2 = column
        else:
            result_img2 = np.concatenate((result_img2, column), axis=1)
    result_img = np.concatenate((result_img1, result_img2), axis=0)
    result_img = (result_img).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, img_name), result_img)
