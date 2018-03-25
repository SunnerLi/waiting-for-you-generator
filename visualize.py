import torchvision_sunner.transforms as sunnertransforms
import numpy as np
import torch
import cv2
import os

def transpose(variable):
    return variable.transpose(1, 2).transpose(2, 3)

def saveTransformResult(output_dir, img_name, 
                        wait_img_var, real_img_var,                 # original image
                        fake_real_var, fake_wait_var,               # transfered space
                        recon_wait_var, recon_real_var):            # reconstructed image
    """
        Visualize for 4 image
    """
    # Reverse normalization
    unNorm = sunnertransforms.UnNormalize()
    wait_img_var, real_img_var = unNorm(wait_img_var), unNorm(real_img_var)
    fake_real_var, fake_wait_var = unNorm(fake_real_var), unNorm(fake_wait_var)
    recon_wait_var, recon_real_var = unNorm(recon_wait_var), unNorm(recon_real_var)

    # BCHW -> BHWC
    wait_img_var, real_img_var = transpose(wait_img_var), transpose(real_img_var)
    fake_real_var, fake_wait_var = transpose(fake_real_var), transpose(fake_wait_var)
    recon_wait_var, recon_real_var = transpose(recon_wait_var), transpose(recon_real_var)

    # Check if output folder exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Merge
    result_img = None
    for idx in range(2):
        # wait -> real -> wait
        wait_img = wait_img_var.data.cpu().numpy()[idx]
        fake_real = fake_real_var.data.cpu().numpy()[idx]
        recon_wait = recon_wait_var.data.cpu().numpy()[idx]
        row = np.concatenate((np.concatenate((wait_img, fake_real), axis=1), recon_wait), axis=1)
        result_img = row if result_img is None else np.concatenate((result_img, row), axis=0)
    for idx in range(2):
        # real -> wait -> real
        real_img = real_img_var.data.cpu().numpy()[idx]
        fake_wait = fake_wait_var.data.cpu().numpy()[idx]
        recon_real = recon_real_var.data.cpu().numpy()[idx]
        row = np.concatenate((np.concatenate((real_img, fake_wait), axis=1), recon_real), axis=1)
        result_img = np.concatenate((result_img, row), axis=0)

    result_img = (result_img).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, img_name), result_img)
