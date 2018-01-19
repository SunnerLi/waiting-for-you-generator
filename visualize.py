import numpy as np
import torch
import cv2
import os

def saveTransformResult(output_dir, img_name, real_variable, wait_vabiable):
    """
        Visualize for 4 image
    """
    # BCHW -> BHWC
    real_variable = real_variable.transpose(1, 2).transpose(2, 3)
    wait_vabiable = wait_vabiable.transpose(1, 2).transpose(2, 3)

    # Check if output folder exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Merge
    result_img = None
    for idx in range(min(4, real_variable.size(0))):
        real_img = real_variable.data.cpu().numpy()[idx]
        wait_img = wait_vabiable.data.cpu().numpy()[idx]
        column = np.concatenate((real_img, wait_img), axis=0)
        if result_img is None:
            result_img = column
        else:
            result_img = np.concatenate((result_img, column), axis=1)
    result_img = (result_img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, img_name), result_img)
