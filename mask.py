from torch.autograd import Variable
import numpy as np
import torch
import cv2

def getMask(img, threshold = 230):
    # Generate make
    r_idx = img[:, :, 0] > threshold
    g_idx = img[:, :, 1] > threshold
    b_idx = img[:, :, 2] > threshold
    filter_idx = r_idx * g_idx * b_idx
    mask = np.copy(img)
    mask[:, :, 0] = img[:, :, 0] * filter_idx
    mask[:, :, 1] = img[:, :, 1] * filter_idx
    mask[:, :, 2] = img[:, :, 2] * filter_idx
    
    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel)
    return np.ones_like(mask) - mask

def getMaskVariable(img_variable, use_cuda = False):
    """
        img_variable type   : torch.variable
        return type         : torch.variable
    """
    mask_result = []
    for img_variable_slice in img_variable:
        img_arr = img_variable_slice.data.cpu().numpy()
        mask = getMask(img_arr)
        mask_result.append(torch.from_numpy(mask))
    mask_result = Variable(torch.stack(mask_result, 0))    
    mask_result = mask_result.cuda() if use_cuda else mask_result
    return mask_result

if __name__ == '__main__':
    img_var = Variable(torch.rand(32, 3, 160, 320))
    mask_var = getMaskVariable(img_var, use_cuda = False)
