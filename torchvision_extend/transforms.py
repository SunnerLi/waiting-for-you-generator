import torchvision.transforms.functional as F
import torch
import cv2

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

        # -------------------------------------------------------
        # Reverse the order 
        # cv2       order: [width, height]
        # pytorch   order: [hdight, width]
        # -------------------------------------------------------
        if len(self.output_size) == 2:
            self.output_size = tuple(reversed(list(self.output_size)))

    def __call__(self, sample):
        real_img, wait_img = sample['real'], sample['wait']
        real_img = cv2.resize(real_img, self.output_size)
        wait_img = cv2.resize(wait_img, self.output_size)
        return {'real': real_img, 'wait': wait_img}

class ToTensor(object):
    def __call__(self, sample):
        real_img, wait_img = sample['real'], sample['wait']
        real_img = torch.from_numpy(real_img)
        wait_img = torch.from_numpy(wait_img)
        real_img = real_img.transpose(1, 2).transpose(0, 1)
        wait_img = wait_img.transpose(1, 2).transpose(0, 1)
        return {'real': real_img, 'wait': wait_img}

class Normalize(object):
    """
        Normalize toward two tensor
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        real_img, wait_img = sample['real'], sample['wait']
        real_img = real_img.float() if type(real_img) == torch.ByteTensor else real_img
        wait_img = wait_img.float() if type(wait_img) == torch.ByteTensor else wait_img       
        real_img = F.normalize(real_img, self.mean, self.std)
        wait_img = F.normalize(wait_img, self.mean, self.std)
        return {'real': real_img, 'wait': wait_img}

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.transpose(0, 1)
        # for slice_tensor in tensor:
        result_tensor = []
        for t, m, s in zip(tensor, self.mean, self.std):
            t = torch.mul(t, s)
            t = t.add_(m)
            result_tensor.append(t)
            # The normalize code -> t.sub_(m).div_(s)
        tensor = torch.stack(result_tensor, 0)
        tensor = tensor.transpose(0, 1)
        return tensor
