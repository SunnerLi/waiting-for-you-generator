from torchvision import transforms
import torchvision.transforms.functional as F
import torch

# -----------------------------------------------------------------------------------
# Normalize definition in torchvision.transforms.transforms.py
# -----------------------------------------------------------------------------------
class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return my_normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# -----------------------------------------------------------------------------------
# Fundtion definition in torchvision.transforms.functional.py
# -----------------------------------------------------------------------------------
def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def my_normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    See ``Normalize`` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    result_tensor = []
    for t, m, s in zip(tensor, mean, std):
        result_tensor.append(t.sub_(m).div_(s))
    return torch.stack(result_tensor, 0)


if __name__ == '__main__':
    # Generate random image which range is (0 ~ 255)
    img = torch.rand(3, 160, 320) * 127.5 + 127.5

    # Define parameter I want to normalize
    my_mean = [127.5, 127.5, 127.5]
    my_std = [127.5, 127.5, 127.5]

    # Adopt official Normalization
    print("max value in tensor before adopt official: ", torch.max(img), img.size())
    norm_op = transforms.Normalize(my_mean, my_std)
    img = norm_op(img)
    print("max value in tensor after  adopt official: ", torch.max(img), img.size())

    # Adopt revision Normalization
    print("max value in tensor before adopt revision: ", torch.max(img), img.size())
    norm_op = Normalize(my_mean, my_std)
    img = norm_op(img)
    print("max value in tensor after  adopt revision: ", torch.max(img), img.size())