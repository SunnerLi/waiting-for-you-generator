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
        return {
            'real': torch.from_numpy(real_img),
            'wait': torch.from_numpy(wait_img)
        }