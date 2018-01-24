import numpy as np
import torch

mask = torch.from_numpy(np.asarray([[0, 0], [1, 0]])).float()
img = torch.from_numpy(np.asarray([[0.5, 0.6], [0.7, 0.8]])).float()
mask = torch.ones_like(mask) - mask
img = torch.mul(mask, img)
print(img)