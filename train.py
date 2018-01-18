from torch.utils import data as Data
from model import CustomCycleGAN
import numpy as np
import torch

if __name__ == '__main__':
    # Generate data
    real_img = np.random.random([2000, 3, 160, 320])
    wait_img = np.random.random([2000, 3, 160, 320])
    dataset = Data.TensorDataset(
        data_tensor = torch.from_numpy(real_img).float(),
        target_tensor = torch.from_numpy(wait_img).float()
    )
    loader = Data.DataLoader(dataset = dataset, batch_size=32, shuffle=True)

    # Train
    model = CustomCycleGAN()
    model.cuda()
    model.train(loader, epoch=1)