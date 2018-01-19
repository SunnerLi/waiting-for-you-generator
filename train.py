import _init_paths
import torchvision_extend.data as mData
from torch.utils import data as Data
from model import CustomCycleGAN
import numpy as np
import torch

def getPretrainedHighestIter(model_folder = './model'):
        import glob
        import os
        prev_model_name_list = glob.glob(os.path.join(model_folder, '*.pth.tar'))
        max_iter = 0
        for name in prev_model_name_list:
            postfix_iter = int(name.split('_')[-1].split('.')[0])
            if max_iter < postfix_iter:
                max_iter = postfix_iter
        return max_iter

def saveModel(self, model, iteration, model_folder = './model'):
    """
        Model name: wait_iter_100.pth.tar
    """
    import glob
    import os
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)      

    # Remove all previous model
    prev_model_name_list = glob.glob(os.path.join(model_folder, '*.pth.tar'))
    for name in prev_model_name_list:
        os.remove(name)

    # Save
    model_iter = self.max_iter + iteration
    model_name = 'wait_iter_' + str(model_iter) + '.pth.tar'
    torch.save(self.state_dict(), os.path.join(model_folder, model_name))

def loadModel(self, model_folder = './model'):
    import glob
    import os
    model_name = 'wait_iter_' + str(self.max_iter) + '.pth.tar'
    model_path = os.path.join(model_folder + model_name)
    if os.path.exists(model_path):
        torch.load(model_path)

if __name__ == '__main__':
    # Generate data
    real_img = np.random.random([2000, 3, 160, 320])
    wait_img = np.random.random([2000, 3, 160, 320])
    dataset = Data.TensorDataset(
        data_tensor = torch.from_numpy(real_img).float(),
        target_tensor = torch.from_numpy(wait_img).float()
    )
    loader = mData.WaitDataLoader(dataset = dataset, batch_size=32, shuffle=True)

    # Train
    model = CustomCycleGAN()
    model.cuda()
    model.train(loader, epoch=1)