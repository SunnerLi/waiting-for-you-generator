import torch.nn as nn
import torch

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

        # Ensure the highest iteration of pretrained model
        self.getPretrainedHighestIter()

    
