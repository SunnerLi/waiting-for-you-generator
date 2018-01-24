from torch.autograd import Variable
import torch.nn as nn
import torch

class GAN(nn.Module):
    def __init__(self, model_folder):
        super(GAN, self).__init__()

        # Ensure the highest iteration of pretrained model
        self.getPretrainedHighestIter(model_folder)

    def getPretrainedHighestIter(self, model_folder = './model'):
        import glob
        import os
        prev_model_name_list = glob.glob(os.path.join(model_folder, '*.pth.tar'))
        self.max_iter = 0
        for name in prev_model_name_list:
            postfix_iter = int(name.split('_')[-1].split('.')[0])
            if self.max_iter < postfix_iter:
                self.max_iter = postfix_iter

    def prepareBatchData(self, batch_real, batch_wait):
        batch_real = Variable(batch_real).cuda()
        batch_wait = Variable(batch_wait).cuda()
        return batch_real, batch_wait

    def saveModel(self, model, iteration, model_folder = './model', model_name = 'generator'):
        """
            Model name: wait_iter_100.pth.tar
        """
        import glob
        import os
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)      

        # Save
        model_iter = self.max_iter + iteration
        model_name = 'wait_iter_' + model_name + '_' + str(model_iter) + '.pth.tar'
        torch.save(model, os.path.join(model_folder, model_name))

    def loadModel(self, model, model_folder = './model', model_name = 'generator'):
        import glob
        import os
        model_name = 'wait_iter_' + model_name + '_' + str(self.max_iter) + '.pth.tar'
        model_path = os.path.join(model_folder + model_name)
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=lambda storage, loc: storage).cuda()
        return model