import os
import torch

class BaseModel(object):
    def initialize(self, isTrain, save_dir = '.', use_gpu = True):
        self.isTrain = isTrain
        self.Tensor = torch.cuda.FloatTensor if use_gpu else torch.Tensor
        self.save_dir = save_dir
        self.use_gpu = use_gpu
        
        # Ensure if the save_dir is exist
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def set_input(self, _input):
        self.input = _input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        if not os.path.exists(os.path.join(self.save_dir, 'model')):
            os.mkdir(os.path.join(self.save_dir, 'model'))
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(os.path.join(self.save_dir, 'model'), save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if self.use_gpu and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(os.path.join(self.save_dir, 'model'), save_filename)
        if os.path.exists(save_path):
            network.load_state_dict(torch.load(save_path))
        else:
            print('pre-train model ( ', save_path, ') not exist.')

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)