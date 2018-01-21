from module import Generator, Discriminator
from visualize import saveTransformResult
from matplotlib import pyplot as plt
from torch.autograd import Variable
from mask import getMaskVariable
from torch.optim import Adam
from gan import GAN
import torchvision_extend.transforms as my_transforms
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

class CustomCycleGAN(GAN):
    # Define cycle index
    INDEX_wait_to_real = 0
    INDEX_real_to_wait = 1

    # Hyper-parameter
    lambda_real_wait_real = 10.
    lambda_wait_real_wait = 100.
    lambda_identity = 0.5

    def __init__(self, input_channel = 3, base_filter = 16, adopt_custom = False, model_folder = './model/', output_folder = './output/'):
        super(CustomCycleGAN, self).__init__()
        # Assign parameters
        self.adopt_custom = adopt_custom
        self.model_folder = model_folder
        self.output_folder = output_folder

        # Compose network components
        print('-' * 20 + ' Build generator ' + '-' * 20)
        self.wait_to_real_generator = Generator(input_channel, base_filter)
        self.real_to_wait_generator = Generator(input_channel, base_filter)
        print(self.wait_to_real_generator)
        print(self.real_to_wait_generator)
        print('-' * 20 + ' Build discriminator ' + '-' * 20)
        self.real_discriminator = Discriminator(input_channel, base_filter)
        self.wait_discriminator = Discriminator(input_channel, base_filter)
        print(self.real_discriminator)
        print(self.wait_discriminator)

        self.wait_to_real_generator_optimizer = Adam(self.wait_to_real_generator.parameters())
        self.real_to_wait_generator_optimizer = Adam(self.real_to_wait_generator.parameters())
        self.real_discriminator_optimizer = Adam(self.real_discriminator.parameters())
        self.wait_discriminator_optimizer = Adam(self.wait_discriminator.parameters())

    def forward(self, wait_variable, real_variable, cycle_idx):
        # 1st cycle
        if cycle_idx == self.INDEX_wait_to_real:  
            real_img = self.wait_to_real_generator(wait_variable)
            wait_img = self.real_to_wait_generator(real_img)
            fake_logits = self.real_discriminator(real_img)
            true_logtis = self.real_discriminator(real_variable)
            return real_img, wait_img, true_logtis, fake_logits
        # 2nd cycle
        elif cycle_idx == self.INDEX_real_to_wait:               
            wait_img = self.real_to_wait_generator(real_variable)
            real_img = self.wait_to_real_generator(wait_img)
            fake_logits = self.wait_discriminator(wait_img)
            true_logtis = self.wait_discriminator(wait_variable)
            return wait_img, real_img, true_logtis, fake_logits
        else:
            print("Invalid cycle index...")
            exit()

    def train(self, loader, epoch=1, verbose_period=2):
        # Load pretrained model
        self.load()
        l1_loss_fn = torch.nn.L1Loss()
        self.wait_to_real_generator_loss_list = []
        self.real_to_wait_generator_loss_list = []
        self.real_discriminator_loss_list = []
        self.wait_discriminator_loss_list = []
        
        for i in range(epoch):
            for j, (batch_real_img, batch_wait_img) in enumerate(loader):
                batch_real_img, batch_wait_img = self.prepareBatchData(batch_real_img, batch_wait_img)

                # Three things should be revised
                # 1. The definition of identity mapping are wrong
                # 2. Maybe we should revise the both generator at the same time (not seperately)
                # 3. Revise plot function, plot three latend space image

                # ------------------------------------------------------------------------
                # 1st cycle 
                # ------------------------------------------------------------------------
                # loss compute
                wait_latent_img, restore_img, true_logtis, fake_logits = self.forward(batch_wait_img, batch_real_img, self.INDEX_wait_to_real)
                self.discriminator_loss = torch.sum((true_logtis - 1) ** 2 + fake_logits ** 2) / 2.
                # usual cycleGAN
                if self.adopt_custom == False:      
                    self.generator_loss = torch.sum((fake_logits - 1) ** 2) / 2. + \
                        self.lambda_wait_real_wait * l1_loss_fn(restore_img, batch_wait_img) + \
                        self.lambda_identity * self.lambda_wait_real_wait * l1_loss_fn(wait_latent_img, batch_wait_img)                 # Identity mapping loss
                # mask cycleGAN
                else:                               
                    mask_var = getMaskVariable(batch_wait_img, use_cuda = True)
                    filted_batch_wait_img = torch.mul(batch_wait_img, mask_var)
                    filted_wait_latent_img = torch.mul(wait_latent_img, mask_var)
                    filted_restore_img = torch.mul(restore_img, mask_var)
                    self.generator_loss = torch.sum((fake_logits - 1) ** 2) / 2. + \
                        self.lambda_wait_real_wait * l1_loss_fn(filted_restore_img, filted_batch_wait_img) + \
                        self.lambda_identity * self.lambda_wait_real_wait * l1_loss_fn(filted_wait_latent_img, filted_batch_wait_img)   # Identity mapping loss

                # 1st cycle parameter update
                self.wait_to_real_generator_optimizer.zero_grad()
                self.real_to_wait_generator_optimizer.zero_grad()
                self.real_discriminator_optimizer.zero_grad()

                self.discriminator_loss.backward(retain_graph=True)
                self.generator_loss.backward()
                discriminator_loss_1st = self.discriminator_loss.data.cpu().numpy()[0]
                generator_loss_1st = self.generator_loss.data.cpu().numpy()[0]

                self.wait_to_real_generator_optimizer.step()
                self.real_to_wait_generator_optimizer.step()
                self.real_discriminator_optimizer.step()

                # ------------------------------------------------------------------------
                # 2nd cycle 
                # ------------------------------------------------------------------------
                # loss compute
                real_latent_img, restore_img, true_logtis, fake_logits = self.forward(batch_wait_img, batch_real_img, self.INDEX_real_to_wait)
                self.discriminator_loss = torch.sum((true_logtis - 1) ** 2 + fake_logits ** 2) / 2.
                self.generator_loss = torch.sum((fake_logits - 1) ** 2) / 2. + \
                    self.lambda_real_wait_real * l1_loss_fn(restore_img, batch_real_img) + \
                    self.lambda_identity * self.lambda_real_wait_real * l1_loss_fn(real_latent_img, batch_real_img)

                # 2nd cycle parameter update
                self.wait_to_real_generator_optimizer.zero_grad()
                self.real_to_wait_generator_optimizer.zero_grad()
                self.wait_discriminator_optimizer.zero_grad()

                self.discriminator_loss.backward(retain_graph=True)
                self.generator_loss.backward()
                discriminator_loss_2nd = self.discriminator_loss.data.cpu().numpy()[0]
                generator_loss_2nd = self.generator_loss.data.cpu().numpy()[0]

                self.real_to_wait_generator_optimizer.step()
                self.wait_to_real_generator_optimizer.step()
                self.wait_discriminator_optimizer.step()

                # ------------------------------------------------------------------------
                # Record
                # ------------------------------------------------------------------------
                if j % verbose_period == 0:
                    print('epoch: ', i, '\titer: ', j, 
                        '\t< 1st cycle >\tgen loss: ', generator_loss_1st, '\tdis loss: ', discriminator_loss_1st,
                        '\t< 2nd cycle >\tgen loss: ', generator_loss_2nd, '\tdis loss: ', discriminator_loss_2nd
                    )
                    self.wait_to_real_generator_loss_list.append(generator_loss_1st)
                    self.real_discriminator_loss_list.append(discriminator_loss_1st)
                    self.real_to_wait_generator_loss_list.append(generator_loss_2nd)
                    self.wait_discriminator_loss_list.append(discriminator_loss_2nd)
                    output_img_name = str(i) + '_' + str(j) + '.png'
                    saveTransformResult(self.output_folder, output_img_name, batch_real_img, batch_wait_img, real_latent_img, wait_latent_img)
                    self.save(i * loader.iter_num + j, model_folder = self.model_folder)
                    break

    def load(self):
        self.real_to_wait_generator = self.loadModel(self.real_to_wait_generator, model_name = 'real_to_wait_generator')
        self.wait_to_real_generator = self.loadModel(self.wait_to_real_generator, model_name = 'wait_to_real_generator')
        self.real_discriminator = self.loadModel(self.real_discriminator, model_name = 'real_discriminator')
        self.wait_discriminator = self.loadModel(self.wait_discriminator, model_name = 'wait_discriminator')

    def save(self, idx, model_folder = './model'):
        import glob
        import os

        # Remove all previous model
        prev_model_name_list = glob.glob(os.path.join(model_folder, '*.pth.tar'))
        for name in prev_model_name_list:
            os.remove(name)
        
        # Save
        self.saveModel(self.real_to_wait_generator, idx, model_name = 'real_to_wait_generator')
        self.saveModel(self.wait_to_real_generator, idx, model_name = 'wait_to_real_generator')
        self.saveModel(self.real_discriminator, idx, model_name = 'real_discriminator')
        self.saveModel(self.wait_discriminator, idx, model_name = 'wait_discriminator')
        
    def storeCSV(self, csv_name = './output.csv'):
        loss_table = {
            'wait_to_real_generator_loss': self.wait_to_real_generator_loss_list,
            'real_to_wait_generator_loss': self.real_to_wait_generator_loss_list,
            'wait_discriminator_loss_list': self.wait_discriminator_loss_list,
            'real_discriminator_loss_list': self.real_discriminator_loss_list
        }
        df = pd.DataFrame.from_dict(loss_table)
        df.to_csv(csv_name)

    def plot(self, period_times = 2, title = '?', fig_name = './output.png'):
        plt.plot(np.arange(len(self.wait_to_real_generator_loss_list)), self.wait_to_real_generator_loss_list, '-o', label='wait_to_real_generator_loss')
        plt.plot(np.arange(len(self.real_to_wait_generator_loss_list)), self.real_to_wait_generator_loss_list, '-o', label='real_to_wait_generator_loss')
        plt.plot(np.arange(len(self.wait_discriminator_loss_list)), self.wait_discriminator_loss_list, '-o', label='wait_discriminator_loss_list')
        plt.plot(np.arange(len(self.real_discriminator_loss_list)), self.real_discriminator_loss_list, '-o', label='real_discriminator_loss_list')
        plt.legend()
        plt.title(title)
        plt.savefig(fig_name)
        plt.gca().clear()
