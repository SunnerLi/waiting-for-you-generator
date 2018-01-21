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
import random
import torch

class CustomCycleGAN(GAN):
    # Hyper-parameter
    lambda_real = 10.
    lambda_wait = 5.
    lambda_identity = 0.5

    def __init__(self, input_channel = 3, base_filter = 32, adopt_mask = False, model_folder = './model/', output_folder = './output/'):
        super(CustomCycleGAN, self).__init__(model_folder)
        # Assign parameters
        self.adopt_mask = adopt_mask
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

        self.generator_optimizer = Adam(list(self.wait_to_real_generator.parameters()) + list(self.real_to_wait_generator.parameters()))
        self.discriminator_optimizer = Adam(list(self.wait_discriminator.parameters()) + list(self.real_discriminator.parameters()))

    def forward(self, x):
        return self.real_to_wait_generator(x)

    def train(self, loader, epoch=1, verbose_period=2):
        # Load pretrained model
        self.load(model_folder = self.model_folder)
        l1_loss_fn = torch.nn.L1Loss()
        self.generator_loss_list = []
        self.discriminator_loss_list = []
        
        for i in range(epoch):
            for j, (batch_real_img, batch_wait_img) in enumerate(loader):
                batch_real_img, batch_wait_img = self.prepareBatchData(batch_real_img, batch_wait_img)

                # ------------------------------------------------------------------------
                # Training discriminator 
                # ------------------------------------------------------------------------
                # True image
                self.discriminator_optimizer.zero_grad()
                true_logits = self.real_discriminator(batch_real_img)
                d1_loss = torch.mean((true_logits - 1) ** 2)
                true_logits = self.wait_discriminator(batch_wait_img)
                d2_loss = torch.mean((true_logits - 1) ** 2)
                d_loss = d1_loss + d2_loss
                d_loss_record = d_loss.data.cpu().numpy()[0]
                d_loss.backward()
                self.discriminator_optimizer.step()

                # Fake image
                self.discriminator_optimizer.zero_grad()
                fake_logits = self.real_discriminator(self.wait_to_real_generator(batch_wait_img))
                d1_loss = torch.mean(fake_logits ** 2)
                fake_logits = self.wait_discriminator(self.real_to_wait_generator(batch_real_img))
                d2_loss = torch.mean(fake_logits ** 2)
                d_loss = d1_loss + d2_loss
                d_loss_record += d_loss.data.cpu().numpy()[0]
                d_loss.backward()
                self.discriminator_optimizer.step()

                # ------------------------------------------------------------------------
                # Training generator
                # ------------------------------------------------------------------------
                # wait -> real -> wait cycle
                self.generator_optimizer.zero_grad()
                fake_real = self.wait_to_real_generator(batch_wait_img)
                fake_logits = self.real_discriminator(fake_real)
                recon_wait = self.real_to_wait_generator(fake_real)
                ident_wait = self.real_to_wait_generator(batch_wait_img)

                adpot_prob = random.random()
                if self.adopt_mask == True and adpot_prob > 0.5:    # mask  cycleGAN
                    mask_var = getMaskVariable(batch_wait_img, use_cuda = True)
                    filted_batch_wait_img = torch.mul(batch_wait_img, mask_var)
                    filted_recon_wait = torch.mul(recon_wait, mask_var)
                    filted_ident_wait = torch.mul(ident_wait, mask_var)
                    g_loss = torch.mean((fake_logits - 1) ** 2) + \
                        self.lambda_wait * l1_loss_fn(filted_recon_wait, filted_batch_wait_img) + \
                        self.lambda_wait * self.lambda_identity * l1_loss_fn(filted_ident_wait, filted_batch_wait_img)      # identity mapping loss                    
                else:                                               # usual cycleGAN
                    g_loss = torch.mean((fake_logits - 1) ** 2) + \
                        self.lambda_wait * l1_loss_fn(recon_wait, batch_wait_img) + \
                        self.lambda_wait * self.lambda_identity * l1_loss_fn(ident_wait, batch_wait_img)                    # identity mapping loss

                g_loss_record = g_loss.data.cpu().numpy()[0]
                g_loss.backward()
                self.generator_optimizer.step()

                # real -> wait -> real cycle
                self.generator_optimizer.zero_grad()
                fake_wait = self.real_to_wait_generator(batch_real_img)
                fake_logits = self.wait_discriminator(fake_wait)
                recon_real = self.wait_to_real_generator(fake_wait)
                ident_real = self.wait_to_real_generator(batch_real_img)
                g_loss = torch.mean((fake_logits - 1) ** 2) + \
                    self.lambda_real * l1_loss_fn(recon_real, batch_real_img) + \
                    self.lambda_real * self.lambda_identity * l1_loss_fn(ident_real, batch_real_img)      # identity mapping loss
                g_loss_record += g_loss.data.cpu().numpy()[0]
                g_loss.backward()
                self.generator_optimizer.step()

                # ------------------------------------------------------------------------
                # Record
                # ------------------------------------------------------------------------
                if j % verbose_period == 0:
                    print('Epoch: %6d \t iter: %6d \t generator loss: %.5f \t discriminator loss: %.5f' % (i, j, d_loss_record, g_loss_record))
                    self.generator_loss_list.append(g_loss_record)
                    self.discriminator_loss_list.append(d_loss_record)
                    output_img_name = str(i) + '_' + str(j) + '.png'
                    saveTransformResult(self.output_folder, output_img_name, 
                        batch_wait_img, batch_real_img,     # original image
                        fake_real, fake_wait,               # transfered space
                        recon_wait, recon_real)             # reconstructed image
                    self.save(i * loader.iter_num + j, model_folder = self.model_folder)
                    
    def load(self, model_folder = None):
        model_folder = self.model_folder if model_folder is None else model_folder
        self.real_to_wait_generator = self.loadModel(self.real_to_wait_generator, model_folder = model_folder, model_name = 'real_to_wait_generator')
        self.wait_to_real_generator = self.loadModel(self.wait_to_real_generator, model_folder = model_folder, model_name = 'wait_to_real_generator')
        self.real_discriminator = self.loadModel(self.real_discriminator, model_folder = model_folder, model_name = 'real_discriminator')
        self.wait_discriminator = self.loadModel(self.wait_discriminator, model_folder = model_folder, model_name = 'wait_discriminator')

    def save(self, idx, model_folder = './model'):
        import glob
        import os

        # Remove all previous model
        prev_model_name_list = glob.glob(os.path.join(model_folder, '*.pth.tar'))
        for name in prev_model_name_list:
            os.remove(name)
        
        # Save
        self.saveModel(self.real_to_wait_generator, idx, model_folder = model_folder, model_name = 'real_to_wait_generator')
        self.saveModel(self.wait_to_real_generator, idx, model_folder = model_folder, model_name = 'wait_to_real_generator')
        self.saveModel(self.real_discriminator, idx, model_folder = model_folder, model_name = 'real_discriminator')
        self.saveModel(self.wait_discriminator, idx, model_folder = model_folder, model_name = 'wait_discriminator')
        
    def storeCSV(self, csv_name = './output.csv'):
        pd.DataFrame.from_dict({
            'generator_loss': self.generator_loss_list,
            'discriminator_loss': self.discriminator_loss_list,
        }).to_csv(csv_name)

    def plot(self, period_times = 2, title = '?', fig_name = './output.png'):
        plt.plot(np.arange(len(self.generator_loss_list)), self.generator_loss_list, '-o', label='generator_loss')
        plt.plot(np.arange(len(self.discriminator_loss_list)), self.discriminator_loss_list, '-o', label='discriminator_loss')
        plt.legend()
        plt.title(title)
        plt.savefig(fig_name)
        plt.gca().clear()