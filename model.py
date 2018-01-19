from visualize import saveTransformResult
from torch.autograd import Variable
from module import Generator, Discriminator
from torch.optim import Adam
from gan import GAN
import torch.nn as nn
import torch

class CustomCycleGAN(GAN):
    # Define cycle index
    INDEX_wait_to_real = 0
    INDEX_real_to_wait = 1

    def __init__(self, input_channel = 3, base_filter = 16, adopt_custom = False):
        super(CustomCycleGAN, self).__init__()
        self.adopt_custom = adopt_custom
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

    def train(self, loader, epoch=1, verbose_period=5):
        self.loadModel()
        for i in range(epoch):
            for j, (batch_real_img, batch_wait_img) in enumerate(loader):
                batch_real_img = Variable(batch_real_img).cuda()
                batch_wait_img = Variable(batch_wait_img).cuda()

                # ------------------------------------------------------------------------
                # 1st cycle 
                # ------------------------------------------------------------------------
                # loss compute
                latent_img, restore_img, true_logtis, fake_logits = self.forward(batch_wait_img, batch_real_img, self.INDEX_wait_to_real)
                self.discriminator_loss = torch.sum((true_logtis - 1) ** 2 + fake_logits ** 2) / 2.
                self.generator_loss = torch.sum((fake_logits - 1) ** 2) / 2. + \
                    torch.mean(torch.abs(batch_wait_img - restore_img))

                # 1st cycle parameter update
                self.wait_to_real_generator_optimizer.zero_grad()
                self.real_to_wait_generator_optimizer.zero_grad()
                self.real_discriminator_optimizer.zero_grad()

                self.discriminator_loss.backward(retain_graph=True)
                self.generator_loss.backward()
                discriminator_loss_1st = self.discriminator_loss
                generator_loss_1st = self.generator_loss

                self.wait_to_real_generator_optimizer.step()
                self.real_to_wait_generator_optimizer.step()
                self.real_discriminator_optimizer.step()

                # ------------------------------------------------------------------------
                # 2nd cycle 
                # ------------------------------------------------------------------------
                # loss compute
                latent_img, restore_img, true_logtis, fake_logits = self.forward(batch_wait_img, batch_real_img, self.INDEX_real_to_wait)
                self.discriminator_loss = torch.sum((true_logtis - 1) ** 2 + fake_logits ** 2) / 2.
                self.generator_loss = torch.sum((fake_logits - 1) ** 2) / 2. + \
                    torch.mean(torch.abs(batch_real_img - restore_img))

                # 2nd cycle parameter update
                self.wait_to_real_generator_optimizer.zero_grad()
                self.real_to_wait_generator_optimizer.zero_grad()
                self.wait_discriminator_optimizer.zero_grad()

                self.discriminator_loss.backward(retain_graph=True)
                self.generator_loss.backward()
                discriminator_loss_2nd = self.discriminator_loss
                generator_loss_2nd = self.generator_loss

                self.real_to_wait_generator_optimizer.step()
                self.wait_to_real_generator_optimizer.step()
                self.wait_discriminator_optimizer.step()

                # ------------------------------------------------------------------------
                # Record
                # ------------------------------------------------------------------------
                if j % verbose_period == 0:
                    print('epoch: ', i, '\titer: ', j, 
                    '\t< 1st cycle >\tgen loss: ', self.generator_loss.data.cpu().numpy()[0], 
                    '\tdis loss: ', self.discriminator_loss.data.cpu().numpy()[0],
                    '\t< 2nd cycle >\tgen loss: ', self.generator_loss.data.cpu().numpy()[0], 
                    '\tdis loss: ', self.discriminator_loss.data.cpu().numpy()[0])
                    saveTransformResult('./output', str(i) + '_' + str(j) + '.png', batch_real_img, restore_img)
                    self.saveModel(i * loader.iter_num + j)