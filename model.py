from module import Generator, Discriminator
from torch.optim import Adam
import torch.nn as nn
import torch

class CustomCycleGAN(nn.Module):
    def __init__(self, input_channel = 3, base_filter = 32, adopt_custom = False):
        self.adopt_custom = adopt_custom
        self.photo_to_real_generator = Generator(input_channel, base_filter)
        self.real_to_photo_generator = Generator(input_channel, base_filter)
        self.real_discriminator = Discriminator(input_channel, base_filter)
        self.photo_discriminator = Discriminator(input_channel, base_filter)

        self.photo_to_real_generator_optimizer = Adam(self.photo_to_real_generator.parameters())
        self.real_to_photo_generator_optimizer = Adam(self.real_to_photo_generator.parameters())
        self.photo_to_real_discriminator_optimizer = Adam(self.photo_to_real_discriminator.parameters())
        self.real_to_photo_discriminator_optimizer = Adam(self.real_to_photo_discriminator.parameters())

    def forward(self, photo_variable, real_variable):
        # Compute 1st cycle loss
        real_image_with_text = self.photo_to_real_generator(photo_variable)
        photo_image_with_text = self.real_to_photo_generator(real_image_with_text)
        fake_logits = self.real_discriminator(real_image_with_text)
        true_logtis = self.real_discriminator(real_variable)
        self.photo_to_real_discriminator_loss = torch.sum((true_logtis - 1) ** 2 + fake_logits ** 2) / 2.
        self.photo_to_real_generator_loss = torch.sum((fake_logits - 1) ** 2) / 2.
        if self.adopt_custom == False:
            self.photo_to_real_cycle_loss = torch.mean(torch.abs(real_image_with_text - photo_image_with_text))
