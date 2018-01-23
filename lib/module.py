from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class Generator(nn.Module):
    def __init__(self, input_channel, base_filter = 32):
        super(Generator, self).__init__()
        self.base_filter = base_filter
        self.input_channel = input_channel

        # Define Encoder part
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, base_filter, 9, padding = 4),
            nn.InstanceNorm2d(base_filter),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter, base_filter, 3, padding = 1),
            nn.InstanceNorm2d(base_filter),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter, base_filter, 4, padding = 1, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filter * 1, base_filter * 2, 3, padding = 1),
            nn.InstanceNorm2d(base_filter * 2),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter * 2, base_filter * 2, 3, padding = 1),
            nn.InstanceNorm2d(base_filter * 2),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter * 2, base_filter * 2, 4, padding = 1, stride = 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filter * 2, base_filter * 4, 3, padding = 1),
            nn.InstanceNorm2d(base_filter * 4),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter * 4, base_filter * 4, 3, padding = 1),
            nn.InstanceNorm2d(base_filter * 4),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter * 4, base_filter * 4, 3, padding = 1),
            nn.InstanceNorm2d(base_filter * 4),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter * 4, base_filter * 4, 3, padding = 1),
            nn.InstanceNorm2d(base_filter * 4),
            nn.LeakyReLU(),
            nn.Conv2d(base_filter * 4, base_filter * 4, 4, padding = 1, stride = 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filter * 4, base_filter * 4, 4, padding = 1, stride = 2),
            nn.InstanceNorm2d(base_filter * 4),
            nn.LeakyReLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(base_filter * 4, base_filter * 4, 4, 2, padding = 1),
            nn.InstanceNorm2d(base_filter * 4),
            nn.LeakyReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(base_filter * 8, base_filter * 2, 4, 2, padding = 1),
            nn.InstanceNorm2d(base_filter * 2),
            nn.LeakyReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(base_filter * 4, base_filter * 1, 4, 2, padding = 1),
            nn.InstanceNorm2d(base_filter * 1),
            nn.LeakyReLU(),
        )
        self.recon = nn.Sequential(
            nn.ConvTranspose2d(base_filter * 2, self.input_channel, 4, 2, padding = 1),
            nn.InstanceNorm2d(self.input_channel * 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.input_channel, self.input_channel, 9, 1, padding = 4),
        )

    def forward(self, x):
        conv1_tensor = self.conv1(x)
        conv2_tensor = self.conv2(conv1_tensor)
        conv3_tensor = self.conv3(conv2_tensor)
        conv4_tensor = self.conv4(conv3_tensor)
        deconv3_tensor = torch.cat((self.deconv3(conv4_tensor), conv3_tensor), 1)
        deconv2_tensor = torch.cat((self.deconv2(deconv3_tensor), conv2_tensor), 1)
        deconv1_tensor = torch.cat((self.deconv1(deconv2_tensor), conv1_tensor), 1)
        recon = F.tanh(self.recon(deconv1_tensor))
        return recon

class Generator1(nn.Module):
    def __init__(self, input_channel, base_filter):
        super(Generator, self).__init__()
        self.base_filter = base_filter

        # Define convolution components
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, base_filter, input_channel, stride = 1, padding = 1),
                nn.BatchNorm2d(base_filter),
                nn.ReLU()
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(base_filter, base_filter * 2, 4, stride = 2, padding = 1),
                nn.BatchNorm2d(base_filter * 2),
                nn.ReLU()
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(base_filter * 2, base_filter * 4, 4, stride = 2, padding = 1),
                nn.BatchNorm2d(base_filter * 4),
                nn.ReLU()
        )

        # Define residual components
        self.res_block1_conv1, self.res_block1_bn1, self.res_block1_relu1 = self.createConvolutionComponents()
        self.res_block1_conv2, self.res_block1_bn2, self.res_block1_relu2 = self.createConvolutionComponents()
        self.res_block2_conv1, self.res_block2_bn1, self.res_block2_relu1 = self.createConvolutionComponents()
        self.res_block2_conv2, self.res_block2_bn2, self.res_block2_relu2 = self.createConvolutionComponents()
        self.res_block3_conv1, self.res_block3_bn1, self.res_block3_relu1 = self.createConvolutionComponents()
        self.res_block3_conv2, self.res_block3_bn2, self.res_block3_relu2 = self.createConvolutionComponents()
        self.res_block4_conv1, self.res_block4_bn1, self.res_block4_relu1 = self.createConvolutionComponents()
        self.res_block4_conv2, self.res_block4_bn2, self.res_block4_relu2 = self.createConvolutionComponents()
        self.res_block5_conv1, self.res_block5_bn1, self.res_block5_relu1 = self.createConvolutionComponents()
        self.res_block5_conv2, self.res_block5_bn2, self.res_block5_relu2 = self.createConvolutionComponents()

        # Define deconvolution components
        self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(base_filter * 4, base_filter * 2, 4, stride = 2, padding = 1),
                nn.BatchNorm2d(base_filter * 2),
                nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(base_filter * 2, base_filter * 1, 4, stride = 2, padding = 1),
                nn.BatchNorm2d(base_filter),
                nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(base_filter, input_channel, 3, stride = 1, padding = 1),
                nn.BatchNorm2d(3),
                nn.ReLU()
        )

    def createConvolutionComponents(self):
        return nn.Conv2d(self.base_filter * 4, self.base_filter * 4, 3, padding = 1), nn.BatchNorm2d(self.base_filter * 4), nn.ReLU()

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        temp = self.res_block1_relu1(self.res_block1_bn1(self.res_block1_conv1(x)))
        x = x + self.res_block1_relu2(self.res_block1_bn2(self.res_block1_conv2(temp)))

        temp = x + self.res_block2_relu1(self.res_block2_bn1(self.res_block2_conv1(x)))
        x = x + self.res_block2_relu2(self.res_block2_bn2(self.res_block2_conv2(temp)))

        temp = x + self.res_block3_relu1(self.res_block3_bn1(self.res_block3_conv1(x))) 
        x = x + self.res_block3_relu2(self.res_block3_bn2(self.res_block3_conv2(temp))) 

        temp = x + self.res_block4_relu1(self.res_block4_bn1(self.res_block4_conv1(x)))
        x = x + self.res_block4_relu2(self.res_block4_bn2(self.res_block4_conv2(x)))

        temp = x + self.res_block5_relu1(self.res_block5_bn1(self.res_block5_conv1(x)))
        x = x + self.res_block5_relu2(self.res_block5_bn2(self.res_block5_conv2(temp)))

        x = self.deconv3(self.deconv2(self.deconv1(x)))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channel, base_filter = 16):
        super(Discriminator, self).__init__()
        self.base_filter = base_filter
        self.input_channel = input_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.base_filter, 3, padding = 1),
            nn.BatchNorm2d(self.base_filter),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.base_filter, self.base_filter * 2, 3, padding = 1),
            nn.BatchNorm2d(self.base_filter * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.base_filter * 2, self.base_filter * 4, 3, padding = 1),
            nn.BatchNorm2d(self.base_filter * 4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.base_filter * 4, self.base_filter * 8, 3, padding = 1),
            nn.BatchNorm2d(self.base_filter * 8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(base_filter * 1600, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

if __name__ == '__main__':
    generator = Discriminator(3, base_filter = 16)
    generator.cuda()
    img_var = Variable(torch.from_numpy(np.random.random([32, 3, 160, 320])).float())
    img_var = img_var.cuda() if torch.cuda.is_available() else img_var
    for i in range(10000):
        generator(img_var)
