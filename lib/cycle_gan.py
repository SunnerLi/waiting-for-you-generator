from util import ImagePool, tensor2im
from collections import OrderedDict
from torch.autograd import Variable
from model import BaseModel
from loss import GANLoss
import itertools
import network
import torch

class CycleGAN(BaseModel):
    use_sigmoid = False

    def __init__(self, save_dir, isTrain = True, input_channel = 3, output_channel = 3, base_filter = 32, batch_size = 32, use_dropout = False, use_gpu = True):
        BaseModel.initialize(self, isTrain, save_dir, use_gpu) 

        # Construct generator
        self.netG_A = network.define_G(input_channel, output_channel, base_filter, use_dropout, use_gpu)
        self.netG_B = network.define_G(input_channel, output_channel, base_filter, use_dropout, use_gpu)

        # Construct discriminator
        if self.isTrain:
            self.netD_A = network.define_D(output_channel, base_filter, self.use_sigmoid, use_gpu)
            self.netD_B = network.define_D(output_channel, base_filter, self.use_sigmoid, use_gpu)

        # Load pre-trained model
        which_epoch = 'latest'
        self.load_network(self.netG_A, 'G_A', which_epoch)
        self.load_network(self.netG_B, 'G_B', which_epoch)
        if self.isTrain:
            self.load_network(self.netD_A, 'D_A', which_epoch)
            self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool() 
            self.fake_B_pool = ImagePool()

            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not self.use_sigmoid, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0002)
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=0.0002)
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=0.0002)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer))

        print('---------- Networks initialized -------------')
        network.print_network(self.netG_A)
        network.print_network(self.netG_B)
        if self.isTrain:
            network.print_network(self.netD_A)
            network.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, _input, AtoB = 'AtoB', use_gpu = True):
        # The input should both not be None during training
        if self.isTrain:
            input_A = _input['A' if AtoB else 'B']
            input_B = _input['B' if AtoB else 'A']
            if use_gpu:
                input_A = input_A.cuda(async=True)
                input_B = input_B.cuda(async=True)
            self.input_A = input_A
            self.input_B = input_B
        else:
            input_A = None
            input_B = None
            if AtoB == 'A':
                if 'A' in _input.keys():
                    input_A = _input['A']
                if 'B' in _input.keys():
                    input_B = _input['B']
            else:
                if 'B' in _input.keys():
                    input_A = _input['B']
                if 'A' in _input.keys():
                    input_B = _input['A']
            if use_gpu:
                if input_A is not None:
                    input_A = input_A.cuda()
                if input_B is not None:
                    input_B = input_B.cuda()
            self.input_A = input_A
            self.input_B = input_B

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        if self.input_A is not None:
            real_A = Variable(self.input_A, volatile=True)
            fake_B = self.netG_A(real_A)
            self.rec_A = self.netG_B(fake_B).data
            self.fake_B = fake_B.data

        if self.input_B is not None:
            real_B = Variable(self.input_B, volatile=True)
            fake_A = self.netG_B(real_B)
            self.rec_B = self.netG_A(fake_A).data
            self.fake_A = fake_A.data

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

    def backward_G(self):
        # Set hyper-parameters
        lambda_idt = 0.5
        lambda_A = 10
        lambda_B = 10

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B
        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        return OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                            ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                            ('idt_A', self.loss_idt_A), ('idt_B', self.loss_idt_B)])

    def get_current_visuals(self):
        if self.input_A is not None:
            real_A = tensor2im(self.input_A)
            fake_B = tensor2im(self.fake_B)
            rec_A  = tensor2im(self.rec_A)
        else:
            real_A, fake_B, rec_A = None, None, None
        if self.input_B is not None:
            real_B = tensor2im(self.input_B)
            fake_A = tensor2im(self.fake_A)
            rec_B  = tensor2im(self.rec_B)
        else:
            real_B, fake_A, rec_B = None, None, None
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        if self.isTrain:
            ret_visuals['idt_A'] = tensor2im(self.idt_A)
            ret_visuals['idt_B'] = tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label)
        self.save_network(self.netD_A, 'D_A', label)
        self.save_network(self.netG_B, 'G_B', label)
        self.save_network(self.netD_B, 'D_B', label)