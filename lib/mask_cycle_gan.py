from mask import getMaskVariable
from cycle_gan import CycleGAN
import random
import torch

class MaskCycleGAN(CycleGAN):
    mask_position = 'B'
    adopt_mask_alpha = 0.5

    def __init__(self, save_dir, isTrain = True, input_channel = 3, output_channel = 3, base_filter = 32, batch_size = 32, use_dropout = False, use_gpu = True):
        super(MaskCycleGAN, self).__init__(save_dir, isTrain, input_channel, output_channel, base_filter, batch_size, use_dropout, use_gpu)

    def backward_G(self):
        # Set hyper-parameters
        lambda_idt = 0.5
        lambda_A = 10
        lambda_B = 10

        # Get the result
        # A -> B -> A
        if lambda_idt > 0:
            idt_B = self.netG_B(self.real_A)
        fake_B = self.netG_A(self.real_A)
        pred_fake_A = self.netD_A(fake_B)
        rec_A = self.netG_B(fake_B)

        # B -> A -> B
        if lambda_idt > 0:
            idt_A = self.netG_A(self.real_B)        
        fake_A = self.netG_B(self.real_B)
        pred_fake_B = self.netD_B(fake_A)
        rec_B = self.netG_A(fake_A)

        # Store image in order to visualize
        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.idt_A = idt_A.data
        self.idt_B = idt_B.data

        # Adopt mask
        real_A = self.real_A
        real_B = self.real_B
        if random.random() < self.adopt_mask_alpha:
            # Mask in A
            mask_var = getMaskVariable(self.real_A, use_cuda = True)
            mask_var = torch.ones_like(mask_var) - mask_var
            real_A = torch.mul(self.real_A, mask_var)
            rec_A  = torch.mul(rec_A, mask_var)
            idt_B  = torch.mul(idt_B, mask_var)
            # Mask in B
            mask_var = getMaskVariable(self.real_B, use_cuda = True)
            mask_var = torch.ones_like(mask_var) - mask_var
            real_B = torch.mul(self.real_B, mask_var)
            rec_B  = torch.mul(rec_B, mask_var)
            idt_A  = torch.mul(idt_A, mask_var)          

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt

            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterionGAN(pred_fake_A, True)

        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterionGAN(pred_fake_B, True)

        # Forward cycle loss
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A

        # Backward cycle loss
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]