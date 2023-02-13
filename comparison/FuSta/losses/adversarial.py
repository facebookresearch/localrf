import utility
from losses import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = 1
        if gan_type == 'T_WGAN_GP':
            self.discriminator = discriminator.Temporal_Discriminator(args)
        elif gan_type == 'FI_GAN':
            self.discriminator = discriminator.FI_Discriminator(args)
        else:
            self.discriminator = discriminator.Discriminator(args, gan_type)
        if gan_type != 'WGAN_GP' and gan_type != 'T_WGAN_GP':
            self.optimizer = utility.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        self.scheduler = utility.make_scheduler(args, self.optimizer)

    def forward(self, fake, real, input_frames=None):
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            if self.gan_type == 'T_WGAN_GP':
                d_fake = self.discriminator(input_frames[0], fake_detach, input_frames[1])
                d_real = self.discriminator(input_frames[0], real, input_frames[1])
            elif self.gan_type == 'FI_GAN':
                d_01 = self.discriminator(input_frames[0], fake_detach)
                d_12 = self.discriminator(fake_detach, input_frames[1])
            else:
                d_fake = self.discriminator(fake_detach)
                d_real = self.discriminator(real)

            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type == 'FI_GAN':
                label_01 = torch.zeros_like(d_01)
                label_12 = torch.ones_like(d_12)
                loss_d = F.binary_cross_entropy_with_logits(d_01, label_01) + F.binary_cross_entropy_with_logits(d_12, label_12)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        if self.gan_type == 'GAN':
            d_fake_for_g = self.discriminator(fake)
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )
        elif self.gan_type == 'FI_GAN':
            d_01_for_g = F.sigmoid(self.discriminator(input_frames[0], fake_detach))
            d_12_for_g = F.sigmoid(self.discriminator(fake_detach, input_frames[1]))
            loss_g = d_01_for_g * torch.log(d_01_for_g + 1e-12) + d_12_for_g * torch.log(d_12_for_g + 1e-12)
            loss_g = loss_g.mean()

        elif self.gan_type.find('WGAN') >= 0:
            d_fake_for_g = self.discriminator(fake)
            loss_g = -d_fake_for_g.mean()

        # Generator loss
        return loss_g
