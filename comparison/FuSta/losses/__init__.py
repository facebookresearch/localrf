import torch.nn as nn
from importlib import import_module
from utility import Module_CharbonnierLoss


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        self.regularize = []

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charb':
                loss_function = Module_CharbonnierLoss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('losses.vgg')
                loss_function = getattr(module, 'VGG')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('losses.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            elif loss_type in ['g_Spatial', 'g_Occlusion', 'Lw', 'Ls']:
                self.regularize.append({
                    'type': loss_type,
                    'weight': float(weight)}
                )
                continue

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        for r in self.regularize:
            print('{:.3f} * {}'.format(r['weight'], r['type']))

        self.loss_module.to('cuda')

    def forward(self, output, gt, input_frames):
        losses = []
        for l in self.loss:
            if l['function'] is not None:
                if l['type'] == 'T_WGAN_GP' or l['type'] == 'FI_GAN':
                    loss = l['function'](output, gt, input_frames)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)
                else:
                    loss = l['function'](output, gt)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)

        # for r in self.regularize:
        #     effective_loss = r['weight'] * output[r['type']]
        #     losses.append(effective_loss)
        loss_sum = sum(losses)

        return loss_sum
