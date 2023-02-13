import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        print(vgg19.features)
        # features = list(vgg19.features)[:36]
        self.vgg19_conv1_2 = torch.nn.Sequential(*list(vgg19.children())[0][:3])
        self.vgg19_conv2_2 = torch.nn.Sequential(*list(vgg19.children())[0][3:8])
        self.vgg19_conv3_2 = torch.nn.Sequential(*list(vgg19.children())[0][8:13])
        self.vgg19_conv4_2 = torch.nn.Sequential(*list(vgg19.children())[0][13:22])
        self.vgg19_conv5_2 = torch.nn.Sequential(*list(vgg19.children())[0][22:31])
        for param in self.vgg19_conv1_2.parameters():
            param.requires_grad = False
        for param in self.vgg19_conv2_2.parameters():
            param.requires_grad = False
        for param in self.vgg19_conv3_2.parameters():
            param.requires_grad = False
        for param in self.vgg19_conv4_2.parameters():
            param.requires_grad = False
        for param in self.vgg19_conv5_2.parameters():
            param.requires_grad = False
        # self.features = nn.ModuleList(features).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, output, gt):
        output = (output - self.mean) / self.std
        gt = (gt - self.mean) / self.std

        vgg_output_conv1_2 = self.vgg19_conv1_2(output)
        vgg_output_conv2_2 = self.vgg19_conv2_2(vgg_output_conv1_2)
        vgg_output_conv3_2 = self.vgg19_conv3_2(vgg_output_conv2_2)
        vgg_output_conv4_2 = self.vgg19_conv4_2(vgg_output_conv3_2)
        vgg_output_conv5_2 = self.vgg19_conv5_2(vgg_output_conv4_2)
        with torch.no_grad():
            vgg_gt_conv1_2 = self.vgg19_conv1_2(gt.detach())
            vgg_gt_conv2_2 = self.vgg19_conv2_2(vgg_gt_conv1_2.detach())
            vgg_gt_conv3_2 = self.vgg19_conv3_2(vgg_gt_conv2_2.detach())
            vgg_gt_conv4_2 = self.vgg19_conv4_2(vgg_gt_conv3_2.detach())
            vgg_gt_conv5_2 = self.vgg19_conv5_2(vgg_gt_conv4_2.detach())

        loss = torch.nn.L1Loss()(output, gt.detach())
        loss += ((torch.nn.L1Loss()(vgg_output_conv1_2, vgg_gt_conv1_2.detach())) / 2.6)
        loss += ((torch.nn.L1Loss()(vgg_output_conv2_2, vgg_gt_conv2_2.detach())) / 4.8)
        loss += ((torch.nn.L1Loss()(vgg_output_conv3_2, vgg_gt_conv3_2.detach())) / 3.7)
        loss += ((torch.nn.L1Loss()(vgg_output_conv4_2, vgg_gt_conv4_2.detach())) / 5.6)
        loss += ((torch.nn.L1Loss()(vgg_output_conv5_2, vgg_gt_conv5_2.detach())) * 10 / 1.5)

        return loss
