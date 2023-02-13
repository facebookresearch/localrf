import torch.nn as nn
import torch


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class Discriminator(nn.Module):
    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        bn = not gan_type == 'WGAN_GP'
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output


class Temporal_Discriminator(nn.Module):
    def __init__(self, args):
        super(Temporal_Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        bn = False
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feature_3d = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        )

        m_features = [
            BasicBlock(out_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, f0, f1, f2):
        f0 = torch.unsqueeze(f0, dim=2)
        f1 = torch.unsqueeze(f1, dim=2)
        f2 = torch.unsqueeze(f2, dim=2)

        x_5d = torch.cat((f0, f1, f2), dim=2)

        x = torch.squeeze(self.feature_3d(x_5d))
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output


class FI_Discriminator(nn.Module):
    def __init__(self, args):
        super(FI_Discriminator, self).__init__()

        in_channels = 6
        out_channels = 64
        depth = 7
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, f0, f1):
        x = torch.cat((f0, f1), dim=1)
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output
