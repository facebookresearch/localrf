import sys

sys.path.append('core')

import torch
import cupy_module.adacof as adacof
import sys
from torch.nn import functional as F
from utility import CharbonnierFunc, moduleNormalize
from torchvision.utils import save_image as imwrite
import cv2
import numpy as np

import softsplat

backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)


# end

def make_model(args):
    return AdaCoFNet(args).cuda()


def conv_function(in_c, out_c, k, p, s):
    return torch.nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, stride=s)


def get_conv_layer():
    conv_layer_base = conv_function
    return conv_layer_base


def get_bn_layer(output_sz):
    return torch.nn.BatchNorm2d(num_features=output_sz)


class ResNet_Block(torch.nn.Module):
    def __init__(self, in_c, in_o, downsample=None):
        super().__init__()
        bn_noise1 = get_bn_layer(output_sz=in_c)
        bn_noise2 = get_bn_layer(output_sz=in_o)

        conv_layer = get_conv_layer()

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        elif downsample:
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = torch.nn.Identity()

        self.ch_a = torch.nn.Sequential(
            bn_noise1,
            torch.nn.ReLU(),
            conv_aa,
            bn_noise2,
            torch.nn.ReLU(),
            conv_ab,
            norm_downsample,
        )

        if downsample or (in_c != in_o):
            self.ch_b = torch.nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = torch.nn.Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b


class GatedConv2d(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_c, out_c, k, p, s):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_c, out_c, k, s, p)
        self.mask_conv2d = torch.nn.Conv2d(in_c, out_c, k, s, p)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_c)
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.gated(mask)
        return x


class GatedConv2d_ResNet_Block(torch.nn.Module):
    def __init__(self, in_c, in_o, downsample=None):
        super().__init__()
        bn_noise1 = get_bn_layer(output_sz=in_c)
        bn_noise2 = get_bn_layer(output_sz=in_o)

        conv_layer = GatedConv2d

        conv_aa = conv_layer(in_c, in_o, 3, 1, 1)
        conv_ab = conv_layer(in_o, in_o, 3, 1, 1)

        conv_b = conv_layer(in_c, in_o, 1, 0, 1)

        if downsample == "Down":
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif downsample == "Up":
            norm_downsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        elif downsample:
            norm_downsample = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            norm_downsample = torch.nn.Identity()

        self.ch_a = torch.nn.Sequential(
            bn_noise1,
            torch.nn.ReLU(),
            conv_aa,
            bn_noise2,
            torch.nn.ReLU(),
            conv_ab,
            norm_downsample,
        )

        if downsample or (in_c != in_o):
            self.ch_b = torch.nn.Sequential(conv_b, norm_downsample)
        else:
            self.ch_b = torch.nn.Identity()

    def forward(self, x):
        x_a = self.ch_a(x)
        x_b = self.ch_b(x)

        return x_a + x_b


class SpatialFeatureNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = ResNet_Block(3, 8)
        self.layer1 = ResNet_Block(8, 8)
        self.layer2 = ResNet_Block(8, 8)
        self.layer3 = ResNet_Block(8, 16)
        self.layer4 = ResNet_Block(16, 16)
        self.layer5 = ResNet_Block(16, 16)
        self.layer6 = ResNet_Block(16, 16)
        self.layer7 = ResNet_Block(16, 32)

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        x_6 = self.layer6(x_5)
        x_7 = self.layer7(x_6)

        return x_7


class RefinementNetwork(torch.nn.Module):
    def __init__(self, with_mask=False, with_proxy=False):
        super().__init__()
        input_ch = 32 + 32
        if with_mask:
            input_ch += 1
        if with_proxy:
            input_ch += 3
        self.layer0 = ResNet_Block(input_ch, 16)
        self.layer1 = ResNet_Block(16, 64, 'Down')
        self.layer2 = ResNet_Block(64, 64, 'Down')
        self.layer3 = ResNet_Block(64, 32)
        self.layer4 = ResNet_Block(32, 32, 'Up')
        self.layer5 = ResNet_Block(32, 32, 'Up')
        self.layer6 = ResNet_Block(32, 32)
        self.layer7 = ResNet_Block(32, 3 + 1)

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        x_6 = self.layer6(x_5)
        x_7 = self.layer7(x_6)

        return x_7[:, 0:3], x_7[:, 3:4]


class GatedConvRefinementNetwork(torch.nn.Module):
    def __init__(self, with_proxy=False, no_pooling=False, single_decoder=False):
        super().__init__()
        input_ch = 32 + 32 + 1
        if with_proxy:
            input_ch += 3
        if no_pooling:
            input_ch = 32 + 1
        if single_decoder:
            input_ch = 32
        self.layer0 = GatedConv2d_ResNet_Block(input_ch, 16)
        self.layer1 = GatedConv2d_ResNet_Block(16, 64, 'Down')
        self.layer2 = GatedConv2d_ResNet_Block(64, 64, 'Down')
        self.layer3 = GatedConv2d_ResNet_Block(64, 32)
        self.layer4 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        self.layer5 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        self.layer6 = GatedConv2d_ResNet_Block(32, 32)
        self.layer7 = GatedConv2d_ResNet_Block(32, 3 + 1)

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        x_6 = self.layer6(x_5)
        x_7 = self.layer7(x_6)

        return x_7[:, 0:3], x_7[:, 3:4]


class AggregationWeightingNetwork(torch.nn.Module):
    def __init__(self, noDL_CNNAggregation=False, CNN_flowError=False):
        super().__init__()
        input_ch = 32 + 32 + 1 + 1
        if noDL_CNNAggregation:
            input_ch = 3 + 3 + 2 + 1 + 1 + 3
        if CNN_flowError:
            input_ch = 32 + 32 + 1 + 1 + 1
        self.layer0 = GatedConv2d_ResNet_Block(input_ch, 16)
        self.layer1 = GatedConv2d_ResNet_Block(16, 64, 'Down')
        self.layer2 = GatedConv2d_ResNet_Block(64, 64, 'Down')
        self.layer3 = GatedConv2d_ResNet_Block(64, 32)
        self.layer4 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        self.layer5 = GatedConv2d_ResNet_Block(32, 32, 'Up')
        self.layer6 = GatedConv2d_ResNet_Block(32, 32)
        self.layer7 = GatedConv2d_ResNet_Block(32, 1)

    def forward(self, x):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_5 = self.layer5(x_4)
        x_6 = self.layer6(x_5)
        x_7 = self.layer7(x_6)

        return x_7


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AdaCoFNet(torch.nn.Module):
    def __init__(self, args):
        super(AdaCoFNet, self).__init__()
        self.args = args

        self.spatialFeatureNetwork = SpatialFeatureNetwork()
        if args.decoder_with_gated_conv <= 0:
            self.refinementNetwork = RefinementNetwork(self.args.decoder_with_mask, self.args.concat_proxy)
        else:
            self.refinementNetwork = GatedConvRefinementNetwork(self.args.concat_proxy, self.args.no_pooling, self.args.single_decoder)
        if self.args.noDL_CNNAggregation > 0 or 'CNN' in self.args.pooling_type:
            self.aggregationWeightingNetwork = AggregationWeightingNetwork(self.args.noDL_CNNAggregation, self.args.pooling_type == 'CNN_flowError')

        if args.beta_learnable > 0:
            self.beta = torch.nn.Parameter(torch.ones(1) * (-20))
        else:
            self.beta = 0.0

        self.splatting_type = args.splatting_type

        # self.kernel_size = args.kernel_size
        # self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        # self.dilation = args.dilation

        # self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        # self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frames, F_kprime_to_k, F_n_to_k_s, F_k_to_n_s):
        # frames: base frame also included
        h0 = int(list(frames[0].size())[2])
        w0 = int(list(frames[0].size())[3])
        h6 = int(list(frames[-1].size())[2])
        w6 = int(list(frames[-1].size())[3])
        if h0 != h6 or w0 != w6:
            sys.exit('Frame sizes do not match')

        GAUSSIAN_FILTER_KSIZE = len(frames)
        gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)

        if self.args.inference_with_frame_selection > 0:
            import numpy as np
            var_lap_values = []
            flow_values = []
            clear_frame_indicator = []

            def variance_of_laplacian(image):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # compute the Laplacian of the image and then return the focus
                # measure, which is simply the variance of the Laplacian
                return cv2.Laplacian(gray, cv2.CV_32F).var()

            for iii, tmp_img in enumerate(frames):
                var_lap_values.append(variance_of_laplacian(np.transpose(tmp_img.cpu().numpy()[0], (1, 2, 0))))
                flow_values.append(np.sum((F_n_to_k_s[iii].cpu().numpy()[0]) ** 2))
            mid_value = sorted(var_lap_values)[len(frames) // 2]
            mid_flow_value = sorted(flow_values)[len(frames) // 2]
            for iii, value in enumerate(var_lap_values):
                if value < mid_value and flow_values[iii] < mid_flow_value:
                    clear_frame_indicator.append(False)
                else:
                    clear_frame_indicator.append(True)
            clear_frames = []
            clear_gaussian_filter = []
            clear_F_n_to_k_s = []
            clear_F_k_to_n_s = []
            for idx, indicator in enumerate(clear_frame_indicator):
                if indicator:
                    clear_frames.append(frames[idx])
                    clear_gaussian_filter.append(gaussian_filter[idx, 0])
                    clear_F_n_to_k_s.append(F_n_to_k_s[idx])
                    clear_F_k_to_n_s.append(F_k_to_n_s[idx])
            frames = clear_frames
            F_n_to_k_s = clear_F_n_to_k_s
            F_k_to_n_s = clear_F_k_to_n_s
            gaussian_filter = np.copy(np.stack(clear_gaussian_filter, 0))
            gaussian_filter = np.expand_dims(gaussian_filter, -1)
            gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

        h_padded = False
        w_padded = False
        minimum_size = 4
        if h0 % minimum_size != 0:
            pad_h = minimum_size - (h0 % minimum_size)
            F_kprime_to_k = F.pad(F_kprime_to_k, (0, 0, 0, pad_h), mode='replicate')
            for i in range(len(frames)):
                frames[i] = F.pad(frames[i], (0, 0, 0, pad_h), mode='replicate')
                F_n_to_k_s[i] = F.pad(F_n_to_k_s[i], (0, 0, 0, pad_h), mode='replicate')
            h_padded = True

        if w0 % minimum_size != 0:
            pad_w = minimum_size - (w0 % minimum_size)
            F_kprime_to_k = F.pad(F_kprime_to_k, (0, pad_w, 0, 0), mode='replicate')
            for i in range(len(frames)):
                frames[i] = F.pad(frames[i], (0, pad_w, 0, 0), mode='replicate')
                F_n_to_k_s[i] = F.pad(F_n_to_k_s[i], (0, pad_w, 0, 0), mode='replicate')
            w_padded = True

        if self.args.noDL_CNNAggregation > 0:
            """no learnable params"""
            """except final aggregation function"""
            W = 256
            H = 256
            tenOnes = torch.ones_like(frames[0])[:, 0:1, :, :]
            tenOnes = torch.nn.ZeroPad2d((W, W, H, H))(tenOnes).detach()
            F_kprime_to_k_pad = torch.nn.ZeroPad2d((W, W, H, H))(F_kprime_to_k)
            tenWarpedFeat = []
            tenWarpedMask = []
            tenWarpedFlow = []
            for idx, feat in enumerate(frames):
                """padding for forward warping"""
                ref_frame_flow = torch.nn.ReplicationPad2d((W, W, H, H))(F_n_to_k_s[idx])
                tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(feat)
                """first forward warping"""
                tenWarpedFirst = softsplat.FunctionSoftsplat(tenInput=tenRef, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
                tenMaskFirst = softsplat.FunctionSoftsplat(tenInput=tenOnes, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
                tenFlowFirst = softsplat.FunctionSoftsplat(tenInput=ref_frame_flow, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
                """second backward warping"""
                tenWarpedSecond = backwarp(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad)
                tenMaskSecond = backwarp(tenInput=tenMaskFirst, tenFlow=F_kprime_to_k_pad)
                tenFlowSecond = backwarp(tenInput=tenFlowFirst, tenFlow=F_kprime_to_k_pad)
                """back to original resolution"""
                tenWarped = tenWarpedSecond[:, :, H:-H, W:-W]
                tenMask = tenMaskSecond[:, :, H:-H, W:-W]
                tenFlow = tenFlowSecond[:, :, H:-H, W:-W]
                tenWarpedFeat.append(tenWarped)
                tenWarpedMask.append(tenMask)
                tenWarpedFlow.append(tenFlow)
            color_tensor = []
            for i in range(len(tenWarpedFeat)):
                color_tensor.append(tenWarpedFeat[i])
            color_tensor = torch.stack(color_tensor, 0)
            weight_tensor = []
            for i in range(len(tenWarpedFeat)):
                weight_tensor.append(self.aggregationWeightingNetwork(torch.cat([tenWarpedFeat[i], tenWarpedMask[i], tenWarpedFlow[i], tenWarpedFeat[len(frames) // 2], tenWarpedMask[len(frames) // 2], torch.abs(tenWarpedFeat[i] - tenWarpedFeat[len(frames) // 2])], 1)))
            weight_tensor = torch.stack(weight_tensor, 0)

            if self.args.gumbel > 0:
                weight_tensor = gumbel_softmax(weight_tensor, hard=True, dim=0)
            else:
                weight_tensor = torch.softmax(weight_tensor, 0)
            total_mask = torch.sum(torch.stack(tenWarpedMask, 0), 0)
            global_average_pooled_feature = torch.sum(color_tensor * weight_tensor, dim=0)
            global_average_pooled_feature = global_average_pooled_feature * torch.where(total_mask > 0, torch.ones_like(total_mask), torch.zeros_like(total_mask))
            if h_padded:
                global_average_pooled_feature = global_average_pooled_feature[:, :, 0:h0, :]
            if w_padded:
                global_average_pooled_feature = global_average_pooled_feature[:, :, :, 0:w0]
            return global_average_pooled_feature

        features = []
        for i in range(len(frames)):
            features.append(self.spatialFeatureNetwork(frames[i]))
        # import numpy as np
        # nnn = feature0.cpu().numpy()
        # print(np.mean(nnn))
        # print(np.std(nnn))
        # print(np.min(nnn))
        # print(np.max(nnn))

        # forward warping + backward warping
        # W = list(features[0].size())[3]//2
        # H = list(features[0].size())[2]//2
        W = 256
        H = 256
        tenOnes = torch.ones_like(features[0])[:, 0:1, :, :]
        tenOnes = torch.nn.ZeroPad2d((W, W, H, H))(tenOnes).detach()
        if self.args.FOV_expansion > 0:
            F_kprime_to_k_pad = torch.nn.ReplicationPad2d((W, W, H, H))(F_kprime_to_k)
        else:
            F_kprime_to_k_pad = torch.nn.ZeroPad2d((W, W, H, H))(F_kprime_to_k)
        tenWarpedFeat = []
        tenWarpedMask = []
        for idx, feat in enumerate(features):
            if self.args.all_backward > 0:
                ref_frame_flow = torch.nn.ReplicationPad2d((W, W, H, H))(F_k_to_n_s[idx])
                tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(feat)
                tenWarpedFirst = backwarp(tenInput=tenRef, tenFlow=ref_frame_flow)
                tenMaskFirst = backwarp(tenInput=tenOnes, tenFlow=ref_frame_flow)
            else:
                """padding for forward warping"""
                ref_frame_flow = torch.nn.ReplicationPad2d((W, W, H, H))(F_n_to_k_s[idx])
                tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(feat)
                """first forward warping"""
                tenWarpedFirst = softsplat.FunctionSoftsplat(tenInput=tenRef, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
                tenMaskFirst = softsplat.FunctionSoftsplat(tenInput=tenOnes, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
            """second backward warping"""
            if self.args.bundle_forward_flow > 0:
                tenWarpedSecond = softsplat.FunctionSoftsplat(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad, tenMetric=None, strType='average')
                tenMaskSecond = softsplat.FunctionSoftsplat(tenInput=tenMaskFirst, tenFlow=F_kprime_to_k_pad, tenMetric=None, strType='average')
            else:
                tenWarpedSecond = backwarp(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad)
                tenMaskSecond = backwarp(tenInput=tenMaskFirst, tenFlow=F_kprime_to_k_pad)
            """back to original resolution"""
            if self.args.FOV_expansion <= 0:
                tenWarped = tenWarpedSecond[:, :, H:-H, W:-W]
                tenMask = tenMaskSecond[:, :, H:-H, W:-W]
            else:
                tenWarped = tenWarpedSecond
                tenMask = tenMaskSecond
            tenWarpedFeat.append(tenWarped)
            tenWarpedMask.append(tenMask)

        # tenMetrics = []
        # tenSoftmaxs = []
        # for i in range(len(frames)):
        #     tenMetrics.append(torch.nn.functional.l1_loss(input=frames[i], target=backwarp(tenInput=target_frame, tenFlow=flows[i]), reduction='none').mean(1, True))
        #     tenSoftmaxs.append(softsplat.FunctionSoftsplat(tenInput=features[i], tenFlow=flows[i], tenMetric=self.beta * tenMetrics[i], strType=self.splatting_type))

        # # backward warping
        # tenSoftmaxs = []
        # for i in range(len(frames)):
        #     # tenSoftmaxs.append(backwarp(tenInput=features[i], tenFlow=flows[i]))
        #     tenSoftmaxs.append(tenWarped)

        # # TODO: mask of backward warping
        # if self.args.pooling_with_mask > 0 or self.args.decoder_with_mask > 0 or self.args.softargmax_with_mask > 0:
        #     def length_sq(x):
        #         return torch.sum(x**2, dim=1, keepdim=True)
        #     tenMasks = []
        #     for i in range(len(frames)):
        #         mag_sq = length_sq(forward_flows[i]) + length_sq(flows[i])
        #         flow_fw_warped = backwarp(forward_flows[i], flows[i])
        #         flow_diff_bw = flows[i] + flow_fw_warped
        #         occ_thresh = 0.01 * mag_sq + 0.5
        #         fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).float()
        #         if self.args.mask_with_proxy_mask > 0:
        #             tenMasks.append(((1.0 - fb_occ_bw)*torch.mean(proxy_mask, dim=1, keepdim=True)).detach())
        #         else:
        #             tenMasks.append((1.0-fb_occ_bw).detach())
        #         # imwrite(fb_occ_bw, 'mask'+str(i)+'.png', range=(0, 1))
        #         # imwrite(frames[i], 'frame'+str(i)+'.png', range=(0, 1))

        # # mask of forward warping
        # if self.args.pooling_with_mask > 0 or self.args.decoder_with_mask > 0 or self.args.softargmax_with_mask > 0:
        #     tenOnes = torch.ones([self.args.batch_size, 1, int(list(frames[0].size())[2]), int(list(frames[0].size())[3])]).cuda().detach()
        #     tenMasks = []
        #     for i in range(len(frames)):
        #         tenMasks.append(softsplat.FunctionSoftsplat(tenInput=tenOnes, tenFlow=flows[i], tenMetric=self.beta * tenMetrics[i], strType=self.splatting_type))

        """Pooling"""
        if self.args.pooling_with_mask <= 0:
            global_average_pooled_feature = torch.stack(tenWarpedFeat, -1).mean(-1)
        else:
            if self.args.pooling_with_center_bias <= 0:
                tmp_list = []
                for i in range(len(frames)):
                    tmp_list.append(tenWarpedFeat[i] * tenWarpedMask[i])
                global_summed_feature = torch.stack(tmp_list, -1).sum(-1)
                global_summed_weights = torch.stack(tenWarpedMask, -1).sum(-1)
                global_average_pooled_feature = global_summed_feature / torch.clamp(global_summed_weights, min=1e-6)
            else:
                if self.args.pooling_type == 'gaussian':
                    # gaussian weights for center bias pooling
                    # GAUSSIAN_FILTER_KSIZE = len(frames)
                    # gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)
                    color_tensor = []
                    weight_tensor = []
                    for i in range(len(frames)):
                        color_tensor.append(tenWarpedFeat[i])
                        weight_tensor.append(tenWarpedMask[i] * gaussian_filter[i, 0])
                    color_tensor = torch.stack(color_tensor, 0)
                    weight_tensor = torch.stack(weight_tensor, 0)
                    output_mask = torch.sum(weight_tensor, dim=0)
                    global_average_pooled_feature = torch.sum(color_tensor * weight_tensor, dim=0) / torch.clamp(torch.sum(weight_tensor, dim=0), min=1e-6)
                elif self.args.pooling_type == 'max':
                    def score_max(x, dim, score):
                        _tmp = [1] * len(x.size())
                        _tmp[dim] = x.size(dim)
                        _tmp[2] = x.size(2)  # channel dim
                        return torch.gather(x, dim, score.max(dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim, 0)

                    # GAUSSIAN_FILTER_KSIZE = len(frames)
                    # gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)
                    color_tensor = []
                    weight_tensor = []
                    for i in range(len(frames)):
                        color_tensor.append(tenWarpedFeat[i])
                        weight_tensor.append(tenWarpedMask[i] * gaussian_filter[i, 0])
                    color_tensor = torch.stack(color_tensor, 0)
                    weight_tensor = torch.stack(weight_tensor, 0)
                    output_mask = torch.sum(weight_tensor, dim=0)
                    global_average_pooled_feature = score_max(color_tensor, 0, weight_tensor)
                elif self.args.pooling_type == 'CNN':
                    # TODO: deep blend
                    color_tensor = []
                    for i in range(len(frames)):
                        color_tensor.append(tenWarpedFeat[i])
                    color_tensor = torch.stack(color_tensor, 0)
                    weight_tensor = []
                    if self.args.inference_with_frame_selection > 0:
                        middle_idx = np.argmax(np.array(gaussian_filter))
                    else:
                        middle_idx = len(frames) // 2
                    for i in range(len(frames)):
                        weight_tensor.append(self.aggregationWeightingNetwork(torch.cat([tenWarpedFeat[i], tenWarpedMask[i], tenWarpedFeat[middle_idx], tenWarpedMask[middle_idx]], 1)))
                    weight_tensor = torch.stack(weight_tensor, 0)
                    if self.args.gumbel > 0:
                        weight_tensor = gumbel_softmax(weight_tensor, hard=True, dim=0)
                    else:
                        weight_tensor = torch.softmax(weight_tensor, 0)
                    global_average_pooled_feature = torch.sum(color_tensor * weight_tensor, dim=0)
                elif self.args.pooling_type == 'CNN_flowError':
                    # consistency
                    def length_sq(x):
                        return torch.sum(x ** 2, dim=1, keepdim=True)

                    flow_errors = []
                    for i in range(len(frames)):
                        flow_fw_warped = backwarp(F_n_to_k_s[i], F_k_to_n_s[i])
                        flow_diff_bw = F_k_to_n_s[i] + flow_fw_warped
                        if self.args.bundle_forward_flow > 0:
                            flow_errors.append(softsplat.FunctionSoftsplat(tenInput=length_sq(flow_diff_bw), tenFlow=F_kprime_to_k, tenMetric=None, strType='average'))
                        else:
                            flow_errors.append(backwarp(tenInput=length_sq(flow_diff_bw), tenFlow=F_kprime_to_k))
                        # flow_errors.append(length_sq(flow_diff_bw))
                    # TODO: deep blend
                    color_tensor = []
                    for i in range(len(frames)):
                        color_tensor.append(tenWarpedFeat[i])
                    color_tensor = torch.stack(color_tensor, 0)
                    weight_tensor = []
                    if self.args.inference_with_frame_selection > 0:
                        middle_idx = np.argmax(np.array(gaussian_filter))
                    else:
                        middle_idx = len(frames) // 2
                    for i in range(len(frames)):
                        if self.args.FOV_expansion > 0:
                            flow_errors[i] = torch.nn.ReplicationPad2d((W, W, H, H))(flow_errors[i])

                        weight_tensor.append(self.aggregationWeightingNetwork(torch.cat([tenWarpedFeat[i], tenWarpedMask[i], flow_errors[i], tenWarpedFeat[middle_idx], tenWarpedMask[middle_idx]], 1)))
                    weight_tensor = torch.stack(weight_tensor, 0)
                    if self.args.gumbel > 0:
                        weight_tensor = gumbel_softmax(weight_tensor, hard=True, dim=0)
                    else:
                        weight_tensor = torch.softmax(weight_tensor, 0)
                    global_average_pooled_feature = torch.sum(color_tensor * weight_tensor, dim=0)

        """decoder"""
        if self.args.no_pooling > 0 and self.args.single_decoder <= 0:
            I_preds = []
            Cs = []
            for i in range(len(frames)):
                I_pred, C = self.refinementNetwork(torch.cat([tenWarpedFeat[i], tenWarpedMask[i]], 1))
                I_preds.append(I_pred)
                Cs.append(C)
        elif self.args.no_pooling <= 0 and self.args.single_decoder > 0:
            I_pred, C = self.refinementNetwork(global_average_pooled_feature)
            if h_padded:
                I_pred = I_pred[:, :, 0:h0, :]
            if w_padded:
                I_pred = I_pred[:, :, :, 0:w0]
            return I_pred
        else:
            if self.args.decoder_with_mask <= 0:
                I_preds = []
                Cs = []
                for i in range(len(frames)):
                    I_pred, C = self.refinementNetwork(torch.cat([tenWarpedFeat[i], global_average_pooled_feature], 1))
                    I_preds.append(I_pred)
                    Cs.append(C)
            else:
                I_preds = []
                Cs = []
                for i in range(len(frames)):
                    I_pred, C = self.refinementNetwork(torch.cat([tenWarpedFeat[i], global_average_pooled_feature, tenWarpedMask[i]], 1))
                    I_preds.append(I_pred)
                    Cs.append(C)

        # TODO: residual detail transfer
        # if self.args.residual_detail_transfer > 0:
        #     # print('with residual detail transfer')
        #     tenOnes = torch.ones([self.args.batch_size, 1, int(list(frames[0].size())[2]), int(list(frames[0].size())[3])]).cuda().detach()
        #     for i in range(len(frames)):
        #         reconstructed_frame, _ = self.refinementNetwork(torch.cat([features[i], features[i], tenOnes], 1))
        #         delta_frame = frames[i] - reconstructed_frame
        #         warped_delta_frame = backwarp(delta_frame, flows[i])
        #         # warped_delta_frame = softsplat.FunctionSoftsplat(tenInput=delta_frame, tenFlow=flows[i], tenMetric=self.beta * tenMetrics[i], strType=self.splatting_type)
        #         if self.args.residual_detail_transfer_with_mask > 0:
        #             I_preds[i] = I_preds[i] + warped_delta_frame * tenMasks[i]
        #         else:
        #             I_preds[i] = I_preds[i] + warped_delta_frame

        # center residual detail transfer (wierd, worse PNSR due to shared weights decoder)
        if self.args.center_residual_detail_transfer > 0:
            tenOnes = torch.ones([self.args.batch_size, 1, int(list(frames[0].size())[2]), int(list(frames[0].size())[3])]).cuda().detach()
            center_idx = len(frames) // 2
            reconstructed_frame, _ = self.refinementNetwork(torch.cat([features[center_idx], features[center_idx], tenOnes], 1))
            delta_frame = frames[center_idx] - reconstructed_frame
            warped_delta_frame = backwarp(delta_frame, F_kprime_to_k)
            # center residual detail transfer with mask
            I_preds[center_idx] = I_preds[center_idx] + warped_delta_frame * tenWarpedMask[center_idx]

        # all frames residual detail transfer
        else:
            if self.args.residual_detail_transfer > 0:
                tenOnes = torch.ones([self.args.batch_size, 1, int(list(frames[0].size())[2]), int(list(frames[0].size())[3])]).cuda().detach()
                for i in range(len(frames)):
                    reconstructed_frame, _ = self.refinementNetwork(torch.cat([features[i], features[i], tenOnes], 1))
                    delta_frame = frames[i] - reconstructed_frame
                    # warped_delta_frame = backwarp(delta_frame, F_kprime_to_k)
                    if self.args.all_backward > 0:
                        ref_frame_flow = torch.nn.ReplicationPad2d((W, W, H, H))(F_k_to_n_s[idx])
                        tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(delta_frame)
                        tenWarpedFirst = backwarp(tenInput=tenRef, tenFlow=ref_frame_flow)
                    else:
                        """padding for forward warping"""
                        ref_frame_flow = torch.nn.ReplicationPad2d((W, W, H, H))(F_n_to_k_s[i])
                        tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(delta_frame)
                        """first forward warping"""
                        tenWarpedFirst = softsplat.FunctionSoftsplat(tenInput=tenRef, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
                    """second backward warping"""
                    if self.args.bundle_forward_flow > 0:
                        tenWarpedSecond = softsplat.FunctionSoftsplat(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad, tenMetric=None, strType='average')
                    else:
                        tenWarpedSecond = backwarp(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad)
                    """back to original resolution"""
                    if self.args.FOV_expansion <= 0:
                        tenWarpedResidual = tenWarpedSecond[:, :, H:-H, W:-W]
                    else:
                        tenWarpedResidual = tenWarpedSecond
                    # center residual detail transfer with mask
                    # if self.args.masked_residual_detail_transfer > 0 and i != len(frames)//2:
                    #     I_preds[i] = I_preds[i] + tenWarpedResidual * torch.clamp((tenWarpedMask[i] - tenWarpedMask[len(frames)//2]), min=0.0)
                    # else:
                    I_preds[i] = I_preds[i] + tenWarpedResidual * tenWarpedMask[i]

        # blending
        # if self.training:
        if not self.args.softargmax_with_mask:
            softmax_conf = torch.softmax(torch.stack(Cs, -1), -1)
        else:
            tmp_list = []
            for i in range(len(frames)):
                tmp_list.append(Cs[i] * tenWarpedMask[i])
            softmax_conf = torch.softmax(torch.stack(tmp_list, -1), -1)

        I_pred = (I_preds[0] * softmax_conf[..., 0])
        for i in range(1, len(frames)):
            I_pred += (I_preds[i] * softmax_conf[..., i])

        if self.args.seamless > 0:
            import numpy as np
            # print(gaussian_filter)
            # print(np.argmax(np.array(gaussian_filter)))
            # print(len(gaussian_filter))
            # print(len(frames))
            # import pdb
            # pdb.set_trace()
            if self.args.inference_with_frame_selection > 0:
                middle_idx = np.argmax(np.array(gaussian_filter))
            else:
                middle_idx = len(frames) // 2


            ref_frame_flow = torch.nn.ReplicationPad2d((W, W, H, H))(F_n_to_k_s[middle_idx])
            tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(frames[middle_idx])
            """first forward warping"""
            tenWarpedFirst = softsplat.FunctionSoftsplat(tenInput=tenRef, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
            """second backward warping"""
            if self.args.bundle_forward_flow > 0:
                tenWarpedSecond = softsplat.FunctionSoftsplat(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad, tenMetric=None, strType='average')
            else:
                tenWarpedSecond = backwarp(tenInput=tenWarpedFirst, tenFlow=F_kprime_to_k_pad)
            """back to original resolution"""
            if self.args.FOV_expansion <= 0:
                tenWarpedKey = tenWarpedSecond[:, :, H:-H, W:-W]
            else:
                tenWarpedKey = tenWarpedSecond

            # """warped key frame"""
            # tenRef = torch.nn.ReplicationPad2d((W, W, H, H))(frames[len(frames)//2])
            # tenWarpedSecond = backwarp(tenInput=tenRef, tenFlow=F_kprime_to_k_pad)
            # """back to original resolution"""
            # if self.args.FOV_expansion <= 0:
            #     tenWarpedKey = tenWarpedSecond[:, :, H:-H, W:-W]
            # else:
            #     tenWarpedKey = tenWarpedSecond
            npWarpedKey = np.transpose(np.clip(tenWarpedKey.cpu().numpy(), 0.0, 1.0)[0, ::-1], (1, 2, 0))
            np_I_pred = np.transpose(np.clip(I_pred.cpu().numpy(), 0.0, 1.0)[0, ::-1], (1, 2, 0))
            np_mask_A = np.clip(tenWarpedMask[middle_idx].cpu().numpy(), 0.0, 1.0)[0, 0, :, :]

            np_mask_B = np.clip(1.0 - tenWarpedMask[middle_idx].cpu().numpy(), 0.0, 1.0)[0, 0, :, :]

            # """np_mask[0] = 0
            # np_mask[-1] = 0
            # np_mask[:, 0] = 0
            # np_mask[:, -1] = 0"""
            #
            # """erosion_size = 10
            # element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
            #                            (erosion_size, erosion_size))
            #
            # np_mask_dst = cv2.erode(np_mask, element)
            # cv2.imwrite('np_mask_dst.png', np_mask_dst)"""
            #
            # cv2.imwrite('np_mask.png', np_mask)
            # cv2.imwrite('np_I_pred.png', np_I_pred)
            # cv2.imwrite('npWarpedKey.png', npWarpedKey)
            #
            # normal_clone = cv2.seamlessClone(np_I_pred, npWarpedKey, np_mask, (np_mask.shape[1] // 2, np_mask.shape[0] // 2), cv2.NORMAL_CLONE)
            # I_pred = torch.from_numpy(np.expand_dims(np.transpose(normal_clone[:, :, ::-1], (2, 0, 1)), 0).astype(np.float32) / 255.0).cuda()

            def GaussianPyramid(img, leveln):
                GP = [img]
                for i in range(leveln - 1):
                    GP.append(cv2.pyrDown(GP[i]))
                return GP

            def LaplacianPyramid(img, leveln):
                LP = []
                for i in range(leveln - 1):
                    next_img = cv2.pyrDown(img)
                    LP.append(img - cv2.pyrUp(next_img, img.shape[1::-1]))
                    img = next_img
                LP.append(img)
                return LP

            def blend_pyramid(LPA, LPB, MA, MB):
                blended = []
                for i, M in enumerate(MA):
                    if len(list(MA[i].shape)) < 3 and len(list(MB[i].shape)) < 3:
                        blended.append(LPA[i] * np.expand_dims(MA[i], -1) + LPB[i] * np.expand_dims(MB[i], -1))
                    else:
                        blended.append(LPA[i] * MA[i] + LPB[i] * MB[i])
                return blended

            def reconstruct(LS):
                img = LS[-1]
                for lev_img in LS[-2::-1]:
                    img = cv2.pyrUp(img, lev_img.shape[1::-1])
                    img += lev_img
                return img

            minimum_multi_band = 32
            pad_h_multi_band = 0
            pad_b_multi_band = 0
            if np_mask_A.shape[0] % minimum_multi_band != 0:
                pad_h_multi_band = (minimum_multi_band - (np_mask_A.shape[0] % minimum_multi_band)) // 2
                pad_b_multi_band = minimum_multi_band - (np_mask_A.shape[0] % minimum_multi_band) - pad_h_multi_band
                print(pad_h_multi_band)
                print(pad_b_multi_band)
                np_mask_A = cv2.copyMakeBorder(np_mask_A, pad_h_multi_band, pad_b_multi_band, 0, 0, cv2.BORDER_REPLICATE)
                np_mask_B = cv2.copyMakeBorder(np_mask_B, pad_h_multi_band, pad_b_multi_band, 0, 0, cv2.BORDER_REPLICATE)
                npWarpedKey = cv2.copyMakeBorder(npWarpedKey, pad_h_multi_band, pad_b_multi_band, 0, 0, cv2.BORDER_REPLICATE)
                np_I_pred = cv2.copyMakeBorder(np_I_pred, pad_h_multi_band, pad_b_multi_band, 0, 0, cv2.BORDER_REPLICATE)

            pad_l_multi_band = 0
            pad_r_multi_band = 0
            if np_mask_A.shape[1] % minimum_multi_band != 0:
                pad_l_multi_band = (minimum_multi_band - (np_mask_A.shape[1] % minimum_multi_band)) // 2
                pad_r_multi_band = minimum_multi_band - (np_mask_A.shape[1] % minimum_multi_band) - pad_l_multi_band
                np_mask_A = cv2.copyMakeBorder(np_mask_A, 0, 0, pad_l_multi_band, pad_r_multi_band, cv2.BORDER_REPLICATE)
                np_mask_B = cv2.copyMakeBorder(np_mask_B, 0, 0, pad_l_multi_band, pad_r_multi_band, cv2.BORDER_REPLICATE)
                npWarpedKey = cv2.copyMakeBorder(npWarpedKey, 0, 0, pad_l_multi_band, pad_r_multi_band, cv2.BORDER_REPLICATE)
                np_I_pred = cv2.copyMakeBorder(np_I_pred, 0, 0, pad_l_multi_band, pad_r_multi_band, cv2.BORDER_REPLICATE)

            MA = GaussianPyramid(np_mask_A, 5)
            MB = GaussianPyramid(np_mask_B, 5)
            LPA = LaplacianPyramid(npWarpedKey, 5)
            LPB = LaplacianPyramid(np_I_pred, 5)
            # Blend two Laplacian pyramidspass
            blended = blend_pyramid(LPA, LPB, MA, MB)
            # Reconstruction process
            frame_A = reconstruct(blended)
            frame_A[frame_A > 1.0] = 1.0
            frame_A[frame_A < 0.0] = 0.0
            # print(frame_A.shape)

            if pad_h_multi_band != 0 or pad_b_multi_band != 0:
                frame_A = frame_A[pad_h_multi_band:-pad_b_multi_band]
            if pad_l_multi_band != 0 or pad_r_multi_band != 0:
                frame_A = frame_A[:, pad_l_multi_band:-pad_r_multi_band]

            # print(frame_A.shape)

            I_pred = torch.from_numpy(np.expand_dims(np.transpose(frame_A[:, :, ::-1], (2, 0, 1)), 0).astype(np.float32)).cuda()

            """cv2.imwrite('_np_mask_A.png', np.round(np_mask_A*255).astype(np.uint8))
            cv2.imwrite('_np_mask_B.png', np.round(np_mask_B*255).astype(np.uint8))
            cv2.imwrite('_frame_A.png', np.round(frame_A*255).astype(np.uint8))
            cv2.imwrite('_npWarpedKey.png', np.round(npWarpedKey*255).astype(np.uint8))
            cv2.imwrite('_np_I_pred.png', np.round(np_I_pred*255).astype(np.uint8))"""

        if h_padded:
            I_pred = I_pred[:, :, 0:h0, :]
        if w_padded:
            I_pred = I_pred[:, :, :, 0:w0]
        return I_pred
