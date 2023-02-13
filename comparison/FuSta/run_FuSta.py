import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable

import softsplat

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import torch.nn.functional as F

import models_arbitrary

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def load_image_list(image_files):
    images = []
    for imfile in image_files:
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

def calc_flow(img1, img2):
    with torch.no_grad():

        images = load_image_list([img1, img2])

        flow_low, flow_up = flow_model(images[0, None], images[1, None], iters=20, test_mode=True)
    return flow_up.detach()

backwarp_tenGrid = {}
backwarp_tenPartial = {}


backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end



def read_homography(H_path):
    xv, yv = np.meshgrid(np.linspace(0, 832 + 2 * 64 - 1, 832 + 2 * 64), np.linspace(0, 448 + 2 * 64 - 1, 448 + 2 * 64))
    H_inv = np.load(H_path)
    if np.sum(np.abs(H_inv)) == 0.0:
        H_inv[0, 0] = 1.0
        H_inv[1, 1] = 1.0
        H_inv[2, 2] = 1.0
    xv_prime = (H_inv[0, 0] * xv + H_inv[0, 1] * yv + H_inv[0, 2]) / (H_inv[2, 0] * xv + H_inv[2, 1] * yv + H_inv[2, 2])
    yv_prime = (H_inv[1, 0] * xv + H_inv[1, 1] * yv + H_inv[1, 2]) / (H_inv[2, 0] * xv + H_inv[2, 1] * yv + H_inv[2, 2])
    flow = np.stack((xv_prime - xv, yv_prime - yv), -1)
    return flow
    
def read_flo(flo_path):
    print(flo_path)
    xv, yv = np.meshgrid(np.linspace(-1, 1, 832 + 2 * 64), np.linspace(-1, 1, 448 + 2 * 64))
    flow = np.load(flo_path)
    flow_u = ((flow[:, :, 0] + xv) + 1.0) / 2.0 * float(832+2*64-1)
    flow_v = ((flow[:, :, 1] + yv) + 1.0) / 2.0 * float(448+2*64-1)
    flow_u -= ((xv + 1.0) / 2.0 * float(832+2*64-1))
    flow_v -= ((yv + 1.0) / 2.0 * float(448+2*64-1))
    flow = np.stack((flow_u, flow_v), -1)
    return flow


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AdaCoF-Pytorch')
    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='adacofnet')

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)

    # Directory Setting
    # parser.add_argument('--train', type=str, default='../VideoStabilization/Adobe240/DeepVideoDeblurring_Dataset/DeepVideoDeblurring_Dataset/quantitative_datasets')
    # parser.add_argument('--out_dir', type=str, default='./output_adacof_train')
    #parser.add_argument('--load', type=str, default='output/checkpoint/model_epoch042.pth')
    parser.add_argument('--load', type=str, default='FuSta_model/checkpoint/model_epoch050.pth')
    #parser.add_argument('--load', type=str, default='output_pooling_with_mask_decoder_with_mask_softargmax_with_mask/checkpoint/model_epoch049.pth')
    # parser.add_argument('--test_input', type=str, default='../VideoStabilization/Adobe240/DeepVideoDeblurring_Dataset/DeepVideoDeblurring_Dataset/quantitative_datasets')
    # parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

    # Learning Options
    # parser.add_argument('--epochs', type=int, default=50, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    # parser.add_argument('--loss', type=str, default='1*VGG', help='loss function configuration')
    # parser.add_argument('--patch_size_h', type=int, default=256, help='Patch size')
    # parser.add_argument('--patch_size_w', type=int, default=256, help='Patch size')

    # parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    # Options for AdaCoF
    # parser.add_argument('--kernel_size', type=int, default=5)
    # parser.add_argument('--dilation', type=int, default=1)

    # Options for network
    parser.add_argument('--pooling_with_mask', type=int, default=1)
    parser.add_argument('--decoder_with_mask', type=int, default=1)
    parser.add_argument('--softargmax_with_mask', type=int, default=0)
    parser.add_argument('--decoder_with_gated_conv', type=int, default=1)
    parser.add_argument('--residual_detail_transfer', type=int, default=1)
    parser.add_argument('--beta_learnable', type=int, default=0)
    parser.add_argument('--splatting_type', type=str, default='softmax')
    # parser.add_argument('--residual_detail_transfer_with_mask', type=int, default=0)
    # parser.add_argument('--mask_with_proxy_mask', type=int, default=0)
    # parser.add_argument('--max_proxy', type=int, default=0)
    parser.add_argument('--concat_proxy', type=int, default=0)
    parser.add_argument('--center_residual_detail_transfer', type=int, default=0)
    parser.add_argument('--pooling_with_center_bias', type=int, default=1)
    parser.add_argument('--pooling_type', type=str, default='CNN_flowError')
    parser.add_argument('--no_pooling', type=int, default=0)
    parser.add_argument('--single_decoder', type=int, default=0)
    parser.add_argument('--noDL_CNNAggregation', type=int, default=0)
    parser.add_argument('--gumbel', type=int, default=0)
    parser.add_argument('--inference_with_frame_selection', type=int, default=0)
    parser.add_argument('--FOV_expansion', type=int, default=1) 
    parser.add_argument('--seamless', type=int, default=1) 
    parser.add_argument('--all_backward', type=int, default=0) 
    parser.add_argument('--bundle_forward_flow', type=int, default=0) 
    
    parser.add_argument('--input_frames_path', type=str)
    parser.add_argument('--warping_field_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--temporal_width', type=int, default=41)
    parser.add_argument('--temporal_step', type=int, default=4)

    args = parser.parse_args()

    model = models_arbitrary.Model(args)

    checkpoint = torch.load(args.load)
    model.load(checkpoint['state_dict'])

    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    
    # RAFT
    flow_model = torch.nn.DataParallel(RAFT(args))
    flow_model.load_state_dict(torch.load('raft_models/raft-things.pth'))
    flow_model = flow_model.module
    flow_model.to('cuda')
    flow_model.eval()

    INPUT_FRAMES_PATH = args.input_frames_path
    CVPR2020_warping_field_path = args.warping_field_path
    OUTPUT_PATH = args.output_path

    GAUSSIAN_FILTER_KSIZE = args.temporal_width
    gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)

    assert (GAUSSIAN_FILTER_KSIZE-1)//2 % args.temporal_step == 0
    
    with torch.no_grad():
    
    
    
        if not os.path.exists(os.path.join(OUTPUT_PATH)):
            os.makedirs(os.path.join(OUTPUT_PATH))

        all_imgs = sorted(glob.glob(os.path.join(INPUT_FRAMES_PATH, '*.jpg')))  # all jpgs in a sequence
        
        tmp_img = cv2.imread(all_imgs[0])
        H = tmp_img.shape[0]
        W = tmp_img.shape[1]

        # temporal padding frames for Gaussian filter
        original_length = len(all_imgs)
        assert original_length > 0

        first_frame = all_imgs[0]
        last_frame = all_imgs[-1]
        all_imgs = [first_frame]*(GAUSSIAN_FILTER_KSIZE//2) + all_imgs + [last_frame]*(GAUSSIAN_FILTER_KSIZE//2)

        large_mask_chain = []

        # delta_x_y = torch.tensor(torch.zeros(original_length, 2), requires_grad=True)
        output_frames = []
        
        for idx in range(GAUSSIAN_FILTER_KSIZE//2, (GAUSSIAN_FILTER_KSIZE//2)+original_length):
            keyframe = all_imgs[idx]
            img_name = os.path.split(keyframe)[-1]
            print(img_name)
            tenSecond = torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename=keyframe, flags=-1)[..., ::-1].transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
            
            

            # if '00196.png' != img_name:
            #     continue

            """smoothed_flow_list = []
            for frame_shift in range(-(GAUSSIAN_FILTER_KSIZE//2), (GAUSSIAN_FILTER_KSIZE//2)+1):
                ref_frame = all_imgs[idx + frame_shift]
                ref_frame_name = os.path.split(ref_frame)[-1]
                # ref_frame_flow_online = calc_flow(keyframe, ref_frame)  # [H, W, 2]
                ref_frame_flow = np.load(os.path.join(pre_calculated_flow_path, category_name, avi_name, img_name[:-4]+'_'+ref_frame_name[:-4]+'.npy'))
                
                print(ref_frame_flow.shape)
                ref_frame_flow = torch.FloatTensor(np.ascontiguousarray(ref_frame_flow.astype(np.float32))).cuda()
                smoothed_flow_list.append(ref_frame_flow * gaussian_filter[frame_shift + GAUSSIAN_FILTER_KSIZE//2, 0])
            smoothed_flow = torch.sum(torch.stack(smoothed_flow_list, dim=0), dim=0, keepdim=False)
            print(torch.mean(smoothed_flow))"""
            if int(img_name[:-4]) == 0:
                tenH_inv = torch.zeros((1, 2, 448 + 2 * 64, 832 + 2 * 64)).cuda()
                tenFlow = torch.zeros((1, 2, 448 + 2 * 64, 832 + 2 * 64)).cuda()
            else:
                if os.path.isfile(os.path.join(CVPR2020_warping_field_path, str(int(img_name[:-4])-1).zfill(5)+'_H_inv.npy')) and os.path.isfile(os.path.join(CVPR2020_warping_field_path, str(int(img_name[:-4])-1).zfill(5)+'.npy')):
                    tenH_inv = torch.FloatTensor(np.ascontiguousarray(read_homography(os.path.join(CVPR2020_warping_field_path, str(int(img_name[:-4])-1).zfill(5)+'_H_inv.npy')).transpose(2, 0, 1)[None, :, :, :])).cuda()
                    if int(img_name[:-4]) == 1:
                        tenFlow = torch.zeros((1, 2, 448 + 2 * 64, 832 + 2 * 64)).cuda()
                    else:
                        tenFlow = torch.FloatTensor(np.ascontiguousarray(read_flo(os.path.join(CVPR2020_warping_field_path, str(int(img_name[:-4])-1).zfill(5)+'.npy')).transpose(2, 0, 1)[None, :, :, :])).cuda()
                else:
                    print('no flow data')
                    continue
                    
            
            """calculate backward flow using inv_H and backward_flow"""
            tenBackFlow = backwarp(tenInput=tenH_inv, tenFlow=tenFlow)
            totalFlowIn832 = (tenBackFlow+tenFlow)[:, :, 64:-64, 64:-64]
            """second backward warping in full resolution"""
            W_ratio = W/(832)
            H_ratio = H/(448)
            totalFlow = F.upsample(totalFlowIn832, size=(H, W), mode='bilinear')
            F_kprime_to_k = torch.stack((totalFlow[:, 0]*W_ratio, totalFlow[:, 1]*H_ratio), dim=1)
            


            sum_color = []
            sum_alpha = []
            input_frames = []
            input_flows = []
            forward_flows = []
            backward_flows = []
            for frame_shift in range(-(GAUSSIAN_FILTER_KSIZE // 2), (GAUSSIAN_FILTER_KSIZE // 2) + 1, int(args.temporal_step)):
                # for frame_shift in [-5, 0, 5]:
                ref_frame = all_imgs[idx + frame_shift]
                ref_frame_name = os.path.split(ref_frame)[-1]
                forward_flow = calc_flow(ref_frame, keyframe)
                # forward_flow = np.load(os.path.join(pre_calculated_flow_path, avi_name, ref_frame_name[:-4] + '_' + img_name[:-4] + '.npy'))
                # forward_flow = torch.FloatTensor(np.ascontiguousarray(forward_flow.astype(np.float32))).cuda()
                
                # somtimes flow encounters nan or very large values
                """forward_flow[forward_flow != forward_flow] = 0
                forward_flow[forward_flow > 448] = 0
                forward_flow[forward_flow < (-448)] = 0"""
                # cut off padding pixels done by RAFT
                flow_H = list(forward_flow.size())[2]
                flow_W = list(forward_flow.size())[3]
                if H != flow_H:
                    top = (flow_H - H) // 2
                    forward_flow = forward_flow[:, :, top:top+H]
                if W != flow_W:
                    left = (flow_W - W) // 2
                    forward_flow = forward_flow[:, :, :, left:left+W]
                print(forward_flow.shape)
                forward_flows.append(forward_flow)
                # forward_flow += smoothed_flow
                # input_flows.append(forward_flow)
                    
                backward_flow = calc_flow(keyframe, ref_frame)
                # backward_flow = np.load(os.path.join(pre_calculated_flow_path, avi_name, img_name[:-4] + '_' + ref_frame_name[:-4] + '.npy'))
                # backward_flow = torch.FloatTensor(np.ascontiguousarray(backward_flow.astype(np.float32))).cuda()
                """backward_flow[backward_flow != backward_flow] = 0
                backward_flow[backward_flow > 448] = 0
                backward_flow[backward_flow < (-448)] = 0"""
                backward_flows.append(backward_flow)
                input_frames.append(torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename=ref_frame, flags=-1)[..., ::-1].transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda())
                
            if H % 4 == 0:
                boundary_cropping_h = 4
            else:
                boundary_cropping_h = 3
            if W % 4 == 0:
                boundary_cropping_w = 4
            else:
                boundary_cropping_w = 3
            input_frames = [x[:, :, boundary_cropping_h:-boundary_cropping_h, boundary_cropping_w:-boundary_cropping_w] for x in input_frames]
            F_kprime_to_k = F_kprime_to_k[:, :, boundary_cropping_h:-boundary_cropping_h, boundary_cropping_w:-boundary_cropping_w]
            forward_flows = [x[:, :, boundary_cropping_h:-boundary_cropping_h, boundary_cropping_w:-boundary_cropping_w] for x in forward_flows]
            backward_flows = [x[:, :, boundary_cropping_h:-boundary_cropping_h, boundary_cropping_w:-boundary_cropping_w] for x in backward_flows]
                
            frame_out = model(input_frames, F_kprime_to_k, forward_flows, backward_flows)
            """output_frames.append(frame_out.detach().cpu())"""
            # if OOM          
            if not os.path.exists('tmp/'):
                os.makedirs('tmp/')                
            np.save('tmp/'+str(len(large_mask_chain)).zfill(5), frame_out.detach().cpu().numpy())
            output_frames.append('tmp/'+str(len(large_mask_chain)).zfill(5)+'.npy')

            """blending methods"""
            WWW = 256
            HHH = 256
            tenOnes = torch.ones_like(input_frames[0])[:, 0:1, :, :]
            tenOnes = torch.nn.ZeroPad2d((WWW, WWW, HHH, HHH))(tenOnes).detach()
            F_kprime_to_k_pad = torch.nn.ReplicationPad2d((WWW, WWW, HHH, HHH))(F_kprime_to_k)
            tenWarpedFeat = []
            tenWarpedMask = []
            for iii, feat in enumerate(input_frames):
                """padding for forward warping"""
                ref_frame_flow = torch.nn.ReplicationPad2d((WWW, WWW, HHH, HHH))(forward_flows[iii])
                """first forward warping"""
                tenMaskFirst = softsplat.FunctionSoftsplat(tenInput=tenOnes, tenFlow=ref_frame_flow, tenMetric=None, strType='average')
                """second backward warping"""
                tenMaskSecond = backwarp(tenInput=tenMaskFirst, tenFlow=F_kprime_to_k_pad)
                """back to original resolution"""
                tenMask = tenMaskSecond
                tenWarpedMask.append(tenMask)
            weight_tensor = torch.stack(tenWarpedMask, 0)
            output_mask = torch.sum(weight_tensor, dim=0)
            output_mask = torch.clamp(output_mask, max=1.0)
            # imwrite(output_mask, str(idx-GAUSSIAN_FILTER_KSIZE//2).zfill(5)+'_mask.png', range=(0, 1))
            
            large_mask_chain.append(output_mask.detach().cpu())

            
            # imwrite(frame_out, os.path.join(OUTPUT_PATH, avi_name, img_name), range=(0, 1))
            
        """loss funstions"""
        """learning_rate = 1e-1
        optimizer = torch.optim.RMSprop([delta_x_y], lr=learning_rate)
        
        f = open('loss.csv', 'w+')
        for step in range(2000):
            data_term = 0.0
            fidelity_term = 0.0
            print(delta_x_y.detach().numpy())
            for iiii in range(len(large_mask_chain)):
                expanded_flow = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(delta_x_y[iiii], 0), 2), 3)
                expanded_flow = expanded_flow.repeat(1, 1, list(large_mask_chain[iiii].size())[2], list(large_mask_chain[iiii].size())[3])
                # expanded_flow = torch.tile(expanded_flow, (1, 1, list(large_mask_chain[iiii].size())[2], list(large_mask_chain[iiii].size())[3]))
                cropped_mask = backwarp(tenInput=large_mask_chain[iiii], tenFlow=expanded_flow.cuda())[:, :, HHH:-HHH, WWW:-WWW]
                summed_mask = torch.sum(1.0 - cropped_mask)
                data_term += summed_mask
                
                fidelity_term += torch.sum(torch.abs(delta_x_y[iiii]))
            smoothness_term = torch.sum(torch.abs(delta_x_y[1:] - delta_x_y[:-1]))
            loss = data_term + 1000*fidelity_term + 1000*smoothness_term
            f.write(str(data_term)+','+str(fidelity_term)+','+str(smoothness_term)+'\n')
            print(data_term)
            print(fidelity_term)
            print(smoothness_term)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f.close()"""
        
        WWW -= boundary_cropping_w
        HHH -= boundary_cropping_h
            
        from maxflow.fastmin import aexpansion_grid
        accumulated_motion_vectors = np.zeros((original_length, 2), np.int32)
        maximum_movement = 3
        for level in range(5, -1, -1):
            # data term / 2^3 level
            data_term = np.zeros((original_length, (2*maximum_movement+1) * (2*maximum_movement+1)))
            print('data term')
            for iiii in range(len(large_mask_chain)):
                print(iiii)
                for uu in range(-maximum_movement, maximum_movement+1):
                    print(uu)
                    for vv in range(-maximum_movement, maximum_movement+1):
                        # converage term
                        motion_vector_u = int(accumulated_motion_vectors[iiii, 0]+uu*(2**level))
                        motion_vector_v = int(accumulated_motion_vectors[iiii, 1]+vv*(2**level))
                        
                        #expanded_flow = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.array([accumulated_motion_vectors[iiii, 0]+uu*(2**level), accumulated_motion_vectors[iiii, 1]+vv*(2**level)])), 0), 2), 3)
                        #expanded_flow = expanded_flow.repeat(1, 1, list(large_mask_chain[iiii].size())[2], list(large_mask_chain[iiii].size())[3])
                        #cropped_mask = backwarp(tenInput=large_mask_chain[iiii].double(), tenFlow=expanded_flow.double().cuda())[:, :, HHH:-HHH, WWW:-WWW]
                        cropped_mask = large_mask_chain[iiii][:, :, HHH+motion_vector_u:-HHH+motion_vector_u, WWW+motion_vector_v:-WWW+motion_vector_v]
                        
                        # cropped_mask = backwarp(tenInput=large_mask_chain[iiii], tenFlow=expanded_flow.cuda())
                        # imwrite(cropped_mask, str(uu)+'_'+str(vv)+'_mask.png', range=(0, 1))
                        summed_mask = (torch.sum(1.0 - cropped_mask)).cpu().numpy()
                        
                        # fidelity term
                        # data_term[iiii, (uu+maximum_movement)*(2*maximum_movement+1)+vv+maximum_movement] = 10.0*(np.abs(uu) + np.abs(vv))*(2**level) + summed_mask
                        data_term[iiii, (uu+maximum_movement)*(2*maximum_movement+1)+vv+maximum_movement] = summed_mask
                        
            
            print('smoothness term')
            smoothness_term = np.zeros(((2*maximum_movement+1) * (2*maximum_movement+1), (2*maximum_movement+1) * (2*maximum_movement+1)))
            for uu in range(-maximum_movement, maximum_movement+1):
                print(uu)
                for vv in range(-maximum_movement, maximum_movement+1):
                    print(vv)
                    for uuuu in range(-maximum_movement, maximum_movement+1):
                        for vvvv in range(-maximum_movement, maximum_movement+1):
                            smoothness_term[(uu+maximum_movement)*(2*maximum_movement+1)+vv+maximum_movement, (uuuu+maximum_movement)*(2*maximum_movement+1)+vvvv+maximum_movement] = (((uu-uuuu)*(2**level))**2 + ((vv-vvvv)*(2**level))**2)
            
            
            
            alpha = 100.0
            labels = aexpansion_grid(data_term,smoothness_term*alpha)  #  [H, W]
            for iiii in range(labels.shape[0]):
                accumulated_motion_vectors[iiii, 0] += (labels[iiii]//(2*maximum_movement+1) - maximum_movement)*(2**level)
                accumulated_motion_vectors[iiii, 1] += (labels[iiii]%(2*maximum_movement+1) - maximum_movement)*(2**level)
            print(accumulated_motion_vectors)
        print(accumulated_motion_vectors)
        np.savetxt("motion_vector.csv", accumulated_motion_vectors, delimiter=",")
        
        loss = 0.0
        for iiii in range(len(large_mask_chain)):
            motion_vector_u = int(accumulated_motion_vectors[iiii, 0])
            motion_vector_v = int(accumulated_motion_vectors[iiii, 1])
            cropped_mask = large_mask_chain[iiii][:, :, HHH+motion_vector_u:-HHH+motion_vector_u, WWW+motion_vector_v:-WWW+motion_vector_v]
            print(motion_vector_u)
            print(motion_vector_v)
            print(cropped_mask.shape)
            """imwrite(output_frames[iiii][:, :, HHH+motion_vector_u:-HHH+motion_vector_u, WWW+motion_vector_v:-WWW+motion_vector_v], os.path.join(OUTPUT_PATH, avi_name, str(iiii+1).zfill(5)+'.png'), range=(0, 1))"""
            # if OOM
            imwrite(torch.from_numpy(np.load(output_frames[iiii]))[:, :, HHH+motion_vector_u:-HHH+motion_vector_u, WWW+motion_vector_v:-WWW+motion_vector_v], os.path.join(OUTPUT_PATH, str(iiii+1).zfill(5)+'.png'), range=(0, 1))
            
            summed_mask = (torch.sum(1.0 - cropped_mask)).cpu().numpy()
            loss += summed_mask
        print(loss)
        
        # loss without adjustment
        loss = 0.0
        for iiii in range(len(large_mask_chain)):
            cropped_mask = large_mask_chain[iiii][:, :, HHH:-HHH, WWW:-WWW]
            summed_mask = (torch.sum(1.0 - cropped_mask)).cpu().numpy()
            loss += summed_mask
        print(loss)


